import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime

# --- 配置区 ---
DATA_DIR = 'stock_data'
OUTPUT_DIR = 'results/online_yin_final'
NAMES_FILE = 'stock_names.csv'

def get_indicators(df):
    df = df.copy()
    # 确保日期升序，这是所有指标计算的基础
    if '日期' in df.columns:
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values(by='日期')
        
    # 1. 核心均线系统
    for m in [5, 10, 20, 60]:
        df[f'ma{m}'] = df['收盘'].rolling(m).mean()
    
    # 2. 通达信标准 MACD 计算
    ema12 = df['收盘'].ewm(span=12, adjust=False).mean()
    ema26 = df['收盘'].ewm(span=26, adjust=False).mean()
    df['dif'] = ema12 - ema26
    df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
    df['macd'] = (df['dif'] - df['dea']) * 2
    
    # 3. 基础动态指标
    df['ma10_up'] = df['ma10'] > df['ma10'].shift(1)
    df['ma20_up'] = df['ma20'] > df['ma20'].shift(1)
    df['v_ma5'] = df['成交量'].rolling(5).mean()
    df['change'] = df['收盘'].pct_change() * 100
    return df

def check_logic(df):
    if len(df) < 60: return None, None
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    # --- 维度 A: 基础过滤 (价格与流动性) ---
    if not (5.0 <= curr['收盘'] <= 35.0) or curr['成交额'] < 800000000:
        return None, None

    # --- 维度 B: MACD 趋势过滤 (通达信买入策略) ---
    # DIF必须在DEA上方（多头区域），且MACD柱子不能太难看
    if not (curr['dif'] > curr['dea'] and curr['macd'] > -0.1):
        return None, None

    # --- 维度 C: 位置过滤 (成功的关键1：拒绝高位，寻找起步) ---
    # 1. 乖离率限制：股价距离 MA20 不能超过 12%，防止在高位回踩时接盘
    if curr['收盘'] > curr['ma20'] * 1.12:
        return None, None
    # 2. 均线斜率：20日线必须是支撑向上状态，确保不是在阴跌
    if not curr['ma20_up']:
        return None, None

    # --- 维度 D: 强势基因与量能断层 (成功的关键2：主力未逃) ---
    recent_15 = df.tail(15)
    # 1. 寻找最近的大阳线（涨幅>7%）
    strong_days = recent_15[recent_15['change'] > 7.0]
    if strong_days.empty:
        return None, None
    
    # 2. 量能断层判定：回踩成交量必须小于最近那根大阳线成交量的 55%
    last_strong_vol = strong_days.iloc[-1]['成交量']
    if curr['成交量'] > last_strong_vol * 0.55:
        return None, None

    # --- 维度 E: 贴线阴线判定 ---
    is_yin = curr['收盘'] < curr['开盘'] or curr['change'] <= 0
    # 极度缩量：成交量小于5日均量的 65% (按你要求的阈值)
    is_shrink = curr['成交量'] < (curr['v_ma5'] * 0.65)
    
    # 贴线精度：收盘价与 MA5 或 MA10 的距离在 1.0% 以内
    bias_m5 = abs(curr['收盘'] - curr['ma5']) / curr['ma5']
    bias_m10 = abs(curr['收盘'] - curr['ma10']) / curr['ma10']
    
    support_ma_key = None
    if bias_m10 <= 0.01:
        support_ma_key = 'MA10'
    elif bias_m5 <= 0.01:
        support_ma_key = 'MA5'
    
    if is_yin and is_shrink and support_ma_key:
        return f"回踩{support_ma_key}极缩阴", support_ma_key
    
    return None, None

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    name_map = {}
    if os.path.exists(NAMES_FILE):
        try:
            n_df = pd.read_csv(NAMES_FILE, dtype={'code': str})
            name_map = dict(zip(n_df['code'], n_df['name']))
        except: pass

    files = glob.glob(f"{DATA_DIR}/*.csv")
    date_str = datetime.now().strftime('%Y-%m-%d')
    results = []

    for f in files:
        try:
            df = pd.read_csv(f)
            df.columns = [c.strip() for c in df.columns]
            df = get_indicators(df)
            match_type, ma_key = check_logic(df)
            
            if match_type:
                code = os.path.basename(f).replace('.csv', '')
                curr_p = df['收盘'].iloc[-1]
                ma_val = df[ma_key.lower()].iloc[-1]
                bias = round((curr_p - ma_val) / ma_val * 100, 2)
                
                results.append({
                    '日期': date_str,
                    '代码': code,
                    '名称': name_map.get(code, '未知'),
                    '当前价': round(curr_p, 2),
                    '形态类型': match_type,
                    '偏离度%': bias,
                    'MACD': round(df['macd'].iloc[-1], 3),
                    '距MA20距离%': round((curr_p/df['ma20'].iloc[-1]-1)*100, 2),
                    '成交额(亿)': round(df['成交额'].iloc[-1] / 100000000, 2)
                })
        except: continue

    if results:
        res_df = pd.DataFrame(results)
        # 排序逻辑：优先选择离MA20近、且偏离度绝对值小的
        res_df['abs_bias'] = res_df['偏离度%'].abs()
        res_df = res_df.sort_values(by=['abs_bias', '距MA20距离%']).drop(columns=['abs_bias'])
        
        save_path = f"{OUTPUT_DIR}/yin_refined_{date_str}.csv"
        res_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"🎯 扫描完成：精选出 {len(res_df)} 个“高胜率”目标。结果已保存。")
    else:
        print("今日未发现符合起步阶段极缩阴条件的信号。")

if __name__ == "__main__":
    main()
