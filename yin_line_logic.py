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
    # 确保日期升序
    if '日期' in df.columns:
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values(by='日期')
        
    # 1. 均线系统 (原有)
    for m in [5, 10, 20, 60]:
        df[f'ma{m}'] = df['收盘'].rolling(m).mean()
    
    # 2. MACD 计算 (原有)
    ema12 = df['收盘'].ewm(span=12, adjust=False).mean()
    ema26 = df['收盘'].ewm(span=26, adjust=False).mean()
    df['dif'] = ema12 - ema26
    df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
    df['macd'] = (df['dif'] - df['dea']) * 2
    
    # 3. RSI 计算 (新增功能：通达信标准6日/12日算法)
    def cal_rsi(series, n):
        delta = series.diff()
        # 简单移动平均计算涨跌幅
        gain = (delta.where(delta > 0, 0)).rolling(n).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(n).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    df['rsi6'] = cal_rsi(df['收盘'], 6)
    df['rsi12'] = cal_rsi(df['收盘'], 12)
    
    # 4. 基础动态指标 (原有)
    df['ma10_up'] = df['ma10'] > df['ma10'].shift(1)
    df['ma20_up'] = df['ma20'] > df['ma20'].shift(1)
    df['v_ma5'] = df['成交量'].rolling(5).mean()
    df['change'] = df['收盘'].pct_change() * 100
    return df

def check_logic(df):
    if len(df) < 60: return None, None
    curr = df.iloc[-1]
    
    # --- 维度 A: 基础过滤 (原有) ---
    if not (5.0 <= curr['收盘'] <= 20.0) or curr['成交额'] < 800000000:
        return None, None

    # --- 维度 B: RSI 强弱过滤 (新增：成功的核心防线) ---
    # 1. RSI6 必须在 50 以上，确保回踩时仍处于强势多头区
    # 2. RSI6 不超过 82，防止追在极端超买的赶顶阶段
    if not (50 <= curr['rsi6'] <= 82):
        return None, None
    # 3. 短期 RSI 强于长期 RSI，确保动力没有出现“死叉”式衰减
    if curr['rsi6'] < curr['rsi12']:
        return None, None

    # --- 维度 C: MACD & 位置过滤 (原有) ---
    if not (curr['dif'] > curr['dea'] and curr['macd'] > -0.1):
        return None, None
    # 乖离限制：防止距离20日线太远
    if curr['收盘'] > df['ma20'].iloc[-1] * 1.12:
        return None, None
    if not curr['ma20_up']:
        return None, None

    # --- 维度 D: 强势基因与量能断层 (原有核心) ---
    recent_15 = df.tail(15)
    strong_days = recent_15[recent_15['change'] > 7.0]
    if strong_days.empty: return None, None
    
    # 量能断层：成交量必须小于最近大阳线成交量的 55%
    if curr['成交量'] > strong_days.iloc[-1]['成交量'] * 0.55:
        return None, None

    # --- 维度 E: 贴线阴线判定 (原有核心调优) ---
    is_yin = curr['收盘'] < curr['开盘'] or curr['change'] <= 0
    # 极度缩量：0.65 阈值
    is_shrink = curr['成交量'] < (curr['v_ma5'] * 0.65)
    
    # 贴线精度：1.0% 以内
    bias_m5 = abs(curr['收盘'] - curr['ma5']) / curr['ma5']
    bias_m10 = abs(curr['收盘'] - curr['ma10']) / curr['ma10']
    
    support_ma_key = None
    if bias_m10 <= 0.01:
        support_ma_key = 'MA10'
    elif bias_m5 <= 0.01:
        support_ma_key = 'MA5'
    
    if is_yin and is_shrink and support_ma_key:
        return f"回踩{support_ma_key}RSI强势", support_ma_key
    
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
                curr = df.iloc[-1]
                ma_val = df[ma_key.lower()].iloc[-1]
                
                results.append({
                    '日期': date_str,
                    '代码': code,
                    '名称': name_map.get(code, '未知'),
                    '当前价': round(curr['收盘'], 2),
                    '形态类型': match_type,
                    'RSI6': round(curr['rsi6'], 2),
                    '偏离度%': round((curr['收盘'] - ma_val) / ma_val * 100, 2),
                    'MACD': round(curr['macd'], 3),
                    '距MA20距离%': round((curr['收盘']/curr['ma20']-1)*100, 2),
                    '成交额(亿)': round(curr['成交额'] / 100000000, 2)
                })
        except: continue

    if results:
        res_df = pd.DataFrame(results)
        # 按照 RSI6 降序排列，越强越靠前
        res_df = res_df.sort_values(by='RSI6', ascending=False)
        
        save_path = f"{OUTPUT_DIR}/yin_rsi_enhanced_{date_str}.csv"
        res_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"🎯 扫描完成：引入RSI强过滤，精选出 {len(res_df)} 个标的。")
    else:
        print("今日未发现符合RSI强势且极度缩量贴线的信号。")

if __name__ == "__main__":
    main()
