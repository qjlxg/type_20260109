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
    # 1. 核心均线系统
    for m in [5, 10, 20, 60]:
        df[f'ma{m}'] = df['收盘'].rolling(m).mean()
    
    # 2. 通达信标准 MACD 计算
    ema12 = df['收盘'].ewm(span=12, adjust=False).mean()
    ema26 = df['收盘'].ewm(span=26, adjust=False).mean()
    df['dif'] = ema12 - ema26
    df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
    df['macd'] = (df['dif'] - df['dea']) * 2
    
    # 3. 趋势指标与成交量均线
    df['ma10_up'] = df['ma10'] > df['ma10'].shift(1)
    df['v_ma5'] = df['成交量'].rolling(5).mean()
    df['change'] = df['收盘'].pct_change() * 100
    return df

def check_logic(df):
    if len(df) < 60: return None, None
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    # --- 过滤逻辑 1: 价格与成交额 ---
    if not (5.0 <= curr['收盘'] <= 30.0) or curr['成交额'] < 800000000: # 稍微放宽至8亿
        return None, None

    # --- 过滤逻辑 2: MACD 确认信号 (通达信买入策略) ---
    # DIF需在DEA上方，或者MACD柱状图拒绝变短（代表多头动能仍在）
    macd_ok = curr['dif'] > curr['dea'] and curr['macd'] > -0.05
    if not macd_ok:
        return None, None

    # --- 过滤逻辑 3: 强势基因与追涨动力 ---
    recent_15 = df.tail(15)
    has_strong_gene = (recent_15['change'] > 9.0).any() # 15天内有过大阳
    # 追涨逻辑：当前收盘价必须在MA20之上，且MA5/MA10金叉或多头
    momentum_ok = curr['收盘'] > curr['ma20'] and curr['ma5'] > curr['ma20']
    
    if not (has_strong_gene and momentum_ok):
        return None, None

    # --- 过滤逻辑 4: 线上阴线回踩 (核心买点) ---
    is_yin = curr['收盘'] < curr['开盘'] or curr['change'] <= 0
    # 缩量：成交量小于5日均量的65%
    is_shrink = curr['成交量'] < (curr['v_ma5'] * 0.65)
    
    # 寻找支撑位：阴线越靠近均线越好（偏离度在1.5%以内）
    support_ma_key = None
    if abs(curr['收盘'] - curr['ma10']) / curr['ma10'] <= 0.015:
        support_ma_key = 'MA10'
    elif abs(curr['收盘'] - curr['ma5']) / curr['ma5'] <= 0.015:
        support_ma_key = 'MA5'
    
    if is_yin and is_shrink and support_ma_key:
        return f"回踩{support_ma_key}缩量阴", support_ma_key
    
    return None, None

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载股票名称映射 (CSV格式: code, name)
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
            # 确保日期升序
            if '日期' in df.columns:
                df = df.sort_values(by='日期')
            
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
                    '贴线偏离%': bias,
                    'MACD值': round(df['macd'].iloc[-1], 3),
                    '成交额(亿)': round(df['成交额'].iloc[-1] / 100000000, 2)
                })
        except Exception as e:
            continue

    if results:
        res_df = pd.DataFrame(results)
        # 按偏离度绝对值排序，寻找最贴线的
        res_df = res_df.reindex(res_df['贴线偏离%'].abs().sort_values().index)
        
        save_path = f"{OUTPUT_DIR}/yin_macd_signals_{date_str}.csv"
        res_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"🎯 扫描完成：结合MACD与回踩逻辑，精选出 {len(res_df)} 个目标。")
    else:
        print("今日未发现符合MACD支撑与贴线阴线条件的信号。")

if __name__ == "__main__":
    main()
