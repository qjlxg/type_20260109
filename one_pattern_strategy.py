import pandas as pd
import os
import glob
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import pytz

"""
战法名称：【一种模式做一万遍】—— 底部平台缩量起爆战法 v3.0 (极致精简版)
核心逻辑：
1. 空间门槛：5-20元，排除 ST, 30开头(创业板), 688(科创板), 8/4(北交所)。
2. 趋势保护：20日均线(MA20)必须向上（今日 MA20 > 5天前 MA20），确保不是阴跌反弹。
3. 缩量待势：过去 15 天箱体振幅严格限制在 15% 以内，且箱体平均成交量小于前一个月的平均量（主力高度控盘）。
4. 爆发强度：涨幅必须在 5% - 10% 之间（大阳线或涨停），且放量倍数（量比） > 2.5 倍。
5. 换手约束：换手率在 3% - 10% 之间的“黄金活跃带”。
"""

def filter_stock(file_path):
    try:
        df = pd.read_csv(file_path)
        # 确保数据量足够计算长周期均线
        if df.empty or len(df) < 60:
            return None
        
        latest = df.iloc[-1]
        code = str(latest['股票代码']).zfill(6)
        close = latest['收盘']
        
        # --- 1. 硬性条件过滤 ---
        if not (5.0 <= close <= 20.0): return None
        if code.startswith(('30', '688', '8', '4')): return None 
        if not (code.startswith('60') or code.startswith('00')): return None

        # --- 2. 趋势与形态计算 ---
        df['MA5'] = df['收盘'].rolling(5).mean()
        df['MA10'] = df['收盘'].rolling(10).mean()
        df['MA20'] = df['收盘'].rolling(20).mean()
        
        # A. 趋势向上：MA20 必须有上行斜率
        if df['MA20'].iloc[-1] <= df['MA20'].iloc[-6]: return None
        
        # B. 极致缩量横盘：过去 15 天振幅 < 15%
        window_15 = df.iloc[-16:-1]
        high_15 = window_15['最高'].max()
        low_15 = window_15['最低'].min()
        amplitude = (high_15 - low_15) / low_15
        if amplitude > 0.15: return None
        
        # C. 筹码集中度：突破前的平均成交量要萎缩 (过去15天均量 < 过去60天均量)
        vol_short = window_15['成交量'].mean()
        vol_long = df['成交量'].iloc[-61:-1].mean()
        if vol_short > vol_long: return None 

        # --- 3. 起爆确认 ---
        # 涨幅强度 (大阳线但非一字板)
        if not (5.0 <= latest['涨跌幅'] <= 10.1): return None
        
        # 价格突破箱体
        if close <= high_15: return None
        
        # 量比要求 (突然放量)
        vol_ratio = latest['成交量'] / vol_short
        if vol_ratio < 2.5: return None
        
        # 换手率要求
        turnover = latest['换手率']
        if not (3.0 <= turnover <= 10.0): return None

        return {
            'code': code,
            'price': close,
            'pct': f"{latest['涨跌幅']}%",
            'turnover': f"{turnover}%",
            'vol_ratio': round(vol_ratio, 2),
            'remark': "缩量横盘+大阳突破"
        }
    except Exception:
        return None

def main():
    files = glob.glob('./stock_data/*.csv')
    with ProcessPoolExecutor() as executor:
        valid_results = [r for r in executor.map(filter_stock, files) if r]
    
    if valid_results:
        names_df = pd.read_csv('stock_names.csv', dtype={'code': str})
        names_df['code'] = names_df['code'].str.zfill(6)
        res_df = pd.merge(pd.DataFrame(valid_results), names_df, on='code', how='left')
        res_df = res_df[~res_df['name'].str.contains('ST', na=False)].dropna()
    else:
        res_df = pd.DataFrame()

    # --- 结果推送 ---
    tz = pytz.timezone('Asia/Shanghai')
    now = datetime.now(tz)
    dir_path = f"results/{now.strftime('%Y-%m')}"
    os.makedirs(dir_path, exist_ok=True)
    
    file_name = f"{dir_path}/pattern_{now.strftime('%Y%m%d_%H%M%S')}.csv"
    if not res_df.empty:
        res_df.to_csv(file_name, index=False, encoding='utf-8-sig')
        print(f"筛选完成！捕获精选标的 {len(res_df)} 只。")
    else:
        print("今日无符合极致精选条件的标的。")

if __name__ == "__main__":
    main()
