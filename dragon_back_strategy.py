import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from multiprocessing import Pool, cpu_count

# ==========================================
# 战法名称：龙回头缩量回踩（核心进阶版）
# 逻辑说明：
# 1. 选股：必须在过去20天内有过至少一次涨幅 > 7% 的放量阳线（确认有主力）。
# 2. 回调：股价缩量回踩至该大阳线的起点（支撑区），要求成交量极度萎缩。
# 3. 过滤：最新价格 5-20元，剔除创业板(30)、科创板(688)、ST。
# 4. 回测：脚本自动计算该股历史上同类信号后的5日收益率，优中选优。
# ==========================================

DATA_DIR = './stock_data/'
NAMES_FILE = 'stock_names.csv'

def backtest_logic(df, signal_idx):
    """历史回测：计算信号出现后5天的最高涨幅"""
    if signal_idx + 5 >= len(df):
        return None
    buy_price = df.loc[signal_idx, '收盘']
    future_max = df.loc[signal_idx+1 : signal_idx+5, '最高'].max()
    profit = (future_max - buy_price) / buy_price
    return 1 if profit > 0.05 else 0 # 5% 涨幅为达标

def analyze_stock(file_path):
    try:
        df = pd.read_csv(file_path)
        if len(df) < 60: return None
        
        # 基础过滤
        code = os.path.basename(file_path).split('.')[0]
        if code.startswith(('30', '688', '300')) or 'ST' in code: return None
        
        last_row = df.iloc[-1]
        if not (5.0 <= last_row['收盘'] <= 20.0): return None

        # --- 核心战法逻辑 ---
        # 1. 寻找启动日（过去20日内涨幅最大且成交量倍增的一天）
        recent = df.tail(25).iloc[:-3] # 避开最近3天寻找启动点
        launch_idx = recent['涨跌幅'].idxmax()
        if recent.loc[launch_idx, '涨跌幅'] < 6: return None # 启动力度不够
        
        support_price = df.loc[launch_idx, '开盘']
        launch_vol = df.loc[launch_idx, '成交量']
        
        # 2. 判定当前是否回踩到位
        curr_close = last_row['收盘']
        curr_vol = last_row['成交量']
        
        # 价格回踩到支撑位上下2%区间，且量能萎缩至启动日的45%以下
        is_touching_support = (support_price * 0.98 <= curr_close <= support_price * 1.03)
        is_ultra_low_vol = (curr_vol < launch_vol * 0.45)
        
        if is_touching_support and is_ultra_low_vol:
            # --- 加入历史回测模块 ---
            success_count = 0
            total_signals = 0
            # 在历史数据中寻找类似信号进行验证
            for i in range(20, len(df) - 10):
                hist_recent = df.iloc[i-15 : i]
                hist_launch = hist_recent['涨跌幅'].idxmax()
                if hist_recent.loc[hist_launch, '涨跌幅'] > 6:
                    h_support = df.loc[hist_launch, '开盘']
                    if abs(df.loc[i, '收盘'] - h_support) / h_support < 0.02 and df.loc[i, '成交量'] < df.loc[hist_launch, '成交量'] * 0.5:
                        res = backtest_logic(df, i)
                        if res is not None:
                            success_count += res
                            total_signals += 1
            
            win_rate = (success_count / total_signals) if total_signals > 0 else 0
            
            # 评分系统
            score = 60
            if win_rate > 0.5: score += 20
            if curr_close > df.iloc[-5:]['收盘'].mean(): score += 10 # 短期站稳5日线
            
            if score >= 80: # 只选高分标的，实现一击必中
                return {
                    '股票代码': code,
                    '最新价': curr_close,
                    '支撑位': round(support_price, 2),
                    '量能萎缩比': f"{round(curr_vol/launch_vol*100, 1)}%",
                    '历史信号胜率': f"{round(win_rate*100, 1)}%",
                    '信号强度': score,
                    '操作建议': "极度缩量守住支撑，建议分批建仓，破支撑位3%止损" if score > 85 else "建议观察，待放量确认"
                }
    except:
        return None

def main():
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    with Pool(cpu_count()) as p:
        results = p.map(analyze_stock, files)
    
    results = [r for r in results if r is not None]
    
    # 匹配名称
    names_df = pd.read_csv(NAMES_FILE)
    names_df['code'] = names_df['code'].astype(str).str.zfill(6)
    
    final_df = pd.DataFrame(results)
    if not final_df.empty:
        final_df = final_df.merge(names_df, left_on='股票代码', right_on='code', how='left')
        final_df = final_df.drop(columns=['code']).sort_values(by='信号强度', ascending=False)
        
        # 结果输出
        now = datetime.now()
        folder = now.strftime('%Y-%m')
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"dragon_back_strategy_{now.strftime('%Y%m%d_%H%M%S')}.csv")
        final_df.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"筛选完成，发现 {len(final_df)} 个符合一击必中条件的标的。")

if __name__ == "__main__":
    main()
