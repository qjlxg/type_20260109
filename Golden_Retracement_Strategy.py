import pandas as pd
import numpy as np
import os
from datetime import datetime
import multiprocessing

# ==========================================
# 战法名称：金线回踩杀入战法 (Golden Retracement)
# 核心逻辑：
# 1. 趋势：股价处于21日均线之上，大趋势向上。
# 2. 诱空：出现“光脚阴线”（收盘价接近全天最低），视为洗盘。
# 3. 支撑：阴线实体底部触及或极度接近21日均线但未放量跌破。
# 4. 确认：次日收阳反包或止跌，确认支撑有效。
# ==========================================

def analyze_stock(file_path, name_dict):
    try:
        df = pd.read_csv(file_path)
        if len(df) < 30: return None
        
        # 基础过滤：排除ST和创业板(30开头)，以及价格区间
        code = os.path.basename(file_path).split('.')[0]
        if code.startswith('30') or "ST" in name_dict.get(code, ""):
            return None
        
        last_close = df.iloc[-1]['收盘']
        if not (5.0 <= last_close <= 20.0):
            return None

        # 计算技术指标
        df['MA21'] = df['收盘'].rolling(window=21).mean()
        df['Vol_MA5'] = df['成交量'].rolling(window=5).mean()
        
        # 获取最近三天的记录 [T-2, T-1, T_Now]
        # T-1 为光脚阴线观察日，T_Now 为确认日
        t_0 = df.iloc[-1]   # 最新一天
        t_1 = df.iloc[-2]   # 前一天（回踩日）
        t_2 = df.iloc[-3]   # 大前天
        
        # 1. 趋势条件：股价在MA21上方运行
        if t_1['收盘'] < t_1['MA21']: return None
        
        # 2. 光脚阴线形态判断 (T-1)
        is_negative = t_1['收盘'] < t_1['开盘']
        # 光脚判断：下影线极其短（收盘价与最低价差距小于实体的5%）
        body_size = abs(t_1['开盘'] - t_1['收盘'])
        lower_shadow = t_1['收盘'] - t_1['最低']
        is_shaved_bottom = lower_shadow <= (body_size * 0.05) if body_size > 0 else False
        
        # 3. 缩量回踩判断
        is_low_volume = t_1['成交量'] < t_1['Vol_MA5'] * 1.2 # 不允许大幅放量跌破
        
        # 4. 支撑强度判断：阴线最低价接近MA21
        touch_ma21 = abs(t_1['最低'] - t_1['MA21']) / t_1['MA21'] < 0.015
        
        # 5. 今日确认信号 (T_Now)
        is_rebound = t_0['收盘'] > t_1['收盘'] # 今日止跌回升
        
        if is_negative and is_shaved_bottom and touch_ma21 and is_rebound:
            # 优中选优评分逻辑
            score = 0
            if t_0['涨跌幅'] > 2: score += 40  # 反弹力度大
            if t_1['成交量'] < t_2['成交量']: score += 30 # 缩量洗盘明显
            if t_0['收盘'] > t_1['开盘']: score += 30 # 完成阳包阴
            
            # 操作建议封装
            advice = "试错观察"
            if score >= 90: advice = "重点关注/一击必中"
            elif score >= 60: advice = "轻仓切入"
            
            return {
                "代码": code,
                "名称": name_dict.get(code, "未知"),
                "当前价": t_0['收盘'],
                "回踩幅度": f"{round((t_1['最低']-t_1['MA21'])/t_1['MA21']*100, 2)}%",
                "信号强度": f"{score}分",
                "操作建议": advice,
                "止损位": t_1['最低']
            }
            
    except Exception:
        return None
    return None

def main():
    data_dir = './stock_data/'
    names_df = pd.read_csv('stock_names.csv')
    name_dict = dict(zip(names_df['code'].astype(str).str.zfill(6), names_df['name']))
    
    file_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    # 并行处理
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        results = pool.starmap(analyze_stock, [(f, name_dict) for f in file_list])
    
    final_list = [r for r in results if r is not None]
    
    # 结果保存
    if final_list:
        output_df = pd.DataFrame(final_list).sort_values(by="信号强度", ascending=False)
        
        # 创建年月目录
        now = datetime.now()
        dir_path = now.strftime('%Y%m')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            
        file_name = f"Golden_Retracement_Strategy_{now.strftime('%Y%m%d_%H%M%S')}.csv"
        save_path = os.path.join(dir_path, file_name)
        output_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"分析完成，发现 {len(final_list)} 只符合条件股票。")
    else:
        print("今日未筛选出符合战法的个股。")

if __name__ == "__main__":
    main()
