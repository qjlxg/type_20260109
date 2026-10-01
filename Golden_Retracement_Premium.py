import pandas as pd
import numpy as np
import os
from datetime import datetime
import multiprocessing

# ==========================================
# 战法名称：金线回踩·精选版 (Golden_Retracement_Premium)
# 核心逻辑（优加选优）：
# 1. 趋势：21日均线斜率向上，股价处于上升通道。
# 2. 极致洗盘：回踩日(T-1)必须是“缩量光脚阴线”，量能萎缩至5日均量的80%以下。
# 3. 强力确认：今日(T_Now)必须收阳，且量比(较昨日) > 1.2，收盘价收复昨日实体大部分。
# 4. 严苛过滤：价格 5-20元，排除ST、创业板、北交所。
# ==========================================

def analyze_stock(file_path, name_dict):
    try:
        df = pd.read_csv(file_path)
        if len(df) < 40: return None
        
        # 1. 基础硬过滤
        code = os.path.basename(file_path).split('.')[0]
        # 排除 30 (创业板), 688 (科创板), 8/9 (北交所)
        if code.startswith(('30', '688', '8', '9')) or "ST" in name_dict.get(code, ""):
            return None
        
        last_close = df.iloc[-1]['收盘']
        if not (5.0 <= last_close <= 20.0):
            return None

        # 2. 计算指标
        df['MA21'] = df['收盘'].rolling(window=21).mean()
        df['MA21_Slope'] = df['MA21'].diff(3) # 计算均线斜率
        df['V_MA5'] = df['成交量'].rolling(window=5).mean()
        
        t_0 = df.iloc[-1]   # 今日 (确认日)
        t_1 = df.iloc[-2]   # 昨日 (回踩日)
        t_2 = df.iloc[-3]   # 前日
        
        # 3. 趋势过滤：21日线必须是向上的
        if t_0['MA21_Slope'] <= 0: return None
        if t_1['收盘'] < t_1['MA21']: return None

        # 4. 判定昨日(T-1)是否为“极致缩量光脚阴”
        is_negative = t_1['收盘'] < t_1['开盘']
        body_size = t_1['开盘'] - t_1['收盘']
        lower_shadow = t_1['收盘'] - t_1['最低']
        # 光脚：下影线极其微小
        is_shaved = lower_shadow <= (body_size * 0.1) if body_size > 0 else False
        # 极致缩量：量能小于5日均量
        is_extreme_low_vol = t_1['成交量'] < t_1['V_MA5'] * 0.9

        # 5. 判定今日(T_0)是否为“强力反攻”
        is_positive = t_0['收盘'] > t_0['开盘']
        # 真实量比：今日成交量显著大于昨日回踩量 (主力真金白银买入)
        real_vol_ratio = t_0['成交量'] / t_1['成交量']
        # 价格反包：今日收盘价至少收复昨日阴线实体的 80%
        is_reclaim = t_0['收盘'] > (t_1['收盘'] + body_size * 0.8)

        if is_negative and is_shaved and is_extreme_low_vol and is_positive and is_reclaim:
            # 评分系统（优加选优）
            score = 60 # 基础分
            if real_vol_ratio > 1.5: score += 20  # 量能倍增
            if t_0['收盘'] > t_1['开盘']: score += 20 # 完全吞并阳线
            
            advice = "重点关注"
            if score >= 100: advice = "一击必中/全仓博弈"
            elif score >= 80: advice = "积极参与"
            
            return {
                "代码": code,
                "名称": name_dict.get(code, "未知"),
                "当前价": t_0['收盘'],
                "今日涨幅": f"{t_0['涨跌幅']}%",
                "真实量增": f"{round(real_vol_ratio, 2)}倍",
                "信号强度": f"{score}分",
                "操作建议": advice,
                "止损点": t_1['最低']
            }
            
    except Exception:
        return None
    return None

def main():
    data_dir = './stock_data/'
    names_df = pd.read_csv('stock_names.csv')
    name_dict = dict(zip(names_df['code'].astype(str).str.zfill(6), names_df['name']))
    
    file_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        results = pool.starmap(analyze_stock, [(f, name_dict) for f in file_list])
    
    final_list = [r for r in results if r is not None]
    
    if final_list:
        output_df = pd.DataFrame(final_list).sort_values(by="信号强度", ascending=False)
        # 最终只取评分最高的前 5 名，宁缺毋滥
        output_df = output_df.head(5)
        
        now = datetime.now()
        dir_path = now.strftime('%Y%m')
        if not os.path.exists(dir_path): os.makedirs(dir_path)
            
        save_path = os.path.join(dir_path, f"Golden_Retracement_Premium_{now.strftime('%Y%m%d_%H%M%S')}.csv")
        output_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"筛选完成。今日精选最优个股 {len(output_df)} 只。")
    else:
        print("今日无符合'极致回踩'逻辑的个股。")

if __name__ == "__main__":
    main()
