# main.py

import matplotlib
from scheduler import Scheduler
from entity import MobileCar, CAR_Y_COORDS
import visualization as vis
from config import *

matplotlib.use('TkAgg')

def main():
    print("欢迎来到“火花杯”数学建模 - 传感器阵列流量调度问题求解器 (v3.0 精确子图+策略导出)")
    
    scheduler = Scheduler()
    
    # 提前计算并记录车辆历史位置 (与之前相同)
    cars_history = []
    temp_cars = [MobileCar(i, CAR_Y_COORDS[i]) for i in range(len(CAR_Y_COORDS))]
    for k in range(TOTAL_TIME):
        for car in temp_cars:
            car.update_position(k)
        cars_history.append([car.coverage_area.copy() for car in temp_cars])

    scheduler.run_simulation()
    
    results = scheduler.get_results()
    
    print("\n正在生成可视化与策略导出...")
    try:
        # 1. 导出详细策略到JSON文件
        vis.export_strategy_to_json(results['snapshots'])
        
        # 2. 绘制性能总览图
        vis.plot_performance_metrics(results)
        
        # 3. 基于详细的快照数据生成动画
        vis.animate_detailed_flow(results['snapshots'], cars_history)
        
    except Exception as e:
        print(f"可视化或导出过程中出现错误: {e}")

    print("\n程序执行完毕。")

if __name__ == "__main__":
    main()