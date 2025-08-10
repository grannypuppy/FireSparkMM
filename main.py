# main.py

import matplotlib
from scheduler import Scheduler
from entity import MobileCar, CAR_Y_COORDS
import visualization as vis
from config import *

matplotlib.use('TkAgg')

def main():
    
    scheduler = Scheduler()
    
    scheduler.run_simulation()
    
    results = scheduler.get_results()
    
    print("\n正在生成可视化与策略导出...")
    try:
        # 1. 导出详细策略到JSON文件
        vis.export_strategy_to_json(results['snapshots'])
        
        # 2. 绘制性能总览图
        vis.plot_performance_metrics(results)
        
        # 3. 基于详细的快照数据生成动画
        vis.animate_detailed_flow(results['snapshots'],  results['flows'])
        
    except Exception as e:
        print(f"可视化或导出过程中出现错误: {e}")

    print("\n程序执行完毕。")

if __name__ == "__main__":
    main()