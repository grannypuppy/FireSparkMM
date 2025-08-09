# main.py

import matplotlib
from scheduler import Scheduler
import visualization as vis

# 设置matplotlib后端，以避免在某些系统上出现GUI问题
matplotlib.use('TkAgg')

def main():
    """
    主函数：初始化、运行仿真并进行可视化
    """
    print("欢迎来到“火花杯”数学建模 - 传感器阵列流量调度问题求解器")
    
    # 1. 初始化调度器
    scheduler = Scheduler()
    
    # 2. 运行仿真
    scheduler.run_simulation()
    
    # 3. 获取并打印结果
    results = scheduler.get_results()
    
    # 4. 可视化结果
    print("\n正在生成可视化图表...")
    try:
        vis.plot_performance_metrics(results)
        vis.animate_simulation(results)
    except Exception as e:
        print(f"可视化过程中出现错误: {e}")
        print("请确保您的图形环境配置正确。")

    print("\n程序执行完毕。")

if __name__ == "__main__":
    main()