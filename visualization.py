# visualization.py

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import networkx as nx
import numpy as np
from config import *

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

def plot_performance_metrics(results):
    """绘制关键性能指标的图表"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("仿真性能评估结果", fontsize=16)

    # 1. 丢包率
    ax1.bar(['已接收', '丢失'], 
            [(1 - results['loss_rate']) * 100, results['loss_rate'] * 100], 
            color=['green', 'red'])
    ax1.set_title("流量接收与丢包率")
    ax1.set_ylabel("百分比 (%)")
    ax1.set_ylim(0, 100)

    # 2. 平均时延
    ax2.bar(['已完成流的平均时延'], [results['avg_delay']], color='skyblue')
    ax2.set_title("平均传输时延")
    ax2.set_ylabel("时间 (秒)")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def animate_simulation(results):
    """创建并保存仿真过程的动画"""
    graph = results['graph']
    link_usage = results['link_usage']
    cars = results['cars']
    
    pos = {node: (node[1], M_ROWS - 1 - node[0]) for node in graph.nodes()} # (y, M-1-x) for correct plot orientation

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_title("传感器网络流量调度仿真 (时间: 0s)")
    plt.box(False)

    # 绘制静态的网格节点
    nx.draw_networkx_nodes(graph, pos, node_size=20, node_color='lightgray', ax=ax)
    
    # 预先创建车辆和链路的绘图对象
    car_patches = [patches.Rectangle((0, 0), 0, 0, alpha=0.3, color=plt.cm.viridis(i/NUM_CARS)) for i in range(NUM_CARS)]
    for patch in car_patches:
        ax.add_patch(patch)
        
    edge_collection = nx.draw_networkx_edges(graph, pos, edge_color='lightgray', width=0.5, ax=ax)

    def update(k):
        ax.set_title(f"传感器网络流量调度仿真 (时间: {k}s)")
        
        # 更新车辆位置
        for i, car in enumerate(cars):
            car.update_position(k)
            if car.coverage_area:
                min_x = min(n[0] for n in car.coverage_area)
                max_x = max(n[0] for n in car.coverage_area)
                min_y = min(n[1] for n in car.coverage_area)
                max_y = max(n[1] for n in car.coverage_area)
                
                # 转换到绘图坐标
                plot_x = min_y - 0.5
                plot_y = M_ROWS - 1 - max_x - 0.5
                width = max_y - min_y + 1
                height = max_x - min_x + 1
                
                car_patches[i].set_bounds(plot_x, plot_y, width, height)

        # 更新链路颜色以表示负载
        edge_colors = []
        for u, v in graph.edges():
            edge = tuple(sorted((u, v)))
            usage = link_usage[k].get(edge, 0)
            # 归一化负载
            norm_usage = min(usage / B_SENSOR, 1.0)
            edge_colors.append(plt.cm.hot_r(norm_usage))
        
        edge_collection.set_edgecolor(edge_colors)

        return [edge_collection] + car_patches

    # 创建动画
    ani = animation.FuncAnimation(fig, update, frames=range(TOTAL_TIME), blit=True, interval=100)
    
    # 保存动画
    print("\n正在生成动画... (这可能需要几分钟)")
    try:
        ani.save('simulation_animation.gif', writer='pillow', fps=10)
        print("动画已保存为 'simulation_animation.gif'")
    except Exception as e:
        print(f"无法保存动画: {e}")
        print("请确保已安装Pillow库 (pip install Pillow)")
    
    plt.show()