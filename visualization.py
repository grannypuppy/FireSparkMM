# visualization.py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import networkx as nx
import numpy as np
import json
from config import *

# (plot_performance_metrics 函数不变)
def plot_performance_metrics(results):
    """绘制关键性能指标的图表"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("仿真性能评估结果 (最大流精确子图版)", fontsize=16)

    ax1.bar(['已接收', '丢失'], 
            [(1 - results['loss_rate']) * 100, results['loss_rate'] * 100], 
            color=['green', 'red'])
    ax1.set_title("流量接收与丢包率")
    ax1.set_ylabel("百分比 (%)")
    ax1.set_ylim(0, 100)

    ax2.bar(['已完成流的平均时延'], [results['avg_delay']], color='skyblue')
    ax2.set_title("平均传输时延")
    ax2.set_ylabel("时间 (秒)")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def export_strategy_to_json(snapshots, filename="strategy_details.json"):
    """将详细的策略快照导出为JSON文件，便于分析。"""
    print(f"\n正在导出详细策略到 {filename}...")
    
    # 自定义JSON编码器来处理元组键和集合
    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, set):
                return list(obj)
            if isinstance(obj, tuple):
                return str(obj) # 将元组转换为字符串
            return json.JSONEncoder.default(self, obj)

    # 转换字典键
    exportable_snapshots = []
    for snap in snapshots:
        # flow_dict的键是节点，可能是元组，需要转换
        flow_dict_str_keys = {}
        for u, neighbors in snap['flow_dict'].items():
            u_str = str(u)
            flow_dict_str_keys[u_str] = {}
            for v, flow in neighbors.items():
                v_str = str(v)
                flow_dict_str_keys[u_str][v_str] = flow
        
        exportable_snap = snap.copy()
        exportable_snap['flow_dict'] = flow_dict_str_keys
        exportable_snapshots.append(exportable_snap)

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(exportable_snapshots, f, cls=CustomEncoder, indent=2)
    print("策略导出完成。")


def animate_detailed_flow(snapshots, cars_history):
    """根据详细的策略快照生成流量动画。"""
    print("\n正在生成详细的流量动画...")
    graph = nx.grid_2d_graph(M_ROWS, N_COLS)
    pos = {node: (node[1], M_ROWS - 1 - node[0]) for node in graph.nodes()}

    fig, ax = plt.subplots(figsize=(16, 10))
    nx.draw_networkx_nodes(graph, pos, node_size=15, node_color='lightgray', ax=ax)
    
    car_patches = [patches.Rectangle((0,0),0,0,alpha=0.2,color=c) for c in ['#1f77b4','#ff7f0e','#2ca02c']]
    for p in car_patches: ax.add_patch(p)
    
    # 预先绘制所有边，后面只更新颜色和宽度
    edge_collection = nx.draw_networkx_edges(graph, pos, edge_color='lightgray', width=0.5, ax=ax)

    def update(k):
        ax.set_title(f"详细流量分配 (时间: {k}s)")
        
        # 更新车辆位置
        cars_at_k = cars_history[k]
        for i, car_coverage in enumerate(cars_at_k):
            if car_coverage:
                # ... (代码与上一版相同) ...
                min_x = min(n[0] for n in car_coverage)
                max_x = max(n[0] for n in car_coverage)
                min_y = min(n[1] for n in car_coverage)
                max_y = max(n[1] for n in car_coverage)
                plot_x, plot_y = min_y - 0.5, M_ROWS - 1 - max_x - 0.5
                width, height = max_y - min_y + 1, max_x - min_x + 1
                car_patches[i].set_bounds(plot_x, plot_y, width, height)

        # 更新链路颜色和宽度以表示流量
        snapshot = snapshots[k]
        flow_dict = snapshot.get('flow_dict', {})
        edge_colors = []
        edge_widths = []

        for u, v in graph.edges():
            flow = 0
            # 流量是双向的，检查两个方向
            if u in flow_dict and v in flow_dict[u]:
                flow += flow_dict[u][v]
            if v in flow_dict and u in flow_dict[v]:
                flow += flow_dict[v][u]
            
            # 归一化流量用于着色和定宽
            norm_flow = min(flow / B_SENSOR, 1.0)
            edge_colors.append(plt.cm.viridis(norm_flow))
            edge_widths.append(0.5 + norm_flow * 3) # 宽度从0.5到3.5变化

        edge_collection.set_edgecolor(edge_colors)
        edge_collection.set_linewidths(edge_widths)
        
        return [edge_collection] + car_patches

    ani = animation.FuncAnimation(fig, update, frames=range(TOTAL_TIME), blit=True, interval=150)
    
    try:
        ani.save('detailed_flow_animation.gif', writer='pillow', fps=10)
        print("动画已保存为 'detailed_flow_animation.gif'")
    except Exception as e:
        print(f"无法保存动画: {e}")
    plt.show()