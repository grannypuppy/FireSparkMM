# visualization.py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import networkx as nx
import numpy as np
import json
from config import *

def plot_performance_metrics(results):
    """
    【最终版】根据用户建议，采用左右横版布局和自定义配色方案。
    左图为每个独立数据流的丢包量柱状图，右图为已完成流的时延折线图。
    """
    # 为了支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    flows = sorted(results['flows'], key=lambda f: f.id)
    
    # 1. 调整为左右横版布局，并缩小尺寸
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("仿真性能评估结果", fontsize=16)

    # --- 左图：各数据流丢包量柱状图 ---
    flow_ids = [f.id for f in flows]
    data_lost = [f.remaining_data for f in flows]
    
    # 2. 采用淡紫色和淡青色的配色方案
    colors = []
    color_map = {
        "completed": '#a7d8de',  # 淡青色
        "partial": '#d3d3d3',    # 淡灰色
        "full_loss": '#d8bde2'   # 淡紫色
    }
    for f in flows:
        if f.status == "completed":
            colors.append(color_map["completed"])
        elif f.remaining_data < FLOW_DATA_SIZE:
            colors.append(color_map["partial"])
        else:
            colors.append(color_map["full_loss"])
            
    ax1.bar(flow_ids, data_lost, color=colors, width=1.0, edgecolor='white', linewidth=0.1)
    ax1.set_title("各数据流丢包量", fontsize=14)
    ax1.set_xlabel("数据流 ID")
    ax1.set_ylabel("丢包数据量 (Mb)")
    ax1.set_xlim(-1, len(flows))
    ax1.set_ylim(0, FLOW_DATA_SIZE * 1.05)
    ax1.grid(axis='y', linestyle='--', alpha=0.6)

    # 创建图例
    legend_patches = [
        patches.Patch(facecolor=color_map["completed"], label=f'无丢包 (已完成) ({len([f for f in flows if f.status == "completed"])})'),
        patches.Patch(facecolor=color_map["partial"], label=f'部分丢包 (未完成) ({len([f for f in flows if f.status == "active"])})'),
        patches.Patch(facecolor=color_map["full_loss"], label=f'完全丢包 (未开始) ({len([f for f in flows if f.status == "pending"])})')
    ]
    ax1.legend(handles=legend_patches, loc='upper right')

    # --- 右图：传输时延折线图 ---
    completed_flows = [f for f in flows if f.status == "completed"]
    if completed_flows:
        completed_flows.sort(key=lambda f: f.id)
        delays = [f.end_time - f.start_time for f in completed_flows]
        avg_delay = np.mean(delays)
        
        x_axis = range(len(completed_flows))
        # --- 修改此处：减小标记尺寸和线条宽度 ---
        ax2.plot(x_axis, delays, color='#aec6cf', marker='o', linestyle='-', markersize=1, linewidth=0.4, label='各流时延')
        ax2.axhline(avg_delay, color='#c0392b', linestyle='--', linewidth=2, label=f'平均时延: {avg_delay:.2f}s')
        
        ax2.set_title("已完成流的传输时延", fontsize=14)
        ax2.set_xlabel("已完成的数据流")
        ax2.set_ylabel("传输时延 (秒)")
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.set_xlim(-1, len(completed_flows))
    else:
        ax2.text(0.5, 0.5, '没有已完成的数据流', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
        ax2.set_title("已完成流的传输时延分布")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()




def convert_keys_to_str(obj):
    """递归地将字典中的元组键转换为字符串，以便JSON序列化。"""
    if isinstance(obj, dict):
        return {str(k): convert_keys_to_str(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_keys_to_str(elem) for elem in obj]
    if isinstance(obj, set):
        return [convert_keys_to_str(elem) for elem in obj]
    return obj

def export_strategy_to_json(snapshots, filename="strategy_details.json"):
    """将详细的策略快照导出为JSON文件，便于分析。"""
    print(f"\n正在导出详细策略到 {filename}...")
    
    exportable_snapshots = []
    for snap in snapshots:
        exportable_snap = {
            'time': snap['time'],
            'flow_value': snap['flow_value'],
            'car_coverages': convert_keys_to_str(snap['car_coverages']),
            'link_flow': convert_keys_to_str(snap['link_flow']),
            'flow_sent_details': convert_keys_to_str(snap['flow_sent_details'])
        }
        exportable_snapshots.append(exportable_snap)

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(exportable_snapshots, f, indent=2)
    print("策略导出完成。")


def animate_detailed_flow(snapshots, flows):
    """根据详细的策略快照生成流量动画。"""
    print("\n正在生成详细的流量动画...")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    graph = nx.grid_2d_graph(M_ROWS, N_COLS)
    pos = {node: (node[1], M_ROWS - 1 - node[0]) for node in graph.nodes()}
    
    flow_id_to_source = {f.id: f.source_node for f in flows}

    fig, ax = plt.subplots(figsize=(18, 10))
    
    node_collection = nx.draw_networkx_nodes(graph, pos, node_size=20, node_color='lightgray', ax=ax)
    edge_collection = nx.draw_networkx_edges(graph, pos, edge_color='lightgray', width=0.5, ax=ax)
    
    car_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    car_patches = [patches.Rectangle((0,0), 0, 0, alpha=0.2, color=c, label=f'车辆 {i}') for i, c in enumerate(car_colors)]
    for p in car_patches:
        ax.add_patch(p)

    # 添加图例
    legend_elements = [
        patches.Patch(facecolor='lightgray', edgecolor='gray', label='空闲传感器'),
        patches.Patch(facecolor='#d62728', edgecolor='gray', label='活动源'),
        patches.Patch(facecolor='#9467bd', edgecolor='gray', label='出口汇聚点'),
    ] + car_patches
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

    def update(k):
        fig.suptitle(f"详细流量分配 (时间: {k}s / {len(snapshots)-1}s)", fontsize=16)
        snapshot = snapshots[k]
        
        # 1. 更新车辆位置
        car_coverages = snapshot['car_coverages']
        all_exit_nodes = set()
        for i in range(NUM_CARS):
            coverage_set = car_coverages.get(i)
            if coverage_set:
                all_exit_nodes.update(coverage_set)
                min_x = min(n[0] for n in coverage_set)
                max_x = max(n[0] for n in coverage_set)
                min_y = min(n[1] for n in coverage_set)
                max_y = max(n[1] for n in coverage_set)
                plot_x, plot_y = min_y - 0.5, M_ROWS - 1 - max_x - 0.5
                width, height = max_y - min_y + 1, max_x - min_x + 1
                car_patches[i].set_bounds(plot_x, plot_y, width, height)
            else:
                car_patches[i].set_bounds(0, 0, 0, 0)

        # 2. 更新链路颜色和宽度
        link_flow = snapshot.get('link_flow', {})
        edge_colors = []
        edge_widths = []
        for u, v in graph.edges():
            # networkx flow_dict is nested. Check both directions.
            flow_uv = link_flow.get(u, {}).get(v, 0)
            flow_vu = link_flow.get(v, {}).get(u, 0)
            total_flow = flow_uv + flow_vu
            
            norm_flow = min(total_flow / B_SENSOR, 1.0) if B_SENSOR > 0 else 0
            edge_colors.append(plt.cm.viridis(norm_flow))
            edge_widths.append(0.5 + norm_flow * 4)

        edge_collection.set_edgecolor(edge_colors)
        edge_collection.set_linewidths(edge_widths)
        
        # 3. 更新节点颜色
        flow_sent_details = snapshot.get('flow_sent_details', {})
        active_source_nodes = {flow_id_to_source[fid] for fid, amount in flow_sent_details.items() if amount > 0}
        
        node_colors = []
        for node in graph.nodes():
            if node in active_source_nodes:
                node_colors.append('#d62728')  # 红色表示活动源
            elif node in all_exit_nodes:
                node_colors.append('#9467bd')  # 紫色表示出口汇聚点
            else:
                node_colors.append('lightgray')
        node_collection.set_color(node_colors)
        
        return [edge_collection, node_collection] + car_patches

    ani = animation.FuncAnimation(fig, update, frames=range(len(snapshots)), blit=True, interval=150)
    
    try:
        ani.save('detailed_flow_animation.gif', writer='pillow', fps=1)
        print("动画已保存为 'detailed_flow_animation.gif'")
    except Exception as e:
        print(f"无法保存动画: {e}")
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    plt.show()