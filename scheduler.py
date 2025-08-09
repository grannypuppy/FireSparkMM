# scheduler.py

import networkx as nx
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from config import *
from entity import DataFlow, MobileCar, get_s2gl_bandwidth

class Scheduler:
    """
    流量调度器，负责整个仿真过程的运行和决策
    """
    def __init__(self):
        self.time_steps = range(0, TOTAL_TIME, TIME_STEP)
        self.graph = self._create_grid_graph()
        self.flows = self._create_flows()
        self.cars = [MobileCar(i, CAR_Y_COORDS[i]) for i in range(NUM_CARS)]
        self.total_flow_sent = 0
        self.snapshots = []

        # 用于记录结果
        self.scheduling_log = [{} for _ in self.time_steps]
        self.link_usage = {t: {edge: 0 for edge in self.graph.edges} for t in self.time_steps}

    def _create_grid_graph(self):
        """创建一个M x N的网格图"""
        G = nx.grid_2d_graph(M_ROWS, N_COLS)
        # networkx中节点是(row, col)顺序，即(x,y)
        return G

    def _create_flows(self):
        """根据规则创建所有1800个数据流"""
        flows = []
        flow_id_counter = 0
        for x in range(M_ROWS):
            for y in range(N_COLS):
                # N为列数。N=30是偶数。
                # 题目规定: N为偶数时t=0, t+30, ... ; N为奇数时t=15, t+30...
                # 此处N=30是网格宽度，是固定的，因此所有传感器遵循同一规则
                # [cite_start]“其中 t=0(N为偶数)或15(N为奇数)” [cite: 44]
                # 此处N是传感器所在列编号y，而非网格总列数
                start_offset = 15 if (y + 1) % 2 != 0 else 0
                
                for t in [start_offset, start_offset + 30, start_offset + 60]:
                    if t < TOTAL_TIME:
                        flows.append(DataFlow(flow_id_counter, (x, y), t))
                        flow_id_counter += 1
        return flows

    def run_simulation(self):
        """运行整个仿真"""
        print("开始仿真调度...")
        for k in tqdm(self.time_steps, desc="仿真进度"):
            self._schedule_at_step_max_flow(k)
        print("仿真结束。")

    def _get_all_shortest_path_edges(self, s_node, e_node):
        """
        获取s_node和e_node之间所有最短路径上的边。
        在网格图中，这构成了一个矩形区域内的所有朝向目标的边。
        """
        sx, sy = s_node
        ex, ey = e_node

        # 如果曼哈顿距离超过限制，则返回空
        if abs(sx - ex) + abs(sy - ey) > MAX_PATH_LENGTH:
            return set()

        path_edges = set()
        x_min, x_max = min(sx, ex), max(sx, ex)
        y_min, y_max = min(sy, ey), max(sy, ey)

        x_dir = 1 if ex > sx else -1
        y_dir = 1 if ey > sy else -1

        for r in range(x_min, x_max + 1):
            for c in range(y_min, y_max + 1):
                # 水平方向的边
                if r != ex:
                    u, v = (r, c), (r + x_dir, c)
                    path_edges.add((u, v))
                # 垂直方向的边
                if c != ey:
                    u, v = (r, c), (r, c + y_dir)
                    path_edges.add((u, v))
        return path_edges

    def _schedule_at_step_max_flow(self, k):
        # 1. 更新状态
        for car in self.cars:
            car.update_position(k)

        active_flows = [f for f in self.flows if f.start_time <= k and f.status != "completed"]
        if not active_flows:
            self.snapshots.append({'time': k, 'flow_value': 0, 'flow_dict': {}})
            return

        source_nodes_map = defaultdict(list)
        for f in active_flows:
            source_nodes_map[f.source_node].append(f)
        
        exit_nodes_all = set.union(*[car.coverage_area for car in self.cars])

        # 2. 【核心改动】构建包含所有最短路径的G_flow子图
        G_flow = nx.DiGraph()
        SUPER_SOURCE, SUPER_SINK = 'S', 'T'
        G_flow.add_nodes_from([SUPER_SOURCE, SUPER_SINK])
        for car in self.cars:
            G_flow.add_edge(f'C_{car.id}', SUPER_SINK, capacity=B_RECEIVE)

        valid_mesh_edges = set()
        valid_exit_nodes = set()

        # a. 确定子图的骨架：所有合法的最短路径边
        for s_node in source_nodes_map:
            for e_node in exit_nodes_all:
                edges = self._get_all_shortest_path_edges(s_node, e_node)
                if edges:
                    valid_mesh_edges.update(edges)
                    valid_exit_nodes.add(e_node)

        # b. 向G_flow中添加节点和边
        all_mesh_nodes = set(n for edge in valid_mesh_edges for n in edge)
        G_flow.add_nodes_from(all_mesh_nodes)

        for u, v in valid_mesh_edges:
            G_flow.add_edge(u, v, capacity=B_SENSOR)

        for s_node, s_flows in source_nodes_map.items():
            if s_node in all_mesh_nodes: # 只有能到达出口的源点才加入
                total_remaining = sum(f.remaining_data for f in s_flows)
                capacity = min(FLOW_UPLOAD_RATE * len(s_flows), total_remaining)
                G_flow.add_edge(SUPER_SOURCE, s_node, capacity=capacity)

        for e_node in valid_exit_nodes:
            bw = get_s2gl_bandwidth(e_node[0], e_node[1], k)
            if bw > 0:
                for car in self.cars:
                    if e_node in car.coverage_area:
                        G_flow.add_edge(e_node, f'C_{car.id}', capacity=bw)
                        break

        # 3. 求解最大流
        flow_value = 0
        flow_dict = {}
        if G_flow.has_node(SUPER_SOURCE) and G_flow.has_node(SUPER_SINK) and nx.has_path(G_flow, SUPER_SOURCE, SUPER_SINK):
            try:
                flow_value, flow_dict = nx.maximum_flow(G_flow, SUPER_SOURCE, SUPER_SINK)
            except Exception:
                pass #TODO 忽略一些可能因图结构产生的异常???!

        # 4. 保存当前时间步的策略快照
        snapshot = {
            'time': k,
            'flow_value': flow_value,
            'flow_dict': flow_dict, # 保存完整的流量分配细节
            'g_flow_structure': {
                'nodes': list(G_flow.nodes()),
                'edges': [(u, v, G_flow.get_edge_data(u, v)) for u, v in G_flow.edges()]
            }
        }
        self.snapshots.append(snapshot)
        
        self.total_flow_sent += flow_value
        
        #TODO 5. 更新数据流状态,如何恢复快照更新
        if flow_value > 0:
            for s_node, s_flows in source_nodes_map.items():
                if SUPER_SOURCE in flow_dict and s_node in flow_dict[SUPER_SOURCE]:
                    flow_out_of_s_node = flow_dict[SUPER_SOURCE][s_node]
                    
                    # 按比例精确分配已发送的流量
                    total_remaining = sum(f.remaining_data for f in s_flows)
                    if total_remaining > 0:
                        for f in s_flows:
                            ratio = f.remaining_data / total_remaining
                            sent_amount = min(f.remaining_data, flow_out_of_s_node * ratio)
                            f.remaining_data -= sent_amount
                            f.flow_sent_history[k] = sent_amount
                            if f.status == 'waiting': f.status = 'active'
                            if f.remaining_data <= 1e-6:
                                f.remaining_data = 0
                                f.status = 'completed'
                                f.end_time = k + 1

    def get_results(self):
        # ... (此函数与上一版类似，但现在可以额外返回snapshots) ...
        # (代码与上一版相同)
        completed_flows = [f for f in self.flows if f.status == "completed"]
        total_data_generated = len(self.flows) * FLOW_DATA_SIZE
        
        total_data_received = self.total_flow_sent
        
        loss_rate = 1 - (total_data_received / total_data_generated) if total_data_generated > 0 else 0
        avg_delay = np.mean([f.end_time - f.start_time for f in completed_flows]) if completed_flows else float('inf')
        
        print("\n--- 性能评估 (最大流精确子图版) ---")
        print(f"总生成流量: {total_data_generated:.2f} Mb")
        print(f"总接收流量: {total_data_received:.2f} Mb")
        print(f"丢包率: {loss_rate:.2%}")
        print(f"完成传输的流数量: {len(completed_flows)} / {len(self.flows)}")
        print(f"已完成流的平均时延: {avg_delay:.2f} 秒")
        
        return {
            "loss_rate": loss_rate, "avg_delay": avg_delay,
            "completed_count": len(completed_flows),
            "flows": self.flows,
            "snapshots": self.snapshots # 返回完整的策略快照
        }
