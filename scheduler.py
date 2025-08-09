# scheduler.py

import networkx as nx
import numpy as np
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
            self._schedule_at_step(k)
        print("仿真结束。")

    def _schedule_at_step(self, k):
        """在单个时间步长k内执行调度逻辑"""
        # 1. 更新系统状态
        for car in self.cars:
            car.update_position(k)
        
        available_car_bw = {car.id: B_RECEIVE for car in self.cars}
        
        # 2. 寻找候选调度任务
        active_flows = [f for f in self.flows if f.start_time <= k and f.status != "completed"]
        
        candidate_tasks = []
        for flow in active_flows:
            for car_id, car in enumerate(self.cars):
                for exit_node in car.coverage_area:
                    # 检查S2GL带宽
                    if get_s2gl_bandwidth(exit_node[0], exit_node[1], k) >= FLOW_UPLOAD_RATE:
                        try:
                            # 寻找最短路径
                            path = nx.shortest_path(self.graph, source=flow.source_node, target=exit_node)
                            hops = len(path) - 1
                            delay = hops * T_SENSOR + T_S2GL
                            waiting_time = k - flow.start_time
                            # (优先级, 任务详情)
                            candidate_tasks.append(
                                ((delay, -waiting_time), (flow, path, car_id)) # 优先等待时间长的
                            )
                        except nx.NetworkXNoPath:
                            continue
        
        # 3. 执行调度 (贪心选择)
        candidate_tasks.sort(key=lambda x: x[0]) # 按(delay, -waiting_time)排序
        
        scheduled_flows_this_step = set()

        for _, (flow, path, car_id) in candidate_tasks:
            if flow.id in scheduled_flows_this_step:
                continue

            # 检查资源是否可用
            is_path_available = True
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                # 保证边的顺序一致
                edge = tuple(sorted((u, v)))
                if self.link_usage[k][edge] + FLOW_UPLOAD_RATE > B_SENSOR:
                    is_path_available = False
                    break
            
            if is_path_available and available_car_bw[car_id] >= FLOW_UPLOAD_RATE:
                # 分配资源
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    edge = tuple(sorted((u, v)))
                    self.link_usage[k][edge] += FLOW_UPLOAD_RATE
                
                available_car_bw[car_id] -= FLOW_UPLOAD_RATE
                
                # 更新流状态
                flow.remaining_data -= FLOW_UPLOAD_RATE * TIME_STEP
                flow.path_history[k] = path
                scheduled_flows_this_step.add(flow.id)
                
                if flow.status == "waiting":
                    flow.status = "active"

                if flow.remaining_data <= 0:
                    flow.remaining_data = 0
                    flow.status = "completed"
                    flow.end_time = k + 1 # 数据在下一秒初到达

    def get_results(self):
        """计算并返回最终的性能指标"""
        completed_flows = [f for f in self.flows if f.status == "completed"]
        
        total_data_generated = len(self.flows) * FLOW_DATA_SIZE
        total_data_received = sum([(f.total_data - f.remaining_data) for f in self.flows])
        
        loss_rate = 1 - (total_data_received / total_data_generated) if total_data_generated > 0 else 0
        
        if completed_flows:
            avg_delay = np.mean([f.end_time - f.start_time for f in completed_flows])
        else:
            avg_delay = float('inf')
            
        print("\n--- 性能评估 ---")
        print(f"总生成流量: {total_data_generated:.2f} Mb")
        print(f"总接收流量: {total_data_received:.2f} Mb")
        print(f"丢包率: {loss_rate:.2%}")
        print(f"完成传输的流数量: {len(completed_flows)} / {len(self.flows)}")
        print(f"已完成流的平均时延: {avg_delay:.2f} 秒")
        
        return {
            "loss_rate": loss_rate,
            "avg_delay": avg_delay,
            "completed_count": len(completed_flows),
            "link_usage": self.link_usage,
            "cars": self.cars,
            "graph": self.graph,
            "flows": self.flows
        }