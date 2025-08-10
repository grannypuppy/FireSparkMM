# entity.py

import numpy as np
from pyparsing import C
from config import *

class DataFlow:
    """定义数据流对象"""
    def __init__(self, flow_id, source_node, start_time):
        self.id = flow_id
        self.source_node = source_node  # (x, y)
        self.start_time = start_time
        self.total_data = FLOW_DATA_SIZE
        self.remaining_data = FLOW_DATA_SIZE
        self.status = "waiting"  # waiting, active, completed
        self.end_time = -1
        self.flow_sent_history = {} # {time_k: amount_sent}

    def __repr__(self):
        return f"Flow({self.id}, Source:{self.source_node}, Start:{self.start_time}, Left:{self.remaining_data:.2f}Mb)"

class MobileCar:
    """定义移动信号接收车对象"""
    def __init__(self, car_id, y_coords):
        self.id = car_id
        self.y_coords = y_coords
        self.speed = (M_ROWS - CAR_COVERAGE_X) / CAR_TRIP_DURATION if CAR_TRIP_DURATION > 0 else 0
        self.coverage_area = set()

    def update_position(self, time_k):
        """根据时间更新车辆位置和覆盖范围"""
        # 假设车辆从x=0开始移动
        # Python索引从0开始，所以x坐标范围是0到M_ROWS-1
        start_x_float = self.speed * time_k
        start_x_int = int(np.ceil(start_x_float))

        self.coverage_area.clear()
        for x_offset in range(CAR_COVERAGE_X):
            current_x = start_x_int + x_offset
            if 0 <= current_x < M_ROWS:
                for y in self.y_coords:
                    self.coverage_area.add((current_x, y))

        if start_x_int == start_x_float:
            for y in self.y_coords:
                self.coverage_area.add((start_x_int + CAR_COVERAGE_X, y))
        else:
            pre_two_point = [(start_x_int-1, y) for y in self.y_coords]
            last_two_point = [(start_x_int+CAR_COVERAGE_X, y) for y in self.y_coords]
            pre_sum = sum(get_s2gl_bandwidth(x, y, time_k) for x, y in pre_two_point)
            last_sum = sum(get_s2gl_bandwidth(x, y, time_k) for x, y in last_two_point)
            if pre_sum >= last_sum:
                for x, y in pre_two_point:
                    self.coverage_area.add((x, y))
            else:
                for x, y in last_two_point:
                    self.coverage_area.add((x, y))

    def __repr__(self):
        return f"Car({self.id}, Y-coords:{self.y_coords}, Coverage Size:{len(self.coverage_area)})"

def get_s2gl_bandwidth(x, y, time_k):
    """计算传感器(x,y)在k时刻到车的瞬时带宽"""
    # y是列编号，对应题目中的m
    # 题目公式: phi = 5 + int(t/3) - m, m为列编号 (1-based)
    # Python索引: y (0-based), 所以 m = y + 1
    phase = 5 + int(time_k / 3) - (y + 1)
    
    # 计算在周期内的相对时间
    time_in_period = (phase + time_k) % BW_PERIOD
    
    if 0 <= time_in_period < 5:
        return (B_PEAK / 5) * time_in_period
    elif 5 <= time_in_period < 10:
        return (B_PEAK / 5) * (10 - time_in_period)
    return 0