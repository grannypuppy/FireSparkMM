# config.py

# 网络参数
M_ROWS = 20  # 传感器网格行数
N_COLS = 30  # 传感器网格列数

# 仿真时间参数
TOTAL_TIME = 90  # 系统总运行时间 (秒)
TIME_STEP = 1    # 时间步长 (秒)

# 传输参数
B_SENSOR = 10     # Mesh网络链路传输容量 (Mbps)
T_SENSOR = 0.05   # Mesh网络单跳传输时延 (秒)
T_S2GL = 0.05     # 传感器到车的传输时延 (秒)

# 移动车辆参数
NUM_CARS = 3
# 车辆覆盖的y坐标(列)，基于Q&A #9的澄清
# 注意：Python索引从0开始，所以原始的(5,6)变为(4,5)
CAR_Y_COORDS = [
    [4, 5],
    [14, 15],
    [24, 25]
]
CAR_COVERAGE_X = 2  # 车辆在x方向(行)的覆盖范围
CAR_TRIP_DURATION = 100  # 车辆单程运行时间 (秒)

# 信号带宽函数参数 (S2GL)
B_PEAK = 20         # S2GL峰值带宽 (Mbps)
BW_PERIOD = 10      # 带宽函数周期 (秒)
B_RECEIVE = 100   # 单辆车最大接收总带宽 (Mbps)

# 数据流参数
TOTAL_FLOWS = 1800         # 数据流总数
FLOW_DATA_SIZE = 10        # 每个流的总数据量 (Mb)
FLOW_UPLOAD_RATE = 5       # 单个流上传至车的固定速率 (Mbps)

# 新算法参数
MAX_PATH_LENGTH = int(TIME_STEP / T_SENSOR) # 最短路径的跳数限制