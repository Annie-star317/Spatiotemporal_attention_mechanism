import torch

# GPU设置
if  torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# DQN超参数
BATCH_SIZE = 64                                 # 样本数量
LR = 0.0001                                       # 学习率
EPSILON = 0.9                                   # greedy policy
GAMMA = 0.99                                     # reward discount
TARGET_REPLACE_ITER = 100                        # 目标网络更新频率(固定不懂的Q网络)
MEMORY_CAPACITY = 100000

# 路径配置
path_list = {
    'resetFlag': 'D:/project_Spatiotemporal_attention_mechanism/python_scripts/resetFlag.txt',
    'resetFlag1': 'D:/project_Spatiotemporal_attention_mechanism/python_scripts/resetFlag1.txt',
    'memory_graph': 'D:/project_Spatiotemporal_attention_mechanism/python_scripts/GNN/memory/memory_graph.pkl',
    'photo_path_real': 'D:/project_Spatiotemporal_attention_mechanism/python_scripts',
    'photo_path': 'D:/project_Spatiotemporal_attention_mechanism/python_scripts/photo',
     
    #DQN 
    'model_path_catch_DQN': 'D:/project_Spatiotemporal_attention_mechanism/python_scripts/DQN/checkpoint/catch',
    'model_path_tai_DQN': 'D:/project_Spatiotemporal_attention_mechanism/python_scripts/DQN/checkpoint/tai',
    'shu_ju_path_DQN':'D:/project_Spatiotemporal_attention_mechanism/python_scripts/DQN/shu_ju.txt',
    'gps_path_DQN':'D:/project_Spatiotemporal_attention_mechanism/python_scripts/DQN/gps.txt',
    'catch_log_path_DQN':'D:/project_Spatiotemporal_attention_mechanism/python_scripts/DQN/log/catch_log/',
    'tai_log_path_DQN':'D:/project_Spatiotemporal_attention_mechanism/python_scripts/DQN/log/tai_log/',

    #PPO
    'model_path_catch_PPO': 'D:/project_Spatiotemporal_attention_mechanism/python_scripts/PPO/checkpoint/catch',
    'model_path_tai_PPO': 'D:/project_Spatiotemporal_attention_mechanism/python_scripts/PPO/checkpoint/tai',
    'shu_ju_path_PPO':'D:/project_Spatiotemporal_attention_mechanism/python_scripts/PPO/shu_ju.txt',
    'gps_path_PPO':'D:/project_Spatiotemporal_attention_mechanism/python_scripts/PPO/gps.txt',
    'catch_log_path_PPO':'D:/project_Spatiotemporal_attention_mechanism/python_scripts/PPO/log/catch_log/',
    'tai_log_path_PPO':'D:/project_Spatiotemporal_attention_mechanism/python_scripts/PPO/log/tai_log/',

    #SAC
    'model_path_catch_SAC': 'D:/project_Spatiotemporal_attention_mechanism/python_scripts/SAC/checkpoint/catch',
    'model_path_tai_SAC': 'D:/project_Spatiotemporal_attention_mechanism/python_scripts/SAC/checkpoint/tai',
    'shu_ju_path_SAC':'D:/project_Spatiotemporal_attention_mechanism/python_scripts/SAC/shu_ju.txt',
    'gps_path_SAC':'D:/project_Spatiotemporal_attention_mechanism/python_scripts/SAC/gps.txt',
    'catch_log_path_SAC':'D:/project_Spatiotemporal_attention_mechanism/python_scripts/SAC/log/catch_log/',
    'tai_log_path_SAC':'D:/project_Spatiotemporal_attention_mechanism/python_scripts/SAC/log/tai_log/'

}

gps_goal = [0.2, 0.175]  # 目标位置坐标1
gps_goal1 = [0.27, 0.225]  # 目标位置坐标2
# SAC超参数
LR_ACTOR = 0.0001 # 策略网络学习率
LR_CRITIC = 0.0001 # 评论家网络学习率
TAU = 0.005
# 机器人运动标准参数
class Darwin_config:
    # 添加最小高度和最小前进位置参数
    min_height = 0.2  # 夹爪最小高度
    min_forward = 0.18  # 夹爪最小前进位置
    gps_goal = [0.2, 0.185]  # 目标位置（高度，前进）
    standard_angle = -32.64359902756043  # 标准角度
    limit = [[-3.14, 3.14], [-3.14, 2.85], [-0.68, 2.30], [-2.25, 0.77], 
             [-1.65, 1.16], [-1.18, 1.63], [-2.42, 0.66], [-0.69, 2.50], 
             [-1.01, 1.01], [-1.00, 0.93], [-1.77, 0.45], [-0.50, 1.68],
             [-0.02, 2.25], [-2.25, 0.03], [-1.24, 1.38], [-1.39, 1.22], 
             [-0.68, 1.04], [-1.02, 0.60], [-1.81, 1.81], [-0.36, 0.94]]  # 限制
    touch_T = [1.0, 1.0]  # 压力传感器目标值
    touch_F = [0.0, 0.0]  # 压力传感器失败值
    acc_low = [480, 450, 580]  # 加速度传感器低值
    acc_high = [560, 530, 700]  # 加速度传感器高值
    gyro_low = [500, 500, 500]  # 陀螺仪低值
    gyro_high = [520, 520, 520]  # 陀螺仪高值





