import torch
import torch.nn as nn
import numpy as np
from python_scripts.Project_config import  path_list, BATCH_SIZE, LR, EPSILON, GAMMA, TARGET_REPLACE_ITER, MEMORY_CAPACITY, device, gps_goal, gps_goal1

# 只保留Net2类，用于第二阶段训练
class Net2(nn.Module):
    def __init__(self, act_dim):
        super().__init__()
        self.fc0 = nn.Linear(in_features=20, out_features=2000)
        self.fc1 = nn.Linear(in_features=2000, out_features=2000)
        self.fc2 = nn.Linear(in_features=2000, out_features=act_dim)

    def forward(self, x):
        x = torch.tensor(x).to(device)
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class DQN2(object):
    def __init__(self):
        # 创建评估网络和目标网络
        self.eval_net, self.target_net = Net2(1).to(device), Net2(1).to(device)
        self.learn_step_counter = 0  # 学习步数记录
        self.memory_counter = 0      # 记忆量计数
        self.memory = np.zeros((MEMORY_CAPACITY, 6))  # 存储空间初始化
        self.optimazer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()  # 使用均方损失函数
        self.loss_func = self.loss_func.to(device)

    def choose_action(self, episode_num, robot_state):
        # 根据训练阶段调整探索率
        # 探索率表示选择最优动作的概率，1-threshold表示随机探索的概率
        if episode_num < 10:
            # 开始时更多随机探索，更少利用
            threshold = 0.1  # 10%选择最优动作，90%随机探索
        elif episode_num < 50:
            threshold = 0.3  # 30%选择最优动作，70%随机探索
        elif episode_num < 100:
            threshold = 0.5  # 50%选择最优动作，50%随机探索
        elif episode_num < 500:
            threshold = 0.7  # 70%选择最优动作，30%随机探索
        else:
            threshold = 0.9  # 90%选择最优动作，10%随机探索
            
        # 打印当前探索率
        print(f"[抬腿] 当前探索率: {threshold}, 周期: {episode_num}")
            
        if np.random.uniform() < threshold:  # 选择最优动作
            action = self.eval_net.forward(robot_state)
            # 打印动作值以进行调试
            action = action.item()
            print(f"[抬腿] 动作值: {action}")
        else:
            # 生成[-1,1]范围内的随机浮点数
            action = np.random.uniform(-1.0, 1.0)
        return action 

    def learn(self, rpm):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print("更新目标网络")
        self.learn_step_counter += 1
        
        b_s, b_a, b_r, b_s_, done = rpm.sample(32)
        loss_all = 0
        
        for i in range(32):
            q_eval = self.eval_net(b_s[i])[int(b_a[i])]
            q_next = self.target_net(b_s_[i])
            q_target = b_r[i] + GAMMA * q_next.max(0)[0]
            loss = self.loss_func(q_eval, q_target)
            loss_all = loss_all + loss
            
        loss_all = loss_all / 32
        self.optimazer.zero_grad()
        loss_all.backward()
        self.optimazer.step()
        
        return loss_all

