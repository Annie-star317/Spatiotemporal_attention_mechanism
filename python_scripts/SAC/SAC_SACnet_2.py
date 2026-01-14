import torch
import torch.nn as nn
import numpy as np
from python_scripts.Project_config import path_list, BATCH_SIZE, LR_ACTOR, LR_CRITIC, GAMMA, TAU, MEMORY_CAPACITY, device, gps_goal, gps_goal1

# 抬腿阶段的策略网络(Actor)
class PolicyNet2(nn.Module):
    def __init__(self, state_dim=20, act_dim=6):
        super(PolicyNet2, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.mean_linear = nn.Linear(256, act_dim)
        self.log_std_linear = nn.Linear(256, act_dim)
        
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        
    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32).to(device)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        
        return mean, log_std
    
    def sample(self, x):
        mean, log_std = self.forward(x)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # 重参数化采样
        action = torch.tanh(x_t)
        
        # 计算log概率，用于策略更新
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob

# 抬腿阶段的Q网络(Critic)
class QNet2(nn.Module):
    def __init__(self, state_dim=20, act_dim=6):
        super(QNet2, self).__init__()
        
        # Q1网络
        self.fc1_q1 = nn.Linear(state_dim + act_dim, 512)
        self.fc2_q1 = nn.Linear(512, 256)
        self.fc3_q1 = nn.Linear(256, 1)
        
        # Q2网络
        self.fc1_q2 = nn.Linear(state_dim + act_dim, 512)
        self.fc2_q2 = nn.Linear(512, 256)
        self.fc3_q2 = nn.Linear(256, 1)
    
    def forward(self, state, action):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        
        # 确保action是张量并且在设备上
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)
        action = action.to(device)
        
        # 合并状态和动作
        x = torch.cat([state, action], dim=-1)
        
        # Q1值
        q1 = torch.relu(self.fc1_q1(x))
        q1 = torch.relu(self.fc2_q1(q1))
        q1 = self.fc3_q1(q1)
        
        # Q2值
        q2 = torch.relu(self.fc1_q2(x))
        q2 = torch.relu(self.fc2_q2(q2))
        q2 = self.fc3_q2(q2)
        
        return q1, q2
    
    def q1(self, state, action):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        
        # 确保action是张量并且在设备上
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)
        action = action.to(device)
        
        # 合并状态和动作
        x = torch.cat([state, action], dim=-1)
        
        # 只计算Q1值
        q1 = torch.relu(self.fc1_q1(x))
        q1 = torch.relu(self.fc2_q1(q1))
        q1 = self.fc3_q1(q1)
        
        return q1

# SAC算法实现 - 抬腿阶段
class SAC2(object):
    def __init__(self):
        self.state_dim = 20
        self.act_dim = 6
        
        # 初始化网络
        self.policy_net = PolicyNet2(self.state_dim, self.act_dim).to(device)
        self.q_net = QNet2(self.state_dim, self.act_dim).to(device)
        self.target_q_net = QNet2(self.state_dim, self.act_dim).to(device)
        
        # 硬拷贝参数到目标网络
        for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(param.data)
        
        # 优化器
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=LR_ACTOR)
        self.q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=LR_CRITIC)
        
        # 自动调整熵权重参数
        self.target_entropy = -self.act_dim  # 目标熵值
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = torch.exp(self.log_alpha)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=LR_ACTOR)
        
        self.learn_step_counter = 0
    
    def choose_action(self, episode_num, robot_state, evaluate=False):
        with torch.no_grad():
            if evaluate:  # 评估模式，使用策略的均值
                mean, _ = self.policy_net.forward(robot_state)
                action = torch.tanh(mean).cpu().numpy()
                # 将连续动作映射为离散动作（兼容原有接口）
                discrete_action = np.argmax(action)
                return discrete_action
            else:  # 训练模式，从分布中采样
                action, _ = self.policy_net.sample(robot_state)
                action = action.cpu().numpy()
                # 将连续动作映射为离散动作（兼容原有接口）
                discrete_action = np.argmax(action)
                return discrete_action
    
    def learn(self, rpm):
        # 从经验回放中采样
        b_s, b_a, b_r, b_s_, done = rpm.sample(BATCH_SIZE)
        
        batch_states = []
        batch_next_states = []
        batch_actions = []
        batch_rewards = []
        batch_dones = []
        
        for i in range(BATCH_SIZE):
            batch_states.append(b_s[i])
            batch_next_states.append(b_s_[i])
            
            # 将离散动作转换为独热编码形式的连续动作向量
            action_one_hot = np.zeros(self.act_dim)
            action_idx = int(b_a[i])
            if 0 <= action_idx < self.act_dim:  # 防止索引越界
                action_one_hot[action_idx] = 1.0
            
            batch_actions.append(torch.FloatTensor(action_one_hot).to(device))
            batch_rewards.append(torch.FloatTensor([b_r[i]]).to(device))
            batch_dones.append(torch.FloatTensor([done[i]]).to(device))
        
        # 转换为批量tensor
        batch_states = torch.FloatTensor(np.array(batch_states)).to(device)
        batch_next_states = torch.FloatTensor(np.array(batch_next_states)).to(device)
        batch_actions = torch.stack(batch_actions).to(device)
        batch_rewards = torch.stack(batch_rewards).to(device)
        batch_dones = torch.stack(batch_dones).to(device)
        
        # 计算目标Q值
        with torch.no_grad():
            next_actions = []
            next_log_probs = []
            
            for i in range(BATCH_SIZE):
                next_action, next_log_prob = self.policy_net.sample(batch_next_states[i])
                next_actions.append(next_action)
                next_log_probs.append(next_log_prob)
            
            next_actions = torch.stack(next_actions).to(device)
            next_log_probs = torch.stack(next_log_probs).to(device)
            
            # 计算目标Q值
            next_q1_values = []
            next_q2_values = []
            
            for i in range(BATCH_SIZE):
                next_q1, next_q2 = self.target_q_net(batch_next_states[i], next_actions[i])
                next_q1_values.append(next_q1)
                next_q2_values.append(next_q2)
            
            next_q1_values = torch.stack(next_q1_values).to(device)
            next_q2_values = torch.stack(next_q2_values).to(device)
            
            next_q_values = torch.min(next_q1_values, next_q2_values)
            next_q_values = next_q_values - self.alpha * next_log_probs
            target_q_values = batch_rewards + (1 - batch_dones) * GAMMA * next_q_values
        
        # 更新Critic
        q1_values = []
        q2_values = []
        
        for i in range(BATCH_SIZE):
            q1, q2 = self.q_net(batch_states[i], batch_actions[i])
            q1_values.append(q1)
            q2_values.append(q2)
        
        q1_values = torch.stack(q1_values).to(device)
        q2_values = torch.stack(q2_values).to(device)
        
        q1_loss = torch.nn.functional.mse_loss(q1_values, target_q_values)
        q2_loss = torch.nn.functional.mse_loss(q2_values, target_q_values)
        q_loss = q1_loss + q2_loss
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        # 更新Policy
        new_actions = []
        log_probs = []
        
        for i in range(BATCH_SIZE):
            new_action, log_prob = self.policy_net.sample(batch_states[i])
            new_actions.append(new_action)
            log_probs.append(log_prob)
        
        new_actions = torch.stack(new_actions).to(device)
        log_probs = torch.stack(log_probs).to(device)
        
        q1_pi = []
        
        for i in range(BATCH_SIZE):
            q1_value = self.q_net.q1(batch_states[i], new_actions[i])
            q1_pi.append(q1_value)
        
        q1_pi = torch.stack(q1_pi).to(device)
        
        policy_loss = (self.alpha * log_probs - q1_pi).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # 更新alpha
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = torch.exp(self.log_alpha)
        
        # 软更新目标网络
        if self.learn_step_counter % 2 == 0:
            for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        
        self.learn_step_counter += 1
        
        return q_loss.item() + policy_loss.item() + alpha_loss.item()

