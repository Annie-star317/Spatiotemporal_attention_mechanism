import torch
import torch.nn as nn
import numpy as np
import torch_geometric
from torch_geometric.data import Data
from python_scripts.Project_config import device
# 超参数
BATCH_SIZE = 64                                 # 样本数量
LR_ACTOR = 0.0001                               # 策略网络学习率
LR_CRITIC = 0.0001                              # 评论家网络学习率
GAMMA = 0.99                                    # 奖励折扣
TAU = 0.005                                     # 软更新系数
MEMORY_CAPACITY = 100000


class FeatureExtractor(nn.Module):
    def __init__(self, node_num):    # 特征提取器，用于提取图像、状态和图结构的特征
        super().__init__()  
        self.node_num = node_num  # 点总数
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=1)  # 卷积层1
        self.relu = nn.ReLU()  # 激活函数
        self.Sigmoid = nn.Sigmoid()  # 激活函数
        self.maxpool1 = nn.MaxPool2d(2, stride=2)  # 池化层1
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=(2, 2))  # 卷积层2
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=1)  # 卷积层3    
        self.fc0 = nn.Linear(in_features=6272, out_features=6000)  # 全连接层0  
        self.fc1 = nn.Linear(in_features=6000, out_features=100)  # 全连接层1
        self.fc2 = nn.Linear(in_features=4, out_features=100)  # 全连接层2
        self.fc3 = nn.Linear(in_features=100, out_features=100)  # 全连接层3
        self.fc4 = nn.Linear(in_features=300, out_features=200)  # 全连接层4
        self.conv_graph1 = torch_geometric.nn.GraphSAGE(1, 1000, 2, aggr='add')    # 图卷积层1
        self.conv_graph2 = torch_geometric.nn.GATConv(1000, 1000, aggr='add')  # 图卷积层2
        self.conv_graph3 = torch_geometric.nn.GraphSAGE(1000, 1000, 2, aggr='add')  # 图卷积层3
        self.conv_graph4 = torch_geometric.nn.GATConv(1000, 1000, aggr='add')  # 图卷积层4
        self.conv_graph5 = torch_geometric.nn.GCNConv(1000, 1000, 2, aggr='add')  # 图卷积层5
        self.fc_graph = nn.Linear(1000, 100)  # 全连接层6


    def create_edge_index(self):    # 创建图的边索引
        ans = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
             1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],

            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
             17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        ]
        return torch.tensor(ans, dtype=torch.long)
    
    # 为图中的节点生成信息
    def creat_x(self, x_graph):
        ans = [[] for i in range(self.node_num)]    # 创建图的节点信息
        for i in range(len(ans)):
            ans[i] = [x_graph[i]]
        return ans
    
    # 结合前面的两个函数生成图神经网络使用的图
    def creat_graph(self, x_graph):   # 生成图神经网络使用的图
        x = torch.as_tensor(self.creat_x(x_graph), dtype=torch.float32)    # 创建图的节点信息  
        # 修改为PyTorch推荐的方式创建张量
        edge_index = torch.as_tensor(self.create_edge_index(), dtype=torch.long)    # 创建图的边索引
        graph = Data(x=x, edge_index=edge_index)    # 创建图
        graph.x = graph.x.to(device)
        graph.edge_index = graph.edge_index.to(device)  # 将图的边索引移动到GPU
        return graph

    def forward(self, x, state, x_graph):
        
        self.graph = self.creat_graph(x_graph) # 创建图
        x = torch.as_tensor(x, dtype=torch.float32).to(device)  # 将输入数据转换为张量并移动到GPU
        # x = torch.unsqueeze(x, dim=0)  # 在第0维增加一个维度
        # 添加通道维度，确保输入是4维的 [batch_size, channels, height, width]
        x = torch.unsqueeze(x, dim=0)  # 在第0维增加一个维度，变成 [1, 128, 128]
        x = torch.unsqueeze(x, dim=1)  # 在第1维增加一个维度，变成 [1, 1, 128, 128]
        x = self.conv1(x)  # 卷积层1
        x = self.relu(x)  # 激活函数
        x = self.conv2(x)  # 卷积层2
        x = self.relu(x)  # 激活函数
        x = self.conv3(x)  # 卷积层3
        x = x.view(x.size(0), -1)  # 展平张量
        x = torch.flatten(x)  # 展平张量
        x = self.fc0(x)  # 全连接层0
        x = self.fc1(x)  # 全连接层1
        min_val1 = torch.min(x)  # 最小值
        max_val1 = torch.max(x)  # 最大值
        normalized_data1 = torch.div(torch.sub(x, min_val1), torch.sub(max_val1, min_val1))  # 归一化 x
        state = torch.as_tensor(state, dtype=torch.float32).to(device)  # 将状态转换为张量并移动到GPU
        state = self.fc2(state)  # 全连接层2
        state = self.fc3(state)  # 全连接层3
        min_val2 = torch.min(state)  # 最小值
        max_val2 = torch.max(state)  # 最大值
        normalized_data2 = torch.div(torch.sub(state, min_val2), torch.sub(max_val2, min_val2))  # 归一化 state
        x_graph = self.creat_graph(x_graph) 
        edge_index = x_graph.edge_index
        x_graph = self.conv_graph1(x_graph.x, edge_index)  # 图卷积层1
        x_graph = self.relu(x_graph)  # 激活函数
        x_graph = self.conv_graph2(x_graph, edge_index)  # 图卷积层2
        x_graph = self.relu(x_graph)  # 激活函数
        x_graph = self.conv_graph3(x_graph, edge_index)  # 图卷积层3
        x_graph = self.relu(x_graph)  # 激活函数
        x_graph = self.conv_graph4(x_graph, edge_index)  # 图卷积层4
        x_graph = self.relu(x_graph)  # 激活函数
        x_graph = self.conv_graph5(x_graph, edge_index)  # 图卷积层5
        x_graph = torch.mean(x_graph, dim=0)  # 全局平均池化
        x_graph = self.fc_graph(x_graph)  # 全连接层6
        min_val3 = torch.min(x_graph)  # 最小值
        max_val3 = torch.max(x_graph)  # 最大值
        normalized_x_graph = torch.div(torch.sub(x_graph, min_val3), torch.sub(max_val3, min_val3))  # 归一化 x_graph
        state_x = torch.cat((normalized_data1, normalized_data2, normalized_x_graph), dim=-1)  # 拼接
        state_x = self.fc4(state_x)  # 全连接层4
        return state_x

# 策略网络（Actor）
class PolicyNetwork(nn.Module):
    def __init__(self, feature_dim, act_dim, node_num):
        super(PolicyNetwork, self).__init__()
        self.feature_extractor = FeatureExtractor(node_num)
        self.mean_linear = nn.Linear(200, act_dim)
        self.log_std_linear = nn.Linear(200, act_dim)
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        
    def forward(self, x, state, x_graph):
        features = self.feature_extractor(x, state, x_graph)
        mean = self.mean_linear(features)
        log_std = self.log_std_linear(features)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std
        
    def sample(self, x, state, x_graph):
        mean, log_std = self.forward(x, state, x_graph)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # 重参数化采样
        action = torch.tanh(x_t)
        
        # 计算log概率，用于策略更新
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob
        
    def get_action(self, x, state, x_graph):
        mean, log_std = self.forward(x, state, x_graph)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        
        return action.detach().cpu().numpy()


# Q值网络（Critic）
class QNetwork(nn.Module):
    def __init__(self, feature_dim, act_dim, node_num):
        super(QNetwork, self).__init__()
        self.feature_extractor = FeatureExtractor(node_num)
        
        # Q1网络
        self.q1_linear1 = nn.Linear(200 + act_dim, 256)
        self.q1_linear2 = nn.Linear(256, 256)
        self.q1_linear3 = nn.Linear(256, 1)
        
        # Q2网络
        self.q2_linear1 = nn.Linear(200 + act_dim, 256)
        self.q2_linear2 = nn.Linear(256, 256)
        self.q2_linear3 = nn.Linear(256, 1)
        
    def forward(self, x, state, x_graph, action):
        features = self.feature_extractor(x, state, x_graph)
        
        # 合并特征和动作
        x1 = torch.cat([features, action], dim=-1)
        x1 = torch.relu(self.q1_linear1(x1))
        x1 = torch.relu(self.q1_linear2(x1))
        q1 = self.q1_linear3(x1)
        
        x2 = torch.cat([features, action], dim=-1)
        x2 = torch.relu(self.q2_linear1(x2))
        x2 = torch.relu(self.q2_linear2(x2))
        q2 = self.q2_linear3(x2)
        
        return q1, q2
    
    def q1(self, x, state, x_graph, action):
        features = self.feature_extractor(x, state, x_graph)
        
        x1 = torch.cat([features, action], dim=-1)
        x1 = torch.relu(self.q1_linear1(x1))
        x1 = torch.relu(self.q1_linear2(x1))
        q1 = self.q1_linear3(x1)
        
        return q1

# SAC算法实现
class SAC(object):
    def __init__(self, act_dim, node_num):
        self.node_num = node_num
        
        # 初始化网络
        self.policy_net = PolicyNetwork(200, act_dim, self.node_num).to(device)
        self.q_net = QNetwork(200, act_dim, self.node_num).to(device)
        self.target_q_net = QNetwork(200, act_dim, self.node_num).to(device)
        
        # 硬拷贝参数到目标网络
        for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(param.data)
            
        # 优化器
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=LR_ACTOR)
        self.q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=LR_CRITIC)
        
        # 自动调整熵权重参数
        self.target_entropy = -act_dim  # 目标熵值
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = torch.exp(self.log_alpha)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=LR_ACTOR)
        
        self.learn_step_counter = 0
        
    def choose_action(self, episode_num, obs, x_graph, evaluate=False):
        x, state = obs
        
        with torch.no_grad():
            if evaluate:  # 评估模式，使用确定性策略
                mean, _ = self.policy_net(x, state, x_graph)
                return torch.tanh(mean).cpu().numpy()
            else:  # 训练模式，使用随机策略
                action, _ = self.policy_net.sample(x, state, x_graph)
                return action.cpu().numpy()
    
    def learn(self, rpm):
        # 从经验回放中采样
        b_o, b_s, b_a, b_r, b_o_, b_s_, done = rpm.sample(BATCH_SIZE)
        x_graph = b_s
        
        batch_states = []
        batch_next_states = []
        batch_actions = []
        batch_rewards = []
        batch_dones = []
        
        for i in range(BATCH_SIZE):
            state = [b_s[i][1], b_s[i][0], b_s[i][5], b_s[i][4]]
            next_state = [b_s_[i][1], b_s_[i][0], b_s_[i][5], b_s_[i][4]]
            
            batch_states.append((b_o[i], state, x_graph[i]))
            batch_next_states.append((b_o_[i], next_state, x_graph[i]))
            batch_actions.append(torch.FloatTensor(b_a[i]).to(device))
            batch_rewards.append(torch.FloatTensor([b_r[i]]).to(device))
            batch_dones.append(torch.FloatTensor([done[i]]).to(device))
        
        # 转换为批量tensor
        batch_actions = torch.stack(batch_actions).to(device)
        batch_rewards = torch.stack(batch_rewards).to(device)
        batch_dones = torch.stack(batch_dones).to(device)
        
        # 计算目标Q值
        with torch.no_grad():
            next_actions = []
            next_log_probs = []
            
            for next_state in batch_next_states:
                next_action, next_log_prob = self.policy_net.sample(*next_state)
                next_actions.append(next_action)
                next_log_probs.append(next_log_prob)
                
            next_actions = torch.stack(next_actions).to(device)
            next_log_probs = torch.stack(next_log_probs).to(device)
            
            # 计算目标Q值
            next_q1_values = []
            next_q2_values = []
            
            for i, next_state in enumerate(batch_next_states):
                next_q1, next_q2 = self.target_q_net(*next_state, next_actions[i])
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
        
        for i, state in enumerate(batch_states):
            q1, q2 = self.q_net(*state, batch_actions[i])
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
        
        for state in batch_states:
            new_action, log_prob = self.policy_net.sample(*state)
            new_actions.append(new_action)
            log_probs.append(log_prob)
            
        new_actions = torch.stack(new_actions).to(device)
        log_probs = torch.stack(log_probs).to(device)
        
        q1_pi = []
        
        for i, state in enumerate(batch_states):
            q1_value = self.q_net.q1(*state, new_actions[i])
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
        
        return q_loss.item(), policy_loss.item(), alpha_loss.item()
