import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from python_scripts.Project_config import path_list, BATCH_SIZE, LR, EPSILON, GAMMA, TARGET_REPLACE_ITER, MEMORY_CAPACITY, device, gps_goal, gps_goal1
import torch_geometric
from torch_geometric.data import Data
from torch.distributions import Normal

# PPO网络结构，包含策略网络和价值网络
class ActorCritic(nn.Module):
    def __init__(self, act_dim, node_num):
        """
        抬腿训练使用的ActorCritic网络
        Args:
            act_dim: 动作维度，抬腿训练中通常为8个动作
            node_num: 图节点数量
        """
        super().__init__()
        self.node_num = node_num
        
        # 图像特征提取网络
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=1)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=1)
        
        # 图像特征全连接层
        self.fc0 = nn.Linear(in_features=6272, out_features=6000)
        self.fc1 = nn.Linear(in_features=6000, out_features=100)
        
        # 状态特征处理层
        self.fc2 = nn.Linear(in_features=4, out_features=100)
        self.fc3 = nn.Linear(in_features=100, out_features=100)
        
        # 图神经网络部分，用于处理关节角度
        self.conv_graph1 = torch_geometric.nn.GraphSAGE(1, 1000, 2, aggr='add')
        self.conv_graph2 = torch_geometric.nn.GATConv(1000, 1000, heads=1, aggr='add')
        self.conv_graph3 = torch_geometric.nn.GraphSAGE(1000, 1000, 2, aggr='add')
        self.conv_graph4 = torch_geometric.nn.GATConv(1000, 1000, heads=1, aggr='add')
        self.conv_graph5 = torch_geometric.nn.GraphSAGE(1000, 1000, 2, aggr='add')
        self.fc_graph = nn.Linear(1000, 100)
        
        # 共享特征层
        self.fc4 = nn.Linear(in_features=300, out_features=200)
        
        # Actor头：输出动作概率
        self.actor_mu = nn.Linear(200, act_dim)  # 输出动作均值
        self.actor_log_sigma = nn.Parameter(torch.zeros(act_dim))  # 可学习的对数标准差
        
        # Critic头：输出状态值
        self.critic = nn.Linear(200, 1)
    
    def create_edge_index(self):
        """创建图边索引"""
        ans = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
             1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
             17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        ]
        return torch.tensor(ans, dtype=torch.long)
    
    def creat_x(self, x_graph):
        """从输入创建图节点特征"""
        ans = [[] for i in range(self.node_num)]
        for i in range(len(ans)):
            ans[i] = [x_graph[i]]
        return ans
    
    def creat_graph(self, x_graph):
        """创建图结构"""
        x = torch.as_tensor(self.creat_x(x_graph), dtype=torch.float32)
        edge_index = torch.as_tensor(self.create_edge_index(), dtype=torch.long)
        graph = Data(x=x, edge_index=edge_index)
        graph.x = graph.x.to(device)
        graph.edge_index = graph.edge_index.to(device)
        return graph

    def forward(self, x, state, x_graph):
        """
        前向传播
        Args:
            x: 图像数据
            state: 状态数据（关节角度等）
            x_graph: 图节点特征数据
        """
        # 创建图
        self.graph = self.creat_graph(x_graph)
        
        # 图像特征处理
        x = torch.as_tensor(x, dtype=torch.float32).to(device)
        x = torch.unsqueeze(x, dim=0)
        x = torch.unsqueeze(x, dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = torch.flatten(x)
        x = self.fc0(x)
        x = self.fc1(x)
        
        # 特征归一化
        min_val1 = torch.min(x)
        max_val1 = torch.max(x)
        normalized_data1 = torch.div(torch.sub(x, min_val1), torch.sub(max_val1, min_val1))
        # 确保normalized_data1是2维的
        if len(normalized_data1.shape) == 1:
            normalized_data1 = normalized_data1.unsqueeze(0)
        
        # 状态特征处理
        state = torch.as_tensor(state, dtype=torch.float32).to(device)
        # 确保状态数据维度正确
        if len(state.shape) == 1:
            state = state.unsqueeze(0)  # 添加批次维度
        # # 处理 state 的数据维度，确保其为 1 * 4
        # if state.shape[1] > 4:
        #     state = state[:, :4]  # 取前四个数据
        # elif state.shape[1] < 4:
        #     # 计算需要补零的数量
        #     padding_size = 4 - state.shape[1]
        #     # 在最后一个维度上补零
        #     state = torch.nn.functional.pad(state, (0, padding_size), mode='constant', value=0)
        # # 确保最终维度为 1 * 4
    # assert state.shape == (1, 4), f"Unexpected state shape after processing: {state.shape}"

        state = self.fc2(state)
        state = self.fc3(state)
        min_val2 = torch.min(state)
        max_val2 = torch.max(state)
        normalized_data2 = torch.div(torch.sub(state, min_val2), torch.sub(max_val2, min_val2))
        
        # 图神经网络处理
        x_graph = self.creat_graph(x_graph)
        edge_index = x_graph.edge_index
        x_graph = self.conv_graph1(x_graph.x, edge_index)
        x_graph = self.relu(x_graph)
        x_graph = self.conv_graph2(x_graph, edge_index)
        x_graph = self.relu(x_graph)
        x_graph = self.conv_graph3(x_graph, edge_index)
        x_graph = self.relu(x_graph)
        x_graph = self.conv_graph4(x_graph, edge_index)
        x_graph = self.relu(x_graph)
        x_graph = self.conv_graph5(x_graph, edge_index)
        x_graph = torch.mean(x_graph, dim=0)
        x_graph = self.fc_graph(x_graph)
        
        # 特征归一化
        min_val3 = torch.min(x_graph)
        max_val3 = torch.max(x_graph)
        normalized_x_graph = torch.div(torch.sub(x_graph, min_val3), torch.sub(max_val3, min_val3))
        # 确保normalized_x_graph是2维的
        if len(normalized_x_graph.shape) == 1:
            normalized_x_graph = normalized_x_graph.unsqueeze(0)
        
        # 特征融合
        state_x = torch.cat((normalized_data1, normalized_data2, normalized_x_graph), dim=-1)
        features = self.fc4(state_x)
        
        # Actor: 输出动作分布的参数
        mu = torch.tanh(self.actor_mu(features))  # 使用 tanh 将均值限制在 [-1, 1]
        log_sigma = self.actor_log_sigma.expand_as(mu)  # 扩展维度以匹配 mu
        sigma = torch.exp(log_sigma)  # 确保标准差为正
        dist = Normal(mu, sigma)  # 创建正态分布
    
        # Critic: 输出状态值
        value = self.critic(features)
    
        return dist, value

class PPO2:
    def __init__(self, node_num, env_information=None):
        """
        PPO算法，专门为抬腿训练调整
        Args:
            node_num: 图节点数量
            env_information: 环境信息
        """
        self.node_num = node_num
        self.env_information = env_information
        
        # PPO超参数
        self.gamma = 0.99  # 折扣因子
        self.gae_lambda = 0.95  # GAE参数
        self.clip_ratio = 0.2  # PPO裁剪参数
        self.policy_update_epochs = 10  # 每批数据更新轮数
        self.value_coef = 0.5  # 值函数损失系数
        self.entropy_coef = 0.02  # 熵正则化系数，对于抬腿训练略微增大
        
        # 创建Actor-Critic网络
        self.policy = ActorCritic(act_dim=1, node_num=self.node_num).to(device)  # 抬腿训练使用8个动作
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.0003)
        
        # 存储轨迹数据
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def choose_action(self, episode_num, obs, x_graph):
        """
        选择动作
        Args:
            episode_num: 当前回合数
            obs: 观察数据
            x_graph: 图特征数据
        Returns:
            动作、对数概率和状态值
        """
        with torch.no_grad():
            dist, value = self.policy(x=obs[0], state=obs[1], x_graph=x_graph)
            action = dist.sample()
            # # 抬腿训练增加一些探索
            # if episode_num < 500:  # 前500回合增加探索
            #     action = dist.sample()
            # else:
            #     # 减少探索，更倾向于选取高概率动作
            #     if np.random.random() < 0.8:
            #         action = torch.argmax(action_probs)
            #     else:
            #         action = dist.sample()
            action_clamped = torch.clamp(action, -1.0, 1.0)        
            log_prob = dist.log_prob(action).sum(dim=-1)
            # 将对数概率和状态值保留到6位小数
            log_prob_rounded = round(log_prob.item(), 6)
            value_rounded = round(value.item(), 6)
           
            return  action_clamped.item(), log_prob_rounded, value_rounded
    
    def store_transition_catch(self, state, action_shoulder, action_arm, reward, next_state, done, value, log_prob):
        """存储轨迹数据"""
        self.states.append(state)
        self.actions.append(action_shoulder)
        self.actions.append(action_arm)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def store_transition_tai(self, state, actions, rewards, next_state, done, value, log_prob):
        """存储轨迹数据"""
        self.states.append(state)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.next_states.append(next_state)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def calculate_advantages(self):
        """计算GAE优势函数"""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(self.rewards))):
            if i == len(self.rewards) - 1:
                next_value = 0
            else:
                next_value = self.values[i + 1]
            
            delta = self.rewards[i] + self.gamma * next_value * (1 - self.dones[i]) - self.values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[i]) * gae
            advantages.insert(0, gae)
        
        return torch.tensor(advantages, dtype=torch.float32).to(device)
    
    def learn(self):
        """PPO学习过程"""
        if len(self.states) < 32:  # 确保有足够的数据，对于抬腿训练可以使用更小的批量
            return 0
            
        # 计算优势函数和回报
        advantages = self.calculate_advantages()
        returns = advantages + torch.tensor(self.values, dtype=torch.float32).to(device)
        
        # 转换为张量
        batch_states = self.states
        batch_actions = torch.tensor(self.actions, dtype=torch.long).to(device)
        batch_log_probs = torch.tensor(self.log_probs, dtype=torch.float32).to(device)
        
        # 归一化优势函数
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO更新
        total_loss = 0
        for _ in range(self.policy_update_epochs):
            # 重新评估动作和值
            all_means = []
            all_log_stds = []
            all_values = []
            
            for i in range(len(batch_states)):
                dist, value = self.policy(x=batch_states[i][0], 
                           state=batch_states[i][1], 
                           x_graph=self.states[i][2])
                all_means.append(dist.mean)
                all_log_stds.append(dist.scale.log())  # 将scale转换为log_std
                all_values.append(value)

            all_means = torch.stack(all_means)
            all_log_stds = torch.stack(all_log_stds)
            all_values = torch.cat(all_values)


            # 计算比率和策略损失
            dist = torch.distributions.Normal(all_means, torch.exp(all_log_stds))
            new_log_probs = dist.log_prob(batch_actions.unsqueeze(-1))
            entropy = dist.entropy().mean()
            
            ratio = torch.exp(new_log_probs - batch_log_probs)
            
            # PPO裁剪目标
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 值函数损失
            value_loss = nn.MSELoss()(all_values, returns)
            
            # 总损失
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # 优化
            self.optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪，防止抬腿训练中梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # 清空轨迹数据
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        return total_loss / self.policy_update_epochs

