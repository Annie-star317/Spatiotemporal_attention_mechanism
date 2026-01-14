import torch
import torch.nn as nn
import numpy as np
import torch_geometric
from torch_geometric.data import Data
from python_scripts.Project_config import device
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, act_dim, node_num):
        super().__init__()
        self.node_num = node_num
        
        # 保留原有的特征提取网络结构
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=1)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=1)
        
        self.fc0 = nn.Linear(in_features=6272, out_features=6000)
        self.fc1 = nn.Linear(in_features=6000, out_features=100)
        self.fc2 = nn.Linear(in_features=4, out_features=100)
        self.fc3 = nn.Linear(in_features=100, out_features=100)
        
        # 图神经网络部分
        self.conv_graph1 = torch_geometric.nn.GraphSAGE(1, 1000, 2, aggr='add')
        self.conv_graph2 = torch_geometric.nn.GATConv(1000, 1000, aggr='add')
        self.conv_graph3 = torch_geometric.nn.GraphSAGE(1000, 1000, 2, aggr='add')
        self.conv_graph4 = torch_geometric.nn.GATConv(1000, 1000, aggr='add')
        self.conv_graph5 = torch_geometric.nn.GCNConv(1000, 1000, 2, aggr='add')
        self.fc_graph = nn.Linear(1000, 100)
        
        # 共享特征层
        self.fc4 = nn.Linear(in_features=300, out_features=200)
        
        self.actor_mu = nn.Sequential(
            nn.Linear(200, act_dim),
            nn.Tanh()  # Tanh激活函数将mu的范围限制在[-1, 1]
        )
        
        self.actor_log_sigma = nn.Parameter(torch.zeros(act_dim))
        
        # Critic头：输出状态值
        self.critic = nn.Linear(200, 1)
    
    # 保留原有的图处理函数
    def create_edge_index(self):
        ans = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
             1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
             17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        ]
        return torch.tensor(ans, dtype=torch.long)
    
    def creat_x(self, x_graph):
        ans = [[] for i in range(self.node_num)]
        for i in range(len(ans)):
            ans[i] = [x_graph[i]]
        return ans
    
    def creat_graph(self, x_graph):
        x = torch.as_tensor(self.creat_x(x_graph), dtype=torch.float32)
        edge_index = torch.as_tensor(self.create_edge_index(), dtype=torch.long)
        graph = Data(x=x, edge_index=edge_index)
        graph.x = graph.x.to(device)
        graph.edge_index = graph.edge_index.to(device)
        return graph

    def forward(self, x, state, x_graph):
        # 特征提取部分与原DQN相同
        self.graph = self.creat_graph(x_graph)
        
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
        
        min_val1 = torch.min(x)
        max_val1 = torch.max(x)
        normalized_data1 = torch.div(torch.sub(x, min_val1), torch.sub(max_val1, min_val1))
        
        state = torch.as_tensor(state, dtype=torch.float32).to(device)
        state = self.fc2(state)
        state = self.fc3(state)
        min_val2 = torch.min(state)
        max_val2 = torch.max(state)
        normalized_data2 = torch.div(torch.sub(state, min_val2), torch.sub(max_val2, min_val2))
        
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
        
        min_val3 = torch.min(x_graph)
        max_val3 = torch.max(x_graph)
        normalized_x_graph = torch.div(torch.sub(x_graph, min_val3), torch.sub(max_val3, min_val3))
        
        state_x = torch.cat((normalized_data1, normalized_data2, normalized_x_graph), dim=-1)
        features = self.fc4(state_x)
        
        # Actor: 输出均值 mu
        mu = self.actor_mu(features)
        
        # 计算标准差 sigma
        # 使用exp来保证sigma是正数。广播log_sigma以匹配mu的批次大小
        log_sigma = self.actor_log_sigma.expand_as(mu)
        sigma = torch.exp(log_sigma)
        
        # 构建正态分布
        dist = Normal(mu, sigma)
        
        # Critic: 输出状态值 (保持不变)
        value = self.critic(features)
        
        return dist, value

class PPO:
    def __init__(self, node_num, env_information):
        self.node_num = node_num
        self.env_information = env_information
        
        # PPO超参数
        self.gamma = 0.99  # 折扣因子
        self.gae_lambda = 0.95  # GAE参数
        self.clip_ratio = 0.2  # PPO裁剪参数
        self.policy_update_epochs = 10  # 每批数据更新轮数
        self.value_coef = 0.5  # 值函数损失系数
        self.entropy_coef = 0.01  # 熵正则化系数
        
        # 创建Actor-Critic网络
        self.policy = ActorCritic(act_dim=1, node_num=self.node_num).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.0003)
        
        # 存储轨迹数据
        self.states = []
        self.action_shoulder = []
        self.action_arm = []
        self.rewards = []
        self.next_states = []
        self.values_shoulder = []
        self.values_arm = []   
        self.log_probs_shoulder = []
        self.log_probs_arm = []
        self.dones = []
    
    def choose_action(self, episode_num, obs, x_graph):
        """
        选择动作，返回动作、对数概率和状态值
        动作值范围：[-1.0, 1.0]
        """
        with torch.no_grad():
            # --- 从正态分布采样 ---
            # self.policy返回的是一个分布对象，而不是概率
            dist, value = self.policy(x=obs[0], state=obs[1], x_graph=x_graph)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            # 将对数概率和状态值保留到6位小数
            log_prob_rounded = round(log_prob.item(), 6)
            value_rounded = round(value.item(), 6)
            action = torch.clamp(action, -1.0, 1.0)
            return action, log_prob_rounded, value_rounded
    
    def store_transition_catch(self, state, action_shoulder, action_arm, reward, next_state, done, value_shoulder, value_arm, log_prob_shoulder, log_prob_arm):
        """
        存储轨迹数据
        动作值范围：[-1.0, 1.0]
        """
        self.states.append(state)
        self.action_shoulder.append(action_shoulder)
        self.action_arm.append(action_arm)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.values_shoulder.append(value_shoulder)
        self.values_arm.append(value_arm)
        self.log_probs_shoulder.append(log_prob_shoulder)
        self.log_probs_arm.append(log_prob_arm)
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
    def get_current_sigma(self):
        return torch.exp(self.policy.actor_log_sigma).item()    
    def learn(self):
        """PPO学习过程"""
        if len(self.states) < 64:  # 确保有足够的数据
            # 清空轨迹数据，确保不会累积过多数据
            self.states = []
            self.action_shoulder = []
            self.action_arm = []
            self.rewards = []
            self.next_states = []
            self.values_shoulder = []
            self.values_arm = []
            self.log_probs_shoulder = []
            self.log_probs_arm = []
            self.dones = []
            return 0
            
        # 设置更安全的最大批次大小，避免内存问题
        max_batch_size = min(len(self.states), 128)
        
        try:
            # 计算优势函数和回报
            with torch.no_grad():  # 在计算优势函数时不需要梯度
                advantages = self.calculate_advantages()
                returns = advantages + torch.tensor(self.values[:max_batch_size], dtype=torch.float32).to(device)
                
                # 归一化优势函数
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # 转换为张量
            batch_states = self.states[:max_batch_size]
            batch_actions_shoulder = torch.tensor(self.action_shoulder[:max_batch_size], dtype=torch.long).to(device)
            batch_actions_arm = torch.tensor(self.action_arm[:max_batch_size], dtype=torch.long).to(device)
            batch_log_probs_shoulder = torch.tensor(self.log_probs_shoulder[:max_batch_size], dtype=torch.float32).to(device)
            batch_log_probs_arm = torch.tensor(self.log_probs_arm[:max_batch_size], dtype=torch.float32).to(device)
            
            # 减少PPO更新次数，确保能快速完成
            update_epochs = min(3, self.policy_update_epochs)  # 最多更新3次
            
            # PPO更新
            total_loss = 0
            for _ in range(update_epochs):
                # 重新评估动作和值，使用批处理减少for循环
                all_action_probs = []
                all_values = []
                
                # 分批处理，避免内存溢出
                batch_size = 32  # 小批量处理
                num_batches = (max_batch_size + batch_size - 1) // batch_size
                
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, max_batch_size)
                    
                    for i in range(start_idx, end_idx):
                        try:
                            # 使用try-except包裹每个处理步骤，确保即使部分数据有问题也能继续
                            action_probs, value = self.policy(
                                x=batch_states[i][0], 
                                state=batch_states[i][1], 
                                x_graph=self.states[i][2]
                            )
                            all_action_probs.append(action_probs)
                            all_values.append(value)
                        except Exception as e:
                            print(f"处理第{i}个样本时出错: {e}")
                            # 对于错误的样本，使用零占位
                            all_action_probs.append(torch.zeros(2, device=device))
                            all_values.append(torch.zeros(1, device=device))
                
                # 确保张量操作不会阻塞
                try:
                    if len(all_action_probs) == 0 or len(all_values) == 0:
                        print("没有有效的动作概率或值，跳过此次更新")
                        continue
                        
                    all_action_probs = torch.stack(all_action_probs)
                    all_values = torch.cat(all_values)
                    
                    # 计算比率和策略损失
                    dist = torch.distributions.Categorical(all_action_probs)
                    new_log_probs = dist.log_prob(batch_actions_shoulder)
                    entropy = dist.entropy().mean()
                    
                    ratio = torch.exp(new_log_probs - batch_log_probs_shoulder)
                    
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
                    
                    # 添加梯度裁剪，避免大梯度
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                except Exception as e:
                    print(f"更新网络时出错: {e}")
            
            # 确保在return前清空数据，防止内存泄漏
            self.states = []
            self.action_shoulder = []
            self.action_arm = []
            self.rewards = []
            self.next_states = []
            self.values_shoulder = []
            self.values_arm = []
            self.log_probs_shoulder = []
            self.log_probs_arm = []
            self.dones = []
            
            # 确保所有张量操作都已完成
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # 清除PyTorch缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return total_loss / max(1, update_epochs)
        except Exception as e:
            print(f"学习过程中出现未处理异常: {e}")
            # 出错时也要清空数据
            self.states = []
            self.action_shoulder = []
            self.action_arm = []
            self.rewards = []
            self.next_states = []
            self.values_shoulder = []
            self.values_arm = []
            self.log_probs_shoulder = []
            self.log_probs_arm = []
            self.dones = []
            return 0