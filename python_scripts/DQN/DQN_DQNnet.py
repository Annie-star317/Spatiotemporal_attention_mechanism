import torch
import torch.nn as nn
import numpy as np
import torch_geometric
from torch_geometric.data import Data
from python_scripts.Project_config import device
# 超参数
BATCH_SIZE = 32                                 # 样本数量
LR = 0.0001                                       # 学习率
EPSILON = 0.9                                   # greedy policy
GAMMA = 0.99                                     # reward discount
TARGET_REPLACE_ITER = 100                        # 目标网络更新频率(固定不懂的Q网络)
MEMORY_CAPACITY = 100000


class Net(nn.Module):
    def __init__(self, act_dim, node_num):    # 初始化网络
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
        self.fc5 = nn.Linear(in_features=200, out_features=act_dim)  # 全连接层5
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
        x = torch.tensor(self.creat_x(x_graph))    # 创建图的节点信息  
        # 修改为PyTorch推荐的方式创建张量
        edge_index = torch.tensor(self.create_edge_index(), dtype=torch.long)    # 创建图的边索引
        graph = Data(x=x, edge_index=edge_index)    # 创建图
        graph.x = graph.x.to(device)
        graph.edge_index = graph.edge_index.to(device)  # 将图的边索引移动到GPU
        return graph

    def forward(self, x, state, x_graph):
        
        self.graph = self.creat_graph(x_graph) # 创建图
        x = torch.tensor(x).to(device)  # 将输入数据转换为张量并移动到GPU
        # x = torch.unsqueeze(x, dim=0)  # 在第0维增加一个维度
        # 添加通道维度，确保输入是4维的 [batch_size, channels, height, width]
        x = torch.unsqueeze(x, dim=0)  # 在第0维增加一个维度，变成 [1, 128, 128]
        x = torch.unsqueeze(x, dim=1)  # 在第1维增加一个维度，变成 [1, 1, 128, 128]
        x = x.float()  # 将张量转换为浮点数
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
        state = torch.tensor(state).to(device)  # 将状态转换为张量并移动到GPU
        state = state.float()  # 将状态转换为浮点数
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
        state_x = self.fc5(state_x)  # 全连接层5
        return state_x 

# 定义DQN_GNN类(定义Q网络以及一个固定的Q网络)
class DQN_GNN(object):
    def __init__(self, node_num, env_information):
        self.node_num = node_num  # 点总数
        self.env_information = env_information  # 环境信息
        # 创建评估网络和目标网络
        self.eval_net,self.target_net = Net(act_dim=2, node_num=self.node_num).to(device),Net(act_dim=2, node_num=self.node_num).to(device) 
        # self.eval_net,self.target_net = Net(2).to('cuda'),Net(2).to('cuda')
        self.learn_step_counter = 0  # 学习步数记录
        self.memory_counter = 0      # 记忆量计数
        self.memory = np.zeros((MEMORY_CAPACITY,6)) # 存储空间初始化，每一组的数据为(s_t,a_t,r_t,s_{t+1})
        self.optimazer = torch.optim.Adam(self.eval_net.parameters(),lr=LR)
        self.loss_func = nn.MSELoss()     # 使用均方损失函数 (loss(xi, yi)=(xi-yi)^2)
        self.loss_func = self.loss_func.to(device)  # 将损失函数移动到GPU

    def choose_action(self, episode_num, obs, x_graph):  # 修改函数参数，只接收obs作为状态输入
    # 使用递减的epsilon，从0.9开始逐渐降低到0.1
        epsilon = max(0.1, 0.90 - episode_num * 0.0001)  # 减小衰减速率，使探索率保持更高
        # print(f'epsilon: {epsilon}')
        random_num = np.random.uniform()
        # print(f'random_num: {random_num}')
        if  random_num > epsilon:  # 如果随机数小于epsilon
            # print("使用网络生成动作")
            actions_value = self.eval_net.forward(x=obs[0], 
                                                  state=obs[1], 
                                                  x_graph=x_graph)  # 前向传播
            # print(f'actions_value: {actions_value}')
            new_actions_value = torch.unsqueeze(actions_value, dim=0)  # 在第0维增加一个维度
            # print(f'new_actions_value: {new_actions_value}')
            action = torch.max(new_actions_value, dim=1)  # 取最大值
            # print(f'action: {action}')
            action = action[1].item()  # 取最大值的索引
            # print(f'action: {action}')
        else:
            # print("随机生成动作")
            action = np.random.randint(0, 2)  # 随机选择动作
        return action
  
    def store_transition(self, o, s, a, r, o_, s_):  # 存储经验
        # This function acts as experience replay buffer
        s = [s]  # 将状态s转换为列表
        s_ = [s_]  # 将状态s_转换为列表
        transition = np.hstack((o, s, [a, r], o_, s_))  # 水平堆叠这些向量
        # if the capacity is full, then use index to replace the old memory with new one
        index = self.memory_counter % MEMORY_CAPACITY  # 计算索引
        self.memory[index, :] = transition  # 将transition存储到memory中
        self.memory_counter += 1  # 记忆量计数器自加1

    def learn(self, rpm):
        """ 用DDPG算法更新 actor 和 critic
        """
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:  # 一开始触发，然后每100步触发
            self.target_net.load_state_dict(self.eval_net.state_dict())  # 将评估网络的参数赋给目标网络
            print("我必须立刻学习")
        self.learn_step_counter += 1  # 学习步数自加1
        b_o, b_s, b_a, b_r, b_o_, b_s_, done = rpm.sample(64)  # 从经验回放缓存中随机采样64个样本
        x_graph = b_s
        edge_index_graph = self.eval_net.create_edge_index()
        loss_all = 0  # 初始化损失值
        for i in range(32):
            # 获得32个trasition的评估值和目标值，并利用损失函数和优化器进行评估网络参数更新
            state = [b_s[i][1], b_s[i][0], b_s[i][5], b_s[i][4]]
            #q_eval = self.eval_net(b_o[i], b_s[i])[int(b_a[i])]  # 因为已经确定在s时候所走的action，因此选定该action对应的Q值
            q_eval = self.eval_net(b_o[i], state, x_graph[i])[int(b_a[i])]  # 因为已经确定在s时候所走的action，因此选定该action对应的Q值
            # q_next 不进行反向传播，故用detach；q_next表示通过目标网络输出32行每个b_s_对应的一系列动作值
            #q_next = self.target_net(b_o_[i], b_s_[i])
            q_next = self.target_net(b_o_[i], state, x_graph[i])
            # 先算出目标值q_target，max(1)[0]相当于计算出每一行中的最大值（注意不是上面的索引了,而是一个一维张量了），view()函数让其变成(32,1)
            q_target = b_r[i] + GAMMA * q_next.max(0)[0]
            # 计算损失值
            loss = self.loss_func(q_eval, q_target)  # 计算损失值
            loss_all = loss_all + loss  # 累加损失值
        loss_all = loss_all / 32  # 计算平均损失值
        self.optimazer.zero_grad()  # 清空上一步的残余更新参数值
        loss_all.backward()  # 误差方向传播
        self.optimazer.step()  # 逐步的梯度优化
        return loss_all
