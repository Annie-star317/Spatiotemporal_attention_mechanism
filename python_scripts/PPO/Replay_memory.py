"""replay_memory.py - 为PPO算法修改的经验回放"""

import random
import collections
import numpy as np
import torch

class ReplayMemory(object):
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)
        
    def append(self, exp):
        """
        存储经验：(观察, 状态, 动作, 对数概率, 奖励, 是否完成, 值函数评估)
        """
        self.buffer.append(exp)
    
    def clear(self):
        """清空经验池"""
        self.buffer.clear()
    
    def sample(self, batch_size):
        """
        为PPO采样经验
        如果buffer中的样本数量小于batch_size，直接返回所有样本
        否则随机抽取batch_size个样本
        """
        if len(self.buffer) <= batch_size:
            mini_batch = self.buffer
        else:
            mini_batch = random.sample(self.buffer, batch_size)
            
        obs_batch, state_batch, action_batch, log_prob_batch, reward_batch, done_batch, value_batch = [], [], [], [], [], [], []
        
        for experience in mini_batch:
            o, s, a, log_prob, r, done, value = experience
            obs_batch.append(o)
            state_batch.append(s)
            action_batch.append(a)
            log_prob_batch.append(log_prob)
            reward_batch.append(r)
            done_batch.append(done)
            value_batch.append(value)

        # 转换为tensor以便于后续计算
        return (
            np.array(obs_batch),
            np.array(state_batch).astype('float32'),
            torch.tensor(action_batch, dtype=torch.float32),
            torch.tensor(log_prob_batch, dtype=torch.float32),
            torch.tensor(reward_batch, dtype=torch.float32),
            torch.tensor(done_batch, dtype=torch.float32),
            torch.tensor(value_batch, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)