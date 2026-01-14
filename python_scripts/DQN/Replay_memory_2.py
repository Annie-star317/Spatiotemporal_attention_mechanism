"""经验回放池模块，用于抬腿训练"""

import random
import collections
import numpy as np
import torch


class ReplayMemory_2:
    """
    抬腿训练使用的经验回放池
    与普通经验回放池不同，这个版本适配了抬腿训练的特殊需求
    """
    
    def __init__(self, max_size):
        """
        初始化经验回放样本池
        Args:
            max_size: 经验池最大容量
        """
        self.buffer = collections.deque(maxlen=max_size)

    def append(self, exp):
        """
        向经验池添加一条经验
        Args:
            exp: 经验数据元组 (状态, 动作, 奖励, 下一状态, 是否结束)
        """
        self.buffer.append(exp)

    def clear(self):
        """清空经验池"""
        self.buffer.clear()

    def sample(self, batch_size):
        """
        从经验池中随机采样
        Args:
            batch_size: 采样批次大小
        Returns:
            采样的经验数据批次
        """
        # 对于抬腿训练，使用所有经验
        mini_batch = self.buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = [], [], [], [], []
        
        for experience in mini_batch:
            # 注意：这里只解包5个值，与DQN_episoid_2.py中存储的数据结构匹配
            s, a, r, s_, done = experience
            state_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_state_batch.append(s_)
            done_batch.append(done)

        return np.array(state_batch).astype('float32'), \
            np.array(action_batch).astype('float32'), \
            np.array(reward_batch).astype('float32'), \
            np.array(next_state_batch).astype('float32'), \
            np.array(done_batch).astype('float32')

    def __len__(self):
        """获取经验池中样本数量"""
        return len(self.buffer)