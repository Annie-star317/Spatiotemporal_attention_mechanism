"""Replay_memory_2.py"""

import random
import collections
import numpy as np

class ReplayMemory_2(object):
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)
        
    def append(self, state, action, reward, next_state, done):
        """
        为SAC算法存储经验，与DQN版本相比少了图像输入
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否完成
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = [], [], [], [], []
        
        for experience in mini_batch:
            s, a, r, n_s, done = experience
            state_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_state_batch.append(n_s)
            done_batch.append(done)

        return np.array(state_batch).astype('float32'),\
            np.array(action_batch).astype('float32'), \
            np.array(reward_batch).astype('float32'),\
            np.array(next_state_batch).astype('float32'),\
            np.array(done_batch).astype('float32')

    def __len__(self):
        return len(self.buffer)