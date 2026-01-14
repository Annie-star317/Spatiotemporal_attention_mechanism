"""replay_memory.py"""

import random
import collections
import numpy as np

class ReplayMemory(object):
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)
        
    def append(self, exp):
        self.buffer.append(exp)
    
    def sample(self, batch_size):
        # 如果缓冲区中的样本数量不足，减小批次大小
        actual_batch_size = min(batch_size, len(self.buffer))
        mini_batch = random.sample(self.buffer, actual_batch_size)
        obs_batch, state_batch, action_batch, reward_batch, next_obs_batch, next_state_batch, done_batch = [], [], [], [], [], [], []
        for experience in mini_batch:
            o, s, a, r, n_o, n_s, done = experience
            obs_batch.append(o)
            state_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(n_o)
            next_state_batch.append(n_s)
            
            done_batch.append(done)

        # 使用较小的批次大小，或者分批处理大型数组
        try:
            return np.array(obs_batch).astype('float32'), np.array(state_batch).astype('float32'),\
                np.array(action_batch).astype('float32'), np.array(reward_batch).astype('float32'),\
                np.array(next_obs_batch).astype('float32'), np.array(next_state_batch).astype('float32'),\
                np.array(done_batch).astype('float32')
        except MemoryError:
            # 如果内存不足，尝试减小批次大小
            print(f"内存不足，将批次大小从{actual_batch_size}减小到{actual_batch_size//2}")
            return self.sample(actual_batch_size // 2)

    def __len__(self):
        return len(self.buffer)