import sys
import os
# 添加项目路径到系统路径
sys.path.append('D:\\project_Spatiotemporal_attention_mechanism')
from python_scripts.DQN import DQN_episoid
from python_scripts.PPO import PPO_episoid_1
from python_scripts.SAC import SAC_episoid
from python_scripts.Project_config import path_list

def main():
    # 直接指定模型路径
    #model_path = "D:/project/python_scripts/DQN/checkpoint/dqn_model_0.ckpt"
    
    #print("将使用DQN进行训练")
    #DQN_episoid.DQN_episoid()#model_path=model_path

    print("将使用PPO进行训练")
    PPO_episoid_1.PPO_episoid_1()

    #print("将使用SAC进行训练")
    #SAC_episoid.SAC_episoid()
    
if __name__ == '__main__':
    print("_________")
    with open(path_list['resetFlag'], 'r+') as file:
        file.write('1')
    main()
