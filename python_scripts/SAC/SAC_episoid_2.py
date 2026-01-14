import torch
import numpy as np
import os
import glob
import math
from python_scripts.Webots_interfaces import Environment
from python_scripts.Project_config import path_list, gps_goal, gps_goal1, device
from python_scripts.SAC.SAC_SACnet_2 import SAC2
from python_scripts.DQN_Log_write import Log_write

def SAC_tai_episoid(sac2=None, existing_env=None, total_episoid=0, episode=0, rpm_2=None, log_writer_tai=None, log_file_latest_tai=None):
    """
    SAC抬腿训练阶段
    
    Args:
        sac2: SAC2算法对象，如果为None则创建新的
        existing_env: 已存在的环境对象
        total_episoid: 抓取训练的总周期数
        episode: 当前抬腿训练的周期
        rpm_2: 经验回放缓冲区
        log_writer_tai: 日志写入器
        log_file_latest_tai: 日志文件路径
    """
    print("开始SAC抬腿训练...")
    
    # 如果没有传入sac2，则创建一个新的实例
    if sac2 is None:
        sac2 = SAC2()
    # 使用已有的环境实例或创建新的
    if existing_env is not None:
        env = existing_env
    else:
        env = Environment()
    
    print("开始抬腿！")
    env.darwin.tai_leg_L1()
    env.darwin.tai_leg_L2()

    # 初始化状态
    count = 0
    return_all = 0
    imgs = []
    goal = 0
    done = 0
    reward = 0
    steps = 0
    catch_flag = 0
    # 获取观察和状态
    print("____________________")
    
    # 抬腿训练循环
    log_writer_tai.add(episode_num=total_episoid)

    while True:
        obs_img, obs_tensor = env.get_img(steps, imgs)
        robot_state = env.get_robot_state()
        # 选择动作
        action = sac2.choose_action(episode_num=episode, robot_state=robot_state)
        
        print(f'第{episode}周期抬腿训练，第{steps}步，动作a: {action}')
        
        # 记录动作
        log_writer_tai.add_action(action)
        # 获取GPS数据
        gps_values = env.print_gps()
        # 设置抓取器状态
        catch_flag = 0.0

        # 执行动作
        next_state, reward, done, good, goal, count = env.step2(
            robot_state, action, steps, catch_flag, 
            gps_values[4], gps_values[0], gps_values[1], gps_values[2], gps_values[3],
        )

        # 计算奖励
        if count == 1:
            x1 = gps_goal1[0] - gps_values[4][1]
            y1 = gps_goal1[1] - gps_values[4][2]
            distance = math.sqrt(x1 * x1 + y1 * y1)
                
            if distance > 0.06:
                reward1 = 0
            elif distance > 0.03:
                reward1 = 0.1
            else:
                reward1 = 1
                    
            reward = reward1
            
        return_all += reward
        steps += 1
        # 获取新的观察
        next_obs_img, next_obs_tensor = env.get_img(steps, obs_img)
        
        # 存储经验
        if good == 1:
            rpm_2.append(robot_state, action, reward, next_state, done)
            
        # 更新状态
        robot_state = env.get_robot_state()
        obs_tensor = next_obs_tensor
            
        # 检查是否结束当前回合
        if steps > 20:
            done = 1
            
        # 定期保存模型
        if episode % 500 == 0:
            save_path = path_list['model_path_tai_SAC'] + f"/sac_model_tai_{total_episoid}_{episode}.ckpt"
            print(f"保存模型到: {save_path}")
            checkpoint = {
                'policy_net': sac2.policy_net.state_dict(),
                'q_net': sac2.q_net.state_dict(),
                'target_q_net': sac2.target_q_net.state_dict(),
                'log_alpha': sac2.log_alpha
            }
            torch.save(checkpoint, save_path)
        # 学习过程
        if len(rpm_2) > 2000 and done == 1:
            # 如果达到目标，保存模型
            # if goal == 1:
            #     save_path = path_list['model_path_tai_SAC'] + f"/sac_model_tai_{total_episoid}_{episode}.ckpt"
            #     checkpoint = {
            #         'policy_net': sac2.policy_net.state_dict(),
            #         'q_net': sac2.q_net.state_dict(),
            #         'target_q_net': sac2.target_q_net.state_dict(),
            #         'log_alpha': sac2.log_alpha
            #     }
            #     torch.save(checkpoint, save_path)
                
            # 学习
            loss = sac2.learn(rpm_2)
                
            # 记录损失值
            log_writer_tai.add(loss=loss)
                
            # 记录结果
            log_writer_tai.add(return_all=return_all)
            log_writer_tai.add(goal=goal)
            
            # # 保持原有的文件记录方式
            # with open(path_list['log_path'] + "/return_leg.txt", 'a') as file:
            #     file.write(f"{return_all},")
                
            # with open(path_list['log_path'] + "/goal_leg.txt", 'a') as file:
            #     file.write(f"{goal},")
            
            # 如果回合结束，重置环境
        print("done:", done)
        if done == 1 or steps > 20:
            print("抬腿回合结束，重置环境...")
            env.darwin._set_left_leg_initpose()  # 重置环境
            print("等待一秒...")
            env.wait(1000)
            imgs = []
            steps = 0
            #episode += 1
            obs, obs_tensor = env.get_img(steps, imgs)
            robot_state = env.get_robot_state()
            
            # 保存日志
            # log_writer_tai.add(action_list=log_writer_tai.action_list)
            log_writer_tai.clear_action()
            log_writer_tai.save_tai(log_file_latest_tai)
            break