import math
import torch
from python_scripts.DQN.DQN_DQNnet_2 import DQN2
from python_scripts.Webots_interfaces import Environment
from python_scripts.Project_config import path_list, BATCH_SIZE, LR, EPSILON, GAMMA, TARGET_REPLACE_ITER, MEMORY_CAPACITY, device, gps_goal, gps_goal1

def DQN_tai_episoid(dqn2_LegUpper=None, dqn2_LegLower=None, dqn2_Ankle=None, existing_env=None ,total_episoid=0, episode=0, rpm_2=None, log_writer_tai=None, log_file_latest_tai=None):
    """
    抬腿训练函数
    Args:
        total_episoid: 总训练周期数
        episode: 抬腿的总周期数
        existing_env: 现有的环境实例
        rpm_2: 经验回放缓冲区
        log_writer_tai: 日志记录器
        log_file_latest_tai: 日志文件路径
        dqn2_LegUpper: DQN2模型实例，如果为None则创建新实例
        dqn2_LegLower: DQN2模型实例，如果为None则创建新实例
        dqn2_Ankle: DQN2模型实例，如果为None则创建新实例
    """
    # 如果没有传入dqn2，则创建一个新的实例
    if dqn2_LegUpper is None:
        dqn2_LegUpper = DQN2()
    if dqn2_LegLower is None:
        dqn2_LegLower = DQN2()
    if dqn2_Ankle is None:
        dqn2_Ankle = DQN2()
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
    
    # 记录回合数
    log_writer_tai.add(episode_num=total_episoid)

    while True:
        obs_img, obs_tensor = env.get_img(steps, imgs)
        robot_state = env.get_robot_state()
        # 选择动作
        action_LegUpper = dqn2_LegUpper.choose_action(episode, robot_state)
        action_LegLower = dqn2_LegLower.choose_action(episode, robot_state)
        action_Ankle = dqn2_Ankle.choose_action(episode, robot_state)
        print("第", steps + 1, "步")
        print(f"选择动作: {action_LegUpper}, {action_LegLower}, {action_Ankle}")
        
        # 记录腿部动作
        log_writer_tai.add_leg_action(action_LegUpper, action_LegLower, action_Ankle)
            
        # 获取GPS数据
        gps_values = env.print_gps()
            
        # 设置抓取器状态
        catch_flag = 0.0
            
        # 生成图像名称
        img_name = f"img{steps}.png"
            
        # 执行动作
        next_state, reward, done, good, goal, count = env.step2(
            robot_state, action_LegUpper, action_LegLower, action_Ankle, steps, catch_flag, 
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
            rpm_2.append((robot_state, action_LegUpper, action_LegLower, action_Ankle, reward, next_state, done))
            
        # 更新状态
        robot_state = env.get_robot_state()
        obs_tensor = next_obs_tensor
            
        # 检查是否结束当前回合
        if steps > 20:
            done = 1
            
        # 定期保存模型
        if episode % 50 == 0:
            save_path = path_list['model_path_tai_DQN'] + f"/dqn_model_tai_{total_episoid}_{episode}.ckpt"
            print(f"保存模型到: {save_path}")
            torch.save(dqn2_LegUpper.eval_net, save_path)
            torch.save(dqn2_LegLower.eval_net, save_path)
            torch.save(dqn2_Ankle.eval_net, save_path)
        # 学习过程
        if len(rpm_2) > 1000 and done == 1:
            # 如果达到目标，保存模型
            if goal == 1:
                save_path = path_list['model_path_tai_DQN'] + f"/dqn_model_tai_{total_episoid}_{episode}.ckpt"
                torch.save(dqn2_LegUpper.eval_net, save_path)
                torch.save(dqn2_LegLower.eval_net, save_path)
                torch.save(dqn2_Ankle.eval_net, save_path)
                
                # 学习
            loss_LegUpper = dqn2_LegUpper.learn(rpm_2)
            loss_LegLower = dqn2_LegLower.learn(rpm_2)
            loss_Ankle = dqn2_Ankle.learn(rpm_2)
            loss_all = loss_LegUpper + loss_LegLower + loss_Ankle    
            # 记录损失值
            log_writer_tai.add(loss=loss_all)
                
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