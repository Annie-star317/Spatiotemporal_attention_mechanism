import math
import torch
from python_scripts.Webots_interfaces import Environment
from python_scripts.Project_config import path_list, BATCH_SIZE, LR, EPSILON, GAMMA, TARGET_REPLACE_ITER, MEMORY_CAPACITY, device, gps_goal, gps_goal1
from python_scripts.PPO.PPO_PPOnet_2 import PPO2 
from python_scripts.PPO_Log_write import Log_write

def PPO_tai_episoid(ppo2_LegUpper=None, ppo2_LegLower=None, ppo2_Ankle=None, existing_env=None ,total_episode=0, episode=0, log_writer_tai=None, log_file_latest_tai=None):

    if ppo2_LegUpper is None:
        ppo2_LegUpper = PPO2()
    if ppo2_LegLower is None:
        ppo2_LegLower = PPO2()
    if ppo2_Ankle is None:
        ppo2_Ankle = PPO2()
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
    log_writer_tai.add(episode_num=total_episode)
    
    while True:
        obs_img, obs_tensor = env.get_img(steps, imgs)
        robot_state = env.get_robot_state()
        # 将机器人状态转换为PPO状态，与PPO_episoid_1.py保持一致
        ppo_state = [robot_state[1], robot_state[0], robot_state[5], robot_state[4]]
        # 选择动作
        action_LegUpper, log_prob_LegUpper, value_LegUpper = ppo2_LegUpper.choose_action(episode_num=episode, 
                                       obs=[obs_img, ppo_state],
                                       x_graph=robot_state)
        action_LegLower, log_prob_LegLower, value_LegLower = ppo2_LegLower.choose_action(episode_num=episode, 
                                       obs=[obs_img, ppo_state],
                                       x_graph=robot_state)
        action_Ankle, log_prob_Ankle, value_Ankle = ppo2_Ankle.choose_action(episode_num=episode, 
                                       obs=[obs_img, ppo_state],
                                       x_graph=robot_state)
        
        # 分别添加动作、对数概率和状态价值到日志
        log_writer_tai.add_action_tai(round(float(action_LegUpper), 4), round(float(action_LegLower), 4), round(float(action_Ankle), 4))
        log_writer_tai.add_log_prob_tai(round(float(log_prob_LegUpper), 4), round(float(log_prob_LegLower), 4), round(float(log_prob_Ankle), 4))
        log_writer_tai.add_value_tai(round(float(value_LegUpper), 4), round(float(value_LegLower), 4), round(float(value_Ankle), 4))

        print("第", steps + 1, "步")
        print(f"{action_LegUpper:.4f}, {action_LegLower:.4f}, {action_Ankle:.4f}")

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
            #rpm_2.append((robot_state, action, reward, next_state, done))
            # 将数据存储到PPO2对象内部
            ppo2_LegUpper.store_transition_tai(
                state=[obs_img, ppo_state, robot_state],
                actions=action_LegUpper,
                rewards=reward,
                next_state=[next_obs_img, robot_state, next_state],
                done=done,
                value=value_LegUpper,
                log_prob=log_prob_LegUpper
            )
            ppo2_LegLower.store_transition_tai(
                state=[obs_img, ppo_state, robot_state],
                actions=action_LegLower,
                rewards=reward,
                next_state=[next_obs_img, robot_state, next_state],
                done=done,
                value=value_LegLower,
                log_prob=log_prob_LegLower
            )
            ppo2_Ankle.store_transition_tai(
                state=[obs_img, ppo_state, robot_state],
                actions=action_Ankle,
                rewards=reward,
                next_state=[next_obs_img, robot_state, next_state],
                done=done,
                value=value_Ankle,
                log_prob=log_prob_Ankle
            )
            
                        
        # 更新状态
        robot_state = env.get_robot_state()
        obs_tensor = next_obs_tensor
            
        # 检查是否结束当前回合
        if steps > 20:
            done = 1
            
        # 定期保存模型
        if episode % 200 == 0 and done == 1:
            save_path = path_list['model_path_tai_PPO'] + f"/ppo_model_tai_{total_episode}_{episode}.ckpt"
            print(f"保存模型到: {save_path}")
            checkpoint = {
                "episode": episode,                      # 只写一次即可
                # 上腿
                "policy_LegUpper":    ppo2_LegUpper.policy.state_dict(),
                "optimizer_LegUpper": ppo2_LegUpper.optimizer.state_dict(),
                # 下腿
                "policy_LegLower":    ppo2_LegLower.policy.state_dict(),
                "optimizer_LegLower": ppo2_LegLower.optimizer.state_dict(),
                # 踝关节
                "policy_Ankle":       ppo2_Ankle.policy.state_dict(),
                "optimizer_Ankle":    ppo2_Ankle.optimizer.state_dict(),
            }
            torch.save(checkpoint, save_path)
        
        #学习过程
        if episode > 100 and done == 1:
            # 如果达到目标，保存模型
            # save_path = path_list['model_path_tai_PPO'] + f"/ppo_model_tai_{total_episode}_{episode}.ckpt"
            # checkpoint = {
            #         'policy': ppo2.policy.state_dict(),
            #         'optimizer': ppo2.optimizer.state_dict(),
            #         'episode': episode
            #     }
            # torch.save(checkpoint, save_path)
            # if goal == 1:
            #     save_path = path_list['model_path_tai_PPO'] + f"/ppo_model_tai_{total_episode}_{episode}.ckpt"
            #     checkpoint = {
            #         'policy': ppo2.policy.state_dict(),
            #         'optimizer': ppo2.optimizer.state_dict(),
            #         'episode': episode
            #     }
            #     torch.save(checkpoint, save_path)
                
            # 学习
            loss_LegUpper = ppo2_LegUpper.learn()
            loss_LegLower = ppo2_LegLower.learn()
            loss_Ankle = ppo2_Ankle.learn()
            loss = loss_LegUpper + loss_LegLower + loss_Ankle
            # 记录损失值
            # log_writer_tai.add(loss_LegUpper=loss_LegUpper)
            # log_writer_tai.add(loss_LegLower=loss_LegLower)
            # log_writer_tai.add(loss_Ankle=loss_Ankle)
            log_writer_tai.add(loss=loss)
                
            # 记录结果
            log_writer_tai.add(return_all=return_all)
            log_writer_tai.add(goal=goal)
            
            # 如果回合结束，重置环境
        print("done:", done)

        # #测试代码
        # if done == 1 :
        #     temp_loss = ppo2.learn()
        #     print("测试代码运行成功")

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
            log_writer_tai.clear()
            log_writer_tai.save_tai(log_file_latest_tai)
            break