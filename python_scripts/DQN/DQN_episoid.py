import torch
from python_scripts.DQN.DQN_DQNnet import DQN_GNN
from python_scripts.DQN.DQN_DQNnet_2 import DQN2
from python_scripts.DQN.Replay_memory import ReplayMemory
from python_scripts.DQN.Replay_memory_2 import ReplayMemory_2
from python_scripts.DQN.DQN_episoid_2 import DQN_tai_episoid
from python_scripts.Webots_interfaces import Environment
# from Data_fusion import data_fusion
from python_scripts.Project_config import path_list, gps_goal, gps_goal1, device
from python_scripts.DQN_Log_write import Log_write

def DQN_episoid(model_path=None):
    dqn_shoulder = DQN_GNN(node_num=19, env_information=None)  # 创建DQN对象
    dqn_arm     = DQN_GNN(node_num=19, env_information=None)  # 创建DQN对象
    
    dqn2_LegUpper = DQN2()
    dqn2_LegLower = DQN2()
    dqn2_Ankle = DQN2()

    # 初始化日志写入器
    log_writer_catch = Log_write()  # 创建抓取日志写入器
    log_writer_tai = Log_write()  # 创建抬腿日志写入器

    tai_episoid = 1
    import os
    import glob
    import re
    # 查找现有的日志文件，确定最新的编号
    # 抓取阶段：
    log_pattern = os.path.join(path_list['catch_log_path_DQN'], 'catch_log_*.json')
    existing_logs = glob.glob(log_pattern)
    latest_num = 0
    if existing_logs:
        # 从文件名中提取编号
        for log_path in existing_logs:
            match = re.search(r'catch_log_(\d+)', log_path)
            if match:
                num = int(match.group(1))
                latest_num = max(latest_num, num)
        # 新的日志文件编号
        new_log_num = latest_num + 1
    else:
        # 没有现有日志文件，从1开始
        new_log_num = 1
    log_file_latest_catch = os.path.join(path_list['catch_log_path_DQN'], f"catch_log_{new_log_num}.json")
    print(f"将使用新的抓取日志目录: {log_file_latest_catch}")

    # 抬腿阶段：
    log_pattern = os.path.join(path_list['tai_log_path_DQN'], 'tai_log_*.json')
    existing_logs = glob.glob(log_pattern)
    latest_num = 0
    if existing_logs:
        # 从文件名中提取编号
        for log_path in existing_logs:
            match = re.search(r'tai_log_(\d+)', log_path)
            if match:
                num = int(match.group(1))
                latest_num = max(latest_num, num)
        # 新的日志文件编号
        new_log_num = latest_num + 1
    else:
        # 没有现有日志文件，从1开始
        new_log_num = 1
    log_file_latest_tai = os.path.join(path_list['tai_log_path_DQN'], f"tai_log_{new_log_num}.json")
    print(f"将使用新抬腿的日志目录: {log_file_latest_tai}")

    # 加载模型
    # 抓取模型加载
    if model_path:  # 如果指定了模型路径
        try:
            # 从指定路径加载模型
            # 加载肩膀模型
            shoulder_model = torch.load(model_path)
            dqn_shoulder.eval_net.load_state_dict(shoulder_model.state_dict())
            dqn_shoulder.target_net.load_state_dict(shoulder_model.state_dict())
            
            # 加载手臂模型
            arm_model = torch.load(model_path)
            dqn_arm.eval_net.load_state_dict(arm_model.state_dict())
            dqn_arm.target_net.load_state_dict(arm_model.state_dict())
            
            # 从文件名中提取周期数
            episode_start = int(model_path.split('_')[-1].split('.')[0])
            print(f"从指定模型加载: {model_path}，从周期 {episode_start} 继续训练")
            print("模型加载成功！")
        except Exception as e:
            print(f"指定模型加载失败: {e}")
            episode_start = 0
    else:  # 如果没有指定模型路径，使用原来的自动查找逻辑
        # 获取所有模型文件
        model_files = glob.glob(path_list['model_path_catch_DQN'] + '/dqn_model_*.ckpt')
        if model_files:
            # 按文件名中的数字排序，获取最新的模型文件
            latest_model = max(model_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            episode_start = int(latest_model.split('_')[-1].split('.')[0])
            print(f"找到最新抓取模型: {latest_model}，从周期 {episode_start} 继续训练")
            
            # 加载肩膀模型
            try:
                shoulder_model = torch.load(latest_model)
                dqn_shoulder.eval_net.load_state_dict(shoulder_model.state_dict())
                dqn_shoulder.target_net.load_state_dict(shoulder_model.state_dict())
                
                # 加载手臂模型
                arm_model = torch.load(latest_model)
                dqn_arm.eval_net.load_state_dict(arm_model.state_dict())
                dqn_arm.target_net.load_state_dict(arm_model.state_dict())
                print("抓取模型加载成功！")
            except Exception as e:
                print(f"抓取模型加载失败: {e}")
                episode_start = 0
        else:
            print("未找到已保存的抓取模型，从头开始训练")
            episode_start = 0
    
    # 抬腿模型加载
    model_files_tai = glob.glob(path_list['model_path_tai_DQN'] + '/dqn_model_tai_*.ckpt')
    if model_files_tai:
        try:
            # 按新的文件名格式排序：dqn_model_tai_{total_episoid}_{episode}.ckpt
            # 定义一个函数来提取total_episoid和episode
            def extract_numbers(filename):
                # 从文件名中提取数字部分
                parts = filename.split('_')
                if len(parts) >= 5:  # 确保文件名格式正确
                    try:
                        total_ep = int(parts[-2])  # 倒数第二个是total_episoid
                        ep = int(parts[-1].split('.')[0])  # 最后一个是episode（去掉.ckpt）
                        return (total_ep, ep)
                    except (ValueError, IndexError):
                        return (0, 0)  # 解析失败时返回默认值
                return (0, 0)
            
            # 按照total_episoid和episode排序，找出最新的模型
            latest_model = max(model_files_tai, key=extract_numbers)
            total_ep, ep = extract_numbers(latest_model)
            print(f"找到最新抬腿模型: {latest_model}，总周期: {total_ep}，抬腿周期: {ep}")
            tai_episoid = ep
            # 加载模型
            dqn2_LegUpper.eval_net = torch.load(latest_model)
            dqn2_LegUpper.target_net.load_state_dict(dqn2_LegUpper.eval_net.state_dict())
            dqn2_LegLower.eval_net = torch.load(latest_model)
            dqn2_LegLower.target_net.load_state_dict(dqn2_LegLower.eval_net.state_dict())
            dqn2_Ankle.eval_net = torch.load(latest_model)
            dqn2_Ankle.target_net.load_state_dict(dqn2_Ankle.eval_net.state_dict())
            print("抬腿模型加载成功！")
        except Exception as e:
            print(f"抬腿模型加载失败: {e}")
    else:
        print("未找到已保存的抬腿模型，从头开始训练")

    #episode_num = episode_start  # 初始化回合计数器
    # 创建经验回放缓存
    rpm_shoulder = ReplayMemory(100000)  # 肩膀舵机的经验回放
    rpm_arm = ReplayMemory(100000)      # 手臂舵机的经验回放  
    rpm_2 = ReplayMemory_2(100000)      # 抬腿舵机的经验回放
    env = Environment()

    for i in range(episode_start, episode_start + 50000):  # 从episode_start开始，最多再训练50000个周期
        log_writer_catch.add(episode_num=i)
        print(f"<<<<<<<<<第{i}周期") # 打印当前周期
        env.reset()
        env.wait(500)   # 等待500ms
        imgs = []  # 初始化图像列表
        steps = 0  # 初始化步数
        return_all = 0  # 初始化总奖励
        obs_img, obs_tensor = env.get_img(steps, imgs)  # 获取初始图像和图像张量
        # log_writer_catch.add(obs_img=obs_img, steps=steps)
        robot_state = env.get_robot_state()  # 获取机器人状态
        # print(f'robot_state: {robot_state}')
        # print(f'robot_state_len: {len(robot_state)}')
        print("____________________")  # 打印初始状态
        while True:
            # print(f'第{episode_num}周期，第{steps}步')
            dqn_state = [robot_state[1], robot_state[0], robot_state[5], robot_state[4]]  # 将机器人状态转换为DQN状态
            # log_writer_catch.add(dqn_state=dqn_state, steps=steps)
            obs = [obs_img, dqn_state]
            # log_writer_catch.add(obs=obs, steps=steps)
            # 将机器人状态转换为张量
            # x_graph = torch.tensor(robot_state, dtype=torch.float32).to(device)
            # x_graph = torch.tensor(robot_state, dtype=torch.float32).unsqueeze(1).to(device)  # 添加维度
            # 输入次数、状态，选择动作
            a_shoulder = dqn_shoulder.choose_action(episode_num=i, 
                                  obs=obs,
                                  x_graph=robot_state)
            a_arm = dqn_arm.choose_action(episode_num=i, 
                                  obs=obs,
                                  x_graph=robot_state)
            print(f'第{i}周期，第{steps}步，动作a_shoulder: {a_shoulder}, 动作a_arm: {a_arm}')
            # env.wait(1000)
            # print('wait 1000ms')

            # 确保动作是标量并限制在 [-1, 1] 范围内
            if torch.is_tensor(a_shoulder):
                a_shoulder = a_shoulder.item()  # 转换为 Python 标量
            #a_shoulder = max(-1.0, min(1.0, a_shoulder))  # 限制在 [-1, 1] 范围内
            
            if torch.is_tensor(a_arm):
                a_arm = a_arm.item()  # 转换为 Python 标量
            #a_arm = max(-1.0, min(1.0, a_arm))  # 限制在 [-1, 1] 范围内

            gps1, gps2, gps3, gps4, foot_gps1 = env.print_gps()  # 获取GPS位置
            if steps >= 19:  # 如果步数大于等于19
                catch_flag = 1.0  # 抓取器状态为1.0
            else:
                catch_flag = 0.0  # 抓取器状态为0.0
            img_name = "img" + str(steps) + ".png"  # 图像名称
            # print("action:", a)
            # 添加动作到日志
            log_writer_catch.add_action(a_shoulder, a_arm)
            # 执行一步动作
            next_state, reward, done, good, goal, count = env.step(robot_state, a_shoulder, a_arm, steps, catch_flag, gps1, gps2, gps3, gps4, img_name)
            print(f'catch_flag: {catch_flag}')
            print(f'done: {done}')
            
            if count == 1:  # 如果计数器为1 
                gps1, gps2, gps3, gps4, foot_gps1 = env.print_gps()  # 获取GPS位置
                x1 = gps_goal[0] - gps1[1]  # 计算目标位置与当前位置的差值
                y1 = gps_goal[1] - gps1[2]
                if x1 > -0.03 and y1 < 0.03:
                    reward1 = 1  # 奖励为1
                elif -0.05 < x1 < -0.03 and 0.03 < y1 < 0.05:
                    reward1 = 1  # 奖励为1
                else:
                    reward1 = 0  # 奖励为0
                reward = reward1  # 奖励为reward1
            return_all = return_all + reward  # 总奖励为当前奖励加上之前的总奖励
            steps += 1  # 步数加1
            next_obs_img, next_obs_tensor = env.get_img(steps, imgs)  # 获取下一个图像和图像张量
            next_obs = [next_obs_img, next_state]
            # print('获取下一个状态更新完毕')
            # 可以修改reward值让其训练速度加快
            if good == 1:  # 如果good为1
                # 将当前状态、动作、奖励、下一个状态、是否完成、是否达到目标添加到经验回放缓存中
                # 存储肩膀舵机经验
                rpm_shoulder.append((obs_img, robot_state, a_shoulder, reward, next_obs_img, next_state, done))
                # 存储手臂舵机经验
                rpm_arm.append((obs_img, robot_state, a_arm, reward, next_obs_img, next_state, done))
            robot_state = env.get_robot_state()  # 获取机器人状态
            obs_tensor = next_obs_tensor  # 更新图像张量
            #if len(rpm) < 5000:  # 如果经验回放缓存小于3000
                #episode_num = 0  # 计数器为0
            if len(rpm_shoulder) > 5000 and done == 1:  # 只有在buffer中存满了数据才会学习
                loss_shoulder = None
                loss_arm = None
                if goal == 1:  # 如果达到目标
                    print("goal = 1")
                    save_path_shoulder = path_list['model_path_catch_DQN'] + '/dqn_model_shoulder_%s.ckpt' % i  # 保存模型
                    save_path_arm = path_list['model_path_catch_DQN'] + '/dqn_model_arm_%s.ckpt' % i
                    torch.save(dqn_shoulder.eval_net, save_path_shoulder)        
                    torch.save(dqn_arm.eval_net, save_path_arm)
                    loss_shoulder = dqn_shoulder.learn(rpm_shoulder)  # 学习
                    loss_arm = dqn_arm.learn(rpm_arm)      #学习          
                    print(f"Loss_shoulder: {loss_shoulder}, Loss_arm: {loss_arm}, Reward: {reward}") 
                    log_writer_catch.add(loss_shoulder=loss_shoulder, loss_arm=loss_arm)
                # 每500步保存一次模型
                if i % 500 == 0:
                    path_shoulder = path_list['model_path_catch_DQN'] + '/dqn_model_shoulder_%s.ckpt' % i
                    torch.save(dqn_shoulder.eval_net, path_shoulder)
                    print(f"保存模型: {path_shoulder}")
                    path_arm = path_list['model_path_catch_DQN'] + '/dqn_model_arm_%s.ckpt' % i
                    torch.save(dqn_arm.eval_net, path_arm)
                    print(f"保存模型: {path_arm}")
                if loss_shoulder is not None and loss_arm is not None:
                    print(f"Loss_shoulder: {loss_shoulder}, Loss_arm: {loss_arm}")
                else:
                    print("Loss values not computed in this step (goal not reached).")
                # 写入总奖励
                log_writer_catch.add(return_all=return_all)
                # 写入目标
                log_writer_catch.add(goal=goal)
                
            success_flag1 = env.darwin.get_touch_sensor_value('grasp_L1_2')

            if catch_flag == 1.0 or done == 1:  # 如果抓取器状态为1.0或完成
                # 写入重置标志
                # if(success_flag1 == 0):
                #     env.reset()  # 重置环境
                env.wait(100)  # 等待100ms
                imgs = []  # 初始化图像列表
                steps = 0  # 初始化步数
                #episode_num = episode_num + 1  # 计数器加1
                # obs, obs_tensor = env.get_img(steps, imgs)  # 获取初始图像和图像张量
                # robot_state = env.get_robot_state()  # 获取机器人状态
                #log_writer_catch.add(action_list=log_writer_catch.action_list)
                log_writer_catch.clear_action()
                log_writer_catch.save_catch(log_file_latest_catch)  # 保存日志
                break
        
        if success_flag1 == 1:
            print("抓取成功，开始抬腿训练...")
            total_episoid = i
            print("tai_episoid:", tai_episoid)
            DQN_tai_episoid(dqn2_LegUpper=dqn2_LegUpper, dqn2_LegLower=dqn2_LegLower, dqn2_Ankle=dqn2_Ankle, existing_env=env, total_episoid=total_episoid, episode=tai_episoid, rpm_2=rpm_2, log_writer_tai=log_writer_tai, log_file_latest_tai=log_file_latest_tai)
            tai_episoid += 1


    # 如果整个训练过程结束，返回抓取成功状态和环境实例
    return False, env