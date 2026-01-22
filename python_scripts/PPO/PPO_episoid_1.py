# 测试
import torch
import shutil
import heapq
import os
import glob
import re
import sys

# 保证在不同机器上也能找到 python_scripts 包：把项目根目录加入 sys.path
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CUR_DIR, os.pardir, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import numpy as np
from python_scripts.PPO.PPO_PPOnet_2 import PPO2
from python_scripts.PPO.PPO_PPOnet_attention import PPO, ActorCritic
from python_scripts.PPO.Replay_memory import ReplayMemory
from python_scripts.PPO.Replay_memory_2 import ReplayMemory_2
from python_scripts.PPO.PPO_episoid_2_1 import PPO_tai_episoid
from python_scripts.Webots_interfaces import Environment
# from Data_fusion import data_fusion
from python_scripts.Project_config import path_list, gps_goal, gps_goal1, device
from python_scripts.PPO_Log_write import Log_write
from python_scripts.PPO.RobotRun1 import RobotRun 
from python_scripts.utils.sensor_utils import wait_for_sensors_stable
from python_scripts.utils.sensor_utils import reset_environment
from copy import deepcopy

class ModelRanking:
    """
    一个用于追踪和管理N个最佳模型的辅助类。
    它使用最小堆来高效地找到当前性能最差的模型。
    """
    def __init__(self, top_n=5):
        self.top_n = top_n
        self.rankings = []
        self.saved_paths = []

    def add_and_manage(self, new_score, new_checkpoint, episode_id, base_dir, success_count):
        """
        核心方法：根据新模型的分数和排行榜情况，决定是否保存模型文件。
        """
        new_entry = (new_score, "")

        should_save = False
        final_save_path = ""

        # 如果排行榜未满，直接保存
        if len(self.rankings) < self.top_n:
            should_save = True
            final_save_path = os.path.join(base_dir, f'ppo_model_success_{episode_id}_{success_count}.ckpt')
        # 如果排行榜已满，但新模型比最差的要好
        elif new_score > self.rankings[0][0]:
            should_save = True
            final_save_path = os.path.join(base_dir, f'ppo_model_success_{episode_id}_{success_count}.ckpt')
            worst_score, worst_path_to_delete = heapq.heappop(self.rankings)
            try:
                os.remove(worst_path_to_delete)
                print(f"删除旧模型文件: {worst_path_to_delete} (成功率: {worst_score:.2f}%)")
            except FileNotFoundError:
                print(f"警告: 试图删除不存在的文件 {worst_path_to_delete}")

        if should_save:
            torch.save(new_checkpoint, final_save_path)
            new_entry = (new_score, final_save_path)
            heapq.heappush(self.rankings, new_entry)
            print(f"模型 {episode_id} (成功率: {new_score:.2f}%, 成功次数: {success_count}) 已保存到 {final_save_path} 并加入排行榜。")
            return final_save_path
        else:
            print(f"模型 {episode_id} (成功率: {new_score:.2f}%, 成功次数: {success_count}) 性能未进入前 {self.top_n}，未保存。")
            return None

    def print_current_rankings(self):
        """打印当前排行榜内容。"""
        if not self.rankings:
            print("当前排行榜为空。")
            return
            
        print("\n--- 基于测试成功率的最佳模型排行榜 ---")
        sorted_rankings = sorted(self.rankings, key=lambda x: x[0], reverse=True)
        for i, (score, path) in enumerate(sorted_rankings, 1):
            # 从文件名中提取成功次数
            filename = os.path.basename(path)
            parts = filename.split('_')
            success_count = parts[-1].split('.')[0]  # 去掉.ckpt后缀
            ep_num = parts[-2]  # episode_id
            print(f"  {i}. Episode {ep_num}: Success Count = {success_count}, Path = {path}")
        print("-----------------------------------------\n")




def PPO_episoid_1(model_path=None, max_steps_per_episode=500): 
    
    policy_shoulder_net = ActorCritic(act_dim=1).to(device)
    policy_arm_net = ActorCritic(act_dim=1).to(device)

    # 优化学习参数：平衡学习率和稳定性
    ppo_shoulder = PPO(policy=policy_shoulder_net, act_dim=1, lr=1e-5, clip_ratio=0.2,  # 稍微提高学习率到1e-5
                         update_epochs=4, minibatch_size=32, gamma=0.99, lam=0.95, entropy_coef=0.1, device=device)  # 增加entropy_coef到0.1，约束sigma减小
    ppo_arm = PPO(policy=policy_arm_net, act_dim=1, lr=1e-5, clip_ratio=0.2,
                   update_epochs=4, minibatch_size=32, gamma=0.99, lam=0.95, entropy_coef=0.1, device=device)

    ppo2_LegUpper = PPO2(node_num=20, env_information=None)  # 创建PPO2对象
    ppo2_LegLower = PPO2(node_num=20, env_information=None)  # 创建PPO2对象
    ppo2_Ankle = PPO2(node_num=20, env_information=None)  # 创建PPO2对象

    # --- 初始化排行榜 ---
    top_n_models = 5
    model_ranking = ModelRanking(top_n=top_n_models)
    
    # 初始化日志写入器
    log_writer_catch = Log_write()  # 创建抓取日志写入器
    log_writer_tai = Log_write()  # 创建抬腿日志写入器
    CHECKPOINT_INTERVAL = 300  # 从500减少到300，更频繁的评估
    NUM_TEST_EPISODES = 50     # 从100减少到50，加快测试速度  
    MAX_TEST_ATTEMPTS = 5 
    tai_episoid = 1
    import os
    import glob
    import re
    # 查找现有的日志文件，确定最新的编号
    # 抓取阶段：
    log_pattern = os.path.join(path_list['catch_log_path_PPO'], 'catch_log_*.json')
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
    log_file_latest_catch = os.path.join(path_list['catch_log_path_PPO'], f"catch_log_{new_log_num}.json")
    print(f"将使用新的抓取日志目录: {log_file_latest_catch}")

    # 抬腿阶段：
    log_pattern = os.path.join(path_list['tai_log_path_PPO'], 'tai_log_*.json')
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
    log_file_latest_tai = os.path.join(path_list['tai_log_path_PPO'], f"tai_log_{new_log_num}.json")
    print(f"将使用新抬腿的日志目录: {log_file_latest_tai}")

    # 加载模型
    # 抓取模型加载
    if model_path:  # 如果指定了模型路径
        try:
            # 从指定路径加载模型
            checkpoint = torch.load(model_path)
            if isinstance(checkpoint, dict) and 'policy_shoulder' in checkpoint:
                # 如果是保存的字典格式 {'policy': state_dict, ...}
                ppo_shoulder.policy.load_state_dict(checkpoint['policy_shoulder'])
                ppo_arm.policy.load_state_dict(checkpoint['policy_arm'])
                # 如果需要加载优化器状态
                if 'optimizer_shoulder' in checkpoint and ppo_shoulder.optimizer:
                    ppo_shoulder.optimizer.load_state_dict(checkpoint['optimizer_shoulder'])
                if 'optimizer_arm' in checkpoint and ppo_arm.optimizer:
                    ppo_arm.optimizer.load_state_dict(checkpoint['optimizer_arm'])
                print("从指定模型加载: {model_path}，模型加载成功！")
                episode_start = int(model_path.split('_')[-1].split('.')[0])
                print(f"从指定模型加载: {model_path}，从周期 {episode_start} 继续训练")
            else:
                # 如果是直接保存的模型或状态字典
                ppo_shoulder.policy.load_state_dict(checkpoint)
                ppo_arm.policy.load_state_dict(checkpoint)
                print("从指定模型加载: {model_path}，模型加载成功！(旧格式)")
                episode_start = 0
        except Exception as e:
            print(f"指定模型加载失败: {e}")
            episode_start = 0
    else:  # 如果没有指定模型路径，使用原来的自动查找逻辑
        # 获取所有模型文件
        model_files = glob.glob(path_list['model_path_catch_PPO'] + '/ppo_model_*.ckpt')
        if model_files:
            # 更健壮地从文件名提取周期号：ppo_model_success_{episode_id}_{success_count}.ckpt，取episode_id
            def extract_episode_id(pathname):
                name = os.path.basename(pathname)
                # 匹配模式：ppo_model_success_{episode_id}_{success_count}.ckpt
                match = re.search(r'ppo_model_success_(\d+)_(\d+)\.ckpt', name)
                if match:
                    episode_id = int(match.group(1))  # 第一组数字是episode_id
                    success_count = int(match.group(2))  # 第二组数字是success_count
                    return episode_id, success_count
                # 如果匹配失败，回退到原来的逻辑
                nums = re.findall(r'\d+', name)
                if nums:
                    try:
                        return int(nums[-2]) if len(nums) >= 2 else int(nums[-1]), 0  # 取倒数第二个数字作为episode_id
                    except Exception:
                        return 0, 0
                return 0, 0

            latest_model = max(model_files, key=lambda x: extract_episode_id(x)[0])  # 按episode_id排序
            episode_start, success_count = extract_episode_id(latest_model)
            print(f"找到最新抓取模型: {latest_model}，从周期 {episode_start} 继续训练 (该模型测试成功次数: {success_count})")
            
            # 加载模型（与原逻辑相同）
            try:
                checkpoint = torch.load(latest_model)
                if isinstance(checkpoint, dict) and 'policy_shoulder' in checkpoint:
                    # 如果是保存的字典格式 {'policy': state_dict, ...}
                    ppo_shoulder.policy.load_state_dict(checkpoint['policy_shoulder'])
                    ppo_arm.policy.load_state_dict(checkpoint['policy_arm'])
                    # 如果需要加载优化器状态
                    if 'optimizer_shoulder' in checkpoint and ppo_shoulder.optimizer:
                        ppo_shoulder.optimizer.load_state_dict(checkpoint['optimizer_shoulder'])
                    if 'optimizer_arm' in checkpoint and ppo_arm.optimizer:
                        ppo_arm.optimizer.load_state_dict(checkpoint['optimizer_arm'])
                    print("抓取模型加载成功！")
                else:
                    # 如果是直接保存的模型或状态字典
                    ppo_shoulder.policy.load_state_dict(checkpoint)
                    ppo_arm.policy.load_state_dict(checkpoint)
                    print("抓取模型加载成功！(旧格式)")
            except Exception as e:
                print(f"抓取模型加载失败: {e}")
                episode_start = 0
        else:
            print("未找到已保存的抓取模型，从头开始训练")
            episode_start = 0
    
    # 抬腿模型加载
    model_files_tai = glob.glob(path_list['model_path_tai_PPO'] + '/ppo_model_tai_*.ckpt')
    if model_files_tai:
        try:
            # 按新的文件名格式排序：ppo_model_tai_{total_episoid}_{episode}.ckpt
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
            print(f"抬腿模型从周期 {tai_episoid} 继续训练")
            # 加载模型
            try:
                checkpoint = torch.load(latest_model)
                if isinstance(checkpoint, dict) and 'policy_LegUpper' in checkpoint:
                    # 如果是保存的字典格式 {'policy': state_dict, ...}
                    ppo2_LegUpper.policy.load_state_dict(checkpoint['policy_LegUpper'])
                    ppo2_LegLower.policy.load_state_dict(checkpoint['policy_LegLower'])
                    ppo2_Ankle.policy.load_state_dict(checkpoint['policy_Ankle'])
                    # 如果需要加载优化器状态
                    if 'optimizer' in checkpoint and ppo2_LegUpper.optimizer:
                        ppo2_LegUpper.optimizer.load_state_dict(checkpoint['optimizer_LegUpper'])
                    if 'optimizer' in checkpoint and ppo2_LegLower.optimizer:
                        ppo2_LegLower.optimizer.load_state_dict(checkpoint['optimizer_LegLower'])
                    if 'optimizer' in checkpoint and ppo2_Ankle.optimizer:
                        ppo2_Ankle.optimizer.load_state_dict(checkpoint['optimizer_Ankle'])
                    print("抬腿模型加载成功！")
                else:
                    # 如果是直接保存的模型或状态字典
                    ppo2_LegUpper.policy.load_state_dict(checkpoint)
                    ppo2_LegLower.policy.load_state_dict(checkpoint)
                    ppo2_Ankle.policy.load_state_dict(checkpoint)
                    print("抬腿模型加载成功！(旧格式)")
            except Exception as e:
                print(f"抬腿模型加载失败: {e}")
        except Exception as e:
            print(f"抬腿模型加载失败: {e}")
    else:
        print("未找到已保存的抬腿模型，从头开始训练")




    episode_num = episode_start  # 初始化回合计数器
    env = Environment()
    success_catch = 0                  # 抓取成功次数
    valid_episode = 0  # 新增标记，有效轮次
    # 用于控制“累计成功回合学习一次”
    success_episode_for_train = 0      # 参与训练计数的成功回合数
    episode_count_for_train = 0        # 总episode计数，用于定期学习

    max_total_episodes = episode_start + 10000

    while episode_num < max_total_episodes:  # 从episode_start开始，最多再训练多少个周期
        i = episode_num
        log_writer_catch.clear()
        log_writer_catch.add(episode_num=i)
        print(f"<<<<<<<<<第{i}周期") # 打印当前周期
        success_flag1 = 0
        # 在 episode 开始之前：冻结 policy_old（用于数据收集时logp）并清空 actor 内部时间 buffer
        ppo_shoulder.start_collection()
        ppo_arm.start_collection()

        env.reset()
        env.wait(500)   # 等待500ms
        # 使用工具函数检查传感器状态
        if not wait_for_sensors_stable(env, max_retries=40, wait_ms=200):
            print("警告: 传感器不稳定，尝试重置环境...")
            reset_environment(env)        
        imgs = []  # 初始化图像列表
        steps = 0  # 初始化步数
        return_all = 0  # 初始化总奖励
        obs_img, obs_tensor = env.get_img(steps, imgs)  # 获取初始图像和图像张量
        robot_state = env.get_robot_state()  # 获取机器人状态
        ppo_state = [robot_state[1], robot_state[0], robot_state[5], robot_state[4]]  # 将机器人状态转换为ppo状态
        ppo_state = torch.tensor(ppo_state, dtype=torch.float32, device=device)
        obs = (obs_tensor, ppo_state)
        print("____________________")  # 打印初始状态
        prev_distance = None
        prev_action_shoulder = 0.0
        prev_action_arm = 0.0
        return_all = 0
        steps = 0
        # 单回合最多 19 步（也可以通过函数参数 max_steps_per_episode 控制，默认 19）
        max_steps_per_episode = min(max_steps_per_episode, 19)
                  
        # 统计当前回合是否最终成功，以及当前回合的样本缓存
        episode_success = 0
        episode_invalid = False
        episode_buffer_shoulder = []
        episode_buffer_arm = []

        
                  
        while True:
            # 1) Select actions (training mode)
            action_shoulder, log_prob_shoulder, value_shoulder = ppo_shoulder.choose_action(obs_img, ppo_state, deterministic=False)
            action_arm, log_prob_arm, value_arm = ppo_arm.choose_action(obs_img, ppo_state, deterministic=False)

            # 将 action 转为 scalar 并进行合理裁剪（与测试保持一致）
            action_shoulder_clipped = np.clip(action_shoulder, -0.5, 0.5)
            action_arm_clipped = np.clip(action_arm, -0.5, 0.5)
            action_shoulder_scalar = float(action_shoulder_clipped.flatten()[0])
            action_arm_scalar = float(action_arm_clipped.flatten()[0])

            print(f'第{i}周期，第{steps}步，肩膀动作: {action_shoulder_scalar:.4f}，手臂动作: {action_arm_scalar:.4f}')           
            # collect sensor info & flags
            gps1, gps2, gps3, gps4, foot_gps1 = env.print_gps()
            # 修复 off-by-one：对下一步判断，使第19步执行时 catch_flag 为 1.0
            catch_flag = 1.0 if (steps + 1) >= 19 else 0.0
            img_name = f"img{steps}.png"

            # 日志记录（保持你原来的逻辑）
            log_writer_catch.add_action_catch(round(action_shoulder_scalar, 4), round(action_arm_scalar, 4))
            log_writer_catch.add_log_prob_catch(round(log_prob_shoulder, 4), round(log_prob_arm, 4))
            log_writer_catch.add_value_catch(round(value_shoulder, 4), round(value_arm, 4))

            # 执行动作
            next_state, reward, done, good, goal, count = env.step(robot_state, action_shoulder_scalar, action_arm_scalar, steps, catch_flag, gps1, gps2, gps3, gps4, img_name)

            # ---------- reward design (改进版：稳定距离奖励，避免震荡) ----------
            if len(gps1) < 3:
                dy, dz = 0.0, 0.0
            else:
                dy = gps_goal1[0] - gps1[1]
                dz = gps_goal1[1] - gps1[2]
            current_distance = (dy ** 2 + dz ** 2) ** 0.5

            if prev_distance is not None:
                # 使用平滑的距离奖励：基于距离的倒数，避免震荡
                distance_improvement = prev_distance - current_distance
                if distance_improvement > 0:  # 只有在接近目标时才给正奖励
                    reward = distance_improvement * 3.0  # 降低系数避免过大梯度
                else:
                    reward = distance_improvement * 1.0  # 远离目标时轻微惩罚

                # 额外奖励：距离越近奖励越大
                proximity_reward = max(0, (1.0 - current_distance / 2.0)) * 0.5  # 距离越近奖励越大
                reward += proximity_reward
            else:
                reward = -current_distance * 0.3  # 减少初始距离惩罚
            prev_distance = current_distance

            # 抓取检测（放宽条件：单手成功即可）
            all_grasp_sensors = [
                env.darwin.get_touch_sensor_value('grasp_L1'),
                env.darwin.get_touch_sensor_value('grasp_L1_1'),
                env.darwin.get_touch_sensor_value('grasp_L1_2'),
                env.darwin.get_touch_sensor_value('grasp_R1'),
                env.darwin.get_touch_sensor_value('grasp_R1_1'),
                env.darwin.get_touch_sensor_value('grasp_R1_2')
            ]
            left_any = any(all_grasp_sensors[0:3])
            right_any = any(all_grasp_sensors[3:6])
            # 放宽成功条件：只要任意一侧有接触就算成功
            success_flag1 = 1 if (left_any or right_any) else 0

            if success_flag1 == 1:
                print(f"current_distance: {current_distance}")
                if current_distance <= 0.15:  # 放宽距离要求从0.1到0.15
                    print("√抓到了目标阶梯")
                    reward += 50.0  # 增加成功奖励到50.0，提供更强的学习信号
                else:
                    reward += 15.0  # 接触但距离远的给中等奖励

            # 温和的失败惩罚，避免过度惩罚
            if done == 1 and success_flag1 != 1:
                if steps < 6:
                    reward -= 2.0  # 从-5.0减少到-2.0
                else:
                    reward -= 3.0  # 从-8.0减少到-3.0
            if done == 1 and steps <= 2 and success_flag1 != 1:
                # 环境不稳，判无效并跳过本回合写入
                episode_invalid = True
                episode_buffer_shoulder = []
                episode_buffer_arm = []

            # 移除动作变化奖励，避免鼓励过大sigma
            # 让熵正则化自然控制探索-利用平衡
            prev_action_shoulder = action_shoulder_scalar
            prev_action_arm = action_arm_scalar

            # 时间惩罚 & 累计 reward
            reward -= steps * 0.05
            return_all += reward
            steps += 1

            # 获取下一个 obs
            next_obs_img, next_obs_tensor = env.get_img(steps, imgs)
            next_state = [next_state[1], next_state[0], next_state[5], next_state[4]]
            # 同步 obs 变量（保持一致）
            obs_img = next_obs_img
            obs_tensor = next_obs_tensor
            ppo_state = torch.tensor(next_state, dtype=torch.float32, device=device)

            # 决定是否存储本步 transition 到回合缓存（先缓存在 episode_buffer_*）
            should_store = True
            if done == 1 and steps <= 2 and success_flag1 != 1:
                should_store = False
            
            if should_store:
                tr = {
                    'obs_img': deepcopy(obs_img),
                    'obs_state': deepcopy(ppo_state.clone().detach().cpu().numpy()) if isinstance(ppo_state, torch.Tensor) else deepcopy(ppo_state),
                    'action_shoulder': deepcopy(action_shoulder),
                    'action_arm': deepcopy(action_arm),
                    'reward': float(reward),
                    'next_obs_img': deepcopy(next_obs_img),
                    'next_state': deepcopy(next_state),
                    'done': int(done),
                    'log_prob_shoulder': float(log_prob_shoulder),
                    'log_prob_arm': float(log_prob_arm),
                    'value_shoulder': float(value_shoulder),
                    'value_arm': float(value_arm)
                }
                episode_buffer_shoulder.append(tr)
                episode_buffer_arm.append(tr)

            robot_state = env.get_robot_state()

            # Episode 结束条件
            if done == 1 or steps >= max_steps_per_episode:
                # 若本回合被判为无效，跳过写入并不计入训练
                if episode_invalid:
                    break

                # 所有回合都参与学习（无论成功失败），但只在有数据时才存储到episode缓冲区
                if len(episode_buffer_shoulder) > 0:
                    # 将episode数据临时存储到PPO缓冲区（不立即计算advantages）
                    for tr in episode_buffer_shoulder:
                        ppo_shoulder.store_transition_catch(
                            obs_img=tr['obs_img'],
                            obs_state=tr['obs_state'],
                            action=tr['action_shoulder'],
                            logp=tr['log_prob_shoulder'],
                            reward=tr['reward'],
                            value=tr['value_shoulder'],
                            done=tr['done']
                        )

                    for tr in episode_buffer_arm:
                        ppo_arm.store_transition_catch(
                            obs_img=tr['obs_img'],
                            obs_state=tr['obs_state'],
                            action=tr['action_arm'],
                            logp=tr['log_prob_arm'],
                            reward=tr['reward'],
                            value=tr['value_arm'],
                            done=tr['done']
                        )

                    episode_success = 1 if success_flag1 == 1 else 0
                else:
                    # 没有有效数据，不参与学习
                    episode_success = 0

                # 清空episode缓存（无论成功或失败）
                episode_buffer_shoulder = []
                episode_buffer_arm = []

                # 触发训练逻辑：每10个episode学习一次，积累数据后再学习
                episode_count_for_train += 1

                # 每10个episode学习一次，使用积累的所有episode数据
                should_learn = (episode_count_for_train % 10 == 0)

                if should_learn:
                    print(f"--- 积累了{episode_count_for_train}个episode，开始学习 ---")

                    # 为所有积累的episode数据计算advantages（一次计算所有数据）
                    ppo_shoulder.finish_path(last_value=0.0)
                    ppo_arm.finish_path(last_value=0.0)

                    # 执行学习，使用所有积累的episode数据
                    stats_shoulder = ppo_shoulder.learn()
                    stats_arm = ppo_arm.learn()

                    # 记录到日志（分别记录 shoulder/arm 以及总损失）
                    try:
                        ls_sh = float(stats_shoulder.get('loss', 0.0))
                        pl_sh = float(stats_shoulder.get('policy_loss', 0.0))
                        vl_sh = float(stats_shoulder.get('value_loss', 0.0))
                        en_sh = float(stats_shoulder.get('entropy', 0.0))
                        logstd_sh = float(stats_shoulder.get('log_std_mean', 0.0))
                        logstd_grad_sh = float(stats_shoulder.get('log_std_grad_norm', 0.0))
                        mean_sigma_sh = float(stats_shoulder.get('mean_sigma', 0.0))
                    except Exception:
                        ls_sh = pl_sh = vl_sh = en_sh = 0.0
                        logstd_sh = logstd_grad_sh = mean_sigma_sh = 0.0

                    try:
                        ls_ar = float(stats_arm.get('loss', 0.0))
                        pl_ar = float(stats_arm.get('policy_loss', 0.0))
                        vl_ar = float(stats_arm.get('value_loss', 0.0))
                        en_ar = float(stats_arm.get('entropy', 0.0))
                        logstd_ar = float(stats_arm.get('log_std_mean', 0.0))
                        logstd_grad_ar = float(stats_arm.get('log_std_grad_norm', 0.0))
                        mean_sigma_ar = float(stats_arm.get('mean_sigma', 0.0))
                    except Exception:
                        ls_ar = pl_ar = vl_ar = en_ar = 0.0
                        logstd_ar = logstd_grad_ar = mean_sigma_ar = 0.0

                    total_loss = ls_sh + ls_ar
                    # 写入日志
                    log_writer_catch.add(loss_shoulder=ls_sh, policy_loss_shoulder=pl_sh, value_loss_shoulder=vl_sh, entropy_shoulder=en_sh, log_std_mean_shoulder=logstd_sh, log_std_grad_shoulder=logstd_grad_sh, mean_sigma_shoulder=mean_sigma_sh)
                    log_writer_catch.add(loss_arm=ls_ar, policy_loss_arm=pl_ar, value_loss_arm=vl_ar, entropy_arm=en_ar, log_std_mean_arm=logstd_ar, log_std_grad_arm=logstd_grad_ar, mean_sigma_arm=mean_sigma_ar)
                    log_writer_catch.add(loss=total_loss)

                    # 学习完成后清空PPO内部缓冲区，为下轮10个episode积累做准备
                    try:
                        ppo_shoulder.reset_buffer()
                    except Exception:
                        pass
                    try:
                        ppo_arm.reset_buffer()
                    except Exception:
                        pass

                    episode_count_for_train = 0  # 重置计数器

                else:
                    print(f"继续积累数据... (当前{episode_count_for_train}/10个episode)")

                base_checkpoint_data = {
                    'policy_shoulder': ppo_shoulder.policy.state_dict(),
                    'optimizer_shoulder': ppo_shoulder.optimizer.state_dict(),
                    'policy_arm': ppo_arm.policy.state_dict(),
                    'optimizer_arm': ppo_arm.optimizer.state_dict(),
                    'episode': i
                }

                 # --- 【逻辑决策点】决定何时进行模型评估 ---
                valid_episode += 1
                print("有效周期",valid_episode)
                is_checkpoint_interval = (valid_episode % CHECKPOINT_INTERVAL == 0) and (valid_episode > 0)
                #is_checkpoint_interval = i % CHECKPOINT_INTERVAL == 0

                if is_checkpoint_interval:
                    print(f"\n--- 周期 {i}: 到达检查点，开始在当前环境进行模型测试 (共 {NUM_TEST_EPISODES} 轮有效) ---")
                    ppo_shoulder.policy.eval()
                    ppo_arm.policy.eval()

                    successful_test_episodes = 0
                    valid_test_cnt   = 0          # 已经跑完的有效轮次
                    total_test_cnt   = 0          # 总共开的轮次（含无效）
                    max_steps_per_test_episode = 500
                     

                    while valid_test_cnt < NUM_TEST_EPISODES:          # 只认有效轮次
                        total_test_cnt += 1
                        print(f"————————————————测试轮次 {valid_test_cnt + 1}/{NUM_TEST_EPISODES} "
                            f"(总开启 {total_test_cnt})——————————————")

                        # -------- 1. 初始化 --------
                        is_test_valid = False
                        for init_try in range(MAX_TEST_ATTEMPTS):
                            env.reset()
                            env.wait(200)
                            if wait_for_sensors_stable(env, max_retries=40, wait_ms=200):
                                is_test_valid = True
                                break
                            print(f"  警告: 传感器不稳定，尝试重置... ({init_try + 1}/{MAX_TEST_ATTEMPTS})")

                        if not is_test_valid:
                            print(f"  ❌ 初始化失败，此轮不计入有效统计。")
                            continue                 # 直接重开一轮

                        # -------- 2. 跑 episode --------
                        test_steps, test_done = 0, False
                        test_imgs = []
                        while not test_done and test_steps < max_steps_per_test_episode:
                            test_obs_img, test_obs_tensor = env.get_img(test_steps, test_imgs)
                            test_robot_state = env.get_robot_state()
                            test_ppo_state = [test_robot_state[1], test_robot_state[0], test_robot_state[5], test_robot_state[4]]  # 将机器人状态转换为ppo状态
                            if len(test_robot_state) < 6:
                                print("  测试警告：robot_state 长度不足，提前结束本轮。")
                                break

                            test_obs = (test_obs_tensor, test_ppo_state)
                            with torch.no_grad():
                                action_shoulder, log_prob_shoulder, value_shoulder = ppo_shoulder.choose_action(test_obs_tensor, test_ppo_state, deterministic=True)
                                action_arm, log_prob_arm, value_arm = ppo_arm.choose_action(test_obs_tensor, test_ppo_state, deterministic=True)


                            # 转 float + 限幅
                            action_shoulder_clipped = np.clip(action_shoulder, -0.5, 0.5)
                            action_arm_clipped      = np.clip(action_arm,      -0.5, 0.5)
                            action_shoulder_t = float(action_shoulder_clipped.flatten()[0])
                            action_arm_t      = float(action_arm_clipped.flatten()[0])
                            test_gps1, test_gps2, test_gps3, test_gps4, _ = env.print_gps()
                            if len(test_gps1) < 3:
                                test_steps += 1
                                continue
                            test_catch_flag = 1.0 if test_steps >= 19 else 0.0
                            _, _, test_done_from_env, _, test_goal_from_env, _ = env.step(
                                test_robot_state, action_shoulder_t, action_arm_t, test_steps,
                                test_catch_flag, test_gps1, test_gps2, test_gps3, test_gps4,
                                f"test_img_{test_steps}.png")
                            
                            if test_done_from_env or test_goal_from_env:
                                test_done = True

                            test_steps += 1

                        # -------- 3. 判定结果 --------
                        # 使用与训练阶段完全相同的成功判定逻辑：检查多个传感器 + 距离条件
                        test_all_grasp_sensors = [
                            env.darwin.get_touch_sensor_value('grasp_L1'),
                            env.darwin.get_touch_sensor_value('grasp_L1_1'),
                            env.darwin.get_touch_sensor_value('grasp_L1_2'),
                            env.darwin.get_touch_sensor_value('grasp_R1'),
                            env.darwin.get_touch_sensor_value('grasp_R1_1'),
                            env.darwin.get_touch_sensor_value('grasp_R1_2')
                        ]
                        test_left_any = any(test_all_grasp_sensors[0:3])
                        test_right_any = any(test_all_grasp_sensors[3:6])
                        test_sensor_triggered = test_left_any or test_right_any

                        # 计算距离（与训练阶段相同的距离计算逻辑）
                        if len(test_gps1) < 3:
                            test_current_distance = float('inf')  # 无法计算距离
                        else:
                            dy = gps_goal1[0] - test_gps1[1]
                            dz = gps_goal1[1] - test_gps1[2]
                            test_current_distance = (dy ** 2 + dz ** 2) ** 0.5

                        # 与训练阶段完全相同的成功判定：传感器触发 AND 距离<=0.15
                        test_success_flag = 1 if (test_sensor_triggered and test_current_distance <= 0.15) else 0

                        early_fail = (test_steps <= 2 and test_success_flag != 1)

                        if early_fail:
                            print(f"  ❌ 过早结束且未成功，此轮无效。")
                            continue
                        elif test_success_flag == 1:
                            successful_test_episodes += 1
                            print(f"  ✓ 测试成功！(传感器:L{test_left_any},R{test_right_any}, 距离:{test_current_distance:.3f})")
                        else:
                            print(f"  ✗ 测试失败。(传感器:L{test_left_any},R{test_right_any}, 距离:{test_current_distance:.3f})")

                        valid_test_cnt += 1          # 只有跑到这里才算完成一次有效测试
                    ppo_shoulder.policy.train()
                    ppo_arm.policy.train() # 两者都切换到训练模式
                    test_success_rate = (successful_test_episodes / NUM_TEST_EPISODES) * 100  # 计算成功率百分比
                    log_writer_catch.add(success_rate=test_success_rate)
                    print(f"\n--- 测试完成：{NUM_TEST_EPISODES}轮测试中成功 {successful_test_episodes} 轮，成功率为 {test_success_rate:.2f}% ---")
                    
                    # --- 【修正】排行榜也使用单一检查点 ---
                    model_ranking.add_and_manage(
                        new_score=test_success_rate,
                        new_checkpoint=base_checkpoint_data,
                        episode_id=i,
                        base_dir=path_list['model_path_catch_PPO'],
                        success_count=successful_test_episodes
                    )

                    model_ranking.print_current_rankings()

                else:
                    # 如果不是检查点周期，跳过测试
                    print(f"\n--- 周期 {i}: 未到达检查点，跳过模型测试 ---")

                # 3. 记录本轮训练日志并重置状态
                print(f"本轮训练累积奖励: {return_all:.2f}, 目标达成: {success_flag1}")
                log_writer_catch.add_log_prob_catch(log_prob_shoulder, log_prob_arm) 
                log_writer_catch.add_value_catch(value_shoulder, value_arm)
                # 使用 ppo_shoulder 或 ppo_arm 都可以，因为它们的 sigma 参数应该是一样的
                current_sigma = ppo_shoulder.get_current_sigma()
                log_writer_catch.add(sigma=current_sigma)
                log_writer_catch.add(return_all=return_all)
                log_writer_catch.add(goal=1 if success_flag1 else 0)
                log_writer_catch.clear()
                log_writer_catch.save_catch(log_file_latest_catch)
                
                # 4. 跳出while循环，开始下一个episode
                break

                
            #success_flag1 = env.darwin.get_touch_sensor_value('grasp_L1_2')

        if episode_invalid:
            print(f"第{i}周期判定为无效，跳过所有日志和训练，重新尝试同一周期编号")
            log_writer_catch.clear()
            continue
        if catch_flag == 1.0 or done == 1:  # 如果抓取器状态为1.0或完成
            env.wait(100)  # 等待100ms
            episode_num = episode_num + 1  # 计数器加1
            log_writer_catch.clear()
          

        if success_flag1 == 1:
            success_catch += 1
            #log_writer_catch.add(success_catch=success_catch)
            log_writer_catch.data['success_catch'] = success_catch
            print("success_catch:", success_catch)
            print("抓取成功，开始抬腿训练...")
            total_episode = i
            print("tai_episoid:", tai_episoid)
            PPO_tai_episoid(ppo2_LegUpper=ppo2_LegUpper, ppo2_LegLower=ppo2_LegLower, ppo2_Ankle=ppo2_Ankle, existing_env=env, total_episode=total_episode, episode=tai_episoid, log_writer_tai=log_writer_tai, log_file_latest_tai=log_file_latest_tai)
            tai_episoid += 1 


    # 如果整个训练过程结束，返回抓取成功状态和环境实例
    return False, env