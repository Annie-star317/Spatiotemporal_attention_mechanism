import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import pandas as pd

def analyze_log8_results(log_file_path):
    """分析 log8 的训练结果"""
    print("正在分析 log8 训练结果...")

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取日志文件失败: {e}")
        return None

    # 提取训练指标
    episode_num = data.get('episode_num', [0])[0]
    print(f"训练总周期数: {episode_num}")

    # 分析损失函数
    loss_shoulder = data.get('loss_shoulder', [])
    loss_arm = data.get('loss_arm', [])
    loss_total = data.get('loss', [])

    # 分析熵
    entropy_shoulder = data.get('entropy_shoulder', [])
    entropy_arm = data.get('entropy_arm', [])

    # 分析策略损失和价值损失
    policy_loss_shoulder = data.get('policy_loss_shoulder', [])
    value_loss_shoulder = data.get('value_loss_shoulder', [])
    policy_loss_arm = data.get('policy_loss_arm', [])
    value_loss_arm = data.get('value_loss_arm', [])

    # 分析对数标准差
    log_std_shoulder = data.get('log_std_mean_shoulder', [])
    log_std_arm = data.get('log_std_mean_arm', [])

    print(f"\n损失函数统计:")
    if loss_shoulder:
        print(f"Shoulder 损失 - 均值: {np.mean(loss_shoulder):.4f}, 标准差: {np.std(loss_shoulder):.4f}")
        print(f"  最小值: {np.min(loss_shoulder):.4f}, 最大值: {np.max(loss_shoulder):.4f}")
    if loss_arm:
        print(f"Arm 损失 - 均值: {np.mean(loss_arm):.4f}, 标准差: {np.std(loss_arm):.4f}")
        print(f"  最小值: {np.min(loss_arm):.4f}, 最大值: {np.max(loss_arm):.4f}")
    if loss_total:
        print(f"总损失 - 均值: {np.mean(loss_total):.4f}, 标准差: {np.std(loss_total):.4f}")
        print(f"  最小值: {np.min(loss_total):.4f}, 最大值: {np.max(loss_total):.4f}")

    print(f"\n熵统计:")
    if entropy_shoulder:
        print(f"Shoulder 熵 - 均值: {np.mean(entropy_shoulder):.4f}, 标准差: {np.std(entropy_shoulder):.4f}")
    if entropy_arm:
        print(f"Arm 熵 - 均值: {np.mean(entropy_arm):.4f}, 标准差: {np.std(entropy_arm):.4f}")

    print(f"\n对数标准差统计:")
    if log_std_shoulder:
        print(f"Shoulder log_std - 均值: {np.mean(log_std_shoulder):.4f}, 标准差: {np.std(log_std_shoulder):.4f}")
    if log_std_arm:
        print(f"Arm log_std - 均值: {np.mean(log_std_arm):.4f}, 标准差: {np.std(log_std_arm):.4f}")

    # 分析训练稳定性 - 检查损失函数的收敛情况
    print(f"\n训练稳定性分析:")

    if len(loss_total) > 100:
        # 计算移动平均
        window_size = 100
        moving_avg_loss = pd.Series(loss_total).rolling(window=window_size).mean()

        # 检查后半部分是否收敛
        first_half = loss_total[:len(loss_total)//2]
        second_half = loss_total[len(loss_total)//2:]

        first_half_mean = np.mean(first_half)
        second_half_mean = np.mean(second_half)

        print(f"前半部分损失均值: {first_half_mean:.4f}")
        print(f"后半部分损失均值: {second_half_mean:.4f}")
        print(f"损失变化: {(second_half_mean - first_half_mean):.4f}")

        if abs(second_half_mean - first_half_mean) < 0.1:
            print("✓ 损失函数相对稳定")
        else:
            print("⚠ 损失函数仍在波动，可能未完全收敛")

    # 分析熵变化趋势
    if len(entropy_shoulder) > 100:
        first_half_entropy = entropy_shoulder[:len(entropy_shoulder)//2]
        second_half_entropy = entropy_shoulder[len(entropy_shoulder)//2:]

        first_entropy_mean = np.mean(first_half_entropy)
        second_entropy_mean = np.mean(second_half_entropy)

        print(f"\n熵变化分析:")
        print(f"前半部分熵均值: {first_entropy_mean:.4f}")
        print(f"后半部分熵均值: {second_entropy_mean:.4f}")
        print(f"熵变化: {(second_entropy_mean - first_entropy_mean):.4f}")

        if second_entropy_mean < first_entropy_mean:
            print("✓ 熵在下降，策略逐渐确定")
        else:
            print("⚠ 熵未明显下降，可能探索不足")

    # 分析动作分布
    shoulder_actions = data.get('shoulder_actions', [])
    arm_actions = data.get('arm_actions', [])

    if shoulder_actions:
        shoulder_flat = [item for sublist in shoulder_actions for item in sublist] if shoulder_actions and isinstance(shoulder_actions[0], list) else shoulder_actions
        print(f"\n动作分布分析:")
        print(f"Shoulder 动作 - 均值: {np.mean(shoulder_flat):.4f}, 标准差: {np.std(shoulder_flat):.4f}")
        print(f"  范围: [{np.min(shoulder_flat):.4f}, {np.max(shoulder_flat):.4f}]")

    if arm_actions:
        arm_flat = [item for sublist in arm_actions for item in sublist] if arm_actions and isinstance(arm_actions[0], list) else arm_actions
        print(f"Arm 动作 - 均值: {np.mean(arm_flat):.4f}, 标准差: {np.std(arm_flat):.4f}")
        print(f"  范围: [{np.min(arm_flat):.4f}, {np.max(arm_flat):.4f}]")

    return data

def plot_training_curves(data):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 损失函数曲线
    if 'loss' in data:
        axes[0, 0].plot(data['loss'], alpha=0.7, label='Total Loss')
        axes[0, 0].set_title('Total Loss over Episodes')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)

    # Shoulder vs Arm 损失
    if 'loss_shoulder' in data and 'loss_arm' in data:
        axes[0, 1].plot(data['loss_shoulder'], alpha=0.7, label='Shoulder Loss', color='blue')
        axes[0, 1].plot(data['loss_arm'], alpha=0.7, label='Arm Loss', color='red')
        axes[0, 1].set_title('Shoulder vs Arm Loss')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # 熵曲线
    if 'entropy_shoulder' in data and 'entropy_arm' in data:
        axes[1, 0].plot(data['entropy_shoulder'], alpha=0.7, label='Shoulder Entropy', color='blue')
        axes[1, 0].plot(data['entropy_arm'], alpha=0.7, label='Arm Entropy', color='red')
        axes[1, 0].set_title('Entropy over Episodes')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Entropy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # 对数标准差曲线
    if 'log_std_mean_shoulder' in data and 'log_std_mean_arm' in data:
        axes[1, 1].plot(data['log_std_mean_shoulder'], alpha=0.7, label='Shoulder Log Std', color='blue')
        axes[1, 1].plot(data['log_std_mean_arm'], alpha=0.7, label='Arm Log Std', color='red')
        axes[1, 1].set_title('Log Standard Deviation over Episodes')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Log Std')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('log8_training_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    log_file_path = r"d:\project_Spatiotemporal_attention_mechanism\python_scripts\PPO\log\catch_log\catch_log_8.json"
    data = analyze_log8_results(log_file_path)

    if data:
        print("\n" + "="*60)
        print("LOG8 训练结果评估完成")
        print("="*60)

        # 生成训练曲线图
        try:
            plot_training_curves(data)
            print("训练曲线图已保存为: log8_training_analysis.png")
        except Exception as e:
            print(f"生成训练曲线图失败: {e}")
