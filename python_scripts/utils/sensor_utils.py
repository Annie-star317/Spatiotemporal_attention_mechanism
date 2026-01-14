import time
from python_scripts.Project_config import Darwin_config

def wait_for_sensors_stable(env, max_retries=30, wait_ms=200):
    """
    等待机器人传感器读数稳定
    
    参数:
        env: Webots 环境实例，需要包含 darwin 机器人和 step 方法
        max_retries: 最大重试次数
        wait_ms: 每次重试等待的毫秒数
        
    返回:
        bool: 如果传感器稳定返回 True，否则返回 False
    """
    print("检查传感器状态...")
    retry_count = 0
    
    while retry_count < max_retries:
        all_stable = True
        acc = env.darwin.accelerometer.getValues()
        gyro = env.darwin.gyro.getValues()
        
        # 检查所有传感器是否都在正常范围内
        for i in range(3):
            acc_ok = Darwin_config.acc_low[i] < acc[i] < Darwin_config.acc_high[i]
            gyro_ok = Darwin_config.gyro_low[i] < gyro[i] < Darwin_config.gyro_high[i]
            
            if not (acc_ok and gyro_ok):
                all_stable = False
                print(f"传感器不稳定 - 轴 {i+1}: acc={acc[i]:.2f} (目标: {Darwin_config.acc_low[i]}-{Darwin_config.acc_high[i]}), gyro={gyro[i]:.2f} (目标: {Darwin_config.gyro_low[i]}-{Darwin_config.gyro_high[i]})")
                break
        
        if all_stable:
            print("所有传感器已稳定")
            return True
            
        retry_count += 1
        print(f"等待传感器稳定... ({retry_count}/{max_retries})")
        env.wait(wait_ms)  # 等待指定毫秒
        env.robot.step(env.timestep)  # 单步执行仿真，让机器人有更多时间稳定
    
    print(f"警告: 达到最大重试次数({max_retries})，可能被卡住")
    return False

def reset_environment(env):
    """
    重置环境并等待传感器稳定
    
    参数:
        env: Webots 环境实例
        
    返回:
        bool: 如果重置后传感器稳定返回 True，否则返回 False
    """
    env.reset()
    env.wait(1000)

    return wait_for_sensors_stable(env)
