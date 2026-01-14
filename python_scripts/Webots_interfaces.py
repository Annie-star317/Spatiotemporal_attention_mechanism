import os
import sys
from controller import Robot, Motor, Motion, LED, Camera, Gyro, Accelerometer, PositionSensor, GPS
from PIL import Image
import numpy as np
import torch
import shutil
sys.path.append('D:/Webots/Webots2021a/Webots/projects/robots/robotis/darwin-op/libraries/python37')
#from managers import RobotisOp2GaitManager, RobotisOp2MotionManager
from python_scripts.Project_config import path_list

class Darwin:
    """Darwin机器人类
    
    功能：封装Darwin机器人的所有操作接口
    
    属性：
        motors: 电机设备列表
        sensors: 传感器设备列表
        camera: 相机设备
        gps: GPS设备组
        touch_sensors: 触碰传感器设备列表
        其他传感器设备...
    """
    def __init__(self, robot):
        # 基础设备初始化
        self.robot = robot
        self.timestep = int(robot.getBasicTimeStep())
        #self.gaitManager = RobotisOp2GaitManager(robot, 'config.ini')
        #self.motionManager = RobotisOp2MotionManager(robot)
        #self.gaitManager.setBalanceEnable(True)
        # 舵机列表初始化
        self.motors = []
        self.motors_sensors = []
        self.motorName = ('ShoulderR', 'ShoulderL', 'ArmUpperR', 'ArmUpperL',
                          'ArmLowerR', 'ArmLowerL', 'PelvYR',    'PelvYL', 
                          'PelvR',     'PelvL',     'LegUpperR', 'LegUpperL', 
                          'LegLowerR', 'LegLowerL', 'AnkleR',    'AnkleL', 
                          'FootR',     'FootL',     'Neck',      'Head', 
                          'GraspL',    'GraspR')
        self._init_motors()
        # LED设备
        self.eyeLed = robot.getDevice('EyeLed')
        self.headLed = robot.getDevice('HeadLed')
        # 传感器设备
        self.camera = robot.getDevice('Camera') # 摄像头
        self.accelerometer = robot.getDevice('Accelerometer')    # 加速度传感器
        self.gyro = robot.getDevice('Gyro')    # 陀螺仪传感器
        # GPS设备组
        self.left_gps1 = robot.getDevice('left_gps1')
        self.right_gps1 = robot.getDevice('right_gps1')
        self.left_gps2 = robot.getDevice('left_gps2')
        self.right_gps2 = robot.getDevice('right_gps2')
        self.foot_gps1 = robot.getDevice('foot_gps1')
        # 启用传感器
        self.enable_sensors()
        # 初始化触碰传感器(字典格式{传感器名称:传感器数值})
        self.touch_sensors = {}
        self._init_touch_sensors() 

    def enable_sensors(self):
        """启用所有传感器"""
        self.camera.enable(self.timestep)
        self.accelerometer.enable(self.timestep)
        self.gyro.enable(self.timestep)
        self.left_gps1.enable(self.timestep)
        self.right_gps1.enable(self.timestep)
        self.left_gps2.enable(self.timestep)
        self.right_gps2.enable(self.timestep)
        self.foot_gps1.enable(self.timestep)

    def _init_touch_sensors(self):
        """初始化触碰传感器"""
        # 定义所有触碰传感器(字典格式{传感器名称:传感器名称})
        self.sensors_config = {
            'grasp_L1'  : 'touch_grasp_L1',
            'grasp_L1_1': 'touch_grasp_L1_1',
            'grasp_L1_2': 'touch_grasp_L1_2',
            'grasp_R1'  : 'touch_grasp_R1',
            'grasp_R1_1': 'touch_grasp_R1_1',
            'grasp_R1_2': 'touch_grasp_R1_2',
            'foot_L1'   : 'touch_foot_L1',
            'foot_L2'   : 'touch_foot_L2',
            'foot_R1'   : 'touch_foot_R1',
            'foot_R2'   : 'touch_foot_R2',
            'arm_L1'    : 'touch_arm_L1',
            'arm_R1'    : 'touch_arm_R1',
            'leg_L1'    : 'touch_leg_L1',
            'leg_L2'    : 'touch_leg_L2',
            'leg_R1'    : 'touch_leg_R1',
            'leg_R2'    : 'touch_leg_R2'
        }
        
        # 初始化并启用每个触碰传感器
        for key, name in self.sensors_config.items():
            sensor = self.robot.getDevice(name)
            sensor.enable(self.timestep)
            self.touch_sensors[key] = sensor
            # print(f'self.touch_sensors_{key}=', self.touch_sensors[key].getValue())

    def _init_motors(self):
        """初始化所有电机和位置传感器"""
        for name in self.motorName:
            motor = self.robot.getDevice(name)
            # 电机不需要enable，创建后即可使用
            self.motors.append(motor)
            # 初始化对应的位置传感器
            sensor = self.robot.getDevice(name + 'S')
            sensor.enable(self.timestep)
            self.motors_sensors.append(sensor)

    def robot_reset(self):
        """重置机器人到初始姿态"""
        a = -np.random.random() * 0.4 + 0.7
        b = -np.random.random() * 0.4 + 1.0
        
        # 设置基础姿态
        self._set_initial_pose(a, b)
        # 设置电机速度
        self._set_motors_velocity()
        # 等待稳定
        for _ in range(24):
            self.robot.step(self.timestep)
    

    def _set_initial_pose(self, a, b):
        """设置机器人初始姿态
        
        参数：
            (a,b are defined in robot_reset())
            a: 肩部角度随机值
            b: 手臂角度随机值
        """
        pose_config = {
            'GraspL': 1, 'GraspR': 1,
            'Neck': 0.2, 'Head': 0,
            'PelvYL': 0.0, 'PelvL': 0.0,
            'LegUpperL': 0.5, 'LegLowerL': -0.2,
            'AnkleL': 0, 'FootL': 0.0,
            'PelvYR': 0.0, 'PelvR': 0.0,
            'LegUpperR': -0.5, 'LegLowerR': 0.2,
            'AnkleR': 0, 'FootR': 0.0,
            'ShoulderL': a, 'ArmUpperL': 0.67, 'ArmLowerL': -b,
            'ShoulderR': -a, 'ArmUpperR': -0.67, 'ArmLowerR': b
        }
        
        for name, position in pose_config.items():
            motor_idx = self.motorName.index(name)
            self.motors[motor_idx].setPosition(position)



    def _set_motors_velocity(self, velocity=1.0):
        """设置所有电机的速度
        
        参数：
            velocity: 目标速度值(原来文件中设置的速度都是1.0)
        """
        for motor in self.motors:
            motor.setVelocity(velocity)

    def get_gps_values(self):
        """获取所有GPS数据
        
        返回：
            tuple: 包含五个GPS数据
        """
        return (
            self.left_gps1.getValues(),
            self.right_gps1.getValues(),
            self.left_gps2.getValues(),
            self.right_gps2.getValues(),
            self.foot_gps1.getValues()
        )
    
    def get_touch_values(self):
        """获取所有触碰传感器的值
        
        返回：
            dict: 包含所有触碰传感器的状态的字典
                键: 传感器名称 (如 'grasp_L1', 'foot_L1' 等)
                值: 传感器读数
                    - 1.0: 传感器被触碰
                    - 0.0: 传感器未被触碰
        """
        return {key: sensor.getValue() for key, sensor in self.touch_sensors.items()}

    def get_touch_sensor_value(self, sensor_name):
        """获取指定触碰传感器的值
        参数:
            sensor_name: 传感器名称，如 'grasp_L1_2'
        返回:
            float: 传感器值，1.0表示被触碰，0.0表示未被触碰
        """
        if sensor_name in self.touch_sensors:
            return self.touch_sensors[sensor_name].getValue()
        return 0.0

    def lock_grasp(self):
        """锁定夹爪位置，确保夹爪保持闭合状态"""
        # 设置夹爪电机位置为闭合状态
        self.motors[20].setPosition(-0.5)
        self.motors[21].setPosition(-0.5)
      
        # 等待夹爪动作完成
        for _ in range(10):
            self.robot.step(self.timestep)

    
    def check_grasp_contact(self):
        """检查抓取器的接触状态（左右各有三个抓取传感器）
        
        返回：
            dict: 包含左右两侧抓取器接触状态的字典
                键: 'left' 或 'right'
                值: bool, 是否接触
        """
        left_grasp = any([
            self.touch_sensors['grasp_L1'].getValue(),
            self.touch_sensors['grasp_L1_1'].getValue(),
            self.touch_sensors['grasp_L1_2'].getValue()
        ])
        
        right_grasp = any([
            self.touch_sensors['grasp_R1'].getValue(),
            self.touch_sensors['grasp_R1_1'].getValue(),
            self.touch_sensors['grasp_R1_2'].getValue()
        ])
        
        return {'left': left_grasp, 'right': right_grasp}    
    #这里是从touch_peng中提取的
    def check_collision(self):
        """检查是否发生碰撞（通过检查手臂和腿部的触碰传感器）
        
        返回：
            bool: 是否发生碰撞
        """
        collision_sensors = [
            'arm_L1', 'arm_R1',
            'leg_L1', 'leg_L2',
            'leg_R1', 'leg_R2'
        ]
        
        return any(self.touch_sensors[sensor].getValue() for sensor in collision_sensors)    

    def get_camera_image(self, step):
        """获取相机图像
        
        参数：
            step: 当前步数
        返回：
            img: 原始图像数组
            img_tensor: 处理后的图像张量
        """
        # 根据当前步数生成图像文件名
        img_name = f"img_{step}.png"
        # 使用Webots的相机API保存图像，100表示图像质量(0-100)
        self.camera.saveImage(img_name, 100)
        
        # 确保目标目录存在
        os.makedirs(path_list['photo_path'], exist_ok=True)
        
        # 直接保存到目标路径，而不是先保存后移动
        try:
            # 如果文件已经存在于源路径，则尝试移动
            if os.path.exists(f"{path_list['photo_path_real']}/{img_name}"):
                shutil.move(f"{path_list['photo_path_real']}/{img_name}", f"{path_list['photo_path']}/{img_name}")
            # 如果文件不存在于源路径，则直接使用目标路径的文件
            elif not os.path.exists(f"{path_list['photo_path']}/{img_name}"):
                # 如果目标路径也没有文件，则复制一个空白图像作为替代
                blank_img = Image.new('L', (128, 128), 255)  # 创建空白图像
                blank_img.save(f"{path_list['photo_path']}/{img_name}")
        except Exception as e:
            print(f"处理图像文件时出错: {e}")
            
        # 使用PIL库打开保存的图像文件
        img = Image.open(f"{path_list['photo_path']}/{img_name}")   
        # 将图像调整为128x128像素大小
        img = img.resize((128, 128))
        # 将图像转换为灰度图像（单通道）
        img = img.convert('L')  
        # 将处理后的图像保存到指定路径，添加"1.png"后缀
        os.makedirs(path_list['photo_path'], exist_ok=True)  # 确保保存路径存在
        #img.save(os.path.join(path_list['photo_path'], img_name))
        img.save(os.path.join(path_list['photo_path'], img_name).replace("\\", "/"))

        # 将PIL图像转换为numpy数组，并归一化到0-1范围
        img_array = np.array(img) / 255.0
        # 将numpy数组转换为PyTorch张量：
        # - unsqueeze(0)添加batch维度
        # - float()转换为浮点类型
        img_tensor = torch.tensor(img_array).unsqueeze(0).float()   
        
        # 返回原始数组和处理后的张量
        return img_array, img_tensor

    def get_robot_state(self):
        """获取机器人关节状态
        
        返回：
            list: 包含所有关节位置的列表
        """
        # robot_state = []
        # print(f'motors_sensors: {self.motors_sensors}')
        # print(f'motors_sensors_len: {len(self.motors_sensors)}')
        # for sensor in self.motors_sensors[:-2]:
            # print(f'sensor: {sensor}')
            # print(f'sensor.getValue(): {sensor.getValue()}')
            # robot_state.append(sensor.getValue())
        robot_state = [sensor.getValue() for sensor in self.motors_sensors[:-2]]
        # print(f'webots_interfaces: robot_state: {robot_state}')
        # print(f'webots_interfaces: robot_state_len: {len(robot_state)}')
        return robot_state

    def check_acceleration_and_gyro(self):
        """检查加速度和陀螺仪数据是否在正常范围内
        
        返回：
            bool: 是否在正常范围内
        """
        acc = self.accelerometer.getValues()
        gyro = self.gyro.getValues()
        #limits 是参考RobotRun2.py和RobotRun1.py的
        acc_limits = {
            'low': [480, 430, 580],
            'high': [560, 530, 700]
        }
        gyro_limits = {
            'low': [500, 500, 500],
            'high': [520, 520, 520]
        }
        
        # 检查加速度
        for i in range(3):
            if not (acc_limits['low'][i] < acc[i] < acc_limits['high'][i]):
                return False
                
        # 检查陀螺仪
        for i in range(3):
            if not (gyro_limits['low'][i] < gyro[i] < gyro_limits['high'][i]):
                return False
                
        return True

    def check_joint_limits(self, positions):
        """检查关节位置是否在限制范围内
        
        参数：
            positions: list, 关节位置列表
        返回：
            bool: 是否所有关节都在限制范围内
        """
        #limits 依然是参考RobotRun2.py和RobotRun1.py的
        limits = [
            [-3.14, 3.14], [-3.14, 2.85], [-0.68, 2.3], 
            [-2.25, 0.77], [-1.65, 1.16], [-1.18, 1.63],
            [-2.42, 0.66], [-0.69, 2.5], [-1.01, 1.01], 
            [-1, 0.93], [-1.77, 0.45], [-0.5, 1.68],
            [-0.02, 2.25], [-2.25, 0.03], [-1.24, 1.38], 
            [-1.39, 1.22], [-0.68, 1.04], [-1.02, 0.6],
            [-1.81, 1.81], [-0.36, 0.94]
        ]
        
        for i in range(len(positions)):
            pos = positions[i]
            if not (limits[i][0] <= pos <= limits[i][1]):
                return False
        return True
    #===============================================以下是固定动作组函数===============================================
    def execute_timed_motion(self, motor_positions, duration, velocity=1.0):
        """执行定时动作
        参数：
            motor_positions: dict, 电机名称和目标位置的映射
            duration: int, 动作持续时间(ms)
            velocity: float, 电机速度{默认1.0}
        """
        timer = 0
        for name, position in motor_positions.items():
            motor = self.motors[self.motorName.index(name)]
            motor.setPosition(position)
            motor.setVelocity(velocity)
        
        while self.robot.step(32) != -1:
            timer += 32
            if timer >= duration:
                break

    def _set_left_leg_initpose(self):
        """设置左腿初始姿态"""
        self.execute_timed_motion({
            'LegUpperL': 1.65, 
            'LegLowerL': -2.2,
            'AnkleL': -0.85
        }, 1500, 1.5)  
        self.execute_timed_motion({
            'LegUpperL': 0.4, 
            'LegLowerL': -0.7,
            'AnkleL': -0.7
        }, 1500, 1.0)
        self.execute_timed_motion({
            'LegLowerL': -0.1,
            'AnkleL': -0.15
        }, 1500, 1.0)

    def tai_leg_L1(self):
        """抬起左腿第一阶段"""
        self.execute_timed_motion({
            'LegLowerL': -0.7,
            'AnkleL': -0.5
        }, 2000)

    def tai_leg_L2(self):
        """抬起左腿第二阶段"""
        self.execute_timed_motion({
            'LegUpperL': 1.65,
            'LegLowerL': -2.2,
            'AnkleL': -0.85
        }, 2000, 2)

    def tai_leg_L3(self):
        """抬起左腿第三阶段"""
        self.execute_timed_motion({
            'LegLowerL': -1.8,
            'AnkleL': -0.45
        }, 1000)

    def tai_leg_L4(self):
        """抬起左腿第四阶段"""
        self.execute_timed_motion({
            'LegUpperL': 1.4,
            'LegLowerL': -1.55,
            'AnkleL': -0.45
        }, 1000)

    def n_tai_leg_L1(self):
        """新抬起左腿第一阶段"""
        self.execute_timed_motion({
            'LegLowerL': -1.45,
            'AnkleL': -0.45
        }, 2000)

    def n_tai_leg_L2(self):
        """新抬起左腿第二阶段"""
        self.execute_timed_motion({
            'LegUpperL': 1.65,
            'LegLowerL': -2.2,
            'AnkleL': -0.85
        }, 2000, 2)

    def n_tai_leg_L3(self):
        """新抬起左腿第三阶段"""
        self.execute_timed_motion({
            'LegLowerL': -1.8,
            'AnkleL': -0.8,
            'AnkleR': -0.8,
            'ArmLowerL': 0.2,
            'ShoulderL': -0.4,
            'ArmLowerR': -0.2,
            'ShoulderR': 0.4
        }, 2000)

    def n_tai_leg_L4(self):
        """新抬起左腿第四阶段"""
        self.execute_timed_motion({
            'LegLowerL': -1.6,
            'AnkleL': -0.45
        }, 1000)

    def tai_leg_R1(self):
        """抬起右腿第一阶段"""
        self.execute_timed_motion({
            'LegLowerR': 0.7,
            'AnkleR': 0.35
        }, 2000)

    def tai_leg_R2(self):
        """抬起右腿第二阶段"""
        self.execute_timed_motion({
            'LegUpperR': -1.65,
            'LegLowerR': 2.2,
            'AnkleR': 0.85
        }, 2000, 2)

    def tai_leg_R3(self):
        """抬起右腿第三阶段"""
        self.execute_timed_motion({
            'LegLowerR': 1.8,
            'AnkleR': 0.45
        }, 1000)

    def tai_leg_R4(self):
        """抬起右腿第四阶段"""
        self.execute_timed_motion({
            'LegUpperR': -1.4,
            'LegLowerR': 1.55,
            'AnkleR': 0.45
        }, 1000)

    def qi_li(self):
        """起立动作"""
        self.execute_timed_motion({
            'LegLowerL': -1.1,
            'LegUpperL': 1.45,
            'LegLowerR': 1.1,
            'LegUpperR': -1.45,
            'ArmLowerL': -0.1,
            'ArmLowerR': 0.1,
            'AnkleL': -0.25,
            'AnkleR': 0.25,
            'ShoulderL': -0.05,
            'ShoulderR': 0.05,
            'GraspL': 0,
            'GraspR': 0
        }, 2000)

    def song_L(self):
        """松开左手"""
        self.execute_timed_motion({
            'GraspL': 1
        }, 1000, 2)

    def song_R(self):
        """松开右手"""
        self.execute_timed_motion({
            'GraspR': 1
        }, 1000, 2)

    # ... 后续可以自己定义

class Environment:
    """基础环境类
    功能：管理仿真环境的基本操作
    属性：
        robot: Webots机器人实例
        timestep: 仿真步长
    """

    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.darwin = Darwin(self.robot)  # Darwin实例
        self.state = None
        self.done = False
        self.isSuccess = False

    def myStep(self):
        """推进仿真一个时间步长"""
        return self.robot.step(self.timestep)

    def wait(self, ms):
        """等待指定时间
        参数：
            ms: 等待毫秒数
        """
        startTime = self.robot.getTime()
        s = ms / 1000
        while s + startTime >= self.robot.getTime():
            self.myStep()

    def reset(self):
        """
        重置状态函数
        """  
        self.darwin.robot_reset()   # 重置机器人状态
        # 以记事本的形式传参，作为标识符
        with open(path_list['resetFlag'], 'r+') as file:
            file.write('0')
        with open(path_list['resetFlag1'], 'r+') as file:
            file.write('0')
        self.done = False
        return self.state

    
    def step(self, state, action_shouder, action_arm, steps, catch_flag, gps1, gps2, gps3, gps4, img_name):
        """执行一步动作
        参数:
            state: 当前状态
            action_shouder: 肩膀舵机动作
            action_arm: 手臂舵机动作
            steps: 步数
            catch_flag: 抓取器状态
            gps1-4: GPS位置信息
            name: 动作名称
        返回:
            tuple: (next_state, reward, done, good, goal, count)
        """
        from python_scripts.DQN.RobotRun1 import RobotRun
        #from python_scripts.PPO.RobotRun1 import RobotRun
        #from python_scripts.SAC.RobotRun1 import RobotRun
        return RobotRun(self.robot, state, action_shouder, action_arm, steps, catch_flag, gps1, gps2, gps3, gps4, img_name).run()
    
    def step2(self, state, action_leg_upper, action_leg_lower, action_ankle, steps, zhua, gps0, gps1, gps2, gps3, gps4):
        from python_scripts.PPO.RobotRun2 import RobotRun2
        return RobotRun2(self.robot, state, action_leg_upper, action_leg_lower, action_ankle, steps, zhua, gps0, gps1, gps2, gps3, gps4).run()
    
    def get_robot_state(self):
        """获取机器人的关节状态，即舵机角度"""
        return self.darwin.get_robot_state()
    
    def get_img(self, step, imgs):
        return self.darwin.get_camera_image(step=step)
    
    def print_gps(self):
        return self.darwin.get_gps_values()

    def get_touch_sensor_value(self, sensor_name):
        return self.darwin.get_touch_sensor_value(sensor_name)

    def lock_grasp(self):
        """锁定夹爪位置，确保夹爪保持闭合状态"""
        return self.darwin.lock_grasp()
        
