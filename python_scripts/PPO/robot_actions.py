"""
机器人特殊动作库
该文件包含了机器人特殊动作的实现，如抬腿、抬臂、调整高度等
"""

import time
import math
import numpy as np
from python_scripts.Project_config import device

class RobotActions:
    """
    机器人特殊动作类
    用于执行机器人的复杂动作，如抬腿、调整姿势等
    """
    
    def __init__(self, robot):
        """
        初始化机器人动作控制
        Args:
            robot: 机器人控制实例
        """
        self.robot = robot
        self.timestep = int(robot.getBasicTimeStep()) if hasattr(robot, 'getBasicTimeStep') else 32
        self.motors = []
        self.motors_sensors = []
        
        # 舵机名称列表
        self.motorName = ('ShoulderR', 'ShoulderL', 'ArmUpperR', 'ArmUpperL',
                          'ArmLowerR', 'ArmLowerL', 'PelvYR', 'PelvYL', 'PelvR',
                          'PelvL', 'LegUpperR', 'LegUpperL', 'LegLowerR', 'LegLowerL',
                          'AnkleR', 'AnkleL', 'FootR', 'FootL', 'Neck', 'Head', 'GraspL',
                          'GraspR')
        
        # 初始化舵机
        self._init_motors()
    
    def _init_motors(self):
        """初始化舵机和传感器"""
        for i in range(len(self.motorName)):
            try:
                motor = self.robot.getDevice(self.motorName[i])
                self.motors.append(motor)
                
                # 初始化传感器
                sensorName = self.motorName[i] + 'S'
                sensor = self.robot.getDevice(sensorName)
                if sensor:
                    sensor.enable(self.timestep)
                    self.motors_sensors.append(sensor)
            except:
                # 如果获取设备失败，添加None作为占位符
                self.motors.append(None)
                self.motors_sensors.append(None)
    
    def myStep(self):
        """执行单步仿真"""
        if hasattr(self.robot, 'step'):
            self.robot.step(self.timestep)
    
    def wait(self, ms):
        """等待一定时间，允许机器人动作完成"""
        if hasattr(self.robot, 'getTime'):
            startTime = self.robot.getTime()
            s = ms / 1000
            while s + startTime >= self.robot.getTime():
                self.myStep()
        else:
            # 如果没有getTime方法，使用循环等待
            steps = int(ms / self.timestep)
            for _ in range(steps):
                self.myStep()
    
    def tai_leg_L1(self):
        """左腿抬起第一阶段"""
        if len(self.motors) < 18:
            return
            
        # 设置左腿舵机角度
        self.motors[11].setPosition(0.4)  # LegUpperL
        self.motors[13].setPosition(-0.4)  # LegLowerL
        self.motors[15].setPosition(-0.2)  # AnkleL
        self.motors[17].setPosition(0.2)  # FootL
        
        # 等待动作完成
        self.wait(300)
    
    def tai_leg_L2(self):
        """左腿抬起第二阶段"""
        if len(self.motors) < 18:
            return
            
        # 设置左腿舵机角度
        self.motors[11].setPosition(0.6)  # LegUpperL
        self.motors[13].setPosition(-0.6)  # LegLowerL
        self.motors[15].setPosition(-0.3)  # AnkleL
        self.motors[17].setPosition(0.3)  # FootL
        
        # 等待动作完成
        self.wait(300)
    
    def tai_leg_L3(self):
        """左腿抬起第三阶段"""
        if len(self.motors) < 18:
            return
            
        # 设置左腿舵机角度
        self.motors[11].setPosition(0.8)  # LegUpperL
        self.motors[13].setPosition(-0.8)  # LegLowerL
        self.motors[15].setPosition(-0.4)  # AnkleL
        self.motors[17].setPosition(0.4)  # FootL
        
        # 等待动作完成
        self.wait(300)
    
    def tai_leg_L4(self):
        """左腿抬起第四阶段"""
        if len(self.motors) < 18:
            return
            
        # 设置左腿舵机角度
        self.motors[11].setPosition(1.0)  # LegUpperL
        self.motors[13].setPosition(-1.0)  # LegLowerL
        self.motors[15].setPosition(-0.5)  # AnkleL
        self.motors[17].setPosition(0.5)  # FootL
        
        # 等待动作完成
        self.wait(300)
    
    def n_tai_leg_L1(self):
        """左腿放下第一阶段"""
        if len(self.motors) < 18:
            return
            
        # 设置左腿舵机角度
        self.motors[11].setPosition(0.8)  # LegUpperL
        self.motors[13].setPosition(-0.8)  # LegLowerL
        self.motors[15].setPosition(-0.4)  # AnkleL
        self.motors[17].setPosition(0.4)  # FootL
        
        # 等待动作完成
        self.wait(300)
    
    def n_tai_leg_L2(self):
        """左腿放下第二阶段"""
        if len(self.motors) < 18:
            return
            
        # 设置左腿舵机角度
        self.motors[11].setPosition(0.6)  # LegUpperL
        self.motors[13].setPosition(-0.6)  # LegLowerL
        self.motors[15].setPosition(-0.3)  # AnkleL
        self.motors[17].setPosition(0.3)  # FootL
        
        # 等待动作完成
        self.wait(300)
    
    def n_tai_leg_L3(self):
        """左腿放下第三阶段"""
        if len(self.motors) < 18:
            return
            
        # 设置左腿舵机角度
        self.motors[11].setPosition(0.4)  # LegUpperL
        self.motors[13].setPosition(-0.4)  # LegLowerL
        self.motors[15].setPosition(-0.2)  # AnkleL
        self.motors[17].setPosition(0.2)  # FootL
        
        # 等待动作完成
        self.wait(300)
    
    def n_tai_leg_L4(self):
        """左腿放下第四阶段"""
        if len(self.motors) < 18:
            return
            
        # 设置左腿舵机角度
        self.motors[11].setPosition(0.5)  # LegUpperL
        self.motors[13].setPosition(-0.2)  # LegLowerL
        self.motors[15].setPosition(0.0)  # AnkleL
        self.motors[17].setPosition(0.0)  # FootL
        
        # 等待动作完成
        self.wait(300)
    
    def song_L(self):
        """松开左夹爪"""
        if len(self.motors) < 21:
            return
            
        self.motors[20].setPosition(1.0)  # GraspL
        self.wait(500)
    
    def song_R(self):
        """松开右夹爪"""
        if len(self.motors) < 22:
            return
            
        self.motors[21].setPosition(1.0)  # GraspR
        self.wait(500)
    
    def tai_leg_R1(self):
        """右腿抬起第一阶段"""
        if len(self.motors) < 18:
            return
            
        # 设置右腿舵机角度
        self.motors[10].setPosition(-0.4)  # LegUpperR
        self.motors[12].setPosition(0.4)  # LegLowerR
        self.motors[14].setPosition(0.2)  # AnkleR
        self.motors[16].setPosition(-0.2)  # FootR
        
        # 等待动作完成
        self.wait(300)
    
    def tai_leg_R2(self):
        """右腿抬起第二阶段"""
        if len(self.motors) < 18:
            return
            
        # 设置右腿舵机角度
        self.motors[10].setPosition(-0.6)  # LegUpperR
        self.motors[12].setPosition(0.6)  # LegLowerR
        self.motors[14].setPosition(0.3)  # AnkleR
        self.motors[16].setPosition(-0.3)  # FootR
        
        # 等待动作完成
        self.wait(300)
    
    def tai_leg_R3(self):
        """右腿抬起第三阶段"""
        if len(self.motors) < 18:
            return
            
        # 设置右腿舵机角度
        self.motors[10].setPosition(-0.8)  # LegUpperR
        self.motors[12].setPosition(0.8)  # LegLowerR
        self.motors[14].setPosition(0.4)  # AnkleR
        self.motors[16].setPosition(-0.4)  # FootR
        
        # 等待动作完成
        self.wait(300)
    
    def tai_leg_R4(self):
        """右腿抬起第四阶段"""
        if len(self.motors) < 18:
            return
            
        # 设置右腿舵机角度
        self.motors[10].setPosition(-1.0)  # LegUpperR
        self.motors[12].setPosition(1.0)  # LegLowerR
        self.motors[14].setPosition(0.5)  # AnkleR
        self.motors[16].setPosition(-0.5)  # FootR
        
        # 等待动作完成
        self.wait(300)
    
    def n_tai_leg_R1(self):
        """右腿放下第一阶段"""
        if len(self.motors) < 18:
            return
            
        # 设置右腿舵机角度
        self.motors[10].setPosition(-0.8)  # LegUpperR
        self.motors[12].setPosition(0.8)  # LegLowerR
        self.motors[14].setPosition(0.4)  # AnkleR
        self.motors[16].setPosition(-0.4)  # FootR
        
        # 等待动作完成
        self.wait(300)
    
    def n_tai_leg_R2(self):
        """右腿放下第二阶段"""
        if len(self.motors) < 18:
            return
            
        # 设置右腿舵机角度
        self.motors[10].setPosition(-0.6)  # LegUpperR
        self.motors[12].setPosition(0.6)  # LegLowerR
        self.motors[14].setPosition(0.3)  # AnkleR
        self.motors[16].setPosition(-0.3)  # FootR
        
        # 等待动作完成
        self.wait(300)
    
    def n_tai_leg_R3(self):
        """右腿放下第三阶段"""
        if len(self.motors) < 18:
            return
            
        # 设置右腿舵机角度
        self.motors[10].setPosition(-0.4)  # LegUpperR
        self.motors[12].setPosition(0.4)  # LegLowerR
        self.motors[14].setPosition(0.2)  # AnkleR
        self.motors[16].setPosition(-0.2)  # FootR
        
        # 等待动作完成
        self.wait(300)
    
    def n_tai_leg_R4(self):
        """右腿放下第四阶段"""
        if len(self.motors) < 18:
            return
            
        # 设置右腿舵机角度
        self.motors[10].setPosition(-0.5)  # LegUpperR
        self.motors[12].setPosition(0.2)  # LegLowerR
        self.motors[14].setPosition(0.0)  # AnkleR
        self.motors[16].setPosition(0.0)  # FootR
        
        # 等待动作完成
        self.wait(300)
    
    def tai_arm_L1(self):
        """左手臂抬起"""
        if len(self.motors) < 6:
            return
            
        # 设置左手臂舵机角度
        self.motors[1].setPosition(1.0)  # ShoulderL
        self.motors[3].setPosition(0.4)  # ArmUpperL
        self.motors[5].setPosition(-0.3)  # ArmLowerL
        
        # 等待动作完成
        self.wait(300)
    
    def tai_arm_R1(self):
        """右手臂抬起"""
        if len(self.motors) < 6:
            return
            
        # 设置右手臂舵机角度
        self.motors[0].setPosition(-1.0)  # ShoulderR
        self.motors[2].setPosition(-0.4)  # ArmUpperR
        self.motors[4].setPosition(0.3)  # ArmLowerR
        
        # 等待动作完成
        self.wait(300)
    
    def tiao_zheng(self):
        """调整机器人姿势"""
        if len(self.motors) < 18:
            return
            
        # 调整手臂姿势
        self.motors[1].setPosition(0.7)  # ShoulderL
        self.motors[0].setPosition(-0.7)  # ShoulderR
        self.motors[3].setPosition(0.67)  # ArmUpperL
        self.motors[2].setPosition(-0.67)  # ArmUpperR
        self.motors[5].setPosition(-1.1)  # ArmLowerL
        self.motors[4].setPosition(1.1)  # ArmLowerR
        
        # 等待动作完成
        self.wait(500)
        
        # 调整腿部姿势
        self.motors[11].setPosition(0.5)  # LegUpperL
        self.motors[10].setPosition(-0.5)  # LegUpperR
        self.motors[13].setPosition(-0.2)  # LegLowerL
        self.motors[12].setPosition(0.2)  # LegLowerR
        
        # 等待动作完成
        self.wait(500)
    
    def sheng_gao1(self):
        """升高机器人身体第一阶段"""
        if len(self.motors) < 18:
            return
            
        # 设置手臂舵机角度
        self.motors[1].setPosition(0.8)  # ShoulderL
        self.motors[0].setPosition(-0.8)  # ShoulderR
        self.motors[3].setPosition(0.7)  # ArmUpperL
        self.motors[2].setPosition(-0.7)  # ArmUpperR
        self.motors[5].setPosition(-1.2)  # ArmLowerL
        self.motors[4].setPosition(1.2)  # ArmLowerR
        
        # 等待动作完成
        self.wait(300)
        
        # 调整腿部
        self.motors[11].setPosition(0.5)  # LegUpperL
        self.motors[10].setPosition(-0.5)  # LegUpperR
        self.motors[13].setPosition(-0.4)  # LegLowerL
        self.motors[12].setPosition(0.4)  # LegLowerR
        
        # 等待动作完成
        self.wait(200)
    
    def la_jin(self):
        """拉近机器人与目标距离"""
        if len(self.motors) < 6:
            return
            
        # 设置手臂舵机角度
        self.motors[1].setPosition(0.9)  # ShoulderL
        self.motors[0].setPosition(-0.9)  # ShoulderR
        self.motors[3].setPosition(0.9)  # ArmUpperL
        self.motors[2].setPosition(-0.9)  # ArmUpperR
        self.motors[5].setPosition(-1.3)  # ArmLowerL
        self.motors[4].setPosition(1.3)  # ArmLowerR
        
        # 等待动作完成
        self.wait(500)
    
    def tiao_zheng1(self):
        """调整机器人最终姿势"""
        if len(self.motors) < 18:
            return
            
        # 调整机器人姿势
        self.motors[1].setPosition(0.9)  # ShoulderL
        self.motors[0].setPosition(-0.9)  # ShoulderR
        self.motors[3].setPosition(0.9)  # ArmUpperL
        self.motors[2].setPosition(-0.9)  # ArmUpperR
        self.motors[5].setPosition(-1.4)  # ArmLowerL
        self.motors[4].setPosition(1.4)  # ArmLowerR
        
        self.motors[11].setPosition(0.6)  # LegUpperL
        self.motors[10].setPosition(-0.6)  # LegUpperR
        self.motors[13].setPosition(-0.6)  # LegLowerL
        self.motors[12].setPosition(0.6)  # LegLowerR
        
        # 等待动作完成
        self.wait(500) 