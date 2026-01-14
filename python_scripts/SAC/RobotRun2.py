"""RobotRun2 控制器 - 用于达尔文OP2机器人爬梯子的第二阶段（抬腿阶段）."""
import math

#import gym
import time
import numpy as np
import os
import cv2

import argparse
import platform
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 将ROOT添加到PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 相对路径
import torch

class RobotRun2:
    """
    控制机器人按照action行动的类，用于爬梯子阶段
    
    该类负责执行机器人的动作，检测传感器状态，计算奖励，并判断是否完成当前回合
    """
    def __init__(self, robot, state, action, step, zhua, gps0, gps1, gps2, gps3, gps4):
        """
        初始化RobotRun2类
        
        参数：
            robot：Webots机器人对象
            state：当前机器人状态（关节角度）
            action：要执行的动作编号
            step：当前步数
            zhua：抓取器状态
            gps0-gps4：GPS传感器数据
        """
        self.robot = robot
        self.timestep = 32  # 仿真时间步长
        self.step = step  # 当前步数
        self.goal = [0.058, 0.0225]  # 目标位置
        self.biao_zhun = -32.64359902756043  # 标准参考值
        self.robot_state = state  # 当前机器人状态
        # 存储GPS数据
        self.gps0 = gps0
        self.gps1 = gps1
        self.gps2 = gps2
        self.gps3 = gps3
        self.gps4 = gps4
        self.action = action  # 当前动作
        
        # 根据动作编号设置不同的关节运动参数
        if action == 0:
            # 动作0：腿部上部向后移动，脚踝向后移动
            self.LegUpper = -0.05
            self.LegLower = 0
            self.Ankle = -0.05
        elif action == 1:
            # 动作1：腿部上部向前移动，脚踝向前移动
            self.LegUpper = 0.05
            self.LegLower = 0
            self.Ankle = 0.05
        elif action == 2:
            # 动作2：腿部下部向后移动，脚踝向后移动
            self.LegUpper = 0
            self.LegLower = -0.05
            self.Ankle = -0.05
        else:
            # 动作3：腿部下部向前移动，脚踝向前移动
            self.LegUpper = 0
            self.LegLower = 0.05
            self.Ankle = 0.05


        self.if_jia = zhua  # 抓取器状态
        self.jie1_Success = False  # 第一阶段是否成功
        self.motors = []  # 电机列表
        self.motors_sensors = []  # 电机传感器列表
        
        # 电机名称列表
        self.motorName = ('ShoulderR', 'ShoulderL', 'ArmUpperR', 'ArmUpperL',
                     'ArmLowerR', 'ArmLowerL', 'PelvYR', 'PelvYL', 'PelvR',
                     'PelvL', 'LegUpperR', 'LegUpperL', 'LegLowerR', 'LegLowerL',
                     'AnkleR', 'AnkleL', 'FootR', 'FootL', 'Neck', 'Head', 'GraspL', 'GraspR')
        
        # 初始化电机和传感器
        for i in range(len(self.motorName)):
            self.motors.append(robot.getDevice(self.motorName[i]))  # 获取电机
            sensorName = self.motorName[i]
            sensorName = sensorName + 'S'  # 传感器名称为电机名称+'S'
            self.motors_sensors.append(self.robot.getDevice(sensorName))  # 获取传感器
            self.motors_sensors[i].enable(self.timestep)  # 启用传感器
            
        # 初始化加速度计和陀螺仪
        self.accelerometer = robot.getDevice('Accelerometer')
        self.gyro = robot.getDevice('Gyro')
        
        # 初始化触摸传感器
        self.touch1 = self.robot.getDevice('touch_foot_L1')
        self.touch1_1 = self.robot.getDevice('touch_foot_L2')
        self.touch1_2 = self.robot.getDevice('touch_foot_L3')
        self.touch5 = self.robot.getDevice('touch_foot_L1')
        self.touch6 = self.robot.getDevice('touch_foot_L2')
        self.touch7 = self.robot.getDevice('touch_foot_R1')
        self.touch8 = self.robot.getDevice('touch_foot_R2')
        self.touch11 = self.robot.getDevice('touch_arm_L1')
        self.touch12 = self.robot.getDevice('touch_arm_R1')
        self.touch13 = self.robot.getDevice('touch_leg_L1')
        self.touch14 = self.robot.getDevice('touch_leg_L2')
        self.touch15 = self.robot.getDevice('touch_leg_R1')
        self.touch16 = self.robot.getDevice('touch_leg_R2')
        
        # 启用触摸传感器
        self.touch1.enable(32)
        self.touch1_1.enable(32)
        self.touch1_2.enable(32)
        self.touch5.enable(32)
        self.touch6.enable(32)
        self.touch7.enable(32)
        self.touch8.enable(32)
        self.touch11.enable(32)
        self.touch12.enable(32)
        self.touch13.enable(32)
        self.touch14.enable(32)
        self.touch15.enable(32)
        self.touch16.enable(32)
        
        # 分组触摸传感器
        self.touch = [self.touch1, self.touch1_1, self.touch1_2]  # 脚部触摸传感器
        self.touch_peng = [self.touch11, self.touch12, self.touch13, self.touch14, self.touch15, self.touch16]  # 碰撞检测触摸传感器
        
        # 初始化状态
        self.future_state = [i for i in self.robot_state]  # 复制当前状态作为未来状态
        
        # 计算下一个状态的关节角度
        self.next = [self.robot_state[11] + self.LegUpper, self.robot_state[13] + self.LegLower, self.robot_state[15] + self.Ankle]

        # 更新未来状态中的关节角度
        self.future_state[11] = self.next[0]  # 左腿上部
        self.future_state[13] = self.next[1]  # 左腿下部
        self.future_state[15] = self.next[2]  # 左脚踝
        print("变化动作")
        print(self.next)

        # 关节角度限制，每个关节的最小和最大角度
        self.limit = [[-3.14, 3.14], [-3.14, 2.85], [-0.68, 2.3], [-2.25, 0.77], [-1.65, 1.16], [-1.18, 1.63],
                      [-2.42, 0.66], [-0.69, 2.5], [-1.01, 1.01], [-1, 0.93], [-1.77, 0.45], [-0.5, 1.68],
                      [-0.02, 2.25], [-2.25, 0.03], [-1.24, 1.38], [-1.39, 1.22], [-0.68, 1.04], [-1.02, 0.6],
                      [-1.81, 1.81], [-0.36, 0.94]]
        
        # 初始化当前状态和下一个状态
        self.now_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.next_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # 初始化触摸传感器值
        self.touch_value = [0.0, 0.0, 0.0]
        
        # 加速度计和陀螺仪的正常范围
        self.acc_low = [480, 450, 580]  # 加速度计下限
        self.acc_high = [560, 530, 700]  # 加速度计上限
        self.gyro_low = [500, 500, 500]  # 陀螺仪下限
        self.gyro_high = [520, 520, 520]  # 陀螺仪上限

    def run(self):
        """
        执行机器人动作并返回结果
        
        返回：
            tuple：（next_state，reward，done，good，goal，count）
                next_state：下一个状态
                reward：奖励值
                done：是否完成回合
                good：是否为有效动作
                goal：是否达到目标
                count：计数器状态
        """
        self.robot.step(32)  # 执行一个仿真步
        
        # 获取传感器数据
        acc = self.accelerometer.getValues()  # 加速度计数据
        gyro = self.gyro.getValues()  # 陀螺仪数据
        
        # 计算与目标的距离
        x1 = self.goal[0] - self.gps0[1]  # x方向距离
        y1 = self.goal[1] - self.gps0[2]  # y方向距离
        
        # 初始化返回值
        goal = 0  # 是否达到目标
        reward = 0  # 奖励值
        reward1 = math.sqrt((x1 * x1) + (y1 * y1))  # 计算欧几里得距离作为基础奖励

        count = 1  # 计数器，用于控制奖励计算
        
        # 检查关节角度是否在限制范围内
        for i in range(len(self.future_state)):
            if self.limit[1][0] <= self.future_state[i] <= self.limit[1][1]:
                continue
            else:
                # 如果超出限制，返回零奖励并结束回合
                reward = 0
                count = 0
                done = 1
                good = 1
                return self.next_state, reward, done, good, goal, count
                
        # 执行多个仿真步
        self.robot.step(32)
        self.robot.step(32)
        self.robot.step(32)
        
        # 检查环境状态
        if_exist = 1
        if if_exist == 1:
            pass
        else:
            # 如果环境状态异常，返回零奖励并结束回合
            reward = 0
            count = 0
            done = 1
            good = 0
            return self.next_state, reward, done, good, goal, count
            
        # 检查加速度计和陀螺仪数据是否在正常范围内
        for i in range(3):
            if self.acc_low[i] < acc[i] < self.acc_high[i] and self.gyro_low[i] < gyro[i] < self.gyro_high[i]:
                continue
            else:
                # 如果传感器数据异常，返回零奖励并结束回合
                reward = 0
                count = 0
                done = 1
                good = 0
                return self.next_state, reward, done, good, goal, count

        # 应用已有的关节角度限制
        leg_upper_pos = max(self.limit[11][0], min(self.limit[11][1], self.next[0]))
        leg_lower_pos = max(self.limit[13][0], min(self.limit[13][1], self.next[1]))
        ankle_pos = max(self.limit[15][0], min(self.limit[15][1], self.next[2]))
        
        # 设置限制后的关节角度
        self.motors[11].setPosition(leg_upper_pos)  # 设置左腿上部位置
        self.motors[13].setPosition(leg_lower_pos)  # 设置左腿下部位置
        self.motors[15].setPosition(ankle_pos)  # 设置左脚踝位置
        
        # 等待动作执行完成
        self.robot.step(32)
        self.robot.step(32)
        self.robot.step(32)
        self.robot.step(32)
        self.robot.step(32)
        self.robot.step(32)
        self.robot.step(32)
        self.robot.step(32)
        
        # 初始化返回值
        done = 0
        reward = reward1  # 使用距离作为奖励
        good = 1
        
        # 检查是否发生碰撞
        for m in range(6):
            if self.touch_peng[m].getValue() == 1.0:
                # 如果发生碰撞，返回零奖励并结束回合
                done = 1
                reward = 0
                good = 1
                count = 0
                return self.next_state, reward, done, good, goal, count



        # 检查脚部触摸传感器
        if self.touch1.getValue() == 1.0 or self.touch1_1.getValue() == 1.0 or self.touch1_2.getValue() == 1.0:
            # 如果脚部接触到物体（梯子），打印传感器值
            print("___________")
            print(self.touch1.getValue())
            print(self.touch1_1.getValue())
            print(self.touch1_2.getValue())
            
            # 等待一段时间
            timer = 0
            while self.robot.step(32) != -1:
                timer += 32
                if timer >= 2000:
                    break
                    
            # 获取触摸传感器值
            for j in range(len(self.touch)):
                self.touch_value[j] = self.touch[j].getValue()
                
            # 根据距离和触摸状态决定奖励和回合结束条件
            if reward1 >= 0.1:
                # 如果距离较远，给予较小奖励
                reward = 0
                goal = 0
                count = 1
                done = 1
                good = 1
            elif self.touch_value[0] == 1 or self.touch_value[1] == 1:
                # 如果脚部成功接触到梯子，给予大奖励
                reward = 100
                goal = 1
                count = 0
                done = 1
                good = 1
                return self.next_state, reward, done, good, goal, count
            else:
                # 其他情况，给予中等奖励
                reward = 20
                goal = 0
                count = 0
                done = 1
                good = 1

        # 获取当前关节角度并检查是否达到目标位置
        for i in range(20):
            self.next_state[i] = self.motors_sensors[i].getValue()  # 获取当前关节角度
            self.cha_zhi = self.next_state[i] - self.future_state[i]  # 计算与目标角度的差值
            if -0.01 < self.cha_zhi < 0.01:
                # 如果差值很小，认为已达到目标位置
                continue
            else:
                # 如果差值较大，认为未达到目标位置
                count = 1
                reward = 0
                done = 1
                goal = 0
                good = 1
                break
                
        # 返回结果
        return self.next_state, reward, done, good, goal, count