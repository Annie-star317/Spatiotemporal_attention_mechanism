from controller import Supervisor, Node
import numpy as np
import operator

DARWININIT = [1.93, 0.235, 0]
DARWINROTATION = [0, 1, 0, 0]
cha = [0, 0.07, 0.07]
DAR = [-0.0176538, 0.332399, -0.00606099]

class SupervisorRobot:
    def __init__(self):
        self.robot = Supervisor()
        self.timestep = 32
        self.Ⅰ = self.robot.getFromDef('Ⅰ')   # 获取世界坐标系中物体的对象
        self.Ⅰ_trans = self.Ⅰ.getField('translation')   # 获取世界坐标系中物体的三维位置对象
        self.Ⅰ_rotation = self.Ⅰ.getField('rotation')   # 获取世界坐标系中物体的三维方向对象
        self.Ⅱ = self.robot.getFromDef('Ⅱ')
        self.Ⅱ_trans = self.Ⅱ.getField('translation')
        self.Ⅱ_rotation = self.Ⅱ.getField('rotation')
        self.Ⅲ = self.robot.getFromDef('Ⅲ')
        self.Ⅲ_trans = self.Ⅲ.getField('translation')
        self.Ⅲ_rotation = self.Ⅲ.getField('rotation')
        self.Ⅳ = self.robot.getFromDef('Ⅳ')
        self.Ⅳ_trans = self.Ⅳ.getField('translation')
        self.Ⅳ_rotation = self.Ⅳ.getField('rotation')
        self.Ⅴ = self.robot.getFromDef('Ⅴ')
        self.Ⅴ_trans = self.Ⅴ.getField('translation')
        self.Ⅴ_rotation = self.Ⅴ.getField('rotation')
        self.Ⅵ = self.robot.getFromDef('Ⅵ')
        self.Ⅵ_trans = self.Ⅵ.getField('translation')
        self.Ⅵ_rotation = self.Ⅵ.getField('rotation')
        self.Ⅶ = self.robot.getFromDef('Ⅶ')
        self.Ⅶ_trans = self.Ⅶ.getField('translation')
        self.Ⅶ_rotation = self.Ⅶ.getField('rotation')
        self.Ⅷ = self.robot.getFromDef('Ⅷ')
        self.Ⅷ_trans = self.Ⅷ.getField('translation')
        self.Ⅷ_rotation = self.Ⅷ.getField('rotation')
        self.Ⅸ = self.robot.getFromDef('Ⅸ')
        self.Ⅸ_trans = self.Ⅸ.getField('translation')
        self.Ⅸ_rotation = self.Ⅸ.getField('rotation')
        self.Ⅹ = self.robot.getFromDef('Ⅹ')
        self.Ⅹ_trans = self.Ⅹ.getField('translation')
        self.Ⅹ_rotation = self.Ⅹ.getField('rotation')
        self.robotis_op2 = self.robot.getFromDef('robotis_op2')
        self.robotis_op2_trans = self.robotis_op2.getField('translation')
        self.robotis_op2_rotation = self.robotis_op2.getField('rotation')


    def wait(self, ms):
        startTime = self.robot.getTime()
        s = ms/1000
        while s+startTime >= self.robot.getTime():
            self.robot.step(self.timestep)

    def reset(self):
        print("++++reset++++")
        a = np.random.uniform(0.015, 0.02)
        DAR = [-0.0176538, 0.332399, -0.00606099]   # 机器人三维坐标的默认值
        sui_ji = [np.random.random() * 0.15 - 0.075, 0, np.random.random() * -0.02]   # 随机化机器人在三维世界中的初始位置坐标
        for i in range(0, 3):
            DAR[i] = DAR[i] + sui_ji[i]
        self.robotis_op2_trans.setSFVec3f(DAR)   # 调整机器人在仿真世界中的位置
        self.wait(100)
        self.robotis_op2_rotation.setSFRotation([0.999989, 0.000865437, 0.00465503, 0.296348])   # 调整机器人在仿真世界中的朝向
        self.wait(100)
        self.robotis_op2_trans.setSFVec3f(DAR)   # 调整机器人在仿真世界中的位置
        self.wait(100)
        
    def resetsimulation(self):
        self.robot.step(self.timestep)
        isremove = True
        while True:
            with open('D:\\project_Spatiotemporal_attention_mechanism\\python_scripts\\resetFlag.txt', 'r') as file:
                flag = file.read()
                if flag == '0':
                    with open('D:\\project_Spatiotemporal_attention_mechanism\\python_scripts\\resetFlag.txt', 'r+') as file:
                        file.write('1')
                    self.robot.simulationResetPhysics()
                    self.reset()
            self.robot.step(self.timestep)
    
supervisorRobot = SupervisorRobot()
supervisorRobot.resetsimulation()