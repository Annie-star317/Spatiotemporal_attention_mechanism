"""RobotRun controller."""
import math

#import gym
import time
import numpy as np
import os
import sys
import cv2

import argparse
import platform

from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
import torch
from python_scripts.Project_config import path_list, gps_goal
# 将上级的上级目录（Train_main）添加到路径
sys.path.append(str(Path(__file__).parent.parent))
from python_scripts.Webots_interfaces import Darwin
from python_scripts.Project_config import Darwin_config
@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                ping = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
                goal = [0, 0, 0, 0, 0, 0]
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    #if save_crop:
                        #save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                        print(c)
                        ping[c][0] = int(xyxy[0].item())
                        ping[c][1] = int(xyxy[1].item())
                        ping[c][2] = int(xyxy[2].item())
                        ping[c][3] = int(xyxy[3].item())
                        class_index=cls#获取属性
                        object_name=names[int(cls)]
                        goal[2 * c] = (ping[c][0] + ping[c][1])/2
                        goal[2 * c + 1] = (ping[c][2] + ping[c][3])/2
                         #print('class index is',class_index.item())#打印属性，由于我们只有一个类，所以是0
                         #print('object_names is',object_name)#打印标签名字，

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
    if len(det):
       return 1
    else:
        return 0

def parse_opt(img_):
    b = './'
    img__ = img_
    abc = '%s%s'%(b,img__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'best.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=abc, help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[160], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', default='True', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt

def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    if_exist = run(**vars(opt))
    return if_exist

class RobotRun(Darwin):
    # 控制机器人按照action行动的类
    # action:
    def __init__(self, robot, state, action_shouder, action_arm, step, catch_flag, gps1, gps2, gps3, gps4, img_name):
        super().__init__(robot)
        self.img_name = img_name    # 名称
        self.step = step    # 步数
        self.robot_state = state  # 机器人状态
        self.gps1 = gps1
        self.gps2 = gps2
        self.gps3 = gps3
        self.gps4 = gps4
        self.gps = [gps1, gps2, gps3, gps4]  # GPS坐标数值列表
        self.action_shouder = action_shouder  # 动作
        self.action_arm = action_arm  # 动作
        
        #print(f"action_arm: {action_arm}, type: {type(action_arm)}")
        #print(f"action_shouder: {action_shouder}, type: {type(action_shouder)}")
        
        # 计算左臂和左肩的目标位置
        current_left_arm = self.robot_state[5]       # 左臂的当前状态在索引5
        current_left_shoulder = self.robot_state[1]  # 左肩的当前状态在索引1
        left_arm_target = 1.25 * action_arm + 0.25      #通过实际仿真应该改变函数表达式，让他变化更小
        left_shoulder_target = 0.2995 * action_shouder - 0.0145     #缩小十倍抓取效果更好

        self.ArmLower = left_arm_target - current_left_arm  # 手臂
        self.ArmLower = max(-0.3, min(0.3, self.ArmLower))  # 限制在-0.5到0.5之间
        self.Shoulder = left_shoulder_target - current_left_shoulder  # 肩部
        self.Shoulder = max(-0.3, min(0.3, self.Shoulder))  # 限制在-0.5到0.5之间

        # print(f"手臂上升: {self.ArmLower}")
        # print(f"肩部上升: {self.Shoulder}")
        # print(f"手臂目标位置: {left_arm_target}")
        # print(f"肩部目标位置: {left_shoulder_target}")

        self.catch_flag = catch_flag  # 抓取标识符
        self.catch_Success_flag = False  # 抓取成功标识符
        #self.small_goal = 0  # 小目标
        # 初始化压力传感器列表
        self.touch = [self.touch_sensors['grasp_L1_1'], 
                      self.touch_sensors['grasp_R1_2']]
        # 压力传感器列表
        self.touch_peng = [self.touch_sensors['arm_L1'], 
                           self.touch_sensors['arm_R1'], 
                           self.touch_sensors['leg_L1'], 
                           self.touch_sensors['leg_L2'], 
                           self.touch_sensors['leg_R1'], 
                           self.touch_sensors['leg_R2']]
        self.future_state = [i for i in self.robot_state]  # 未来状态
        # 下一个状态
        self.next = [self.robot_state[1] + self.Shoulder,  #左肩
                     self.robot_state[0] - self.Shoulder,  #右肩
                     self.robot_state[5] + self.ArmLower,  #左臂
                     self.robot_state[4] - self.ArmLower]  #右臂
        # print(f"左肩: {self.robot_state[1]}")
        # print(f"右肩: {self.robot_state[0]}")
        # print(f"左臂: {self.robot_state[5]}")
        # print(f"右臂: {self.robot_state[4]}")
        self.future_state[1] = self.next[0]  # 未来状态[1] = 下一个状态[0]
        self.future_state[0] = self.next[1]  # 未来状态[0] = 下一个状态[1]
        self.future_state[5] = self.next[2]  # 未来状态[5] = 下一个状态[2]
        self.future_state[4] = self.next[3]  # 未来状态[4] = 下一个状态[3]
        # print(f"左肩: {self.future_state[1]}")
        # print(f"右肩: {self.future_state[0]}")
        # print(f"左臂: {self.future_state[5]}")
        # print(f"右臂: {self.future_state[4]}")
    
        self.now_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 当前状态
        self.next_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 下一个状态
        self.touch_value = [0.0, 0.0]  # 压力传感器值
        self.return_flag_list ={'reward':0,
                                'done'  :0,
                                'good'  :0,
                                'goal'  :0,
                                'count' :0} # 标识符列表 reward, done, good, goal, count

    def run(self):
        self.robot.step(32)  # 机器人步长
        acc = self.accelerometer.getValues()  # 加速度传感器值
        gyro = self.gyro.getValues()  # 陀螺仪值
        x1 = Darwin_config.gps_goal[0] - self.gps[1][1]  # 高度 目标位置x1与当前位置x1的差值
        x2 = Darwin_config.gps_goal[0] - self.gps[2][1]  # 高度 目标位置x2与当前位置x2的差值
        y1 = Darwin_config.gps_goal[1] - self.gps[1][2]  # 前进 目标位置y1与当前位置y1的差值
        y2 = Darwin_config.gps_goal[1] - self.gps[2][2]  # 前进 目标位置y2与当前位置y2的差值
        reward1 = 20 - 200 * math.sqrt((x1 * x1) + (y1 * y1))  # 奖励1
        reward2 = 20 - 200 * math.sqrt((x2 * x2) + (y2 * y2))  # 奖励2
        tan1 = x1 / y1  # 角度1正切值
        tan2 = x2 / y2  # 角度2正切值
        angle1 = math.degrees(math.atan(tan1))  # 角度1
        angle2 = math.degrees(math.atan(tan2))  # 角度2
        delta_angle1 = abs(angle1 - Darwin_config.standard_angle)   # 角度1与标准角度差值的绝对值
        delta_angle2 = abs(angle2 - Darwin_config.standard_angle)   # 角度2与标准角度差值的绝对值
        """
        if angle1 < Darwin_config.standard_angle:    # 角度1小于标准角度
            delta_angle1 = Darwin_config.standard_angle - angle1   # 角度1与标准角度的差值
        else:   # 角度1大于标准角度
            delta_angle1 = angle1 - Darwin_config.standard_angle
        if angle2 < Darwin_config.standard_angle:    # 角度2小于标准角度 
            delta_angle2 = Darwin_config.standard_angle - angle2    # 角度2与标准角度的差值
        else:
            delta_angle2 = angle2 - Darwin_config.standard_angle   # 角度2大于标准角度
        """
        reward3 = 20 - delta_angle1  # 奖励3=20-角度1与标准角度的差值
        reward4 = 20 - delta_angle2  # 奖励4=20-角度2与标准角度的差值
        # 添加对夹爪高度的奖励 - 鼓励夹爪位置更高（左边靠前的gps）
        # height_reward = 20 - 200 * math.sqrt((self.gps[0][1] - Darwin_config.min_height) ** 2)
        # # 添加对夹爪前进程度的奖励 - 鼓励夹爪位置更前
        # forward_reward = 20 - 200 * math.sqrt((self.gps[0][2] - Darwin_config.min_forward) ** 2)
        if reward3 <= -20:  # 奖励3小于-20
            reward3 = -20  # 奖励3= -20
        if reward4 <= -20:  # 奖励4小于-20
            reward4 = -20  # 奖励4= -20
        self.return_flag_list.update({'count':1})
        # 遍历未来状态
        for i in range(len(self.future_state)): 
            if Darwin_config.limit[1][0] <= self.future_state[i] <= Darwin_config.limit[1][1]:    # 角度1在限制范围内
                continue
            else:
                self.return_flag_list.update({'reward':0, 'count':0, 'done':1, 'good':1})
                # 返回下一个状态，奖励，完成，好，目标，计数
                # print('角度1超出限制,catch_flag: 0,done:1--------->341')
                print("设置done=1，原因：future_state超出限制")  
                return self.next_state, \
                       self.return_flag_list['reward'], \
                       self.return_flag_list['done'], \
                       self.return_flag_list['good'], \
                       self.return_flag_list['goal'], \
                       self.return_flag_list['count']
        self.robot.step(32)  # 机器人步长
        # self.robot.step(32)  # 机器人步长
        # self.robot.step(32)  # 机器人步长
        if_exist = 1  # 初始化存在标识符为1    
        if if_exist == 1:    # 如果存在标识符为1
            pass
        # 存在标识符不为1
        else:
            self.return_flag_list.update({'reward':0, 'count':0, 'done':1, 'good':0})
            # 返回下一个状态，奖励，完成，好，目标，计数
            print('存在标识符不为1,catch_flag: 0,done:1------------->390')
            return self.next_state, \
                   self.return_flag_list['reward'], \
                   self.return_flag_list['done'], \
                   self.return_flag_list['good'], \
                   self.return_flag_list['goal'], \
                   self.return_flag_list['count']
        # 遍历加速度传感器
        for i in range(3):
            # 加速度传感器值在限制范围内
            if Darwin_config.acc_low[i] < acc[i] < Darwin_config.acc_high[i] and \
               Darwin_config.gyro_low[i] < gyro[i] < Darwin_config.gyro_high[i]:
                continue
            # 加速度传感器值不在限制范围内
            else:
                self.return_flag_list.update({'reward':0, 'count':0, 'done':1, 'good':0})
                # 返回下一个状态，奖励，完成，好，目标，计数
                print('加速度传感器值不在限制范围内,catch_flag: 0,done:1--------->405')
                return self.next_state, \
                       self.return_flag_list['reward'], \
                       self.return_flag_list['done'], \
                       self.return_flag_list['good'], \
                       self.return_flag_list['goal'], \
                       self.return_flag_list['count']
        # 如果catch_flag为0，即还没有抓到
        
        if self.catch_flag == 0.0:
            # 执行动作到下一状态
            self.motors[1].setPosition(self.next[0])  # 电机1设置位置
            self.motors[0].setPosition(self.next[1])  # 电机0设置位置
            self.motors[5].setPosition(self.next[2])  # 电机5设置位置
            self.motors[4].setPosition(self.next[3])  # 电机4设置位置
            # 获取当前4个舵机的位置
            current_positions = [
                self.motors_sensors[1].getValue(),  # 左肩
                self.motors_sensors[0].getValue(),  # 右肩
                self.motors_sensors[5].getValue(),  # 右臂
                self.motors_sensors[4].getValue()   # 左臂
            ]

            # 目标位置
            target_positions = [
                self.next[0],  # 左肩
                self.next[1],  # 右肩
                self.next[2],  # 左臂
                self.next[3],  # 右臂
            ]

            # 循环直到所有舵机都到达目标位置或超时
            timeout = 100  # 设置最大循环次数以防止无限循环
            for _ in range(timeout):
                # 执行一个仿真步长
                if self.robot.step(32) == -1:
                    break

                # 获取当前所有舵机的位置
                current_positions = [self.motors_sensors[i].getValue() for i in range(20)]

                # 检查是否所有舵机都已到达目标位置
                all_in_position = True
                for i in range(len(target_positions)):
                    # 使用一个小的容忍度（例如0.01弧度）来判断是否到达
                    if abs(target_positions[i] - current_positions[i]) > 0.01:
                        all_in_position = False
                        break
                
                # 如果所有舵机都已就位，则退出循环
                if all_in_position:
                    break

            # self.robot.step(32)  # 机器人步长
            # self.robot.step(32)  # 机器人步长
            # self.robot.step(32)  # 机器人步长
            # self.robot.step(32)  # 机器人步长
            # self.robot.step(32)  # 机器人步长
            # self.robot.step(32)  # 机器人步长
            # self.robot.step(32)  # 机器人步长
            # self.robot.step(32)  # 机器人步长
            # self.robot.step(32)  # 机器人步长
            # self.robot.step(32)  # 机器人步长
            # self.robot.step(32)  # 机器人步长
            # self.robot.step(32)  # 机器人步长
            # self.robot.step(32)  # 机器人步长
            # self.robot.step(32)  # 机器人步长
            # self.robot.step(32)  # 机器人步长
            # self.robot.step(32)  # 机器人步长
            # self.robot.step(32)  # 机器人步长
            # self.robot.step(32)  # 机器人步长
            # self.robot.step(32)  # 机器人步长
            # self.robot.step(32)  # 机器人步长
            # self.robot.step(32)  # 机器人步长
            # self.robot.step(32)  # 机器人步长
            # self.robot.step(32)  # 机器人步长
            # self.robot.step(32)  # 机器人步长
            # self.robot.step(32)  # 机器人步长
            # print(f'catch_flag={self.catch_flag}, done=0---------->479')
            self.return_flag_list.update({'done':0, 'reward':reward1 + reward2, 'good':1})
            # 遍历压力传感器
            for m in range(6):
                # 压力传感器值为1.0
                if self.touch_peng[m].getValue() == 1.0:
                    print(f'catch_flag={self.catch_flag}, done=1---------->480ppo')
                    self.return_flag_list.update({'done':1, 'reward':0, 'good':1, 'count':0})
                    # 返回下一个状态，奖励，完成，好，目标，计数
                    return self.next_state, \
                           self.return_flag_list['reward'], \
                           self.return_flag_list['done'], \
                           self.return_flag_list['good'], \
                           self.return_flag_list['goal'], \
                           self.return_flag_list['count']
            # print(f'grasp_L1=', self.touch_sensors['grasp_L1'].getValue())
            # print(f'grasp_L1_1=', self.touch_sensors['grasp_L1_1'].getValue())
            # print(f'grasp_L1_2=', self.touch_sensors['grasp_L1_2'].getValue())
            # print(f'grasp_R1=', self.touch_sensors['grasp_R1'].getValue())
            # print(f'grasp_R1_1=', self.touch_sensors['grasp_R1_1'].getValue())
            # print(f'grasp_R1_2=', self.touch_sensors['grasp_R1_2'].getValue())
            # 遍历压力传感器
            if self.touch_sensors['grasp_L1'].getValue() == 1.0 or \
               self.touch_sensors['grasp_L1_1'].getValue() == 1.0 or \
               self.touch_sensors['grasp_L1_2'].getValue() == 1.0 or \
               self.touch_sensors['grasp_R1'].getValue() == 1.0 or \
               self.touch_sensors['grasp_R1_1'].getValue() == 1.0 or \
               self.touch_sensors['grasp_R1_2'].getValue() == 1.0:
                # 打印压力传感器值
                # print("___________")
                # print(self.touch_sensors['grasp_L1'].getValue())
                # print(self.touch_sensors['grasp_L1_1'].getValue())
                # print(self.touch_sensors['grasp_L1_2'].getValue())
                # print(self.touch_sensors['grasp_R1'].getValue())
                # print(self.touch_sensors['grasp_R1_1'].getValue())
                # print(self.touch_sensors['grasp_R1_2'].getValue())

                timer = 0  # 计时器
                self.motors[21].setPosition(-0.5)  # 电机21设置位置 
                self.motors[20].setPosition(-0.5)  # 电机20设置位置
                while self.robot.step(32) != -1:
                    timer += 32  # 计时器增加32 
                    if timer >= 3000:
                        print(timer)
                        print("等待时间过长：", timer,'----------------------------->532')
                        break
                # 遍历压力传感器    
                for j in range(len(self.touch)):
                    self.touch_value[j] = self.touch[j].getValue()  # 压力传感器值
                sucess = np.array_equal(self.touch_value, Darwin_config.touch_T)  # 成功标识符=压力传感器值与目标值相等    
                sucess = np.int(sucess)  # 成功标识符=1
                faild = np.array_equal(self.touch_value, Darwin_config.touch_F)  # 失败标识符=压力传感器值与失败值相等
                faild = np.int(faild)  # 失败标识符=1
                # 失败标识符=1且步长小于等于5
                if faild == 1 and self.step <= 5:
                    self.return_flag_list.update({'reward':0, 'count':1, 'done':1, 'good':1})
                    print("失败标识符=1且步长小于等于5")
                    # 写入数据
                    with open(path_list['shu_ju_path_DQN'], 'a') as file:
                        file.write('0')
                        file.write(",")
                        file.close()
                # 失败=1且步长大于5
                elif faild == 1 and self.step > 5:
                    self.return_flag_list.update({'count':1, 'done':1, 'good':1})
                    print("失败=1且步长大于5")
                # 成功=1
                elif sucess == 1:
                    # 奖励1+奖励2小于20
                    if (reward1 + reward2) < 20:
                        self.return_flag_list.update({'count':1, 'done':1, 'good':1})
                    # 奖励1+奖励2大于等于20 
                    else:
                        self.return_flag_list.update({'count':0, 'done':1, 'good':1, 'goal':1})
                        print("俺抓到了")  
                        # 写入数据
                        with open(path_list['gps_path_DQN'], 'a') as file:
                            gpss = [self.gps1, self.gps2, self.gps3, self.gps4]  # 目标位置
                            gpss1 = str(gpss)  # 目标位置字符串 
                            file.write(gpss1)  # 写入目标位置字符串
                            file.write(",")  # 写入逗号
                            file.close()  # 关闭文件
                # 成功=0
                else:
                    # 奖励1+奖励2小于20
                    if (reward1 + reward2) < 20:
                        self.return_flag_list.update({'count':1, 'done':1, 'good':1})
                    # 奖励1+奖励2大于等于20 
                    else:
                        self.return_flag_list.update({'count':1, 'done':1, 'good':1})
            # 遍历电机传感器
            else:
                for i in range(20):
                    # print(f'self.future_state[{i}]={self.future_state[i]}')
                    # print(f'self.motors_sensors[{i}].getValue()={self.motors_sensors[i].getValue()}')
                    # print(f'self.robot_state[{i}]={self.robot_state[i]}')
                    self.next_state[i] = self.motors_sensors[i].getValue()  # 舵机位置传感器值
                    # self.next_state[i] = self.robot_state[i]
                    # print(f'self.next_state[{i}]={self.next_state[i]}')
                    self.cha_zhi = self.next_state[i] - self.future_state[i]  # 当前值与未来值的差值
                    # print(f'i={i}, cha_zhi={self.cha_zhi}, done={self.return_flag_list["done"]}')
                    # if -100 < self.cha_zhi < 100:  # 差值在-0.005到0.005之间
                    if -0.005 < self.cha_zhi < 0.005:  # 差值在-0.005到0.005之间
                        # print(f'i1={i}, 当前值={self.next_state[i]:.4f}, 目标值={self.future_state[i]:.4f}, 差值={self.cha_zhi:.4f}, done={self.return_flag_list["done"]}')
                        # print('----------------------------->577')
                        continue
                    else:
                        print(f'catch_flag={self.catch_flag}, done=1----------------->585PPO')
                        self.return_flag_list.update({'count':1, 'done':1, 'good':1})
                        print(f'i2={i}, 当前值={self.next_state[i]:.4f}, 目标值={self.future_state[i]:.4f}, 差值={self.cha_zhi:.4f}, done={self.return_flag_list["done"]}')
                        break
                        # continue
        # 否则catch_flag为非0，即已经抓到了
        else:
            timer = 0  # 计时器
            self.motors[21].setPosition(-0.5)  # 电机21设置位置
            self.motors[20].setPosition(-0.5)  # 电机20设置位置
            while self.robot.step(32) != -1:
                timer += 32  # 计时器增加32
                if timer >= 3000:  # 计时器大于等于2000
                    break
            # 遍历压力传感器
            for j in range(len(self.touch)):
                self.touch_value[j] = self.touch[j].getValue()  # 压力传感器值
            sucess = np.array_equal(self.touch_value, Darwin_config.touch_T)  # 成功=压力传感器值与目标值相等
            sucess = np.int(sucess)  # 成功=1
            faild = np.array_equal(self.touch_value, Darwin_config.touch_F)  # 失败=压力传感器值与失败值相等
            faild = np.int(faild)  # 失败=1
            # 失败=1且步长小于等于5
            if faild == 1 and self.step <= 5:
                self.return_flag_list.update({'reward':0, 'count':1, 'done':1, 'good':1})
                print("失败=1且步长小于等于5")
            # 失败=1且步长大于5
            elif faild == 1 and self.step > 5:
                self.return_flag_list.update({'reward':0, 'count':1, 'done':1, 'good':1})
                print("失败=1且步长大于5")
            # 成功=1
            elif sucess == 1:
                # 奖励1+奖励2小于20
                if (reward1 + reward2) < 20:
                    self.return_flag_list.update({'reward':0, 'count':1, 'done':1, 'good':1})
                # 奖励1+奖励2大于等于20
                else:
                    self.return_flag_list.update({'reward':100, 'count':0, 'done':1, 'good':1, 'goal':1})
                    print(self.return_flag_list['reward'])  # 打印奖励
                    print("俺抓到了")  # 打印"俺抓到了"
                    # 写入数据
                    with open(path_list['gps_path_DQN'], 'a') as file:
                        gpss = str([self.gps1, self.gps2, self.gps3, self.gps4])  # 目标位置
                        file.write(gpss)  # 写入目标位置字符串
                        file.write(",")  # 写入逗号
                        file.close()  # 关闭文件
            # 奖励1+奖励2大于等于20
            else:
                # 奖励1+奖励2小于20
                self.return_flag_list.update({'count':1, 'done':1, 'good':1})
                """
                if (reward1 + reward2) < 20:
                    self.return_flag_list.update({'count':1, 'done':1, 'good':1})
                else:
                    self.return_flag_list.update({'count':1, 'done':1, 'good':1})
                """
        # print(f'正常情况输出，catch_flag={self.catch_flag}, done=', self.return_flag_list['done'])
        # 返回下一个状态，奖励，完成，好，目标，计数
        return self.next_state, \
               self.return_flag_list['reward'], \
               self.return_flag_list['done'], \
               self.return_flag_list['good'], \
               self.return_flag_list['goal'], \
               self.return_flag_list['count']
               #self.small_goal
    
