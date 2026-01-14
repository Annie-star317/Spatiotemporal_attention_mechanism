import numpy as np
import json
import re
from datetime import datetime
# import torch # 如果你想使用 isinstance 检查张量，请取消注释

class CustomJSONEncoder(json.JSONEncoder):
    """自定义 JSON 编码器，处理 NumPy 数组、PyTorch 张量和 datetime 对象"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # elif isinstance(obj, torch.Tensor): # 如果环境中确定有 torch，这是更健壮的方式
        #     return obj.cpu().detach().numpy().tolist()
        elif "torch.Tensor" in str(type(obj)): # 使用字符串检查作为备选方案
            # 确保在调用 .cpu() 和 .detach() 之前对象确实是张量
            try:
                return obj.cpu().detach().numpy().tolist()
            except AttributeError:
                # 如果对象声称是 Tensor 但没有这些方法，按原样处理
                pass
        elif isinstance(obj, datetime):
            return obj.isoformat()
        # 处理 Python 的基本数字类型，以防万一传入的是 NumPy 数字类型
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)):
            return None # 或者选择其他合适的表示方式
        return super().default(obj)

class Log_write:
    def __init__(self):
        """初始化日志记录器"""
        self.data = {
            'start time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'save time': [],
            'shoulder_actions': [[]],  # 存储肩膀动作
            'arm_actions': [[]]        # 存储手臂动作
        }
        # 注意：不再有 self.action_list 实例变量

    def add_action(self, a_shoulder, a_arm):
        """
        将肩膀和手臂动作分别添加到对应的列表中
        """
        # 确保动作列表存在
        if 'shoulder_actions' not in self.data or not self.data['shoulder_actions']:
            self.data['shoulder_actions'] = [[]]
        if 'arm_actions' not in self.data or not self.data['arm_actions']:
            self.data['arm_actions'] = [[]]

        # 如果 action 是 Tensor 或 NumPy scalar，获取其 Python 基本类型值
        # 处理肩膀动作
        shoulder_item = a_shoulder
        if hasattr(a_shoulder, 'item') and callable(a_shoulder.item):
            try:
                shoulder_item = a_shoulder.item()
            except Exception as e:
                print(f"Warning: Could not call .item() on shoulder action {a_shoulder}: {e}")
                # 保持原样
        
        # 处理手臂动作
        arm_item = a_arm
        if hasattr(a_arm, 'item') and callable(a_arm.item):
            try:
                arm_item = a_arm.item()
            except Exception as e:
                print(f"Warning: Could not call .item() on arm action {a_arm}: {e}")
                # 保持原样

        # 检查转换后的类型是否为基本数字类型（int 或 float）
        if not isinstance(shoulder_item, (int, float)):
            print(f"Warning: Adding non-standard action type '{type(shoulder_item).__name__}' to action list. Value: {shoulder_item}")
        if not isinstance(arm_item, (int, float)):
            print(f"Warning: Adding non-standard action type '{type(arm_item).__name__}' to action list. Value: {arm_item}")

        # 将动作添加到最后一个（当前活动的）序列中
        self.data['shoulder_actions'][-1].append(shoulder_item)
        self.data['arm_actions'][-1].append(arm_item)
        
    def add_leg_action(self, leg_upper_action, leg_lower_action, ankle_action):
        """
        将腿部动作（大腿、小腿、脚踝）添加到对应的列表中
        
        Args:
            leg_upper_action: 大腿动作值
            leg_lower_action: 小腿动作值
            ankle_action: 脚踝动作值
        """
        # 确保动作列表存在
        for action_name in ['leg_upper_actions', 'leg_lower_actions', 'ankle_actions']:
            if action_name not in self.data:
                self.data[action_name] = [[]]
            elif not self.data[action_name]:
                self.data[action_name] = [[]]
        
        # 处理每个动作
        actions = {
            'leg_upper_actions': leg_upper_action,
            'leg_lower_actions': leg_lower_action,
            'ankle_actions': ankle_action
        }
        
        for action_key, action_value in actions.items():
            # 如果 action 是 Tensor 或 NumPy scalar，获取其 Python 基本类型值
            action_item = action_value
            if hasattr(action_value, 'item') and callable(action_value.item):
                try:
                    action_item = action_value.item()
                except Exception as e:
                    print(f"Warning: Could not call .item() on {action_key} action {action_value}: {e}")
                    
            # 检查转换后的类型是否为基本数字类型（int 或 float）
            if not isinstance(action_item, (int, float)):
                print(f"Warning: Adding non-standard {action_key} type '{type(action_item).__name__}'. Value: {action_item}")
                
            # 将动作添加到对应的列表中
            self.data[action_key][-1].append(float(action_item))

    def add(self, **kwargs):
        """
        向日志添加任意键值对数据。
        如果键不存在，则创建它并初始化为空列表，然后追加值。
        """
        for key, value in kwargs.items():
            # 确保我们不意外地覆盖核心列表，如 'action_list'
            if key == 'action_list':
                 print(f"Warning: Attempted to use reserved key 'action_list' in add(). Please use add_action() for actions.")
                 continue
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)

    def clear_action(self):
        """
        标记当前动作序列的结束（例如，一个 episode 结束）。
        准备开始记录下一个新的动作序列。
        会在 self.data['action_list'] 中添加一个新的空列表。
        """
        # 只有当最后一个动作列表非空时，才添加新的空列表，
        # 避免在连续调用 clear_action 时产生多个连续的空列表。  
        for key in ['shoulder_actions', 'arm_actions']:
            if key in self.data and self.data[key]:
                if self.data[key][-1]:  # 检查最后一个列表是否非空
                    self.data[key].append([])
            else:
                self.data[key] = [[]]

    def get(self, key):
        """获取指定键的所有历史记录值（一个列表）"""
        return self.data.get(key, [])

    def save_catch(self, file_path):
        """将当前所有日志数据保存到指定的 JSON 文件中"""
        self.data['save time'].append(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print(f'保存日志...')

        # --- 数据准备 ---
        # 创建一个数据的深拷贝用于保存，以防修改影响原始数据
        # 使用 JSON 序列化和反序列化是一种简单的深拷贝方式，能处理嵌套结构
        try:
            data_to_save = json.loads(json.dumps(self.data, cls=CustomJSONEncoder))
        except Exception as e:
             print(f"Error creating deep copy of data for saving: {e}")
             # 可以选择是继续使用 self.data 还是中止保存
             data_to_save = self.data # 回退到使用原始数据（浅拷贝）

        # 修改episode_num，只保留最大值
        if 'episode_num' in data_to_save and data_to_save['episode_num']:
            max_episode = max(data_to_save['episode_num'])
            data_to_save['episode_num'] = [max_episode]

        # 可选：移除 action_list 末尾的空列表
        # 这通常是期望的行为，因为它代表一个已结束但未记录任何动作的序列
        for action_key in ['shoulder_actions', 'arm_actions']:
            if action_key in data_to_save and data_to_save[action_key] and not data_to_save[action_key][-1]:
                data_to_save[action_key] = data_to_save[action_key][:-1]

        # --- JSON 序列化 ---
        try:
            # 使用自定义编码器将准备好的数据转换为 JSON 字符串
            json_data = json.dumps(data_to_save, cls=CustomJSONEncoder, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error during JSON serialization: {e}")
            return # 无法序列化，中止保存

        #--- 格式化特定字段 ---
        # # 处理 episode_num 字段，将其格式化为多行
        # if 'episode_num' in data_to_save and len(data_to_save['episode_num']) > 10:
        #     # 将 episode_num 格式化为每行最多 10 个数字
        #     episode_chunks = []
        #     chunk_size = 20
        #     episodes = data_to_save['episode_num']
            
        #     for i in range(0, len(episodes), chunk_size):
        #         chunk = episodes[i:i+chunk_size]
        #         episode_chunks.append(json.dumps(chunk))
            
        #     # 创建格式化的 episode_num 字符串
        #     formatted_episodes = "[\n        " + ",\n        ".join(episode_chunks) + "\n    ]"
            
        #     # 在 JSON 字符串中替换 episode_num 部分
        #     pattern_episode = r'"episode_num":\s*\[\s*[\s\S]*?\]'
        #     json_data = re.sub(pattern_episode, f'"episode_num": {formatted_episodes}', json_data)
        
        # --- 正则表达式格式化 action_list ---
        # 改进的模式，尝试更灵活地匹配数字列表（整数）
        pattern = r'\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]'

        def compress_list(match):
            # 获取匹配到的列表内容（包括括号）
            list_content = match.group(0)
            # 移除内部的所有换行符和多余的空白，替换为单个空格
            compressed_content = re.sub(r'\s+', ' ', list_content)
            # 标准化逗号后的空格
            compressed_content = re.sub(r'\s*,\s*', ', ', compressed_content)
            # 标准化括号内外的空格
            compressed_content = re.sub(r'\[\s+', '[', compressed_content)
            compressed_content = re.sub(r'\s+\]', ']', compressed_content)
            return compressed_content.strip()

        try:
            # 应用正则表达式替换，注意 re.MULTILINE 可能有助于跨行匹配
            formatted_json = re.sub(pattern, compress_list, json_data, flags=re.MULTILINE)
        except Exception as e:
            print(f"Error during regex formatting: {e}")
            formatted_json = json_data # 如果正则出错，回退到未格式化的 JSON

        # --- 文件写入 ---
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(formatted_json)
            print(f"日志保存成功")
        except IOError as e:
            print(f"错误：无法写入日志文件: {e}")
        except Exception as e:
            print(f"保存过程中发生意外错误: {e}")

    def save_tai(self, file_path):
        """将当前所有日志数据保存到指定的 JSON 文件中"""
        self.data['save time'].append(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print(f'保存日志...')

        # --- 数据准备 ---
        # 创建一个数据的深拷贝用于保存，以防修改影响原始数据
        # 使用 JSON 序列化和反序列化是一种简单的深拷贝方式，能处理嵌套结构
        try:
            data_to_save = json.loads(json.dumps(self.data, cls=CustomJSONEncoder))
        except Exception as e:
             print(f"Error creating deep copy of data for saving: {e}")
             # 可以选择是继续使用 self.data 还是中止保存
             data_to_save = self.data # 回退到使用原始数据（浅拷贝）

        # # 修改episode_num，只保留最大值
        # if 'episode_num' in data_to_save and data_to_save['episode_num']:
        #     max_episode = max(data_to_save['episode_num'])
        #     data_to_save['episode_num'] = [max_episode]
        #     print(f"仅保存最大episode值: {max_episode}")

        # 可选：移除 action_list 末尾的空列表
        # 这通常是期望的行为，因为它代表一个已结束但未记录任何动作的序列
        for action_key in ['shoulder_actions', 'arm_actions']:
            if action_key in data_to_save and data_to_save[action_key] and not data_to_save[action_key][-1]:
                data_to_save[action_key] = data_to_save[action_key][:-1]

        # --- JSON 序列化 ---
        try:
            # 使用自定义编码器将准备好的数据转换为 JSON 字符串
            json_data = json.dumps(data_to_save, cls=CustomJSONEncoder, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error during JSON serialization: {e}")
            return # 无法序列化，中止保存

        #--- 格式化特定字段 ---
        # 处理 episode_num 字段，将其格式化为多行
        if 'episode_num' in data_to_save and len(data_to_save['episode_num']) > 10:
            # 将 episode_num 格式化为每行最多 10 个数字
            episode_chunks = []
            chunk_size = 20
            episodes = data_to_save['episode_num']
            
            for i in range(0, len(episodes), chunk_size):
                chunk = episodes[i:i+chunk_size]
                episode_chunks.append(json.dumps(chunk))
            
            # 创建格式化的 episode_num 字符串
            formatted_episodes = "[\n        " + ",\n        ".join(episode_chunks) + "\n    ]"
            
            # 在 JSON 字符串中替换 episode_num 部分
            pattern_episode = r'"episode_num":\s*\[\s*[\s\S]*?\]'
            json_data = re.sub(pattern_episode, f'"episode_num": {formatted_episodes}', json_data)
        
        # --- 正则表达式格式化 action_list ---
        # 改进的模式，尝试更灵活地匹配数字列表（整数）
        pattern = r'\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]'

        def compress_list(match):
            # 获取匹配到的列表内容（包括括号）
            list_content = match.group(0)
            # 移除内部的所有换行符和多余的空白，替换为单个空格
            compressed_content = re.sub(r'\s+', ' ', list_content)
            # 标准化逗号后的空格
            compressed_content = re.sub(r'\s*,\s*', ', ', compressed_content)
            # 标准化括号内外的空格
            compressed_content = re.sub(r'\[\s+', '[', compressed_content)
            compressed_content = re.sub(r'\s+\]', ']', compressed_content)
            return compressed_content.strip()

        try:
            # 应用正则表达式替换，注意 re.MULTILINE 可能有助于跨行匹配
            formatted_json = re.sub(pattern, compress_list, json_data, flags=re.MULTILINE)
        except Exception as e:
            print(f"Error during regex formatting: {e}")
            formatted_json = json_data # 如果正则出错，回退到未格式化的 JSON

        # --- 文件写入 ---
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(formatted_json)
            print(f"日志保存成功")
        except IOError as e:
            print(f"错误：无法写入日志文件: {e}")
        except Exception as e:
            print(f"保存过程中发生意外错误: {e}")