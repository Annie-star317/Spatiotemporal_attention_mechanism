import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 尝试导入 scipy，如果失败则用 None 代替
try:
    from scipy.signal import savgol_filter
except ImportError:
    savgol_filter = None
import numpy as np

class SilentChartApp:
    def __init__(self, root):
        self.root = root
        self.root.title("静默数据可视化工具")
        self.file_path = ""
        self.raw_data = None

        # --- 【新增】设置 matplotlib 全局字体和样式 ---
        # 这两行对于显示中文至关重要
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 'SimHei' 是黑体，你也可以用 'Microsoft YaHei' 等
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        # 使用 seaborn 风格美化图表
        import matplotlib
        matplotlib.style.use('seaborn')
        # -----------------------------

        # 初始化所有Tkinter变量
        self.smooth_enabled = tk.BooleanVar(value=False)
        self.window_length = tk.IntVar(value=5)
        self.polyorder = tk.IntVar(value=2)

        # 初始化界面组件
        self.create_widgets()
        self.init_chart()

    def create_widgets(self):
        # 创建控制面板
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        # 文件选择组件
        ttk.Button(
            control_frame,
            text="打开JSON文件",
            command=self.open_file
        ).pack(side=tk.LEFT, padx=5)

        # 坐标轴选择组件
        ttk.Label(control_frame, text="X轴字段:").pack(side=tk.LEFT, padx=5)
        self.x_axis_combobox = ttk.Combobox(control_frame, width=20)
        self.x_axis_combobox.pack(side=tk.LEFT, padx=5)

        ttk.Label(control_frame, text="Y轴字段:").pack(side=tk.LEFT, padx=5)
        self.y_axis_combobox = ttk.Combobox(control_frame, width=20)
        self.y_axis_combobox.pack(side=tk.LEFT, padx=5)

        # 操作按钮
        ttk.Button(
            control_frame,
            text="刷新图表",
            command=self.update_chart
        ).pack(side=tk.LEFT, padx=5)

        control_right = ttk.Frame(control_frame)
        control_right.pack(side=tk.RIGHT)
        
        # 保存按钮
        ttk.Button(
            control_right,
            text="保存图表",
            command=self.save_chart
        ).pack(side=tk.LEFT, padx=5)
        
        # 平滑功能控件
        ttk.Checkbutton(
            control_right,
            text="平滑曲线",
            variable=self.smooth_enabled,
            command=self.toggle_smooth_options
        ).pack(side=tk.LEFT, padx=5)
        
        self.smooth_frame = ttk.Frame(control_right)
        
        ttk.Label(self.smooth_frame, text="窗口:").pack(side=tk.LEFT)
        ttk.Spinbox(
            self.smooth_frame,
            from_=3,
            to=21,
            increment=2,
            textvariable=self.window_length,
            width=5
        ).pack(side=tk.LEFT)
        
        ttk.Label(self.smooth_frame, text="阶数:").pack(side=tk.LEFT)
        ttk.Spinbox(
            self.smooth_frame,
            from_=2,
            to=5,
            textvariable=self.polyorder,
            width=3
        ).pack(side=tk.LEFT)

    def toggle_smooth_options(self):
        """控制平滑参数控件的显示"""
        if self.smooth_enabled.get():
            self.smooth_frame.pack(side=tk.LEFT, padx=5)
        else:
            self.smooth_frame.pack_forget()

    def apply_smoothing(self, y_data):
        """应用Savitzky-Golay平滑"""
        if not savgol_filter:
            messagebox.showerror("错误", "请先安装scipy库：pip install scipy")
            return y_data
            
        window = self.window_length.get()
        polyorder = self.polyorder.get()
        
        # 确保 window 是奇数
        if window % 2 == 0:
            window += 1
            
        # 确保不大于数据长度
        if window > len(y_data):
            window = len(y_data) - 1 if len(y_data) % 2 == 0 else len(y_data)
            if window < 3: # 窗口不能太小
                messagebox.showwarning("警告", "数据点太少，无法应用平滑。")
                return y_data

        try:
            # 应用平滑
            smoothed_data = savgol_filter(y_data, window, polyorder)
            # 将超出原始数据范围的平滑值截断，避免曲线跑偏
            # 平滑可能会导致值超出原始min/max范围
            smoothed_data = np.clip(smoothed_data, min(y_data), max(y_data))
            return smoothed_data
        except Exception as e:
            messagebox.showerror("平滑错误", f"窗口长度 {window} 和阶数 {polyorder} 不匹配或数据有问题: {str(e)}")
            return y_data

    def save_chart(self):
        """保存图表到文件"""
        if not hasattr(self, 'figure') or len(self.figure.axes) == 0:
            messagebox.showwarning("警告", "请先生成图表")
            return
            
        filetypes = [
            ('PNG 图片', '*.png'),
            ('JPEG 图片', '*.jpg'),
            ('PDF 文档', '*.pdf'),
            ('SVG 矢量图', '*.svg')
        ]
        
        path = filedialog.asksaveasfilename(
            filetypes=filetypes,
            defaultextension=".png"
        )
        
        if path:
            try:
                self.figure.savefig(path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("成功", f"图表已保存至：{path}")
            except Exception as e:
                messagebox.showerror("保存失败", str(e))

    def init_chart(self):
        # 初始化图表
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def open_file(self):
        self.file_path = filedialog.askopenfilename(
            filetypes=[("JSON文件", "*.json")]
        )
        if self.file_path and self.load_data():
            # 'num' 代表 X轴是序号
            keys = ['num'] + list(self.raw_data.keys())
            self.x_axis_combobox['values'] = keys
            self.y_axis_combobox['values'] = list(self.raw_data.keys())
            self.x_axis_combobox.set('num')
            # 默认选择第一个数据字段作为Y轴
            if self.raw_data:
                self.y_axis_combobox.set(next(iter(self.raw_data.keys())))

    def load_data(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f: # 指定 utf-8 编码
                data = json.load(f)
                
            # 强制转换为字典格式
            if not isinstance(data, dict):
                # 如果是列表，转换为 {field_name: list_of_values} 的格式
                if isinstance(data, list) and data:
                    # 假设列表里的第一个字典的键就是所有数据键
                    first_item_keys = data[0].keys()
                    new_data = {key: [] for key in first_item_keys}
                    for item in data:
                        for key in first_item_keys:
                            new_data[key].append(item.get(key))
                    self.raw_data = new_data
                else:
                    messagebox.showerror("数据错误", "JSON文件格式不正确，无法解析。")
                    return False
            else:
                # 如果是字典，处理其值
                self.raw_data = {}
                for k, v in data.items():
                    if not isinstance(v, list):
                        self.raw_data[k] = [v]
                    else:
                        self.raw_data[k] = v
            
            # 加载数据后自动更新一次图表
            self.update_chart()
            return True
        except Exception as e:
            messagebox.showerror("加载错误", f"无法加载文件: {e}")
            return False

    def validate_entries(self):
        x_key = self.x_axis_combobox.get().strip()
        y_key = self.y_axis_combobox.get().strip()
        return (
            x_key and y_key and
            (x_key == 'num' or x_key in self.raw_data) and
            y_key in self.raw_data
        )

    def process_data(self, x_key, y_key):
        try:
            raw_y = self.raw_data[y_key]
            y_data = []
            for y in raw_y:
                try:
                    y_data.append(float(y))
                except (ValueError, TypeError):
                    pass # 跳过非数值

            x_data = []
            if x_key == 'num':
                x_data = list(range(len(y_data)))
            else:
                raw_x = self.raw_data[x_key]
                for x in raw_x:
                    try:
                        x_data.append(float(x))
                    except (ValueError, TypeError):
                        pass

            min_length = min(len(x_data), len(y_data))
            if min_length == 0:
                messagebox.showwarning("数据警告", f"找不到有效数值对，请检查 {x_key} 和 {y_key} 字段。")
                return None, None
            return x_data[:min_length], y_data[:min_length]
        except Exception as e:
            messagebox.showerror("处理错误", f"数据处理失败: {e}")
            return None, None

    def update_chart(self):
        if not self.raw_data or not self.validate_entries():
            return
            
        x_key = self.x_axis_combobox.get().strip()
        y_key = self.y_axis_combobox.get().strip()
        x_data, y_data = self.process_data(x_key, y_key)
        
        if x_data and y_data: # 确保数据有效
            self.draw_chart(x_data, y_data, x_key, y_key)

    def draw_chart(self, x_data, y_data, x_label, y_label):
        """优化后的绘图方法，风格更清晰、明了，专注于趋势展示"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # --- 绘制原始数据线 (可选，推荐) ---
        # 先画一条非常原始、非常淡的线作为背景，展示所有数据点
        ax.plot(x_data, y_data, color='lightgrey', linewidth=0.8, alpha=0.6, label='原始数据', zorder=1)

        # --- 绘制平滑后的主曲线 ---
        if self.smooth_enabled.get():
            y_smoothed = self.apply_smoothing(y_data)
            # 主曲线使用与之前不同的颜色和样式
            ax.plot(x_data, y_smoothed, 
                    color='#1f77b4',  # 保留您喜欢的蓝色
                    linewidth=2.5,    # 保持线宽
                    solid_capstyle='round', # 圆角线帽，让曲线更平滑
                    zorder=2)         # 确保这条线在原始数据线之上
            label_text = f'{y_label} (平滑后)'
        else:
            # 如果没有平滑，主曲线就是原始数据本身，但用更突出的样式
            ax.plot(x_data, y_data, 
                    color='#1f77b4', 
                    linewidth=2.5,
                    solid_capstyle='round',
                    zorder=2)
            label_text = y_label

        # --- 清晰而简洁的图例 ---
        # 只显示最重要的线（平滑后的或原始数据）
        ax.legend(
            handles=[
                ax.lines[-1] # 总是显示较新的那条线（平滑的或原始的）
            ],
            labels=[label_text],
            loc='best', 
            frameon=True,
            framealpha=0.9,
            facecolor='white',
            edgecolor='grey'
        )

        # --- 精炼的标题和标签 ---
        # 标题简单直接，放在图表内部，更节省空间
        ax.set_title(f'{y_label} 的变化趋势', fontsize=12, pad=20, y=1.02)
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        
        # --- 美化坐标轴和网格 ---
        # 添加非常淡的网格线，提升读图体验
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # 可选：让图表的“骨架”（上边框和右边框）不可见，更清爽
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        self.figure.tight_layout()
        self.canvas.draw()



if __name__ == "__main__":
    root = tk.Tk()
    app = SilentChartApp(root)
    root.mainloop()
