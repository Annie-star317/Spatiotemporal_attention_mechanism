import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.colors as mcolors
from collections import OrderedDict

try:
    from scipy.signal import savgol_filter
except ImportError:
    savgol_filter = None

class MultiCurveChartApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Curve Data Visualization Tool")
        
        # 存储多个文件的数据
        self.file_data = OrderedDict()  # 文件名 -> 数据字典
        self.file_labels = {}  # 文件名 -> 自定义标签
        self.file_colors = {}  # 文件名 -> 颜色
        
        # 扩展颜色列表，支持更多文件
        self.available_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
            '#FF8A80', '#82B1FF', '#B388FF', '#A7FFEB', '#FFAB91',
            '#F8BBD9', '#E1F5FE', '#F3E5F5', '#E8F5E8', '#FFF9C4'
        ]
        self.color_index = 0
        
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
        
        # 文件操作区域
        file_frame = ttk.LabelFrame(control_frame, text="File Management", padding=5)
        file_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y)
        
        ttk.Button(
            file_frame,
            text="Add JSON Files",
            command=self.add_multiple_files  # 支持多文件选择
        ).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(
            file_frame,
            text="Clear All Files",
            command=self.clear_all_files
        ).pack(side=tk.LEFT, padx=2)
        
        # 坐标轴选择组件
        axis_frame = ttk.Frame(control_frame)
        axis_frame.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(axis_frame, text="X Axis Field:").pack(side=tk.LEFT, padx=2)
        self.x_axis_combobox = ttk.Combobox(axis_frame, width=15)
        self.x_axis_combobox.pack(side=tk.LEFT, padx=2)
        
        ttk.Label(axis_frame, text="Y Axis Field:").pack(side=tk.LEFT, padx=2)
        self.y_axis_combobox = ttk.Combobox(axis_frame, width=15)
        self.y_axis_combobox.pack(side=tk.LEFT, padx=2)
        
        # 操作按钮
        ttk.Button(
            axis_frame,
            text="Refresh Chart",
            command=self.update_chart
        ).pack(side=tk.LEFT, padx=5)
        
        # 右侧控制面板
        control_right = ttk.Frame(control_frame)
        control_right.pack(side=tk.RIGHT)
        
        # 保存按钮
        ttk.Button(
            control_right,
            text="Save Chart",
            command=self.save_chart
        ).pack(side=tk.LEFT, padx=5)
        
        # 平滑功能控件
        ttk.Checkbutton(
            control_right,
            text="Smooth Curve",
            variable=self.smooth_enabled,
            command=self.toggle_smooth_options
        ).pack(side=tk.LEFT, padx=5)
        
        self.smooth_frame = ttk.Frame(control_right)
        
        ttk.Label(self.smooth_frame, text="Window:").pack(side=tk.LEFT)
        ttk.Spinbox(
            self.smooth_frame,
            from_=3,
            to=21,
            increment=2,
            textvariable=self.window_length,
            width=5
        ).pack(side=tk.LEFT)
        
        ttk.Label(self.smooth_frame, text="Order:").pack(side=tk.LEFT)
        ttk.Spinbox(
            self.smooth_frame,
            from_=2,
            to=5,
            textvariable=self.polyorder,
            width=3
        ).pack(side=tk.LEFT)
        
        # 创建文件列表和图表区域
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左侧文件列表
        list_frame = ttk.LabelFrame(main_frame, text="Loaded Files", padding=5)
        list_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        self.file_listbox = tk.Listbox(list_frame, width=30, height=20)
        self.file_listbox.pack(fill=tk.BOTH, expand=True)
        
        # 文件列表按钮
        btn_frame = ttk.Frame(list_frame)
        btn_frame.pack(fill=tk.X, pady=2)
        
        ttk.Button(
            btn_frame,
            text="Rename Line",
            command=self.rename_selected_file
        ).pack(side=tk.LEFT, padx=1)
        
        ttk.Button(
            btn_frame,
            text="Remove Selected",
            command=self.remove_selected_file
        ).pack(side=tk.LEFT, padx=1)
        
        # 图表区域
        chart_frame = ttk.Frame(main_frame)
        chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.chart_frame = chart_frame
        
    def toggle_smooth_options(self):
        """控制平滑参数控件的显示"""
        if self.smooth_enabled.get():
            self.smooth_frame.pack(side=tk.LEFT, padx=5)
        else:
            self.smooth_frame.pack_forget()
    
    def apply_smoothing(self, y_data):
        """应用Savitzky-Golay平滑"""
        if not savgol_filter:
            messagebox.showerror("Error", "Please install scipy library: pip install scipy")
            return y_data
            
        window = self.window_length.get()
        polyorder = self.polyorder.get()
        
        # 确保窗口为奇数且小于数据长度
        window = min(len(y_data)//2*2-1, window)  # 最大可用奇数
        window = max(3, window)
        if window % 2 == 0:
            window += 1
            
        try:
            return savgol_filter(y_data, window, polyorder)
        except Exception as e:
            messagebox.showerror("Smoothing Error", str(e))
            return y_data
    
    def save_chart(self):
        """保存图表到文件"""
        if not hasattr(self, 'figure') or len(self.figure.axes) == 0:
            messagebox.showwarning("Warning", "Please generate a chart first")
            return
            
        filetypes = [
            ('PNG Image', '*.png'),
            ('JPEG Image', '*.jpg'),
            ('PDF Document', '*.pdf'),
            ('SVG Vector', '*.svg')
        ]
        
        path = filedialog.asksaveasfilename(
            filetypes=filetypes,
            defaultextension=".png"
        )
        
        if path:
            try:
                self.figure.savefig(path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Chart saved to: {path}")
            except Exception as e:
                messagebox.showerror("Save Failed", str(e))
    
    def init_chart(self):
        # 初始化图表
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def add_multiple_files(self):
        """添加多个JSON文件 - 支持同时选择多个文件"""
        file_paths = filedialog.askopenfilenames(
            filetypes=[("JSON files", "*.json")],
            title="Select JSON Files - Multiple Selection Supported"
        )
        
        if file_paths:
            loaded_count = 0
            error_files = []
            
            for file_path in file_paths:
                try:
                    if self.load_file_data(file_path):
                        filename = file_path.split('/')[-1]  # 获取文件名
                        
                        # 检查是否已经加载了该文件
                        if filename in self.file_data:
                            # 如果文件已存在，添加数字后缀
                            base_name = filename.replace('.json', '')
                            counter = 1
                            while f"{base_name}_{counter}.json" in self.file_data:
                                counter += 1
                            filename = f"{base_name}_{counter}.json"
                        
                        self.file_data[filename] = self.temp_data
                        
                        # 分配颜色
                        color = self.available_colors[self.color_index % len(self.available_colors)]
                        self.file_colors[filename] = color
                        self.color_index += 1
                        
                        # 设置默认标签为文件名
                        self.file_labels[filename] = filename
                        
                        loaded_count += 1
                except Exception as e:
                    error_files.append(f"{file_path}: {str(e)}")
            
            if loaded_count > 0:
                # 更新文件列表显示
                self.update_file_list()
                # 更新坐标轴选项
                self.update_axis_options()
                
                success_msg = f"{loaded_count} file(s) loaded successfully"
                if error_files:
                    success_msg += f"\n\nFailed files:\n" + "\n".join(error_files[:3])  # 最多显示3个错误
                
                messagebox.showinfo("Load Result", success_msg)
    
    def load_file_data(self, file_path):
        """加载单个文件的数据"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # 强制转换为字典格式
            if not isinstance(data, dict):
                data = {"data": [data]}
                
            # 转换所有值为列表
            processed_data = {}
            for k, v in data.items():
                if not isinstance(v, list):
                    processed_data[k] = [v]
                else:
                    processed_data[k] = v
                    
            self.temp_data = processed_data
            return True
        except Exception as e:
            return False
    
    def update_file_list(self):
        """更新文件列表显示"""
        self.file_listbox.delete(0, tk.END)
        for filename in self.file_data.keys():
            label = self.file_labels.get(filename, filename)
            color = self.file_colors.get(filename, 'black')
            self.file_listbox.insert(tk.END, f"{label} ({filename})")
            
            # 设置颜色（如果可能的话）
            try:
                index = list(self.file_data.keys()).index(filename)
                self.file_listbox.itemconfig(index, {'fg': color})
            except:
                pass
    
    def update_axis_options(self):
        """更新坐标轴选项"""
        if not self.file_data:
            return
            
        # 获取所有文件的共同字段
        all_keys = set()
        for data in self.file_data.values():
            all_keys.update(data.keys())
        
        if all_keys:
            keys = ['num'] + list(all_keys)
            self.x_axis_combobox['values'] = keys
            self.y_axis_combobox['values'] = list(all_keys)
            
            # 设置默认值
            if not self.x_axis_combobox.get():
                self.x_axis_combobox.set('num')
            if not self.y_axis_combobox.get() and all_keys:
                self.y_axis_combobox.set(next(iter(all_keys)))
    
    def rename_selected_file(self):
        """重命名选中的文件线条"""
        selection = self.file_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a file first")
            return
        
        index = selection[0]
        filename = list(self.file_data.keys())[index]
        
        new_label = simpledialog.askstring(
            "Rename Line",
            f"Enter new line name for '{filename}':",
            initialvalue=self.file_labels.get(filename, filename)
        )
        
        if new_label and new_label.strip():
            self.file_labels[filename] = new_label.strip()
            self.update_file_list()
            self.update_chart()
    
    def remove_selected_file(self):
        """移除选中的文件"""
        selection = self.file_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a file first")
            return
        
        index = selection[0]
        filename = list(self.file_data.keys())[index]
        
        if messagebox.askyesno("Confirm", f"Remove file '{filename}'?"):
            del self.file_data[filename]
            del self.file_labels[filename]
            del self.file_colors[filename]
            
            self.update_file_list()
            self.update_chart()
            
            if not self.file_data:
                self.x_axis_combobox['values'] = []
                self.y_axis_combobox['values'] = []
    
    def clear_all_files(self):
        """清空所有文件"""
        if not self.file_data:
            return
            
        if messagebox.askyesno("Confirm", "Clear all files?"):
            self.file_data.clear()
            self.file_labels.clear()
            self.file_colors.clear()
            self.color_index = 0
            
            self.file_listbox.delete(0, tk.END)
            self.x_axis_combobox['values'] = []
            self.y_axis_combobox['values'] = []
            
            self.update_chart()
    
    def validate_entries(self):
        """验证输入"""
        x_key = self.x_axis_combobox.get().strip()
        y_key = self.y_axis_combobox.get().strip()
        
        if not x_key or not y_key:
            return False
            
        if not self.file_data:
            return False
            
        # 检查至少有一个文件包含所需的字段
        for data in self.file_data.values():
            if y_key in data:
                if x_key == 'num' or x_key in data:
                    return True
        
        return False
    
    def process_file_data(self, filename, data, x_key, y_key):
        """处理单个文件的数据"""
        try:
            # 获取Y轴数据并过滤无效值
            raw_y = data[y_key]
            y_data = []
            for y in raw_y:
                try:
                    y_data.append(float(y))
                except (ValueError, TypeError):
                    continue
            
            # 生成X轴数据
            if x_key == 'num':
                x_data = list(range(len(y_data)))
            else:
                x_data = []
                for x in data[x_key]:
                    try:
                        x_data.append(float(x))
                    except (ValueError, TypeError):
                        continue
            
            # 自动对齐数据长度并限制X值小于10000
            min_length = min(len(x_data), len(y_data))
            filtered_x, filtered_y = [], []
            for i in range(min_length):
                if x_data[i] < 10000:
                    filtered_x.append(x_data[i])
                    filtered_y.append(y_data[i])
            
            if not filtered_x:
                return None
            
            return filtered_x, filtered_y
        except:
            return None
    
    def update_chart(self):
        """更新图表 - 确保显示所有曲线"""
        if not self.validate_entries():
            return
            
        x_key = self.x_axis_combobox.get().strip()
        y_key = self.y_axis_combobox.get().strip()
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        lines_plotted = 0
        
        # 调试信息
        print(f"Attempting to plot {len(self.file_data)} files...")
        print(f"X axis: {x_key}, Y axis: {y_key}")
        
        # 为每个文件绘制线条
        for filename, data in self.file_data.items():
            print(f"Processing file: {filename}")
            print(f"Available fields: {list(data.keys())}")
            
            if y_key not in data:
                print(f"Skipping {filename}: missing {y_key}")
                continue
                
            processed = self.process_file_data(filename, data, x_key, y_key)
            if not processed or not all(len(d) > 0 for d in processed):
                print(f"Skipping {filename}: no valid data")
                continue
                
            x_data, y_data = processed
            print(f"Plotting {filename}: {len(x_data)} data points")
            
            # 应用平滑
            if self.smooth_enabled.get():
                y_data = self.apply_smoothing(y_data)
            
            # 获取线条标签和颜色
            line_label = self.file_labels.get(filename, filename)
            color = self.file_colors.get(filename, 'steelblue')
            
            # 绘制线条
            ax.plot(
                x_data, y_data,
                linestyle='-',
                color=color,
                linewidth=1,
                label=line_label,
                marker='None',
                markersize=4,
                alpha=0.8
            )
            ax.set_yscale('log')
            lines_plotted += 1
        
        print(f"Total lines plotted: {lines_plotted}")
        
        if lines_plotted > 0:
            # 使用英文标题，去除空白字符
            chart_title = f"Multi-File Data Comparison - {y_key}"
            ax.set_title(chart_title, fontsize=14, pad=15)
            ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
            
            # 使用英文标签
            x_label = x_key if x_key != 'num' else 'Data Point Index'
            ax.set_xlabel(x_label, fontsize=12, labelpad=10)
            ax.set_ylabel(y_key, fontsize=12, labelpad=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            self.figure.tight_layout()
            self.canvas.draw()
        else:
            messagebox.showwarning("Warning", "No data to plot")


if __name__ == "__main__":
    root = tk.Tk()
    app = MultiCurveChartApp(root)
    root.mainloop()