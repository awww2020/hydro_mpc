import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline


'''
将1200分钟的流量序列切成若干个t分钟的窗口，每个窗口之间可能有重叠。
学习特征并随机生成N组
'''

# 生成切片函数
def generate_slices(flow_data, slice_length):
    slices = []
    for i in range(len(flow_data) - slice_length + 1):
        slices.append(flow_data[i:i + slice_length])
    return slices

# 时间变形函数
def time_warping(slice_data, sigma=0.2, knot=4):
    orig_steps = np.arange(len(slice_data))
    random_points = np.random.normal(1, sigma, knot)
    random_points = np.clip(random_points, 0.1, 2)  # 保证变形幅度在合理范围
    spline = CubicSpline(np.linspace(0, len(slice_data)-1, knot), random_points)
    warping_steps = spline(orig_steps)
    warped_slice = np.interp(warping_steps, orig_steps, slice_data)
    return warped_slice

# 幅度变形函数
def magnitude_warping(slice_data, sigma=0.2, knot=4):
    random_points = np.random.normal(1, sigma, knot)
    spline = CubicSpline(np.linspace(0, len(slice_data)-1, knot), random_points)
    warping_factors = spline(np.arange(len(slice_data)))
    warped_slice = slice_data * warping_factors
    return warped_slice

# 抖动函数
def jittering(slice_data, sigma=0.03):
    noise = np.random.normal(0, sigma, len(slice_data))
    jittered_slice = slice_data + noise
    return jittered_slice

# 生成增强数据集
def generate_augmented_data(flow_data, slice_length, num_slices):
    slices = generate_slices(flow_data, slice_length)
    augmented_data = []
    for _ in range(num_slices):
        selected_slice = random.choice(slices)
        augmented_slice = random.choice([
            selected_slice,
            time_warping(selected_slice),
            magnitude_warping(selected_slice),
            jittering(selected_slice)
        ])
        augmented_data.append(augmented_slice)
    return augmented_data

# 示例流量数据（实际数据应为1200分钟的流量序列）
file_path = '上游流量.xlsx'
data = pd.read_excel(file_path)

# 提取时间和流量数据
time_data = data['时间（min）'].values
flow_data = data['流量'].values

# 确认时间数据是按升序排列的
if not np.all(np.diff(time_data) > 0):
    print("时间数据没有按升序排列。请检查数据。")
else:
    print("时间数据按升序排列。")

# 检查时间数据是否有重复值
if len(time_data) != len(np.unique(time_data)):
    print("时间数据包含重复值。请检查数据。")
else:
    print("时间数据没有重复值。")

# 创建时间插值函数
interp_function = interp1d(time_data, flow_data, kind='linear', fill_value="extrapolate")

# 定义新的等时间间隔
new_time_data = np.arange(0, 1201, 1)  # 从0到1200分钟，以1分钟为间隔

# 使用插值函数计算新时间点的流量数据
new_flow_data = interp_function(new_time_data)

# 可视化
plt.plot(new_flow_data)
plt.xlabel('Time (minute)')
plt.ylabel('Flow (m³/s)')
plt.title('new_time_data')
plt.show()

# 生成增强数据集
t = 6  # 定义t时间间隔数
N = 36  # 定义要生成的随机数据组数
augmented_data = generate_augmented_data(new_flow_data, slice_length=t, num_slices=N)

# 打印部分生成的数据并可视化
for i, aug_slice in enumerate(augmented_data[:36]):  # 显示前5组增强数据
    # plt.plot(aug_slice, label=f'Slice {i+1}')
    plt.plot(aug_slice)
plt.xlabel('Time (minute)')
plt.ylabel('Flow (m³/s)')
plt.title('Augmented Data (Flow Rate)')
plt.legend()
plt.show()