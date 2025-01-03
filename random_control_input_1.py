import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

# 读取实际数据文件
file_path = '上游流量.xlsx'
data = pd.read_excel(file_path)

# 提取时间和流量数据
time_data = data['时间（min）'].values
flow_data = data['流量'].values

# 创建时间插值函数
interp_function = interp1d(time_data, flow_data, kind='linear', fill_value="extrapolate")

# 定义新的等时间间隔
new_time_data = np.arange(0, time_data[-1], 1)  # 以1分钟为间隔

# 进行插值
interpolated_flow_data = interp_function(new_time_data)

# 绘制插值后的流量数据
plt.figure(figsize=(10, 4))
plt.plot(time_data, flow_data, 'o', label='Original Data')
plt.plot(new_time_data, interpolated_flow_data, '-', label='Interpolated Data')
plt.xlabel('Time (minute)')
plt.ylabel('Flow (m³/s)')
plt.title('Interpolated Flow Data')
plt.legend()
plt.show()

# 查找突发变化点
peaks, _ = find_peaks(interpolated_flow_data, prominence=0.5)
troughs, _ = find_peaks(-interpolated_flow_data, prominence=0.5)

# 计算突发变化的持续时间
peak_durations = np.diff(peaks)  # 突发变化的持续时间
trough_durations = np.diff(troughs)

# 打印突发变化的持续时间
print("Peak durations (minutes):", peak_durations)
print("Trough durations (minutes):", trough_durations)

# 计算平均突发变化持续时间
average_change_duration = np.mean(np.concatenate([peak_durations, trough_durations]))
print("Average change duration (minutes):", average_change_duration)

# 计算突发变化次数
num_changes = len(peaks) + len(troughs)
print("Number of changes:", num_changes)



# 计算基本统计特征
mean_flow = np.mean(interpolated_flow_data)
std_flow = np.std(interpolated_flow_data)
max_flow = np.max(interpolated_flow_data)
min_flow = np.min(interpolated_flow_data)

# 打印提取的特征
print(f"Mean Flow: {mean_flow}")
print(f"Standard Deviation of Flow: {std_flow}")
print(f"Max Flow: {max_flow}")
print(f"Min Flow: {min_flow}")


# 傅里叶变换分析流量数据的频率成分
N = len(interpolated_flow_data)
T = 1.0  # 采样间隔为1，时间单位为分钟

yf = fft(interpolated_flow_data)
xf = fftfreq(N, T)[:N//2]

# 绘制频率成分
plt.figure(figsize=(10, 4))
plt.plot(xf, 2.0/N * np.abs(yf[:N//2]))
plt.grid()
plt.xlabel('Frequency (1/minute)')
plt.ylabel('Amplitude')
plt.title('Frequency Components of Flow Data')
plt.show()

# 找到主导频率
dominant_freq = xf[np.argmax(2.0/N * np.abs(yf[:N//2]))]
print(f"Dominant Frequency: {dominant_freq}")


# 定义生成控制输入的函数
def generate_base_signal(duration, mean, std, dominant_freq):
    t = np.arange(duration)
    base_signal = mean + std * np.sin(2 * np.pi * dominant_freq * t)
    return base_signal


def add_random_disturbances(base_signal, noise_std, max_change_per_minute):
    noise = np.random.normal(0, noise_std, len(base_signal))
    disturbed_signal = base_signal + noise

    # 限制每分钟的最大变化幅度
    for i in range(1, len(disturbed_signal)):
        change = disturbed_signal[i] - disturbed_signal[i - 1]
        if abs(change) > max_change_per_minute:
            disturbed_signal[i] = disturbed_signal[i - 1] + np.sign(change) * max_change_per_minute

    return disturbed_signal


def add_sudden_changes(signal, num_changes, change_magnitude, change_duration):
    if change_duration >= len(signal):
        print("Change duration is too long, skipping sudden changes.")
        return signal

    for _ in range(num_changes):
        change_start = np.random.randint(0, len(signal) - change_duration)
        change = change_magnitude * (2 * np.random.rand() - 1)
        for i in range(change_duration):
            signal[change_start + i] += change / change_duration
    return signal


def apply_physical_constraints(signal, u_min, u_max):
    return np.clip(signal, u_min, u_max)


def generate_control_inputs(duration, mean, std, noise_std, u_min, u_max, change_magnitude, change_duration,
                            dominant_freq, max_change_per_minute):
    # Step 1: Generate base signal
    base_signal = generate_base_signal(duration, mean, std, dominant_freq)

    # Step 2: Add random disturbances
    disturbed_signal = add_random_disturbances(base_signal, noise_std, max_change_per_minute)

    # Step 3: Add sudden changes
    signal_with_changes = add_sudden_changes(disturbed_signal, num_changes, change_magnitude=change_magnitude,
                                             change_duration=change_duration)

    # Step 4: Apply physical constraints
    constrained_signal = apply_physical_constraints(signal_with_changes, u_min, u_max)

    return constrained_signal


# 参数设置
duration = 20  # 10分钟
u_min, u_max = min_flow, max_flow  # 控制输入范围
noise_std = std_flow  # 噪声标准差
change_magnitude = std_flow * 0.2  # 突发变化幅度
change_duration = int(average_change_duration)  # 突发变化持续时间
max_change_per_minute = std_flow * 0.1  # 每分钟最大变化幅度
dominant_freq = 1 / np.mean(peak_durations)  # 主导频率

# 生成10分钟的控制输入
control_inputs = generate_control_inputs(duration, mean_flow, std_flow, noise_std, u_min, u_max, change_magnitude, change_duration, dominant_freq, max_change_per_minute)

# 可视化
plt.plot(control_inputs)
plt.xlabel('Time (minute)')
plt.ylabel('Flow (m³/s)')
plt.title('Generated 10-minute Control Input (Flow Rate) with Realistic Variations')
plt.show()



