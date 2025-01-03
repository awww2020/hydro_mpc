import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

# 读取并插值上游流量数据
file_path = '上游流量.xlsx'
data = pd.read_excel(file_path)

time_data = data['时间（min）'].values
flow_data = data['流量'].values

# 创建时间插值函数
interp_function = interp1d(time_data, flow_data, kind='linear', fill_value="extrapolate")

# 定义新的等时间间隔
new_time_data = np.arange(0, 1201, 1)  # 从0到1200分钟，以1分钟为间隔

# 使用插值函数计算新时间点的流量数据
upstream_flow_data = interp_function(new_time_data)

# 可视化上游流量数据
plt.plot(new_time_data, upstream_flow_data)
plt.xlabel('Time (minute)')
plt.ylabel('Flow (m³/s)')
plt.title('Interpolated Upstream Flow Data')
plt.show()


# 生成下游流量数据
def generate_downstream_flow(length, base_flow=3.0, variation=1.0, sigma=0.2):
    downstream_flow = np.ones(length) * base_flow
    control_points = np.linspace(0, length - 1, num=10)  # 控制点数量，可以调整
    for i in range(1, len(control_points)):
        change = np.random.uniform(-variation, variation)
        downstream_flow[int(control_points[i - 1]):int(control_points[i])] += change

    # 加入随机扰动
    noise = np.random.normal(0, sigma, length)
    downstream_flow += noise
    downstream_flow = np.clip(downstream_flow, 0, None)  # 确保流量为正值
    return downstream_flow


# 生成多种下游流量模式
def generate_varied_downstream_flows(length, num_patterns=5):
    patterns = []
    for _ in range(num_patterns):
        base_flow = np.random.uniform(2.0, 4.0)  # 基础流量在2.0到4.0之间随机变化
        variation = np.random.uniform(0.5, 2.0)  # 变化幅度在0.5到2.0之间随机变化
        sigma = np.random.uniform(0.1, 0.5)  # 噪声标准差在0.1到0.5之间随机变化
        patterns.append(generate_downstream_flow(length, base_flow, variation, sigma))
    return patterns


# 生成多个下游流量过程
num_downstream_flows = 100  # 生成100组下游流量过程
downstream_flows = [flow for _ in range(num_downstream_flows) for flow in
                    generate_varied_downstream_flows(len(new_time_data))]


# 生成初始水位条件
def generate_initial_water_levels(length, num_levels=10):
    levels = []
    for _ in range(num_levels):
        base_level = np.random.uniform(1.0, 3.0)  # 基础水位在1.0到3.0之间随机变化
        sigma = np.random.uniform(0.2, 0.8)  # 噪声标准差在0.2到0.8之间随机变化
        levels.append(np.clip(np.random.normal(base_level, sigma, length), 0, None))
    return levels


# 生成初始水位数据
num_initial_levels = 100  # 生成100组初始水位
initial_levels = [level for _ in range(num_initial_levels) for level in
                  generate_initial_water_levels(len(new_time_data))]


# 生成训练样本
def generate_training_samples(upstream_flow, downstream_flows, initial_levels, slice_length=10):
    training_samples = []
    for i in range(0, len(upstream_flow) - slice_length, slice_length):
        for downstream_flow in downstream_flows:
            for initial_level in initial_levels:
                sample = {
                    'upstream_flow': upstream_flow[i:i + slice_length],
                    'downstream_flow': downstream_flow[i:i + slice_length],
                    'initial_level': initial_level[i],
                    'hydrodynamic_process': compute_hydrodynamic_process(upstream_flow[i:i + slice_length],
                                                                         downstream_flow[i:i + slice_length],
                                                                         initial_level[i])  # 计算水动力过程的函数
                }
                training_samples.append(sample)
    return training_samples


# 示例计算水动力过程的函数
def compute_hydrodynamic_process(upstream_flow, downstream_flow, initial_level):
    # 这里只是一个示例，实际计算水动力过程可能需要使用专业的水力学模型
    process = upstream_flow + downstream_flow + initial_level  # 简单示例
    return process


# 生成训练样本
training_samples = generate_training_samples(upstream_flow_data, downstream_flows, initial_levels, slice_length=10)

# 将训练样本转换为适合模型训练的格式
X = []
y = []
for sample in training_samples:
    X.append([sample['upstream_flow'], sample['downstream_flow'], sample['initial_level']])
    y.append(sample['hydrodynamic_process'])

X = np.array(X)
y = np.array(y)

# 训练模型（例如使用神经网络）
# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练神经网络模型
model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000)
model.fit(X_train, y_train)

# 评估模型性能
score = model.score(X_test, y_test)
print(f'Model R^2 score: {score}')

# 示例预测
sample_index = 0
predicted_process = model.predict([X_test[sample_index]])
actual_process = y_test[sample_index]

plt.plot(predicted_process.flatten(), label='Predicted')
plt.plot(actual_process.flatten(), label='Actual')
plt.xlabel('Time Step')
plt.ylabel('Hydrodynamic Process')
plt.title('Predicted vs Actual Hydrodynamic Process')
plt.legend()
plt.show()
