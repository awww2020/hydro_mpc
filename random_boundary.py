import numpy as np
from scipy.integrate import solve_ivp
import geo_pre_cal.read_net as rn

# 定义明渠参数
length = 4000.0  # 渠道长度 (m)
width = 10.0  # 渠道宽度 (m)
slope = 0.001  # 底坡
roughness = 0.03  # 曼宁糙率系数

# 生成动态边界条件
def generate_boundary_conditions():
    # 随机选择变化类型
    # change_type = np.random.choice(['linear', 'pulse', 'sinusoidal'])
    change_type = 'linear'
    # 基础参数
    Q_base = np.random.uniform(1.0, 5.0)
    h_base = np.random.uniform(1.0, 3.0)
    t = np.linspace(0, 300, 4)  # 3分钟，一分钟一个点

    # 生成流量和水位序列
    if change_type == 'linear':
        k_Q = np.random.uniform(-0.05/60, 0.05/60)
        k_h = np.random.uniform(-0.005, 0.005)
        Q_up = Q_base + k_Q * t
        h_down = h_base + k_h * t
    '''
    elif change_type == 'pulse':
        Q_up = Q_base + 2.0 * (t > 150)  # 在150秒时突增
        h_down = h_base * np.ones_like(t)
    elif change_type == 'sinusoidal':
        Q_up = Q_base + 1.0 * np.sin(0.1 * t)
        h_down = h_base + 0.5 * np.cos(0.05 * t)
    '''
    return Q_up, h_down, t


def simulate_channel(Q_up, h_down, t):
    # 河网水动力模拟主函数
    network = rn.Net

    # 初始化
    dt = 60  # 60秒时间步长
    h = []
    u = []

    for i in range(len(t)):
        # 设置边界条件
        boundaries = {
            'upstream_Q': Q_up[i],
            'downstream_Z': h_down[i]
        }

        # 求解当前时刻
        z = network.solve_time_step(dt, boundaries)

        h.append(z.mean())  # 示例提取结果
        u.append(z.std())

    return np.array(h), np.array(u)


# 生成并保存训练数据
num_scenarios = 5 # 生成1000组数据
dataset = []

for i in range(num_scenarios):
    Q_up, h_down, t = generate_boundary_conditions()
    print(i,'Q_up,h_down',Q_up, h_down)
    h, u = simulate_channel(Q_up, h_down, t)
