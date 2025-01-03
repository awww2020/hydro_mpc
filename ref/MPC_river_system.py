import numpy as np
from scipy.optimize import minimize

# system parameters
A = 1.0  # state transition matrix 状态矩阵
Bu = 1.0  # control input matrix  输入矩阵
Bd = 1.0  # disturbance matrix  扰动矩阵

# MPC parameters
N = 10  # prediction horizon
Q = 1.0  # state cost
R = 0.1  # control cost

# system model 通过这个模型来实现根据输入（流量）预测输出（水位）
def model(x, u, d):
    # x 状态变量（跟目标水位之间的差值） u是控制变量，闸门流量 d 扰动，可包括降雨和侧向流动
    return A * x + Bu * u + Bd * d

# cost function 损失函数
def cost(u, x0, x_ref, d):
    x = x0
    cost = 0.0
    for i in range(N):
        x = model(x, u[i], d[i])
        cost += Q * (x - x_ref)**2 + R * u[i]**2
    return cost

# initial state 初始状态
x = 0.0

# reference state 参考状态
x_ref = 1.0

# control sequence 控制序列
u = np.zeros(N)

# disturbance sequence (downstream water level) 扰动序列
d = np.random.normal(size=N)  # for example

# MPC loop
for i in range(100):
    # solve optimization problem
    res = minimize(cost, u, args=(x, x_ref, d))
    u = res.x

    # apply first control input to system
    x = model(x, u[0], d[0])

    # shift control sequence and disturbance sequence
    u = np.roll(u, -1)
    u[-1] = 0.0
    d = np.roll(d, -1)

    print("step", i, ":", "x =", x, "u =", u[0])

