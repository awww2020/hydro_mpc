import torch
import numpy as np

'''
我有四个坐标位置假设为0，1，2，3  每个坐标有个变量定义为Q，则有四个变量Q1,Q2,Q3,Q4,
我通过建立一个神经网络模型可得到Q2，Q3和Q1,Q4的关系，即输入Q1,Q4得到Q2，Q3。在实际边界Q1和Q4非恒定变化过程中，
我们只能实测得到Q3，现在我想用EKF去融合Q2、Q3的预测值和Q3的测量值，从而得到Q2和Q3的校正值，帮我写这个代码
'''


# 定义神经网络
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = torch.nn.Linear(2, 3)
        self.layer2 = torch.nn.Linear(3, 2)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.layer1(x))
        x = self.tanh(self.layer2(x))
        return x

# 初始化网络和优化器
net = NeuralNetwork()

def neural_network_jacobian(x):
    x.requires_grad_(True)
    y = net(x)
    jacobian = torch.autograd.functional.jacobian(lambda x: net(x), x)
    return y, jacobian[0]

# 初始化
Q_est = torch.tensor([initial_Q2, initial_Q3], dtype=torch.float32)  # 初始状态估计
P = torch.eye(2)  # 初始误差协方差矩阵
Q = torch.eye(2) * 0.01  # 过程噪声协方差矩阵
R = torch.tensor([[0.01]])  # 观测噪声协方差矩阵

# 预测步骤
Q1_current, Q4_current = get_current_Q1_Q4()  # 获取当前的Q1和Q4
inputs = torch.tensor([[Q1_current, Q4_current]], dtype=torch.float32)
Q_pred, F = neural_network_jacobian(inputs)
# Q_pred是神经网络对状态的预测，F是计算得到的雅可比矩阵，表示神经网络输出对于输入的敏感度。

Q_est = Q_pred.squeeze()  # 更新状态估计
P = F @ P @ F.t() + Q

# 更新步骤
H = torch.tensor([[0, 1]])  # 观测矩阵
actual_Q3 = get_actual_Q3()  # 获取实测的Q3
y = actual_Q3 - Q_est[1]  # 观测残差
S = H @ P @ H.t() + R
K = P @ H.t() @ torch.linalg.inv(S)
Q_est = Q_est + K @ y
P = (torch.eye(2) - K @ H) @ P

# 输出校正后的Q2和Q3
corrected_Q2, corrected_Q3 = Q_est
print(f"Corrected Q2: {corrected_Q2}, Corrected Q3: {corrected_Q3}")
