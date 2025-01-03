import read_net as rn
from read_net import Net
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # 在Windows上使用PyCharm进行开发时，默认的交互式框架是Tkinter
import matplotlib.pyplot as plt

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU:-1; GPU0: 0; GPU1: 1;
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# 定义模型
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(1, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc4(x)
        return x

# 数据

Zs = []
Bs = []
for m in range(rn.Net.nb):
    for n in range(rn.Net.Bra[m].ns+1):
        # print('m=',m,'n=',n)
        Z = torch.tensor([ rn.Net.Bra[m].Sec[n].Lay[i].cote for i in range(Net.Bra[0].Sec[0].npas) ], dtype=torch.float32)
        B = torch.tensor([rn.Net.Bra[m].Sec[n].Lay[i].bt for i in range(Net.Bra[0].Sec[0].npas)], dtype=torch.float32)

        # 转换数据的形状以匹配模型输入
        Z = Z.view(-1, 1)
        B = B.view(-1, 1)

        Zs.append(Z)
        Bs.append(B)

# 初始化模型和优化器
models = [NN() for _ in range(26)]
optimizers = [optim.Adam(model.parameters(), lr=0.001) for model in models]
optimizers_LB = [torch.optim.LBFGS(model.parameters(), lr=0.001,
                                  max_iter=10000,
                                  max_eval=None,
                                  tolerance_grad=1e-6,
                                  tolerance_change=1e-11,
                                  history_size=100,
                                  line_search_fn='strong_wolfe') for model in models]

losses = [[] for _ in range(12)]
# 检查是否已经存在训练好的模型
if not os.path.exists('models_B'):
    os.makedirs('models_B')

for i in range(26):
    model_path = f'models_B/model_{i}.pth'

    # 如果模型已存在，则加载模型
    if os.path.exists(model_path):
        models[i] = torch.load(model_path)
    else:
        def closure():
            optimizers_LB[i].zero_grad()
            outputs = models[i](Zs[i])
            loss = nn.MSELoss()(outputs, Bs[i])
            losses.append(loss.item())
            loss.backward()
            return loss

        # 训练模型
        for epoch in range(15000):  # 迭代次数可以自己调整
            optimizers[i].zero_grad()
            outputs = models[i](Zs[i])
            loss = nn.MSELoss()(outputs, Bs[i])  # 我们使用均方误差作为损失函数
            loss.backward()
            optimizers[i].step()
            # 打印每个epoch的损失
            losses[i].append(loss.item())

        loss = optimizers_LB[i].step(closure)

        # 打印最终的损失
        print(f'Final Loss: {losses[i][-1]}')
        # 保存模型
        torch.save(models[i], model_path)

        # 使用模型预测
        Z_test = Zs[i]
        B_pred = models[i](Z_test)

        # 绘制真实值和预测值
        plt.figure()
        plt.plot(Zs[i].detach().numpy(), Bs[i].detach().numpy(), 'g', label='True value')
        plt.plot(Z_test.detach().numpy(), B_pred.detach().numpy(), 'r--', label='Predicted value')
        plt.xlabel('Z')
        plt.ylabel('B')
        plt.legend()
        plt.savefig(f'Figs/HP/Z_B_{i}.pdf', dpi=300)

        # 绘制损失下降图
        plt.figure()
        plt.plot(np.log10(losses[i]), 'b')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss during training')
        plt.savefig(f'Figs/HP/Z_B_loss_{i}.pdf', dpi=300)

print('finished')
