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
As = []
Zs_mean = []
Zs_std = []

for m in range(rn.Net.nb):
    for n in range(rn.Net.Bra[m].ns+1):
        # print('m=',m,'n=',n)
        Z = torch.tensor([ rn.Net.Bra[m].Sec[n].Lay[i].cote for i in range(Net.Bra[0].Sec[0].npas) ], dtype=torch.float32)
        A = torch.tensor([ rn.Net.Bra[m].Sec[n].Lay[i].st for i in range(Net.Bra[0].Sec[0].npas) ], dtype=torch.float32)

        # 转换数据的形状以匹配模型输入
        Z = Z.view(-1, 1)
        A = A.view(-1, 1)

        Zs.append(Z)
        As.append(A)

        mean = torch.mean(Z)
        std = torch.std(Z)

        Zs_mean.append(mean)
        Zs_std.append(std)

# 初始化模型和优化器
models = [NN() for _ in range(26)]
#print('Zs[0]',Zs[0])
#print('As[0]',As[0])
optimizers = [optim.Adam(model.parameters(), lr=0.001) for model in models]
optimizers_LB = [torch.optim.LBFGS(model.parameters(), lr=0.001,
                                  max_iter=10000,
                                  max_eval=None,
                                  tolerance_grad=1e-6,
                                  tolerance_change=1e-11,
                                  history_size=100,
                                  line_search_fn='strong_wolfe') for model in models]

for i in range(26):
    torch.save(Zs_mean[i], f"models_A/Zs_mean_{i}.pth")
    torch.save(Zs_std[i], f"models_A/Zs_std_{i}.pth")

losses = [[] for _ in range(26)]

# 检查是否已经存在训练好的模型
if not os.path.exists('models_A'):
    os.makedirs('models_A')

loaded_Zs_mean = [None] * 26
loaded_Zs_std = [None] * 26

for i in range(26):
    model_path = f'models_A/model_{i}.pth'
    Zs_mean_path = f'models_A/Zs_mean_{i}.pth'
    Zs_std_path = f'models_A/Zs_std_{i}.pth'
    print('i',i)
    # 如果模型已存在，则加载模型
    # 如果 Zs_mean 已存在，则加载 Zs_mean
    if os.path.exists(Zs_mean_path):
        loaded_Zs_mean[i] = torch.load(Zs_mean_path)
        loaded_Zs_std[i] = torch.load(Zs_std_path)

    if os.path.exists(model_path):
        models[i] = torch.load(model_path)
        '''
        if i == 0 :
            print(loaded_Zs_mean[i])
            print(loaded_Zs_std[i])
            normalized_input = (torch.tensor(10.2232) - loaded_Zs_mean[i] )/ loaded_Zs_std[i]
            normalized_input = normalized_input.view(1, 1).to(dtype=torch.float32)
            print('As[0]',models[0](normalized_input))
        这边后面没有归一化！所以也不需要
        '''
    else:
        def closure():
            optimizers_LB[i].zero_grad()
            outputs = models[i](Zs[i])
            loss = nn.MSELoss()(outputs, As[i])
            losses.append(loss.item())
            loss.backward()
            return loss

        # 训练模型
        for epoch in range(10000):  # 迭代次数可以自己调整
            optimizers[i].zero_grad()
            outputs = models[i](Zs[i])
            loss = nn.MSELoss()(outputs, As[i])  # 我们使用均方误差作为损失函数
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
        A_pred = models[i](Z_test)

        # 绘制真实值和预测值
        plt.figure()
        plt.plot(Zs[i].detach().numpy(), As[i].detach().numpy(), 'g', label='True value')
        plt.plot(Z_test.detach().numpy(), A_pred.detach().numpy(), 'r--', label='Predicted value')
        plt.xlabel('Z')
        plt.ylabel('A')
        plt.legend()
        plt.savefig(f'Figs/HP/Z_A_{i}.pdf', dpi=300)

        # 绘制损失下降图
        plt.figure()
        plt.plot(np.log10(losses[i]), 'b')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss during training')
        plt.savefig(f'Figs/HP/Z_A_loss_{i}.pdf', dpi=300)

print('finished')
