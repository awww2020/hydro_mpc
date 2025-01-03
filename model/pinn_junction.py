#!/usr/bin/env python2
#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
# from pyDOE import lhs         #Latin Hypercube Sampling
import time
import torch
import torch.autograd as autograd         # computation graph
from torch import Tensor                  # tensor node in the computation graph
import torch.nn as nn                     # neural networks
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Set default dtype to float32
torch.set_default_dtype(torch.float)

# PyTorch random number generator
torch.manual_seed(24)
import os
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

class HydroNet_1D(nn.Module):
    # Neural Network
    def __init__(self, layers_U, layers_Z, device, f_hat, Net, lb, ub,  t_c, N_c):
        super().__init__()  # call __init__ from parent class
        'activation function'
        self.activation = nn.Tanh()
        # self.activation = nn.ReLU()
        'loss function'
        self.loss_function = nn.MSELoss(reduction ='mean') # 平方求平均
        'Initialise neural network as a list using nn.Modulelist'
        self.layers_U = layers_U
        self.layers_Z = layers_Z

        self.linears_U = nn.ModuleList([nn.Linear(layers_U[i], layers_U[i + 1]) for i in range(len(layers_U) - 1)])
        self.linears_Z = nn.ModuleList([nn.Linear(layers_Z[i], layers_Z[i + 1]) for i in range(len(layers_Z) - 1)])

        self.bns_U = nn.ModuleList([nn.BatchNorm1d(layers_U[i + 1]) for i in range(len(layers_U) - 3)])
        self.bns_Z = nn.ModuleList([nn.BatchNorm1d(layers_Z[i + 1]) for i in range(len(layers_Z) - 3)])

        self.iter = -1  # For the Optimizer

        'Xavier Normal Initialization'
        for i in range(len(layers_U) - 1):
            nn.init.xavier_normal_(self.linears_U[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linears_U[i].bias.data)

        for i in range(len(layers_Z)-1):
            nn.init.xavier_normal_(self.linears_Z[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linears_Z[i].bias.data)

        self.device = device
        self.f_hat = f_hat

        self.Net = Net

        self.loss_u_train_log = []
        self.loss_z_train_log = []
        self.loss_pde_train_log = []
        self.loss_val_train_log = []
        self.weight_l_u_log = []
        self.weight_l_z_log = []

        self.alpha = 0.1
        self.weight_l_u = 1.0
        self.weight_l_z = 1.0
        self.weight_l_pde = 1.0

        # 梯度存储(i层所有)
        self.grad_l_u = []
        self.grad_l_z = []
        self.grad_l_pde1 = []
        self.grad_l_pde2 = []

        self.adaptive_u_list = []
        self.adaptive_z_list = []

        self.method = '1'

        self.lb = torch.from_numpy(lb).to(self.device)
        self.ub = torch.from_numpy(ub).to(self.device)
        self.t_c = t_c
        self.N_c = N_c



        self.models_A = [NN() for _ in range(26)]
        self.models_B = [NN() for _ in range(26)]
        self.models_K = [NN() for _ in range(26)]

    'foward pass'
    def forward_U(self, x):
        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x)   # Creates a Tensor from a numpy.ndarray
        a = x.float()         # 将对象转换成一个浮点型数据
        # 加batch normalization
        for i in range(len(self.layers_U) - 2):
            if i == 0:
                b = self.linears_U[i](a)
                a = self.activation(b)
            else:
                b = self.linears_U[i](a)
                b = self.bns_U[i - 1](b)
                a = self.activation(b)
        a = self.linears_U[-1](a)

        return a

    def forward_Z(self, x):
        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x)   # Creates a Tensor from a numpy.ndarray
        a = x.float()         # 将对象转换成一个浮点型数据
        for i in range(len(self.layers_Z)-2):
            if i == 0:
                b = self.linears_Z[i](a)
                a = self.activation(b)
            else:
                b = self.linears_Z[i](a)
                b = self.bns_Z[i-1](b)
                a = self.activation(b)
        a = self.linears_Z[-1](a)
        return a

    'Loss Functions'
    # Loss BC
    def lossU(self, x_U, y_U):
        #print('x_U.shape',x_U.shape)
        #print('y_U.shape',y_U.shape)
        loss_U = self.loss_function(self.forward_U(x_U), y_U)
        return loss_U

    def lossZ(self, x_Z, y_Z):
        loss_Z = self.loss_function(self.forward_Z(x_Z), y_Z)
        return loss_Z
    '''
    def lossBC(self, x_BC, y_BC):
        loss_BC = self.loss_function(self.forward(x_BC), y_BC)
        return loss_BC

    def lossIC(self, x_IC, y_IC):
        loss_IC = self.loss_function(self.forward(x_IC), y_IC)
        return loss_IC
    '''
    # Loss PDE
    def lossPDE(self, x_PDE):
        '''
        :param x_PDE: 方程控制点的坐标
        :return:
        '''
        # 这边要特别注意，传进来的X_Pde应该是没有正则化之前的，仅用正则化后的坐标去得到Z,Q的值，但是求导还是要用最初的值
        # 最初的坐标

        x_clone = x_PDE.clone()
        x_clone.requires_grad = True # Enable differentiation
        # 标准化输入
        X_clone_n = 2 *(x_clone - self.lb) /(self.ub-self.lb) - 1
        # X_clone[:, 0:1] = 0.0
        # print('X_clone',X_clone)

        Z = self.forward_Z(X_clone_n)
        U = self.forward_U(X_clone_n)
        g = 9.81
        Z_REF_list = []
        for i in range(len(self.Net.Bra)):
            for j in range(self.Net.Bra[i].ns + 1):
                Z_REF_i = torch.tensor([[self.Net.Bra[i].Sec[j].zref + 0.01]]).to(self.device)
                Z_REF_list.append(Z_REF_i)

        Z_REF = torch.cat(Z_REF_list, dim=0)

        Z_REF = Z_REF.repeat(self.t_c, 1)
        Z_MAX = Z_REF + 10
        Z1 = torch.where(Z < Z_REF, Z_REF, Z)
        Z2 = torch.where(Z1 < Z_MAX, Z1, Z_MAX)

        # 面积，输运系数，河宽
        '''
        A = np.zeros((self.t * self.N_c, 1))
        K = np.zeros((self.t * self.N_c, 1))
        B = np.zeros((self.t * self.N_c, 1))
        for i in range(0, self.t * self.N_c , self.N_c):
        #    A[i:i+self.N_c], K[i:i+self.N_c], B[i:i+self.N_c] = self.pro_cal(Z2[i:i+self.N_c])
            for j in range(26):
                A[i:i+self.N_c] = [self.models_A[j](z) for z in Z2[i:i+self.N_c]]
                B[i:i+self.N_c] = [self.models_B[j](z) for z in Z2[i:i+self.N_c]]
                K[i:i+self.N_c] = [self.models_K[j](z) for z in Z2[i:i+self.N_c]]
        A = torch.from_numpy(A).to(torch.float32).to(self.device)
        K = torch.from_numpy(K).to(torch.float32).to(self.device)
        B = torch.from_numpy(B).to(torch.float32).to(self.device)
        '''

        A = torch.zeros((self.t_c * self.N_c, 1), device=self.device)
        B = torch.zeros_like(A)
        K = torch.zeros_like(A)
        '''
        K = torch.zeros((self.t * self.N_c, 1), device=self.device)
        B = torch.zeros((self.t * self.N_c, 1), device=self.device)
        '''

        for i in range(0, self.t_c * self.N_c, self.N_c):
            # 取出子块
            z_chunk = Z2[i:i + self.N_c]

            # .item() 方法用于从单值张量中获取Python数值
            A_values = [model(z_chunk[j]).item() for j, model in enumerate(self.models_A)]
            B_values = [model(z_chunk[j]).item() for j, model in enumerate(self.models_B)]
            K_values = [model(z_chunk[j]).item() for j, model in enumerate(self.models_K)]

            # 将上面获得的输出值转换回PyTorch张量，并确保它们具有正确的形状（列向量）

            A_chunk = torch.tensor(A_values, device=self.device).view(-1, 1)
            B_chunk = torch.tensor(B_values, device=self.device).view(-1, 1)
            K_chunk = torch.tensor(K_values, device=self.device).view(-1, 1)

            A[i:i + self.N_c] = A_chunk
            B[i:i + self.N_c] = B_chunk
            K[i:i + self.N_c] = K_chunk

        '''
        for i in range(0, self.t * self.N_c, self.N_c):
            for j in range(self.N_c): # 26
                A[i:i + self.N_c] = torch.tensor([self.models_A[j](z) for z in Z2[i:i + self.N_c]],
                                                 device=self.device).view(-1, 1)
                B[i:i + self.N_c] = torch.tensor([self.models_B[j](z) for z in Z2[i:i + self.N_c]],
                                                 device=self.device).view(-1, 1)
                K[i:i + self.N_c] = torch.tensor([self.models_K[j](z) for z in Z2[i:i + self.N_c]],
                                                 device=self.device).view(-1, 1)
        '''

        Q = A * U

        Z_x = autograd.grad(Z, x_clone, torch.ones([x_clone.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True)[0] # first derivative
        #Z_xx = autograd.grad(Z_x, x_clone, torch.ones(x_clone.shape).to(self.device), retain_graph=True, create_graph=True)[0] # second derivative
        U_x = autograd.grad(U, x_clone, torch.ones([x_clone.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True)[0]
        Q_x = autograd.grad(Q, x_clone, torch.ones([x_clone.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True)[0]
        QU_x = autograd.grad(Q*U, x_clone, torch.ones([x_clone.shape[0], 1]).to(self.device), retain_graph=True,
                             create_graph=True)[0]
        # Q2_A_x = autograd.grad(Q2/A, x_clone, torch.ones([x_clone.shape[0], 1]).to(self.device), retain_graph=True,
        #                      create_graph=True)[0]

        Z_x_x2 = Z_x[:, [1]]
        U_x_x2 = U_x[:, [1]]
        Q_x_x2 = Q_x[:, [1]]
        QU_x_x2 = QU_x[:, [1]]

        Z_x_x3 = Z_x[:, [2]]
        Q_x_x3 = Q_x[:, [2]]

        B_split = torch.chunk(B, self.t_c, dim=0)
        Z_x_x3_split = torch.chunk(Z_x_x3, self.t_c, dim=0)
        Q_x_x2_split = torch.chunk(Q_x_x2, self.t_c, dim=0)
        Q_x_x3_split = torch.chunk(Q_x_x3, self.t_c, dim=0)
        QU_x_x2_split = torch.chunk(QU_x_x2, self.t_c, dim=0)
        Z_x_x2_split = torch.chunk(Z_x_x2, self.t_c, dim=0)
        Q_split = torch.chunk(Q, self.t_c, dim=0)
        K_split = torch.chunk(K, self.t_c, dim=0)
        A_split = torch.chunk(A, self.t_c, dim=0)
        # 节点增加代码
        Z2_split = torch.chunk(A, self.t_c, dim=0)
        self.f_hat_split = torch.chunk(self.f_hat, self.t_c, dim=0)

        total_loss = 0
        total_loss_eq = 0

        for i in range(self.t_c):  # Assuming there are 4 time steps
            # 河网分段的标识
            c = [[0, 5], [5, 13], [13, 22], [22, 26]]
            # 遍历分支
            for j in range(len(self.Net.Bra)):
                # 从 B_split[i] 中提取一个子集，该子集的行范围是从 c[j][0] 到 c[j][1]
                B_sub = B_split[i][(c[j][0]):(c[j][1]), :]
                Z_x_x3_sub = Z_x_x3_split[i][(c[j][0]):(c[j][1]), :]
                Q_x_x2_sub = Q_x_x2_split[i][(c[j][0]):(c[j][1]), :]
                Q_x_x3_sub = Q_x_x3_split[i][(c[j][0]):(c[j][1]), :]
                QU_x_x2_sub = QU_x_x2_split[i][(c[j][0]):(c[j][1]), :]
                Z_x_x2_sub = Z_x_x2_split[i][(c[j][0]):(c[j][1]), :]
                Q_sub = Q_split[i][(c[j][0]):(c[j][1]), :]
                K_sub = K_split[i][(c[j][0]):(c[j][1]), :]
                A_sub = A_split[i][(c[j][0]):(c[j][1]), :]
                self.f_hat_sub = self.f_hat_split[i][(c[j][0]):(c[j][1]), :]

                f_c = B_sub * Z_x_x3_sub + Q_x_x2_sub
                f_m = Q_x_x3_sub + QU_x_x2_sub + g * A_sub * (
                            Z_x_x2_sub + abs(Q_sub) * Q_sub / (K_sub ** 2))


                loss_pde = self.loss_function(f_c, self.f_hat_sub) + self.loss_function(f_m, self.f_hat_sub)
                total_loss += loss_pde

        for i in range(self.t_c):  # Assuming there are 4 time steps
            # 河网分段的标识
            c = [[0, 5], [5, 13], [13, 22], [22, 26]]
            # 节点增加代码,这边考虑流量守恒
            loss_eq = Q_split[i][(c[0][1] - 1):(c[0][1]), :] \
                      - Q_split[i][(c[1][0]):(c[1][0] + 1), :] \
                      - Q_split[i][(c[2][0]):(c[2][0] + 1), :] \
                      + Q_split[i][(c[1][1] - 1):(c[1][1]), :] \
                      + Q_split[i][(c[2][1] - 1):(c[2][1]), :] \
                      - Q_split[i][(c[3][0]):(c[3][0] + 1), :]
            total_loss_eq += self.loss_function(loss_eq, torch.zeros((1, 1)))
            total_loss = total_loss + total_loss_eq

        self.iter = self.iter + 1.

        return total_loss

    def pro_cal(self, Z):
        s_list = []
        k_list = []
        b_list = []

        def binary_search(cotes, value):
            low, high = 0, len(cotes) - 1
            if value >= cotes[-1]:
                return len(cotes) - 1

            while low <= high:
                mid = (low + high) // 2
                if cotes[mid] <= value < cotes[mid + 1]:
                    return mid
                elif cotes[mid] < value:
                    low = mid + 1
                else:
                    high = mid - 1
            return -1

        # 二分法寻找
        for i_bra in range(len(self.Net.Bra)):
            s = np.zeros((self.Net.Bra[i_bra].ns + 1, 1))
            k = np.zeros((self.Net.Bra[i_bra].ns + 1, 1))
            b = np.zeros((self.Net.Bra[i_bra].ns + 1, 1))

            for i in range((self.Net.Bra[i_bra].ns + 1)):
                cotes = [self.Net.Bra[i_bra].Sec[i].zref + j * self.Net.Bra[i_bra].Sec[i].pas for j in
                         range(self.Net.Bra[i_bra].Sec[i].npas)]

                j = binary_search(cotes, Z[i][0])
                if j == -1:
                    s[i][0] = 0
                    k[i][0] = 0
                    b[i][0] = 0

                elif j == len(cotes) - 1 or pd.isnull(Z[i][0]):

                    sw = self.Net.Bra[i_bra].Sec[i].Lay[j].st
                    bw = self.Net.Bra[i_bra].Sec[i].Lay[j].bt
                    s1w = self.Net.Bra[i_bra].Sec[i].Lay[j].s1
                    s2lw = self.Net.Bra[i_bra].Sec[i].Lay[j].s2l
                    s2rw = self.Net.Bra[i_bra].Sec[i].Lay[j].s2r

                    p1w = self.Net.Bra[i_bra].Sec[i].Lay[j].p1
                    p2lw = self.Net.Bra[i_bra].Sec[i].Lay[j].p2l
                    p2rw = self.Net.Bra[i_bra].Sec[i].Lay[j].p2r

                    # k=1/n*R^(2/3)*A
                    kw = s1w * (s1w / p1w) ** (2. / 3) / self.Net.Bra[i_bra].Sec[i].nc + s2lw * (s2lw / p2lw) ** (2. / 3) / \
                         self.Net.Bra[i_bra].Sec[i].nf + s2rw * (s2rw / p2rw) ** (2. / 3) / self.Net.Bra[i_bra].Sec[i].nf
                    s[i][0] = sw
                    k[i][0] = kw
                    b[i][0] = bw
                else:
                    delta = Z[i][0] - cotes[j]
                    sw = self.Net.Bra[i_bra].Sec[i].Lay[j].st + delta / self.Net.Bra[i_bra].Sec[i].pas * (
                                self.Net.Bra[i_bra].Sec[i].Lay[j + 1].st - self.Net.Bra[i_bra].Sec[i].Lay[j].st)  # 断面面积
                    bw = self.Net.Bra[i_bra].Sec[i].Lay[j].bt + delta / self.Net.Bra[i_bra].Sec[i].pas * (
                            self.Net.Bra[i_bra].Sec[i].Lay[j + 1].bt - self.Net.Bra[i_bra].Sec[i].Lay[j].bt)
                    # pw = self.Lay[i].pt + delta / pas * (self.Lay[i+1].pt - self.Lay[i+1].pt)  # 断面湿周
                    # kw = sw * (sw / pw) ** (2.0 / 3) / self.nc  糙率一样采用这个公式

                    # 主槽 滩地糙率不一样的时候用
                    s1w = self.Net.Bra[i_bra].Sec[i].Lay[j].s1 + delta / self.Net.Bra[i_bra].Sec[i].pas * (
                                self.Net.Bra[i_bra].Sec[i].Lay[j + 1].s1 - self.Net.Bra[i_bra].Sec[i].Lay[j].s1)
                    s2lw = self.Net.Bra[i_bra].Sec[i].Lay[j].s2l + delta / self.Net.Bra[i_bra].Sec[i].pas * (
                                self.Net.Bra[i_bra].Sec[i].Lay[j + 1].s2l - self.Net.Bra[i_bra].Sec[i].Lay[j].s2l)
                    s2rw = self.Net.Bra[i_bra].Sec[i].Lay[j].s2r + delta / self.Net.Bra[i_bra].Sec[i].pas * (
                                self.Net.Bra[i_bra].Sec[i].Lay[j + 1].s2r - self.Net.Bra[i_bra].Sec[i].Lay[j].s2r)

                    p1w = self.Net.Bra[i_bra].Sec[i].Lay[j].p1 + delta / self.Net.Bra[i_bra].Sec[i].pas * (
                                self.Net.Bra[i_bra].Sec[i].Lay[j + 1].p1 - self.Net.Bra[i_bra].Sec[i].Lay[j].p1)
                    p2lw = self.Net.Bra[i_bra].Sec[i].Lay[j].p2l + delta / self.Net.Bra[i_bra].Sec[i].pas * (
                                self.Net.Bra[i_bra].Sec[i].Lay[j + 1].p2l - self.Net.Bra[i_bra].Sec[i].Lay[j].p2l)
                    p2rw = self.Net.Bra[i_bra].Sec[i].Lay[j].p2r + delta / self.Net.Bra[i_bra].Sec[i].pas * (
                                self.Net.Bra[i_bra].Sec[i].Lay[j + 1].p2r - self.Net.Bra[i_bra].Sec[i].Lay[j].p2r)

                    kw = s1w * (s1w / p1w) ** (2. / 3) / self.Net.Bra[i_bra].Sec[i].nc + s2lw * (s2lw / p2lw) ** (2. / 3) / \
                         self.Net.Bra[i_bra].Sec[i].nf + s2rw * (s2rw / p2rw) ** (2. / 3) / self.Net.Bra[i_bra].Sec[i].nf

                    s[i][0] = sw
                    k[i][0] = kw
                    b[i][0] = bw

            s_list.append(s)
            k_list.append(k)
            b_list.append(b)

        s_array = np.concatenate(s_list, axis=0)
        k_array = np.concatenate(k_list, axis=0)
        b_array = np.concatenate(b_list, axis=0)
        return s_array, k_array, b_array

    def loss(self, x_U, y_U, x_Z, y_Z, x_PDE):

        loss_u = self.lossU(x_U, y_U)
        loss_z = self.lossZ(x_Z, y_Z)
        loss_pde = self.lossPDE(x_PDE)

        L2_reg = torch.tensor(0.0, requires_grad=True)
        for param in self.parameters():
            L2_reg = L2_reg + torch.norm(param)

        if self.method == '1':
            loss_val = self.weight_l_u * loss_u + self.weight_l_z * loss_z + loss_pde
        elif self.method == '2':
            loss_val = weights_q * loss_u + weights_z * loss_z + weights_pde * loss_pde
        else:
            loss_val = loss_u + loss_z + loss_pde

        # print('iter', self.iter)
        if self.iter % 1000 == 0:
            print('-------------训练损失输出------------------')
            print('loss_u', loss_u.detach().cpu().numpy())
            print('loss_z', loss_z.detach().cpu().numpy())
            print('loss_pde', loss_pde.detach().cpu().numpy())
            print('loss_val', loss_val.detach().cpu().numpy())
            print('self.weight_l_u', self.weight_l_u)
            print('self.weight_l_z', self.weight_l_z)

        # 权重调整
        if self.iter % 1000 == 0:
            # loss梯度计算
            for param in self.parameters():
                param.requires_grad = True

            for i in range(len(self.layers_U) - 1):
                #    print('linears_q_q_grade', i, autograd.grad(loss_u, self.linears_Q[i].weight, retain_graph=True)[0])
                self.grad_l_u.append(autograd.grad(loss_u, self.linears_U[i].weight, retain_graph=True)[0])

            # loss_z.backward()
            for i in range(len(self.layers_Z) - 1):
                self.grad_l_z.append( autograd.grad(loss_z, self.linears_Z[i].weight, retain_graph=True)[0] )

            # loss_pde.backward(retain_graph=True)
            for i in range(len(self.layers_U) - 1):
                self.grad_l_pde1.append(autograd.grad(loss_pde, self.linears_U[i].weight, retain_graph=True)[0])

            for i in range(len(self.layers_Z) - 1):
                self.grad_l_pde2.append( autograd.grad(loss_pde, self.linears_Z[i].weight, retain_graph=True)[0] )


            # 权重存储(每层一个)
            self.adaptive_u_list = []
            self.adaptive_z_list = []

            # 权重更新
            for i in range(len(self.layers_U) - 1):
                self.adaptive_u_list.append(
                    torch.max(torch.abs(self.grad_l_pde1[i])) / torch.mean(torch.abs(self.grad_l_u[i])))

            for i in range(len(self.layers_Z) - 1):
                self.adaptive_z_list.append(
                    torch.max(torch.abs(self.grad_l_pde2[i])) / torch.mean(torch.abs(self.grad_l_z[i])))

            self.weight_l_u = torch.max(torch.stack(self.adaptive_u_list)) * self.alpha + self.weight_l_u * (1.0 - self.alpha)
            self.weight_l_z = torch.max(torch.stack(self.adaptive_z_list)) * self.alpha + self.weight_l_z*(1.0 - self.alpha)

            # 重置梯度存储
            self.grad_l_u = []
            self.grad_l_z = []

            self.grad_l_pde1 = []
            self.grad_l_pde2 = []

        self.loss_val_train_log.append(loss_val.detach().cpu().numpy())
        self.loss_u_train_log.append(loss_u.detach().cpu().numpy())
        self.loss_z_train_log.append(loss_z.detach().cpu().numpy())
        self.loss_pde_train_log.append(loss_pde.detach().cpu().numpy())
        self.weight_l_u_log.append(self.weight_l_u.detach().cpu().numpy())
        self.weight_l_z_log.append(self.weight_l_z.detach().cpu().numpy())
        # '''
        return loss_val, loss_pde

    # 存储NN系数
    def save_NN(self, filepath):
        weights_U = [layer.weight.data for layer in self.linears_U]
        biases_U = [layer.bias.data for layer in self.linears_U]
        weights_Z = [layer.weight.data for layer in self.linears_Z]
        biases_Z = [layer.bias.data for layer in self.linears_Z]
        with open(filepath, 'wb') as f:
            pickle.dump([weights_U, biases_U, weights_Z, biases_Z], f)
        print("Successfully save NN parameters...")

    # 加载NN系数,layer之间的权重系数
    def load_NN(self, filepath):
        with open(filepath, 'rb') as f:
            weights_U, biases_U, weights_Z, biases_Z = pickle.load(f)
        for i, layer in enumerate(self.linears_U):
            layer.weight.data = weights_U[i]
            layer.bias.data = biases_U[i]

        for i, layer in enumerate(self.linears_Z):
            layer.weight.data = weights_Z[i]
            layer.bias.data = biases_Z[i]
        print("Successfully load NN parameters...")

    def save_weight(self, filepath):
        weight_array = np.array([self.weight_l_u, self.weight_l_z])
        np.savetxt(filepath, weight_array.T, fmt='%1.4e')
        print("Successfully save weights...")

    def load_weight(self, filepath):
        weight_array = np.loadtxt(filepath)
        weight_array = weight_array.T
        self.weight_l_u = weight_array[0]
        self.weight_l_z = weight_array[1]
        print("Successfully load weights...")
