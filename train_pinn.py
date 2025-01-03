#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 2024/4/16
@author: XIAO LUO
"""

import os
from utils.data import generate_data_points, generate_collocation_points, load_data

import numpy as np
import matplotlib
matplotlib.use('TkAgg') # 在Windows上使用PyCharm进行开发时，默认的交互式框架是Tkinter
# matplotlib.use('module://backend_interagg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib.ticker import FuncFormatter

import torch
from model.nn_model import NN
from model.pinn_junction import HydroNet_1D
import time
import geo_pre.read_net as rn

np.random.seed(24)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU:-1; GPU0: 0; GPU1: 1;
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def add_noise(insignal):
    # 用于给输入信号添加噪声
    # target_snr_db = 500
    target_snr_db = 500
    insignal = torch.from_numpy(insignal)
    # Calculate signal power and convert to dB
    sig_avg = torch.mean(insignal)
    print('sig_avg', sig_avg)
    sig_avg_db = 10 * torch.log10(sig_avg)
    print('sig_avg_db', sig_avg_db)
    # Calculate noise according to [2] then convert to watts 计算噪声的目标平均功率，噪声的目标平均功率从分贝（dB）转换为线性形式
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg = 10 ** (noise_avg_db / 10)
    # Generate an sample of white noise
    mean_noise = 0
    noise = torch.randn(insignal.size()) * torch.sqrt(noise_avg)
    sig_noise = insignal + noise
    return sig_noise.numpy()

def plot_Z(x2, y_pred, y_exact,t):
    '''
    plt.figure()
    plt.xlabel('x2')
    plt.ylabel('Z/m')
    plt.plot(x2, y_pred, 'x-', label='prediction')
    plt.plot(x2, y_exact, '+-', label='exact')
    plt.legend()
    plt.savefig(f'{output_file}/Z_prediction.png', dpi=300)
    '''

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))  # Create 4 subplots, in a 2 by 2 grid.

    # Split the data into 4 equal parts for the 4 time steps.
    x2_split = np.split(x2, t)
    y_pred_split = np.split(y_pred, t)
    y_exact_split = np.split(y_exact, t)
    '''
    time_steps = [360,720]
    for i, j in enumerate(time_steps):
        axs[i].plot(x2_split[j], y_pred_split[j], 'x-', label='prediction')
        axs[i].plot(x2_split[j], y_exact_split[j], '+-', label='exact')
        axs[i].set_xlabel('x2')
        axs[i].set_ylabel('Z/m')
        axs[i].legend()
        axs[i].set_xlabel(f'x2\n\nTime step {(i*60)} s')

    plt.tight_layout()  # Adjust subplot parameters to give specified padding.
    plt.savefig(f'{output_file}/Z_prediction.png', dpi=300)
    '''
    slices = [slice(0, 5), slice(5, 13), slice(13, 22), slice(22, 26)]
    for ax, s in zip(axs.ravel(), slices):
        ax.plot(x2_split[t-1][s], y_pred_split[t-1][s], 'x-', label='prediction')
        ax.plot(x2_split[t-1][s], y_exact_split[t-1][s], '+-', label='exact')
        ax.set_xlabel('x2')
        ax.set_ylabel('Z/m')
        ax.legend()

    # 设置标题
    axs[0, 0].set_title('Bra 1')
    axs[0, 1].set_title('Bra 2')
    axs[1, 0].set_title('Bra 3')
    axs[1, 1].set_title('Bra 4')

    plt.tight_layout()  # Adjust subplot parameters to give specified padding.
    plt.savefig(f'{output_file}/Z_prediction.png', dpi=300)
    plt.savefig(f'{output_file}/Z_prediction.eps', dpi=300)

def plot_U(x2, y_pred, y_exact,t):

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))  # Create 4 subplots, in a 2 by 2 grid.
    # Split the data into 4 equal parts for the 4 time steps.
    x2_split = np.split(x2, t)
    y_pred_split = np.split(y_pred, t)
    y_exact_split = np.split(y_exact, t)
    '''
    # time_steps = [360, 540, 720, 1440]
    time_steps = [360]
    for i, j in enumerate(time_steps):
        axs[i].plot(x2_split[j], y_pred_split[j], 'x-', label='prediction')
        axs[i].plot(x2_split[j], y_exact_split[j], '+-', label='exact')
        axs[i].set_xlabel('x2')
        axs[i].set_ylabel('U/(m/s)')
        axs[i].legend()
        axs[i].set_xlabel(f'x2\n\nTime step {(i*60)} s')

    plt.tight_layout()  # Adjust subplot parameters to give specified padding.
    plt.savefig(f'{output_file}/U_prediction.png', dpi=300)
    '''

    slices = [slice(0, 5), slice(5, 13), slice(13, 22), slice(22, 26)]
    for ax, s in zip(axs.ravel(), slices):

        ax.plot(x2_split[t-1][s], y_pred_split[t-1][s], 'x-', label='prediction')
        ax.plot(x2_split[t-1][s], y_exact_split[t-1][s], '+-', label='exact')
        ax.set_xlabel('x2')
        ax.set_ylabel('U/(m/s)')
        ax.legend()

    # 设置标题
    axs[0, 0].set_title('Bra 1')
    axs[0, 1].set_title('Bra 2')
    axs[1, 0].set_title('Bra 3')
    axs[1, 1].set_title('Bra 4')

    plt.tight_layout()  # Adjust subplot parameters to give specified padding.
    plt.savefig(f'{output_file}/U_prediction.png', dpi=300)
    plt.savefig(f'{output_file}/U_prediction.eps', dpi=300)

def plot_Z_3D(x1, x2, y_pred, y_exact, t):

    # Split the data into 4 equal parts for the 4 time steps.
    x1_split = np.split(x1, t)
    x2_split = np.split(x2, t)
    y_pred_split = np.split(y_pred, t)
    y_exact_split = np.split(y_exact, t)
    '''
    # time_steps = [360, 540, 720, 1440]
    time_steps = [360]
    for i, j in enumerate(time_steps):
        scatter = axs[i].scatter(x1_split[j], x2_split[j], y_exact_split[j] - y_pred_split[j])
        axs[i].set_xlabel(f'x1\n\nTime step {(i*60)} s')
        axs[i].set_ylabel('x2')
        axs[i].set_zlabel('Z/m')
        # axs[i].legend()

    plt.tight_layout()  # Adjust subplot parameters to give specified padding.
    plt.savefig(f'{output_file}/Z_prediction_3D.png', dpi=300)
    '''
def plot_U_3D(x1, x2, y_pred, y_exact, t):


    # Split the data into 4 equal parts for the 4 time steps.
    x1_split = np.split(x1, t)
    x2_split = np.split(x2, t)
    y_pred_split = np.split(y_pred, t)
    y_exact_split = np.split(y_exact, t)
    '''
    # time_steps = [360, 540, 720, 1440]
    time_steps = [360]

    for i, j in enumerate(time_steps):
        for k in np.unique(x1_split[j]):  # For each unique value in x1
            # Create a mask to select only the data for the current x1 value
            mask = x1_split[j] == k
            y = np.squeeze(y_exact_split[j][mask] - y_pred_split[j][mask])
            axs[i].plot(x1_split[j][mask], x2_split[i][mask], y)
        axs[i].set_xlabel(f'x1\n\nTime step {(i*60)} s')
        axs[i].set_ylabel('x2')
        axs[i].set_zlabel('U/(m/s)')

    plt.tight_layout()  # Adjust subplot parameters to give specified padding.
    plt.savefig(f'{output_file}/U_prediction_3D.png', dpi=300)
    '''


if __name__ == "__main__":
    # 象征着程序主入口
    # plt.switch_backend('Agg')
    # Device configuration
    # 用于作为tensor或者model被分配到的位置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device', device)
    if device == 'cuda':
        print(torch.cuda.get_device_name())


    #读取方法2：直接读取成array
    X = np.loadtxt("data/x_case-12h.txt")
    data = np.loadtxt("data/data_case-12h-1.txt")
    # X是所有时空点的集合，每一个时空点都有一个对应的Q值，所以X的长度跟data长度一致

    # Paths
    data_path = os.path.join('data/data.npz')
    weights_path = os.path.join('weights')

    lb, ub, input_dim, output_dim, X_test, Y_test, X_star, Y_star = load_data(data_path)

    # 数据处理
    # 测试集
    X_test = X
    U_test = data[:, 0:1]
    Z_test = data[:, 1:2]

    U_test = add_noise(U_test)
    Z_test = add_noise(Z_test)

    X_test = X[:18746, :]
    X_test_orig = X_test.copy()
    U_test = U_test[:18746, :]
    Z_test = Z_test[:18746, :]

    # 训练集
    N = X.shape[0]  # 读取矩阵第一维度的长度,这边即行，有多少个空间加时间点
    N_f = N         # 用12个点去约束PDE

    # Specify input domain bounds 指定边界 返回横纵坐标的范围
    lb, ub = X.min(0), X.max(0)  # min(0)返回该矩阵中每一列的最小值，max(0)返回该矩阵中每一列的最大值 若为1则返回每一行 1*2

    # Training data 得到训练数据，划分/抽取数据集
    #1.划分用哪些值来进行约束
    # 实测值
    N_U = 26    # U点实测值约束的空间点个数，这边实测约束点可以小于总共的点
    N_Z = 26    # Z点实测值约束的空间点个数
    t = 25    # pde约束时间间隔 0,60,..600,43200
    N_c = 26  # 河网总断面数
    t_output = 721  # 训练数据间隔

    def extract_rows(data, interval):
        # Define the desired values in the third column based on the given interval
        desired_values = np.arange(0, 43201, interval)
        # 从 data 数组中筛选出那些第三列的值在 desired_values 中的行
        filtered_data = data[np.isin(data[:, 2], desired_values)]
        return filtered_data

    X_U = extract_rows(X, 1800)  # equivalent to the previous calculation
    # Extract the corresponding rows from U_test using the indices of the filtered X_test
    U_test_indices = np.where(np.isin(X[:, 2], np.arange(0, 43201, 1800)))[0]
    Y_U = U_test[U_test_indices, :]
    print('X_U.shape',X_U.shape)
    print('Y_U.shape',Y_U.shape)
    # X_U = X[idx_u, :]
    # Y_U = U_test[idx_u, :]

    X_Z = extract_rows(X, 1800)  # equivalent to the previous calculation
    # Extract the corresponding rows from U_test using the indices of the filtered X_test
    Z_test_indices = np.where(np.isin(X[:, 2], np.arange(0, 43201, 1800)))[0]
    Y_Z = Z_test[Z_test_indices, :]


    # ！！！改实测限制的话，X_U, Y_U, X_Z, Y_Z,这几个是要改的
    #
    #idx_z = np.arange(0, N_c*t)
    #X_Z = X[idx_z, :]       # !!!坐标中随机提出Z点实测值约束的时空坐标
    #Y_Z = Z_test[idx_z,:]   # 该时空点上的Z值
    '''
    idx_c = np.random.choice(N, N_c)  # 从数组、列表或元组中随机抽取,N必须是一维的！*1 的数组
    X_U = X[idx_c, :]        # 坐标中随机提出U点实测值约束的坐标
    Y_U = U_test[idx_c,:]    # 坐标中随机提出的X_U个空间点上的U值

    X_Z = X[idx_c, :]       # 坐标中随机提出Z点实测值约束的坐标
    Y_Z = Z_test[idx_c,:]   # 坐标中随机提出的X_Z个空间点上的Z值
    '''
    # 这边约束也直接用上面选出的点，或者可以设置不同的
    #idx_f = np.random.choice(N, N_f)
    X_f = extract_rows(X, 1800)
    print(X_f.shape)
    Y_f = np.zeros((26*13, 1))

    feature = 'Normalization_2'
    if feature == 'Normalization_1':
        X_U = (X_U - lb) - 0.5 * (ub - lb)  # 使得坐标左右对称，成为正方形
        X_Z = (X_Z - lb) - 0.5 * (ub - lb)
        # X_f = (X_f - lb) - 0.5 * (ub - lb)
        X_test =  (X_test - lb) - 0.5 * (ub - lb)

    if feature == 'Normalization_2':
        X_U = 2.0 * (X_U - lb) / (ub - lb) - 1
        X_Z = 2.0 * (X_Z - lb)/(ub - lb) - 1
        # X_f = 2.0 * (X_f - lb)/(ub - lb) - 1
        X_test = 2.0 * (X_test - lb)/(ub - lb) - 1

    if feature == 'scale_minmax':
        '''归一化'''
        X_U = (X_U - lb) / (ub - lb)  # 使得坐标左右对称，成为正方形
        X_Z = (X_Z - lb)/(ub - lb)
        # X_f = (X_f - lb)/(ub - lb)
        X_test = (X_test - lb)/(ub - lb)

    ''' loop case
    X_U[:, 0:1] = 0.0
    X_Z[:, 0:1] = 0.0
    X_test[:, 0:1] = 0.0
    '''
    # 这边生成的是训练的输入坐标
    x1_u = X_U[:, 0:1]  # 第一列 但是是矩阵不是向量
    x2_u = X_U[:, 1:2]  # 第二列
    x3_u = X_U[:, 2:3]  # 第二列
    X_U_train = torch.from_numpy(np.hstack((x1_u, x2_u, x3_u)))
    Y_U_train = torch.from_numpy(Y_U)

    x1_z = X_Z[:, 0:1]
    x2_z = X_Z[:, 1:2]
    x3_z = X_Z[:, 2:3]
    X_Z_train = torch.from_numpy(np.hstack((x1_z, x2_z, x3_z)))
    Y_Z_train = torch.from_numpy(Y_Z)

    X_f_train = torch.from_numpy(X_f)

    output_file = 'figs/test'

    # tunning Parameters
    steps = 100
    lr = 0.01
    layers_U = np.array([3, 30, 30, 30, 1])
    layers_Z = np.array([3, 30, 30, 30, 1])

    # Store tensors to GPU
    X_U_train = X_U_train.float().to(device)  # Training Points
    Y_U_train = Y_U_train.float().to(device)  # Training Points
    X_Z_train = X_Z_train.float().to(device)  # Training Points
    Y_Z_train = Y_Z_train.float().to(device)  # Training Points
    X_f_train = X_f_train.float().to(device)  # Collocation Points
    f_hat = torch.zeros(X_f_train.shape[0], 1).to(device)  # to minimize function

    # X_test = X_test[:, 1:2] # @仅第二列
    X_test = torch.from_numpy(X_test)
    U_test = torch.from_numpy(U_test)
    Z_test = torch.from_numpy(Z_test)

    print("Original shapes for X and Y:", X_test.shape, U_test.shape)
    print("Final training data:", X_U_train.shape, Y_U_train.shape)
    print("Total collocation points:", X_f_train.shape)

    X_test= X_test.float().to(device)  # the input dataset (complete)
    U_test = U_test.float().to(device)  # the real solution
    Z_test = Z_test.float().to(device)  # the real solution

    # Create model
    PINN = HydroNet_1D(layers_U, layers_Z, device, f_hat, rn.Net, lb, ub, t, N_c)
    PINN.to(device)


    for i in range(26):
        model_path = f'geo_pre/models_A/model_{i}.pth'
        if os.path.exists(model_path):
            PINN.models_A[i] = torch.load(model_path)

    for i in range(26):
        model_path = f'geo_pre/models_B/model_{i}.pth'
        if os.path.exists(model_path):
            PINN.models_B[i] = torch.load(model_path)

    for i in range(26):
        model_path = f'geo_pre/models_K/model_{i}.pth'
        if os.path.exists(model_path):
            PINN.models_K[i] = torch.load(model_path)

    weights_mode = 'load'
    if weights_mode == 'load':
        PINN.load_NN('init_model/hydro.pkl')
        PINN.load_weight('init_model/weights.txt')

    print(PINN)
    params = list(PINN.parameters())

    optimizer_AdamW = torch.optim.AdamW(PINN.parameters(), lr=lr, weight_decay=0.01, amsgrad=False)

    'L-BFGS Optimizer'
    optimizer_LB = torch.optim.LBFGS(PINN.parameters(), lr=0.001,
                                  max_iter=steps,
                                  max_eval=None,
                                  tolerance_grad=1e-15,
                                  tolerance_change=1e-15,
                                  history_size=100,
                                  line_search_fn='strong_wolfe')

    iter = steps

    def closure():
        global iter
        optimizer_LB.zero_grad()
        loss, loss_p = PINN.loss(X_U_train, Y_U_train, X_Z_train, Y_Z_train, X_f_train)
        loss_train_log.append(loss.detach().cpu().numpy())
        loss.backward()
        iter = iter +1
        with torch.no_grad():
            test_loss_U = PINN.lossU(X_test, U_test)
            test_loss_Z = PINN.lossZ(X_test, Z_test)
            loss_test_U_log.append(test_loss_U.cpu().numpy())
            loss_test_Z_log.append(test_loss_Z.cpu().numpy())
        # print(loss.cpu().detach().numpy())
        U_pred = PINN.forward_U(X_test)
        Z_pred = PINN.forward_Z(X_test)
        pred_output = True
        if pred_output == True:
            print('-------------LB训练结果预测输出------------------')
            print('iter', iter)
            print('loss',loss)
            print('loss_pde',loss_p)
            print('test_loss_U',test_loss_U)
            print('test_loss_Z',test_loss_Z)
        return loss

    start_time = time.time()

    # 调用模型进行训练（优化权重和偏差）
    loss_train_log = []
    loss_test_U_log = []
    loss_test_Z_log = []
    for i in range(steps):
        time1 = time.time()
        if i == 0:
            print("Training Loss-----Test Loss")
        # 训练集输出间隔
        if i % 1000 == 0:
            print('i', i)
        loss,loss_p = PINN.loss(X_U_train, Y_U_train, X_Z_train, Y_Z_train, X_f_train) # use mean squared error X_f_train 先不进行正则化
        loss_train_log.append(loss.detach().cpu().numpy())
        optimizer_AdamW.zero_grad()
        loss.backward()
        # 更新模型参数
        optimizer_AdamW.step()

        # elapsed = time.time() - time1
        # print('single Training time: %.2f' % (elapsed))

        # 记录测试集的loss
        with torch.no_grad():
            test_loss_U = PINN.lossU(X_test, U_test)
            test_loss_Z = PINN.lossZ(X_test, Z_test)
            loss_test_U_log.append(test_loss_U.cpu().numpy())
            loss_test_Z_log.append(test_loss_Z.cpu().numpy())
            if i % 1000 == 0:
                print("Test Loss U:", test_loss_U)
                print("Test Loss Z:", test_loss_Z)

        U_pred = PINN.forward_U(X_test)
        Z_pred = PINN.forward_Z(X_test)
        pred_output = False
        if i == (steps-1):
            print('-------------AdamW最终训练结果预测输出------------------')
            print('U_pred', np.squeeze(U_pred.detach().cpu().numpy()))
            print('Z_pred', np.squeeze(Z_pred.detach().cpu().numpy()))

    # loss = optimizer_LB.step(lambda: closure(steps))
    loss = optimizer_LB.step(closure)

    final_test_loss_U = loss_test_U_log[-1]
    final_test_loss_Z = loss_test_Z_log[-1]
    print("Final Test Loss U:", final_test_loss_U)
    print("Final Test Loss Z:", final_test_loss_Z)

    U_pred = PINN.forward_U(X_test)
    Z_pred = PINN.forward_Z(X_test)
    print('U_pred', np.squeeze(U_pred.detach().cpu().numpy()))
    print('Z_pred', np.squeeze(Z_pred.detach().cpu().numpy()))

    U_pred_array = np.squeeze(U_pred.detach().cpu().numpy())
    Z_pred_array = np.squeeze(Z_pred.detach().cpu().numpy())

    np.savetxt(f'{output_file}/U_pred.csv', U_pred_array, delimiter=",")
    np.savetxt(f'{output_file}/Z_pred.csv', Z_pred_array, delimiter=",")

    elapsed = time.time() - start_time
    print('Training time: %.2f' % (elapsed))

    # 6：模型评估，计算误差
    # 计算RMSE
    rmse_U = np.sqrt(np.mean((U_test.detach().cpu().numpy() - U_pred.detach().cpu().numpy()) ** 2))
    rmse_Z = np.sqrt(np.mean((Z_test.detach().cpu().numpy() - Z_pred.detach().cpu().numpy()) ** 2))

    # 计算MAPE，这里假设U_test和Z_test都不包含零，或者你已经处理了这种情况
    mape_U = np.mean(np.abs((U_test.detach().cpu().numpy() - U_pred.detach().cpu().numpy()) / U_test.detach().cpu().numpy())) * 100
    mape_Z = np.mean(np.abs((Z_test.detach().cpu().numpy() - Z_pred.detach().cpu().numpy()) / Z_test.detach().cpu().numpy())) * 100

    print("RMSE for U:", rmse_U)
    print("RMSE for Z:", rmse_Z)
    print("MAPE for U:", mape_U)
    print("MAPE for Z:", mape_Z)


    # Plot 绘图 7：数据后处理
    # 测试集的loss绘图


    plt.figure()
    plt.plot(np.log10(PINN.loss_val_train_log), 'b', label='val_train')
    plt.plot(np.log10(PINN.loss_u_train_log), 'g', label='u_train')
    plt.plot(np.log10(PINN.loss_z_train_log), 'r', label='z_train')
    plt.plot(np.log10(PINN.loss_pde_train_log), 'y', label='pde_train')
    plt.ylabel('loss value')
    plt.xlabel('iter_num')

    # 定义一个将对数坐标转换回原始坐标的函数
    def format_func(value, tick_number):
        return f'$10^{value:.0f}$'
    formatter = FuncFormatter(format_func)
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.legend()
    plt.savefig(f'{output_file}/train_loss_value_log10.png', dpi=300)
    plt.savefig(f'{output_file}/train_loss_value_log10.eps', dpi=300)

    plt.figure()
    plt.plot(np.log10(loss_test_U_log), 'g--', label='u_test')
    plt.plot(np.log10(loss_test_Z_log), 'r--', label='z_test')
    plt.ylabel('loss value')
    plt.xlabel('iter_num')

    # 定义一个将对数坐标转换回原始坐标的函数
    def format_func(value, tick_number):
        return f'$10^{value:.0f}$'
    formatter = FuncFormatter(format_func)
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.legend()
    plt.savefig(f'{output_file}/test_loss_value_log10.png', dpi=300)
    plt.savefig(f'{output_file}/test_loss_value_log10.eps', dpi=300)

    plt.figure()
    plt.plot(PINN.loss_val_train_log, 'b', label='val_train')
    plt.plot(PINN.loss_u_train_log, 'g', label='u_train')
    plt.plot(PINN.loss_z_train_log, 'r', label='z_train')
    plt.plot(PINN.loss_pde_train_log, 'y', label='pde_train')
    plt.plot(loss_test_U_log, 'g--', label='u_test')
    plt.plot(loss_test_Z_log, 'r--', label='z_test')
    plt.ylabel('loss value')
    plt.xlabel('iter_num')
    plt.legend()
    plt.savefig(f'{output_file}/loss_value.png', dpi=300)
    plt.savefig(f'{output_file}/loss_value.eps', dpi=300)

    plt.figure()
    plt.plot(PINN.weight_l_u_log, 'g--', label='weight_l_u')
    plt.plot(PINN.weight_l_z_log, 'r--', label='weight_l_z')
    plt.ylabel('weight')
    plt.xlabel('iter_num')
    plt.legend()
    plt.savefig(f'{output_file}/weight.png', dpi=300)
    plt.savefig(f'{output_file}/weight.eps', dpi=300)

    plt.figure()
    plt.plot(np.log10(PINN.weight_l_u_log), 'g--', label='weight_l_u')
    plt.plot(np.log10(PINN.weight_l_z_log), 'r--', label='weight_l_z')
    plt.ylabel('weight')
    plt.xlabel('iter_num')
    # 定义一个将对数坐标转换回原始坐标的函数
    def format_func(value, tick_number):
        return f'$10^{value:.0f}$'
    formatter = FuncFormatter(format_func)
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.legend()
    plt.savefig(f'{output_file}/weight_log10.png', dpi=300)
    plt.savefig(f'{output_file}/weight_log10.eps', dpi=300)

    # 测试集的结果与真实值比较
    arr_x1 = X_test[:, 0]
    arr_x2 = X_test_orig[:, 1]

    plot_Z(arr_x2, Z_pred.detach().cpu().numpy(), Z_test.detach().cpu().numpy(), t_output)
    plot_U(arr_x2, U_pred.detach().cpu().numpy(), U_test.detach().cpu().numpy(), t_output)

    #plot_Z_3D(arr_x1, arr_x2, Z_pred.detach().cpu().numpy(), Z_test.detach().cpu().numpy(), t)
    #plot_U_3D(arr_x1, arr_x2, U_pred.detach().cpu().numpy(), U_test.detach().cpu().numpy(), t)

    # 保存数据到一个.npz文件
    np.savez(f'{output_file}/z_data.npz', Z_pred=Z_pred.detach().cpu().numpy(), Z_test=Z_test.detach().cpu().numpy())
    np.savez(f'{output_file}/u_data.npz', U_pred=U_pred.detach().cpu().numpy(), U_test=U_test.detach().cpu().numpy())

    with open(f'{output_file}/hyperpara.txt', 'a') as f:
        f.write(f'steps={steps}\n')
        f.write(f'lr={lr}\n')
        f.write(f'PINN={PINN}\n')
        f.write(f'feature={feature}\n')
        f.write(f'elapsed={elapsed}\n')
        f.write(f'test_loss_U={test_loss_U}\n')
        f.write(f'test_loss_Z={test_loss_Z}\n')


