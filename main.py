#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 2024/4/16
@author: XIAO LUO
"""

import logging
import os

import numpy as np
import torch

from controller.mpc import MPC
import train_pinn
from utils.data import load_ref_trajectory, load_data
from utils.plotting import plot_input_sequence, plot_states, plot_absolute_error, animate
# from utils.system import f


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    # Paths
    data_path = os.path.join('../data/data.npz')
    weights_path = os.path.join('../weights')

    lb, ub, input_dim, output_dim, _, _, _, _ = load_data(data_path)

    # Hyper parameter 神经网络超参数
    N_l = 4
    N_n = 64
    layers = [input_dim, *N_l * [N_n], output_dim]

    #===============设置MPC模型参数=============
    logging.info('MPC parameters:')
    H = 5  # 预测未来5步（horizon）
    logging.info(f'\tH:\t{H}')
    u_ub = np.array([0.5, 0.5])
    u_lb = - u_ub

    # ==============定义MPC目标参考值=============
    X_ref, T_ref = load_ref_trajectory('./data')

    x0 = X_ref[0] # 从(1220,4)取出第一行，返回一个形状为 (4,) 的数组，包含第一行的所有列的数据，也可写成X_ref[0,：]
    T_ref = T_ref[:-H, 0] # 从(1220,1)取出前-h行,第1列
    # [:-H] 表示从数组中去掉最后 H 个元素，通常这样做是因为在仿真的最后几步可能无法应用完整的预测窗口。

    tau = T_ref[1] - T_ref[0] # 采样时间间隔
    logging.info(f'\ttau:\t{tau}') # 0.2 计算时间为0，24.4s

    # Initialization
    # pinn = ManipulatorInformedNN(layers, lb, ub)
    # Load pretrained weights
    # pinn.load_weights(weights_path)


    # ==============初始化MPC控制器=============
    # 这边的f是system import过来的
    controller = MPC(f, train_pinn.PINN, u_ub=u_ub, u_lb=u_lb,
                     t_sample=tau, H=H,
                     Q=torch.diag(torch.tensor([1, 1, 0, 0], dtype=torch.float64)),
                     R=1e-6 * torch.eye(2, dtype=torch.float64))


    # ============== Testing self loop prediction =============
    # 自循环预测测试，这边就是测试PINN跟真实值的差别
    ''' 先不管真实值要在程序中实现计算
    H_sl = 20

    # Generate self loop prediction input sequence 生成控制输入序列U_sl
    U1_sl = 0.5 * np.sin(np.linspace(0, 2 * np.pi, H_sl))
    U2_sl = - U1_sl
    U_sl = np.hstack((U1_sl[:, np.newaxis], U2_sl[:, np.newaxis]))

    # Initial state 系统起始状态
    x0_sl = np.zeros(4)

    # Simulate plant system 未来20步的参考序列,solve_ivp进行求解
    X_ref_sl = controller.sim_open_loop_plant(x0_sl, U_sl,
                                              t_sample=tau,
                                              H=H_sl)

    # Simulate PINN system PINN未来20步的预测序列
    X_sl = controller.sim_open_loop(x0_sl, U_sl,
                                    t_sample=tau,
                                    H=H_sl)

    T_sl = np.arange(0., H_sl * tau + tau, tau)

    # 比较绘图
    plot_input_sequence(T_sl, U=np.vstack((U_sl, U_sl[-1:, :])))
    plot_states(T_sl, X_ref_sl, X_sl)
    plot_absolute_error(T_sl, X_ref_sl, X_sl)
    '''
    # ============== Testing closed loop =============
    #  闭环控制测试，这种仿真模式考虑反馈，并使用控制器的MPC策略来优化控制输入。
    X_mpc, U_mpc, X_pred = controller.sim(x0, X_ref, T_ref)
    # 用于在给定的初始状态 x0、参考状态轨迹 X_ref 和时间点 T_ref 下进行闭环仿真

    plot_input_sequence(T_ref, U_mpc)
    plot_states(T_ref, X_ref[:-H], Z_mpc=X_mpc)
    plot_absolute_error(T_ref, X_ref[:-H], Z_mpc=X_mpc)
    animate(X_ref[:-H], [X_mpc], ['MPC'], fps=1 / tau)