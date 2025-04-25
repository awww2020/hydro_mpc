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
import tensorflow as tf

from controller.mpc import MPC
import train_pinn
from train_pinn import Hydro_NN
from utils.data import load_ref_trajectory, load_data, load_data_sc
from utils.plotting import  plot_input_sequence, plot_aim_states
from utils.system import f

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    # Paths
    data_path = os.path.join('./data/data_0.npz')
    data_sc_path = os.path.join('./data/data_sc_4min.npz')
    weights_path_z = os.path.join('./weights/easy_checkpoint_model_z/')
    weights_path_u = os.path.join('./weights/easy_checkpoint_model_u/')

    lb, ub, input_dim, output_dim, _, _, _, _, _, _ = load_data(data_path)
    X_sc, Y_z_sc, Y_u_sc = load_data_sc(data_sc_path)

    # Hyper parameter 神经网络超参数
    N_l = 3
    N_n = 40
    layers = [input_dim, *N_l * [N_n], output_dim]

    #===============设置MPC模型参数=============
    logging.info('MPC parameters:')
    H = 30  # 预测未来15步（horizon），15个4分钟，预见期1小时
    logging.info(f'\tH:\t{H}')
    # 控制目标的上下限
    u_ub = np.array([4,  0.01, 4])
    u_lb = np.array([0, -0.01, 0])

    # ==============定义MPC目标参考值=============
    # X_ref, T_ref = load_ref_trajectory('./data')
    # shape = (289, 1)
    shape = (300, 1)
    fill_value = -320.
    X_ref = np.full(shape, fill_value)

    # x0 = X_ref[0] # 取出第一行，包含第一行的所有列的数据，也可写成X_ref[0,：]
    x0 = np.array([-0.187, -1.187, -2.183, -2.999, -3.2])
    x0 = x0 * 100
    print('x0', x0, x0.shape)
    # T_ref = T_ref[:-H, 0] # 从(1220,1)取出前-h行,第1列
    # [:-H] 表示从数组中去掉最后 H 个元素，通常这样做是因为在仿真的最后几步可能无法应用完整的预测窗口。
    # 定义起始值、步长和元素数量
    start = 0
    step = 240
    num_elements = 300
    # 使用 numpy.arange 生成等差数列
    array_1d = np.arange(start, start + num_elements * step, step)
    # 确保数组的形状为 (289, 1)
    T_ref = array_1d.reshape(300, 1)

    tau = 240 # 采样时间间隔
    logging.info(f'\ttau:\t{tau}') # 240 s

    # Initialization
    pinn = Hydro_NN(layers, lb, ub)
    # Load pretrained weights
    pinn.load_weights(weights_path_z, weights_path_u)

    # ==============初始化MPC控制器=============
    # 这边的f是system import过来的
    print('初始化')
    controller = MPC(f, pinn.model_z, u_ub=u_ub, u_lb=u_lb,
                     t_sample=tau, H=H,
                     Q=tf.linalg.tensor_diag(tf.constant([1], dtype=tf.float64)),
                     R=1e-3 * tf.eye(1, dtype=tf.float64))
    # 输入个数变了的话这边Q和R也要改，R的个数等于控制变量u(这边只控制下游，因此是1)的个数,Q的列数=考虑的目标参数的个数



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
    X_ref_sl = controller.sim_open_loop_plant(x0_sl, U_sl,t_sample=tau,H=H_sl)

    # Simulate PINN system PINN未来20步的预测序列
    X_sl = controller.sim_open_loop(x0_sl, U_sl,t_sample=tau,H=H_sl)

    T_sl = np.arange(0., H_sl * tau + tau, tau)

    # 比较绘图
    plot_input_sequence(T_sl, U=np.vstack((U_sl, U_sl[-1:, :])))
    plot_states(T_sl, X_ref_sl, X_sl)
    plot_absolute_error(T_sl, X_ref_sl, X_sl)
    '''
    # ============== Testing closed loop =============
    #  闭环控制测试，这种仿真模式考虑反馈，并使用控制器的MPC策略来优化控制输入。
    print('X_ref', X_ref.shape)

    print('T_ref', T_ref.shape)
    U_dis = X_sc[:len(T_ref),6:8]
    # print('U_dis', U_dis)
    print('U_dis.shape', U_dis.shape)

    X_mpc, U_mpc, X_pred = controller.sim(x0, X_ref, T_ref, U_dis)

    output_path = "results_111.txt"
    with open(output_path, "w") as f:
        # 写入表头
        f.write("t\tX_sc\tU\tY_z_pred\tY_u_sc\n")

        # 假设 X_sc、Y_z_pred、Y_z_sc 都是形如 (N, d) 或 (N,) 的 numpy 数组
        for t, z_0, u, z, v in zip(controller.t_list, controller.z_0_list, controller.u_list, controller.z_list,controller.v_list):
            # 如果是多维向量，用 list() 或 tolist() 转成可打印的格式
            t_str = ",".join(map(str, t.tolist())) if hasattr(t, "tolist") else str(t)
            z_0_str = ",".join(map(str, z_0.tolist())) if hasattr(z_0, "tolist") else str(z_0)
            u_str = ",".join(map(str, u.tolist())) if hasattr(u, "tolist") else str(u)
            z_str = ",".join(map(str, z.tolist())) if hasattr(z, "tolist") else str(z)
            v_str = ",".join(map(str, v.tolist())) if hasattr(v, "tolist") else str(v)
            f.write(f"{t_str}\t{z_0_str}\t{u_str}\t{z_str}\t{v_str}\n")

    # 最大绝对误差（百分比）
    MAE = max(controller.errors_pct)
    # IAQ 已经在循环中累积完毕
    IAQ = controller.iaq_accum - abs(controller.u_history[0]-controller.u_history[-1])
    print(f"=== 控制性能指标 ===")
    print(f"MAE (最大绝对水位误差百分比): {MAE:.2f}%")
    print(f"IAQ (综合绝对排量变化): {IAQ:.2f} m³/s")

    # 用于在给定的初始状态 x0、参考状态轨迹 X_ref 和时间点 T_ref 下进行闭环仿真

    plot_input_sequence(T_ref[:-H], U_mpc[:-H],filename='控制输入优化结果')
    print(X_mpc.shape)
    plot_aim_states(T_ref[:-H], X_ref[:-H]/100, Z_mpc=X_mpc[:-H]/100,filename='目标状态水位')
    # plot_absolute_error(T_ref, X_ref, Z_mpc=X_mpc)