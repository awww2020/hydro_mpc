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

from controller.mpc_velocity import MPC
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

    # Hyper parameter
    N_l = 3
    N_n = 40
    layers = [input_dim, *N_l * [N_n], output_dim]

    logging.info('MPC parameters:')
    H = 30
    logging.info(f'\tH:\t{H}')
    u_ub = np.array([4,  0.01, 4])
    u_lb = np.array([0, -0.01, 0])

    X_ref = np.tile([-320, 0.25], (300, 1))

    x0 = np.array([-0.187, -1.187, -2.183, -2.999, -3.2])
    x0 = x0 * 100
    print('x0', x0, x0.shape)
    start = 0
    step = 240
    num_elements = 300
    array_1d = np.arange(start, start + num_elements * step, step)
    T_ref = array_1d.reshape(300, 1)

    tau = 240
    logging.info(f'\ttau:\t{tau}') # 240 s

    # Initialization
    pinn = Hydro_NN(layers, lb, ub)
    # Load pretrained weights
    pinn.load_weights(weights_path_z, weights_path_u)

    print('初始化')
    controller = MPC(f, pinn.model_z,pinn.model_u, u_ub=u_ub, u_lb=u_lb,
                     t_sample=tau, H=H,
                     Q=tf.linalg.tensor_diag(tf.constant([1], dtype=tf.float64)),
                     R=1e-3 * tf.eye(1, dtype=tf.float64))
    # ============== Testing self loop prediction =============

    # ============== Testing closed loop =============
    print('X_ref', X_ref.shape)

    print('T_ref', T_ref.shape)
    U_dis = X_sc[:len(T_ref),6:8]
    # print('U_dis', U_dis)
    print('U_dis.shape', U_dis.shape)

    X_mpc, U_mpc, X_pred = controller.sim(x0, X_ref, T_ref, U_dis)

    output_path = "results_222.txt"
    with open(output_path, "w") as f:
        f.write("t\tX_sc\tU\tY_z_pred\tY_u_sc\n")

        for t, z_0, u, z, v in zip(controller.t_list, controller.z_0_list, controller.u_list, controller.z_list,controller.v_list):
            t_str = ",".join(map(str, t.tolist())) if hasattr(t, "tolist") else str(t)
            z_0_str = ",".join(map(str, z_0.tolist())) if hasattr(z_0, "tolist") else str(z_0)
            u_str = ",".join(map(str, u.tolist())) if hasattr(u, "tolist") else str(u)
            z_str = ",".join(map(str, z.tolist())) if hasattr(z, "tolist") else str(z)
            v_str = ",".join(map(str, v.tolist())) if hasattr(v, "tolist") else str(v)
            f.write(f"{t_str}\t{z_0_str}\t{u_str}\t{z_str}\t{v_str}\n")

    MAE = max(controller.errors_pct)
    IAQ = controller.iaq_accum - abs(controller.u_history[0]-controller.u_history[-1])
    print(f"=== 控制性能指标 ===")
    print(f"MAE (最大绝对水位误差百分比): {MAE:.2f}%")
    print(f"IAQ (综合绝对排量变化): {IAQ:.2f} m³/s")

    plot_input_sequence(T_ref[:-H], U_mpc[:-H],filename='控制输入优化结果')
    print(X_mpc.shape)
    plot_aim_states(T_ref[:-H], X_ref[:-H,0:1]/100, Z_mpc=X_mpc[:-H]/100,filename='目标状态水位')
