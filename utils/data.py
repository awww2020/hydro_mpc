import os

import numpy as np
from pyDOE import lhs
# from scipy.io import loadmat

'''
加载目标路径和加载输入输出数据
'''

def load_data(path):
    """
    Loads reference data and input bounds.

    :param path: path of the reference data, stored in 'pendulum.npz'
    :return
    np.ndarray lb: lower bounds of the inputs of the training data,
    np.ndarray ub: upper bounds of the inputs of the training data,
    int input_dim: dimension of the inputs,
    int output_dim: dimension of the outputs,
    np.ndarray X_test: input tensor of the testing data,
    np.ndarray Y_test output tensor of the testing data,
    np.ndarray X_test: input tensor of the testing data,
    np.ndarray Y_test output tensor of the testing data,
    """

    npzfile = np.load(path)

    # Lower and upper bound
    lb = npzfile['lb']
    ub = npzfile['ub']

    # All data
    X_star = npzfile['X']
    Y_z_star = npzfile['Y_z']
    Y_u_star = npzfile['Y_u']

    X_test = npzfile['X_test']
    Y_z_test = npzfile['Y_z_test']
    Y_u_test = npzfile['Y_u_test']

    input_dim = X_star.shape[1]
    output_dim = Y_z_star.shape[1]

    return lb, ub, input_dim, output_dim, X_test, Y_z_test, Y_u_test, X_star, Y_z_star, Y_u_star

def load_data_sc(path):
    """
    加载实测数据

    :param path: path of the reference data, stored in 'pendulum.npz'
    :return
    """

    npzfile = np.load(path)

    # All data 这边star表示实际过程中的真实值
    X_star = npzfile['X']
    Y_z_star = npzfile['Y_z']
    Y_u_star = npzfile['Y_u']

    return X_star, Y_z_star, Y_u_star


def generate_data_points(N_z, lb, ub):
    X_data = np.hstack((np.zeros((N_z, 1)), lb[1:] + (ub[1:] - lb[1:]) * lhs(len(ub) - 1, N_z)))
    # 生成（100，1）的全零数组作为X_data的第一列，（100，6）作为X_data的后六列
    Y_data = X_data[:, 1:5] # 提取其中四列(状态变量)
    return X_data, Y_data


def generate_collocation_points(N_phys, lb, ub):
    X_phys = lb + (ub - lb) * lhs(len(ub), N_phys)
    return X_phys


def load_ref_trajectory(path):
    '''
    X_12_ref = loadmat(os.path.join(path, 'y_soll.mat'))['y_soll'].T # 读取并转置 (1220, 2)，每列代表一个变量
    X_34_ref = loadmat(os.path.join(path, 'Dy_soll.mat'))['Dy_soll'].T
    X_ref = np.hstack((X_12_ref, X_34_ref)) # 水平堆叠 (1220,4)

    T_ref = loadmat(os.path.join(path, 't_soll.mat'))['t_soll'].T # (1220,1)
    # t_soll: 形状为 (1, 1220)，表示这个矩阵是一行1220列，通常用于表示时间点的序列。这表明有1220个时间点的数据。


    freq = 10
    # 稀疏化处理，从原始数据中每隔10个数据点取一个点
    X_ref = X_ref[::freq]
    T_ref = T_ref[::freq] # (122,1)
    '''
    X_ref = []
    T_ref = []
    return X_ref, T_ref