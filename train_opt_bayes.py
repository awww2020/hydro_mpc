import os

import click
import logging
import time

import numpy as np
import tensorflow as tf

from model.pinn import PINN
from utils.data import generate_data_points, generate_collocation_points, load_data,load_data_sc
from utils.plotting import animate, plot_states_z, plot_states_u, plot_input_sequence, plot_absolute_error_z, plot_absolute_error_u
from utils.system import M_tensor, k_tensor, q_tensor, B_tensor
from skopt import gp_minimize
from skopt.space import Integer, Real


class Hydro_NN(PINN):
    """
    Class used to represent the Manipulator Informed Neural Network, child of PINN.
    """

    def __init__(self, layers, lb, ub, X_f=None):
        """
        Constructor.

        :param list layers: widths of the layers
        :param np.ndarray lb: lower bound of the inputs of the training data
        :param np.ndarray ub: upper bound of the inputs of the training data
        :param np.ndarray X_f: collocation points
        """

        super().__init__(layers, lb, ub)

        self.t = None
        self.x0 = None
        self.u = None

        if X_f is not None:
            self.set_collocation_points(X_f)

    def set_collocation_points(self, X_f):
        self.t = self.tensor(X_f[:, 0:1])  # 取第1列，索引为0
        self.x0 = self.tensor(X_f[:, 1:6]) # 取第2列到第4列
        self.u = self.tensor(X_f[:, 6:9])  # 取第6列到第7列

    @tf.function
    def f_model(self, X_f=None):
        """
        The actual Physics Informed Neural Network for the approximation of the equation of motion of the
        Schunk PowerCube Serial Robot.

        :return: tf.Tensor: the prediction of the PINN 预测值
        返回物理方程项的预测值
        """
        # del_V = del_t*(Q_out- Q_in)
        # del_V = 1/2*B*L(del_h_in+del_h_out)

        if X_f is None:
            t = self.t
            x0 = self.x0
            u = self.u
        else:
            t = self.tensor(X_f[:, 0:1])
            x0 = self.tensor(X_f[:, 1:6])
            u = self.tensor(X_f[:, 6:9])

        y_z_pred = self.model_z(tf.concat([t, x0, u], axis=1))
        y_u_pred = self.model_u(tf.concat([t, x0, u], axis=1))

        # print(y_z_pred)
        # print('y_z_pred.shape',y_z_pred.shape)

        z0 = tf.constant([-1.4, -2.4, -3.4, -4.4, -5.4], dtype=y_z_pred.dtype)

        f_pred_list = []

        for i in range(4):
            # 提取第 i 和第 i+1 个断面的预测值
            h_i = y_z_pred[:, i:i + 1] - z0[i]
            h_ip1 = y_z_pred[:, i + 1:i + 2] - z0[i + 1]

            B_i = 1 + 3 * h_i
            B_ip1 = 1 + 3 * h_ip1

            A_i = (2 + 3 * h_i) / 2
            A_ip1 = (2 + 3 * h_ip1) / 2

            Q_i = A_i * y_u_pred[:, i:i + 1]
            Q_ip1 = A_ip1 * y_u_pred[:, i + 1:i + 2]

            # 计算物理项 f_pred (T-1, 1)
            f = 500 * 0.5 * (B_i[:-1] + B_ip1[:-1]) * (h_i[1:] - h_i[:-1] + h_ip1[1:] - h_ip1[:-1]) \
                - 600 * (Q_ip1[:-1] - Q_i[:-1])
            f_pred_list.append(f)

        f_pred = tf.concat(f_pred_list, axis=1)

        return f_pred

def objective(params):
    N_layer, N_neuron, learning_rate = params
    layers = [input_dim, *N_layer * [N_neuron], output_dim]
    pinn = Hydro_NN(layers, lb, ub)
    # 训练
    logging.info(
        f'\Start training of the PINN with N_layer={N_layer}, N_neurons={N_neuron}, learning_rate={learning_rate}')
    start_time = time.time()
    pinn.set_collocation_points(X_star)
    pinn.fit(X_star, Y_z_star, Y_u_star, epochs, X_test, Y_z_test, Y_u_test, optimizer='adam', learning_rate=learning_rate,
             val_freq=1000, log_freq=1000)
    end_time = time.time()
    logging.info(f'\tTraining time: {end_time - start_time} seconds')
    loss_z = pinn.mean_squared_error_z.numpy()
    return loss_z

def train_and_save_best_model(best_params):
    N_layer = best_params['N_layer']
    N_neuron = best_params['N_neurons']
    learning_rate = best_params['learning_rate']
    print(N_layer, N_neuron, learning_rate)
    layers = [input_dim, *N_layer * [N_neuron], output_dim]
    pinn_best = Hydro_NN(layers, lb, ub)
    logging.info(f'Start training of the best PINN with N_layer={N_layer}, N_neurons={N_neuron}, learning_rate={learning_rate}')
    start_time = time.time()
    pinn_best.fit(X_star, Y_z_star, Y_u_star, epochs, X_test, Y_z_test, Y_u_test, optimizer='adam', learning_rate=learning_rate,
             val_freq=1000, log_freq=1000)
    loss_z = pinn_best.mean_squared_error_z.numpy()

    end_time = time.time()
    logging.info(f'Training time: {end_time - start_time} seconds')
    # Save the best model weights
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)
    pinn_best.save_weights_z(os.path.join(weights_path, 'best_checkpoint/model_z/.'))
    pinn_best.save_weights_u(os.path.join(weights_path, 'best_checkpoint/model_u/.'))
    logging.info(f'Saved best model weights to {os.path.join(weights_path, "best_checkpoint/")}')

    return pinn_best, loss_z

if __name__ == "__main__":
    # LOAD_WEIGHTS = True
    LOAD_WEIGHTS = False
    TRAIN_NET = True

    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    logging.info("TensorFlow version: {}".format(tf.version.VERSION))
    logging.info("Eager execution: {}".format(tf.executing_eagerly()))

    # Hyper parameter
    N_train = 1
    # epochs = 500000
    epochs = 5000
    N_phys = 20000
    N_data = 100

    logging.info(f'Epochs: {epochs}')

    # Paths
    data_path = os.path.join('./data/data_0.npz')
    data_sc_path = os.path.join('./data/data_sc_4min.npz')
    weights_path = os.path.join('./weights')

    print('加载数据')
    lb, ub, input_dim, output_dim, X_test, Y_z_test,Y_u_test, X_star, Y_z_star, Y_u_star = load_data(data_path)
    X_sc, Y_z_sc, Y_u_sc = load_data_sc(data_sc_path)

    # 定义参数空间
    space = [
        Integer(3,4, name='N_layer'),
        Integer(20,40, name='N_neurons'),
        Real(0.001, 0.1, name='learning_rate')
    ]
    # 贝叶斯优化
    result = gp_minimize(objective, space, n_calls=20, n_random_starts=10, random_state=0)

    best_params = {
        'N_layer': result.x[0],
        'N_neurons': result.x[1],
        'learning_rate': result.x[2]
    }
    best_loss = result.fun
    logging.info(f'Best parameters: {best_params} with loss: {best_loss}')
    # 使用最优参数重新训练模型并保存权重
    pinn_best, best_loss = train_and_save_best_model(best_params)

    print('评价')
    # PINN Evaluation 这边的X_test应该是实测数据的连续时段的
    Y_z_pred, Y_u_pred, F_pred = pinn_best.predict(X_sc)

    # t_step = X_test[1, 0] - X_test[0, 0]
    tau = 600 # 10 分钟
    # T = np.arange(t_step, 12 * tau + t_step, t_step)
    T = np.arange(0, 119 * tau , tau)

    plot_input_sequence(T, X_sc[:-1, 6:], filename='beyas控制输入')

    plot_states_z(T, Y_z_sc[:-1, :], Y_z_pred[:-1, :], filename='beyas水位')
    plot_absolute_error_z(T, Y_z_sc[:-1, :], Z_pred=Y_z_pred[:-1, :], filename='beyas水位误差对比图')

    plot_states_u(T, Y_u_sc[:-1, :], Y_u_pred[:-1, :], filename='beyas流速')
    plot_absolute_error_u(T, Y_u_sc[:-1, :], Z_pred=Y_u_pred[:-1, :], filename='beyas流速误差对比图')

    if click.confirm('Do you want to save (overwrite) the models weights?'):
        pinn.save_weights_z(rf'{weights_path}/easy_checkpoint_model_z/')
        pinn.save_weights_u(rf'{weights_path}/easy_checkpoint_model_u/')






