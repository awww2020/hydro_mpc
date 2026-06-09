import os

try:
    import click
except ImportError:
    click = None
import logging
import time

import numpy as np
import tensorflow as tf
from model_v1.nn import NN
from model_v1.pinn import PINN
from utils.data import generate_collocation_points, load_data,load_data_sc
from utils.plotting import plot_states_z, plot_states_u, plot_input_sequence, plot_absolute_error_z, plot_absolute_error_u

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
    '''
    @tf.function
    def f_model(self, X_f=None):
        """
        The actual Physics Informed Neural Network for the approximation of the equation.
        :return: tf.Tensor: the prediction of the PINN 预测值
        返回物理方程项的预测值
        """
        # del_V = del_t*(Q_out- Q_in)
        # del_V = 1/2*B*L(del_h_in+del_h_out)
        x = tf.convert_to_tensor(X_f, dtype=tf.float64)

        y_z_pred = self.model_z(x)  # [N, output_dim]
        y_u_pred = self.model_u(x)  # [N, output_dim]

        f_pred_list = []
        # z0单位m
        z0 = tf.constant([-1.4, -2.4, -3.4, -4.4, -5.4], dtype=y_z_pred.dtype)

        for i in range(4):
            # 提取第 i 和第 i+1 个断面的预测值
            # 断面水深 h_i和h_ip1 单位 m
            h_i = y_z_pred[:, i:i + 1]/100 - z0[i]
            h_ip1 = y_z_pred[:, i + 1:i + 2]/100 - z0[i + 1]

            # 断面河宽 单位 m
            B_i = 1.0 + 3.0 * h_i
            B_ip1 = 1.0 + 3.0 * h_ip1

            # 断面面积 单位 m
            A_i = (2.0 + 3.0 * h_i) / 2
            A_ip1 = (2.0 + 3.0 * h_ip1) / 2

            Q_i = A_i * y_u_pred[:, i:i + 1]
            Q_ip1 = A_ip1 * y_u_pred[:, i + 1:i + 2]

            # 计算物理项 f_pred (T-1, 1)

            f = 500.0 * 0.5 * ( (B_i[:-1] + B_i[1:]) * (h_i[1:] - h_i[:-1]) +(B_ip1[:-1] + B_ip1[1:])*(h_ip1[1:] - h_ip1[:-1]) ) \
                - 240.0 * (Q_ip1[:-1] - Q_i[:-1])
            f_pred_list.append(f)

        f_pred = tf.concat(f_pred_list, axis=1)

        return f_pred
    '''

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
    epochs = 20000
    N_phys = 10000
    N_data = 100

    logging.info(f'Epochs: {epochs}')
    logging.info(f'N_data: {N_data}')
    logging.info(f'N_phys: {N_phys}')

    # Paths
    data_path = os.path.join('./data/data_0.npz')
    data_sc_path = os.path.join('./data/data_sc_4min.npz')
    weights_path = os.path.join('./weights')
    weights_path_z = os.path.join('E:/program/hydro_MPC/weights/easy_checkpoint_model_z/')
    weights_path_u = os.path.join('E:/program/hydro_MPC/weights/easy_checkpoint_model_u/')

    print('加载数据')
    lb, ub, input_dim, output_dim, X_test, Y_z_test,Y_u_test, X_star, Y_z_star, Y_u_star = load_data(data_path)
    X_sc, Y_z_sc, Y_u_sc = load_data_sc(data_sc_path)

    Y_z_test = Y_z_test/100
    Y_z_star = Y_z_star/100
    Y_z_sc = Y_z_sc/100
    # X_star是以60秒为间隔 (9353, 9)
    print('X_star',X_star.shape)
    # 以data加载的水位以m为单位，流速以m/s为单位
    # X_test是以60秒为间隔 (990, 9)
    '''
    print(Y_z_test)
    print(Y_z_star)
    print(Y_z_sc)
    '''
    N_layer = 3
    N_neurons = 40
    layers = [input_dim, *N_layer * [N_neurons], output_dim]

    # PINN initialization 初始化
    print('pinn初始化')
    pinn = Hydro_NN(layers, lb, ub)
    pinn.strategy = os.environ.get('PINN_STRATEGY', pinn.strategy).lower()
    if pinn.strategy not in {'lra', 'relobralo', 'gradnorm', 'grad'}:
        raise ValueError(f"Unknown PINN_STRATEGY: {pinn.strategy}")
    pinn.set_constraint_count(os.environ.get('PINN_NC', 5))
    pinn.name = os.environ.get('PINN_RUN_NAME', 'PINN_5_grad')

    if LOAD_WEIGHTS:
        pinn.load_weights(weights_path_z,weights_path_u)

    print(X_star.shape)
    print(Y_z_star.shape)
    print(X_test.shape)
    # PINN training 训练
    if TRAIN_NET:
        for i in range(N_train):  #训练2次，保障可重复性
            # Generate training data via LHS
            print('lb', lb)
            print('ub', ub)
            # X_data, Y_data = generate_data_points(N_data, lb, ub)  # 100
            logging.info(f'\t{i + 1}/{N_train} Start training of the PINN')
            pinn.set_collocation_points(X_star)
            start_time = time.time()
            pinn.fit(X_star, Y_z_star, Y_u_star, epochs, X_test, Y_z_test, Y_u_test, optimizer='adam', learning_rate=0.01,
                     val_freq=1000, log_freq=1000)

    print('评价')
    # PINN Evaluation 这边的X_test应该是实测数据的连续时段的
    print(X_sc)
    # Y_z_pred, Y_u_pred, F_pred = pinn.predict(X_sc)
    Y_z_pred, Y_u_pred = pinn.predict(X_sc)
    print(Y_z_pred)
    print(Y_z_pred.shape)

    # t_step = X_test[1, 0] - X_test[0, 0]
    tau = 240 # 10 分钟
    # T = np.arange(t_step, 12 * tau + t_step, t_step)
    T = np.arange(0, 300 * tau , tau)  # 20小时有72000秒，120个600秒
    print(T.shape)
    print(X_sc.shape)
    print(Y_z_sc.shape)

    # 2. 将数据写入 txt 文件
    output_path = "results_prediction.txt"
    with open(output_path, "w") as f:
        # 写入表头
        f.write("X_sc\tY_z_pred\tY_z_sc\n")

        # 假设 X_sc、Y_z_pred、Y_z_sc 都是形如 (N, d) 或 (N,) 的 numpy 数组
        for x, y_pred, y_true in zip(X_sc, Y_z_pred/100, Y_z_sc/100):
            # 如果是多维向量，用 list() 或 tolist() 转成可打印的格式
            x_str = ",".join(map(str, x.tolist())) if hasattr(x, "tolist") else str(x)
            yp_str = ",".join(map(str, y_pred.tolist())) if hasattr(y_pred, "tolist") else str(y_pred)
            yt_str = ",".join(map(str, y_true.tolist())) if hasattr(y_true, "tolist") else str(y_true)
            f.write(f"{x_str}\t{yp_str}\t{yt_str}\n")

    #plot_input_sequence(T, X_sc[:-1, 6:9],filename=f'{pinn.name}_控制输入')

    #plot_states_z(T, Y_z_sc[:-1, :], Y_z_pred[:-1, :],filename=f'{pinn.name}_水位')
    #plot_absolute_error_z(T, Y_z_sc[:-1, :], Z_pred = Y_z_pred[:-1, :],filename=f'{pinn.name}_水位误差对比图')

    #plot_states_u(T, Y_u_sc[:-1, :], Y_u_pred[:-1, :], filename=f'{pinn.name}_流速')
    #plot_absolute_error_u(T, Y_u_sc[:-1, :], Z_pred=Y_u_pred[:-1, :], filename=f'{pinn.name}_流速误差对比图')

    pinn.plot_train_results()
    pinn.reset_train_results

    if os.environ.get('PINN_SKIP_SAVE_PROMPT') == '1':
        print('Skip saving model weights because PINN_SKIP_SAVE_PROMPT=1')
    elif click is None:
        print('Skip saving model weights because click is not installed.')
    elif click.confirm('Do you want to save (overwrite) the models weights?'):
        pinn.save_weights_z(rf'{weights_path}/easy_checkpoint_model_z/')
        pinn.save_weights_u(rf'{weights_path}/easy_checkpoint_model_u/')
