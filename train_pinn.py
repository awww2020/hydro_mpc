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
        self.t = self.tensor(X_f[:, 0:1])
        self.x0 = self.tensor(X_f[:, 1:6])
        self.u = self.tensor(X_f[:, 6:9])

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
    print('X_star',X_star.shape)
    '''
    print(Y_z_test)
    print(Y_z_star)
    print(Y_z_sc)
    '''
    N_layer = 3
    N_neurons = 40
    layers = [input_dim, *N_layer * [N_neurons], output_dim]

    # PINN initialization
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
    # PINN training
    if TRAIN_NET:
        for i in range(N_train):
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
    # PINN Evaluation
    print(X_sc)
    # Y_z_pred, Y_u_pred, F_pred = pinn.predict(X_sc)
    Y_z_pred, Y_u_pred = pinn.predict(X_sc)
    print(Y_z_pred)
    print(Y_z_pred.shape)

    # t_step = X_test[1, 0] - X_test[0, 0]
    tau = 240
    # T = np.arange(t_step, 12 * tau + t_step, t_step)
    T = np.arange(0, 300 * tau , tau)
    print(T.shape)
    print(X_sc.shape)
    print(Y_z_sc.shape)

    output_path = "results_prediction.txt"
    with open(output_path, "w") as f:
        f.write("X_sc\tY_z_pred\tY_z_sc\n")

        for x, y_pred, y_true in zip(X_sc, Y_z_pred/100, Y_z_sc/100):
            x_str = ",".join(map(str, x.tolist())) if hasattr(x, "tolist") else str(x)
            yp_str = ",".join(map(str, y_pred.tolist())) if hasattr(y_pred, "tolist") else str(y_pred)
            yt_str = ",".join(map(str, y_true.tolist())) if hasattr(y_true, "tolist") else str(y_true)
            f.write(f"{x_str}\t{yp_str}\t{yt_str}\n")

    pinn.plot_train_results()
    pinn.reset_train_results

    if os.environ.get('PINN_SKIP_SAVE_PROMPT') == '1':
        print('Skip saving model weights because PINN_SKIP_SAVE_PROMPT=1')
    elif click is None:
        print('Skip saving model weights because click is not installed.')
    elif click.confirm('Do you want to save (overwrite) the models weights?'):
        pinn.save_weights_z(rf'{weights_path}/easy_checkpoint_model_z/')
        pinn.save_weights_u(rf'{weights_path}/easy_checkpoint_model_u/')
