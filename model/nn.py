import abc
import logging
import os
import time
import datetime

from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.python.keras import Sequential, backend
from tensorflow.python.keras.layers import Dense, InputLayer, Lambda, Dropout, LayerNormalization, Activation

from optimizer.lbfgs import LBFGS
from utils.plotting import new_fig, save_fig

CHECKPOINTS_PATH = os.path.join('../checkpoints')


class NN(object, metaclass=abc.ABCMeta):
    """
    Abstract class used to represent a Neural Network.
    """

    def __init__(self, layers: list, lb: np.ndarray, ub: np.ndarray) -> None:
        """
        Constructor.

        :param list layers: widths of the layers
        :param np.ndarray lb: lower bounds of the inputs of the training data
        :param np.ndarray ub: upper bounds of the inputs of the training data
        """

        self.checkpoints_dir = CHECKPOINTS_PATH

        self.dtype = "float64"
        # Descriptive Keras model
        backend.set_floatx(self.dtype)

        self.input_dim = layers[0]
        self.output_dim = layers[-1]

        # Keras Sequential Model
        self.model_z = Sequential()
        self.model_u = Sequential()

        # Input Layer
        self.model_z.add(InputLayer(input_shape=(self.input_dim,)))
        self.model_u.add(InputLayer(input_shape=(self.input_dim,)))

        lb = tf.constant(lb, dtype=tf.float64)
        ub = tf.constant(ub, dtype=tf.float64)

        # Normalization Layer
        self.model_z.add(Lambda(lambda X: 2.0 * (X - lb) / (ub - lb) - 1.0))
        self.model_u.add(Lambda(lambda X: 2.0 * (X - lb) / (ub - lb) - 1.0))

        # Hidden Layer
        '''
        for layer_width in layers[1:-1]:
            self.model_z.add(Dense(layer_width, activation=tf.nn.tanh,
                                 kernel_initializer='glorot_normal'))
            # self.model_z.add(Dropout(rate=0.3))

        for layer_width in layers[1:-1]:
            self.model_u.add(Dense(layer_width, activation=tf.nn.tanh,
                                   kernel_initializer='glorot_normal'))
        '''
        for layer_width in layers[1:-1]:
            self.model_z.add(Dense(layer_width,
                                   activation=None,
                                   kernel_initializer='glorot_normal'))
            # 添加 LayerNorm
            self.model_z.add(LayerNormalization())
            # 再加激活
            self.model_z.add(Activation('tanh'))



        for layer_width in layers[1:-1]:
            self.model_u.add(Dense(layer_width,
                                   activation=None,
                                   kernel_initializer='glorot_normal'))
            # 添加 LayerNorm
            self.model_u.add(LayerNormalization())
            # 再加激活
            self.model_u.add(Activation('tanh'))


        # Output Layer :
        self.model_z.add(Dense(self.output_dim))
        self.model_u.add(Dense(self.output_dim))

        self.optimizer = None
        self.loss_object = tf.keras.losses.MeanSquaredError()

        self.start_time = None
        self.prev_time = None

        # Store metrics
        self.train_loss_results_z = {}
        self.train_loss_results_u = {}
        self.train_accuracy_results_z = {}
        self.train_accuracy_results_u = {}
        self.train_time_results = {}
        self.train_pred_results_z = {}
        self.train_pred_results_u = {}

        self.early_stop_wait_z = 0
        self.early_stop_best_loss_z = float('inf')
        self.early_stop_triggered_z = False

        self.early_stop_wait_u = 0
        self.early_stop_best_loss_u = float('inf')
        self.early_stop_triggered_u = False

    def tensor(self, X):
        """
        Converts a list or numpy array to a tf.tensor.

        :param list or nd.array X:
        :return: tf.tensor: tensor of X
        """
        return tf.convert_to_tensor(X, dtype=self.dtype)

    def summary(self):
        """
        Pipes the Keras.model.summary function to the logging.
        """

        self.model_z.summary(print_fn=lambda x: logging.info(x))
        self.model_u.summary(print_fn=lambda x: logging.info(x))

    @tf.function
    def train_step(self, x, y_z, y_u):
        """
        Performs training step during training.

        :param tf.tensor x: (batched) input tensor of training data
        :param tf.tensor y: (batched) output tensor of training data
        :return: float loss: the corresponding current loss value
        """
        def loss_f(y, y_pred):
            penalty_weight = 10.0
            # 基础 MSE
            L_data = tf.reduce_mean(tf.square(y - y_pred))
            # 初值：输入第5列（Python 索引 4）
            y0 = x[:, 5:6]
            # 预测第5个分量
            y5 = y_pred[:, 4:5]
            # 允许的最大变化量 Δ_max = x[:,0] * 0.001/60
            delta_max = x[:, 0:1] * (0.01 / 60.0)
            # 超出部分
            violation = tf.nn.relu(tf.abs(y5 - y0) - delta_max)
            # 惩罚项
            penalty = tf.reduce_mean(tf.square(violation))

            return L_data + penalty_weight * penalty

        with tf.GradientTape(persistent=True) as tape:
            y_z_pred = self.model_z(x)
            y_u_pred = self.model_u(x)
            # f_pred = self.f_model(x)
            # loss_z = self.loss_object(y_z, y_z_pred)
            loss_z = self.loss_object(y_z, y_z_pred)
            loss_u = self.loss_object(y_u, y_u_pred)
            # loss_data = tf.reduce_mean(tf.square(f_pred))
            loss = loss_z + loss_u
        # gradients = tape.gradient(loss, self.model_u.trainable_variables)
        gradients_z = tape.gradient(loss, self.model_z.trainable_variables)
        gradients_u = tape.gradient(loss, self.model_u.trainable_variables)

        # self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.optimizer.apply_gradients(zip(gradients_z, self.model_z.trainable_variables))
        self.optimizer.apply_gradients(zip(gradients_u, self.model_u.trainable_variables))

        return loss_z, loss_u

    def fit(self, x, y_z, y_u, epochs=2000, x_test=None, y_z_test=None, y_u_test=None, optimizer='adam', learning_rate=0.1,
            load_best_weights=False, val_freq=1000, log_freq=1000, verbose=1):
        """
        Performs the neural network training phase.

        :param tf.tensor x: input tensor of the training dataset
        :param tf.tensor y: output tensor of the training dataset
        :param int epochs: number of training epochs
        :param tf.tensor x_test: input tensor of the test dataset, used to evaluate current accuracy
        :param tf.tensor y_test: output tensor of the test dataset, used to evaluate current accuracy
        :param str optimizer: name of the optimizer, choose from 'adam' or 'lbfgs'
        :param bool load_best_weights: flag to determine if the best weights corresponding to the best
        accuracy are loaded after training
        """

        x = self.tensor(x)
        y_z = self.tensor(y_z)
        y_u = self.tensor(y_u)

        self.start_time = time.time()
        self.prev_time = self.start_time

        if optimizer == 'adam':
            self.train_adam(x, y_z, y_u, epochs, x_test, y_z_test, y_u_test, learning_rate, val_freq, log_freq, verbose)
        elif optimizer == 'lbfgs':
            self.train_lbfgs(x, y_z, y_u, epochs, x_test, y_z_test, y_u_test, learning_rate, val_freq, log_freq, verbose)

        [self.mean_squared_error_z, errors_z, Y_z_pred] = self.evaluate_z(x_test, y_z_test)
        [self.mean_squared_error_u, errors_u, Y_u_pred] = self.evaluate_u(x_test, y_z_test)

        if load_best_weights is True:
            self.load_weights()

    def train_adam(self, x, y_z, y_u, epochs=2000, x_test=None, y_z_test=None, y_u_test=None, learning_rate=0.1, val_freq=1000, log_freq=1000,
                   verbose=1):
        """
        Performs the neural network training, using the adam optimizer.

        :param tf.tensor x: input tensor of the training dataset
        :param tf.tensor y: output tensor of the training dataset
        :param int epochs: number of training epochs
        :param tf.tensor x_test: input tensor of the test dataset, used to evaluate accuracy
        :param tf.tensor y_test: output tensor of the test dataset, used to evaluate accuracy
        """

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        epoch_loss_z = tf.keras.metrics.Mean(name='epoch_loss_z')
        epoch_loss_u = tf.keras.metrics.Mean(name='epoch_loss_u')
        if verbose:
            logging.info(f'Start ADAM optimization')

        for epoch in range(1, epochs + 1):
            loss_z, loss_u = self.train_step(x, y_z, y_u)
            # Track progress
            epoch_loss_z.update_state(loss_z)  # Add current batch loss
            epoch_loss_u.update_state(loss_u)
            self.epoch_callback_adam(epoch, epoch_loss_z.result(), epoch_loss_u.result(), epochs, x_test, y_z_test, y_u_test, val_freq, log_freq,
                                verbose)

    def joint_loss(self, x, y_z, y_u):
        y_z_pred = self.model_z(x)
        y_u_pred = self.model_u(x)
        loss_z = self.loss_object(y_z, y_z_pred)
        loss_u = self.loss_object(y_u, y_u_pred)
        return loss_z + loss_u

    def train_lbfgs(self, x, y_z, y_u, epochs=2000, x_test=None, y_z_test=None, y_u_test=None, learning_rate=1.0, val_freq=1000, log_freq=1000,
                    verbose=1):
        """
        Performs the neural network training, using the L-BFGS optimizer.

        :param tf.tensor x: input tensor of the training dataset
        :param tf.tensor y: output tensor of the training dataset
        :param int epochs: number of training epochs
        :param tf.tensor x_test: input tensor of the test dataset, used to evaluate accuracy
        :param tf.tensor y_test: output tensor of the test dataset, used to evaluate accuracy
        """

        # train the model with L-BFGS solver
        if verbose:
            logging.info(f'Start L-BFGS optimization')

        # trainable_variables = self.model_z.trainable_variables + self.model_u.trainable_variables

        optimizer = LBFGS()

        self.penalty_weight = 1.0

        def loss_f(y, y_pred):
            # 基础 MSE
            L_data = tf.reduce_mean(tf.square(y - y_pred))
            # 初值：输入第5列（Python 索引 4）
            y0 = x[:, 1:6]
            # 预测第5个分量
            y5 = y_pred[:, 0:5]
            # 允许的最大变化量 Δ_max = x[:,0] * 0.001/60
            delta_max = x[:, 0:1] * (0.01 / 60.0)
            # 超出部分
            violation = tf.nn.relu(tf.abs(y5 - y0) - delta_max)
            # 惩罚项
            penalty = tf.reduce_mean(tf.square(violation))
            print('L_data',L_data, 'violation',penalty)

            return L_data + self.penalty_weight * penalty

        '''
        optimizer.minimize(
            self.model, self.loss_object, x, y_z, y_u, self.epoch_callback, epochs, x_test=x_test, y_z_test=y_z_test, y_u_test=y_u_test,
            val_freq=val_freq, log_freq=log_freq, verbose=verbose, learning_rate=learning_rate)
        '''

        try:
            optimizer.minimize(
                self.model_z, self.loss_object, x, y_z, self.epoch_callback_z, epochs, x_test=x_test, y_test=y_z_test,
                val_freq=val_freq, log_freq=log_freq, verbose=verbose, learning_rate=learning_rate)
        except StopIteration as e:
            print(e)

        try:
            optimizer.minimize(
                self.model_u, self.loss_object, x, y_u, self.epoch_callback_u, epochs=5000, x_test=x_test, y_test=y_u_test,
                val_freq=val_freq, log_freq=log_freq, verbose=verbose, learning_rate=0.05)
        except StopIteration as e:
            print(e)

        '''
        val_freq 作用: 指定验证数据评估的频率。
        示例: 每 5 个 epoch 评估一次测试数据，val_freq=5
        log_freq  作用: 指定日志记录的频率。
        示例: 每 10 个 epoch 记录一次日志，log_freq=10
        verbose 作用: 控制训练过程中的输出信息详细程度。
        取值: 0 (不输出), 1 (进度条), 2 (每个 epoch 输出一行)
        示例: verbose=1
        '''

    '''
    def predict(self, x):
        """
        Calls the model prediction function and returns the prediction on an input tensor.

        :param tf.tensor x: input tensor
        :return: tf.tensor: output tensor
        """
        return self.model.predict(x)
    '''

    def predict_z(self, x):
        """
        Calls the model prediction function and returns the prediction on an input tensor.

        :param tf.tensor x: input tensor
        :return: tf.tensor: output tensor
        """
        return self.model_z.predict(x)

    def predict_u(self, x):
        """
        Calls the model prediction function and returns the prediction on an input tensor.

        :param tf.tensor x: input tensor
        :return: tf.tensor: output tensor
        """
        return self.model_u.predict(x)

    def plot_train_results(self, basename=None):
        """
        Visualizes the training metrics Loss resp. Accuracy over epochs.

        :param str basename: used to save the figure with this name, if None the figure is not saved
        """

        fig = new_fig()
        ax = fig.add_subplot(111)
        fig.suptitle(f'{self.name} - Training Metrics')

        ax.set_ylabel('Loss')
        ax.set_yscale('log')
        ax.plot(self.train_loss_results_z.keys(), self.train_loss_results_z.values(), label='Loss')
        if self.train_accuracy_results_z:
            ax.set_ylabel("Loss / Accuracy")
            ax.plot(self.train_accuracy_results_z.keys(), self.train_accuracy_results_z.values(), label='Accuracy')
        ax.set_xlabel("Epoch", fontsize=14)
        ax.legend(loc='best')
        if basename is not None:
            save_fig(fig, f'{basename}_train_metrics')
        fig.tight_layout()
        plt.show()

    def train_results_z(self):
        """
        Returns the training metrics stored in dictionaries.
        :return: dict: loss over epochs, dict: accuracy over epochs,
        dict: predictions (on the testing dataset) over epochs
        """

        return self.train_loss_results_z, self.train_accuracy_results_z, self.train_pred_results_z

    def train_results_u(self):
        """
        Returns the training metrics stored in dictionaries.
        :return: dict: loss over epochs, dict: accuracy over epochs,
        dict: predictions (on the testing dataset) over epochs
        """

        return self.train_loss_results_u, self.train_accuracy_results_u, self.train_pred_results_u


    def reset_train_results(self):
        """
        Clears the training metrics.
        """
        self.train_loss_results_z = {}
        self.train_loss_results_u = {}
        self.train_accuracy_results_z = {}
        self.train_accuracy_results_u = {}
        self.train_pred_results_z = {}
        self.train_pred_results_u = {}

    def get_weights(self):
        """
        Returns the model weights.

        :return: tf.tensor model weights
        """
        return self.model_z.get_weights(), self.model_u.get_weights()

    def set_weights(self, weights_z, weights_u):
        return self.model_z.set_weights(weights_z), self.model_u.set_weights(weights_u)

    def save_weights_z(self, path):
        """
        Saves the model weights under a specified path.

        :param str path: path where the weights are saved
        """
        Path(path).mkdir(parents=True, exist_ok=True)
        # 这边的save_weights是model自带的函数
        self.model_z.save_weights(path)

    def save_weights_u(self, path):
        """
        Saves the model weights under a specified path.

        :param str path: path where the weights are saved
        """
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model_u.save_weights(path)

    def load_weights(self, path_z=None, path_u=None):
        """
        Loads the model weights from a specified path.

        :param str path: path where the weights are saved,
        if None the weights are assumed to be saved at the checkpoints directory
        """

        if path_z is None:
            path_z = self.checkpoints_dir
        if path_u is None:
            path_u = self.checkpoints_dir

        self.model_z.load_weights(path_z)
        self.model_u.load_weights(path_u)

        logging.info(f'\tWeights loaded from {path_z},{path_u}')

    def get_epoch_duration(self):
        """
        Measures the time for a training epoch.

        :return: float time per epoch
        """
        now = time.time()
        epoch_duration = datetime.datetime.fromtimestamp(now - self.prev_time).strftime("%M:%S.%f")[:-4]
        self.prev_time = now
        return epoch_duration

    def get_elapsed_time(self):
        """
        Measures the time since training start.
        :return: float elapsed time
        """

        return datetime.timedelta(seconds=int(time.time() - self.start_time))

    def epoch_callback_adam(self, epoch, epoch_loss_z, epoch_loss_u, epochs, x_val=None, y_z_val=None, y_u_val=None, val_freq=1000, log_freq=1000,
                       verbose=1):
        """
        Callback function, which is called after each epoch, to produce proper training logging
        and keep track of training metrics.

        :param int epoch: current epoch
        :param float epoch_loss: current loss value
        :param int epochs: number of training epochs
        :param tf.tensor x_val: input tensor of the test dataset, used to evaluate current accuracy
        :param tf.tensor y_val: output tensor of the test dataset, used to evaluate current accuracy
        :param int val_freq: number of epochs passed before trigger validation
        :param int log_freq: number of epochs passed before each logging
        """
        self.train_loss_results_z[epoch] = epoch_loss_z
        self.train_loss_results_u[epoch] = epoch_loss_u
        elapsed_time = self.get_elapsed_time()
        self.train_time_results[epoch] = elapsed_time

        if epoch % val_freq == 0 or epoch == 1:
            length = len(str(epochs))
            log_str = f'\tEpoch: {str(epoch).zfill(length)}/{epochs},\t' \
                      f'Loss: {epoch_loss_z:.4e},{epoch_loss_u:.4e} '

            if x_val is not None and y_z_val is not None:
                [mean_squared_error_z, errors_z, Y_z_pred] = self.evaluate_z(x_val, y_z_val)
                self.train_accuracy_results_z[epoch] = mean_squared_error_z
                self.train_pred_results_z[epoch] = Y_z_pred
                log_str += f',\tAccuracy (MSE): {mean_squared_error_z:.4e}'
                if mean_squared_error_z <= min(self.train_accuracy_results_z.values()):
                    self.save_weights_z(os.path.join(self.checkpoints_dir, 'easy_checkpoint_model_z'))

            if x_val is not None and y_u_val is not None:
                [mean_squared_error_u, errors_u, Y_u_pred] = self.evaluate_u(x_val, y_u_val)
                self.train_accuracy_results_u[epoch] = mean_squared_error_u
                self.train_pred_results_u[epoch] = Y_u_pred
                log_str += f',\tAccuracy (MSE): {mean_squared_error_u:.4e}'
                if mean_squared_error_u <= min(self.train_accuracy_results_u.values()):
                    self.save_weights_u(os.path.join(self.checkpoints_dir, 'easy_checkpoint_model_u'))


            if (epoch % log_freq == 0 or epoch == 1) and verbose == 1:
                log_str += f',\t Elapsed time: {elapsed_time} (+{self.get_epoch_duration()})'
                logging.info(log_str)

        if epoch == epochs and x_val is None and y_z_val is None:
            self.save_weights_z(os.path.join(self.checkpoints_dir, 'easy_checkpoint_model_z'))

        if epoch == epochs and x_val is None and y_u_val is None:
            self.save_weights_u(os.path.join(self.checkpoints_dir, 'easy_checkpoint_model_u'))

    def epoch_callback_z(self, epoch, epoch_loss, epochs, x_val=None, y_val=None, val_freq=1000, log_freq=1000,
                       verbose=1):
        """
        Callback function, which is called after each epoch, to produce proper training logging
        and keep track of training metrics.

        :param int epoch: current epoch
        :param float epoch_loss: current loss value
        :param int epochs: number of training epochs
        :param tf.tensor x_val: input tensor of the test dataset, used to evaluate current accuracy
        :param tf.tensor y_val: output tensor of the test dataset, used to evaluate current accuracy
        :param int val_freq: number of epochs passed before trigger validation
        :param int log_freq: number of epochs passed before each logging
        """
        self.train_loss_results_z[epoch] = epoch_loss
        elapsed_time = self.get_elapsed_time()
        self.train_time_results[epoch] = elapsed_time

        # -------- Early stopping 判断逻辑 --------
        patience = 20
        min_delta = 1e-6

        if epoch % val_freq == 0 or epoch == 1:
            length = len(str(epochs))
            log_str = f'\tEpoch: {str(epoch).zfill(length)}/{epochs},\t' \
                      f'Loss: {epoch_loss:.4e}'

            if x_val is not None and y_val is not None:
                [mean_squared_error_z, errors_z, Y_z_pred] = self.evaluate_z(x_val, y_val)
                self.train_accuracy_results_z[epoch] = mean_squared_error_z
                self.train_pred_results_z[epoch] = Y_z_pred
                log_str += f',\tAccuracy (MSE): {mean_squared_error_z:.4e}'
                if mean_squared_error_z <= min(self.train_accuracy_results_z.values()):
                    self.save_weights_z(os.path.join(self.checkpoints_dir, 'easy_checkpoint_model_z'))

                # ---------- Early stopping 检查逻辑 ----------
                if mean_squared_error_z < self.early_stop_best_loss_z - min_delta:
                    self.early_stop_best_loss_z = mean_squared_error_z
                    self.early_stop_wait_z = 0
                else:
                    self.early_stop_wait_z += 1
                    if self.early_stop_wait_z >= patience:
                        self.early_stop_triggered_z = True
                        logging.info(
                            f"[EarlyStopping] Stop at epoch {epoch}, best val MSE: {self.early_stop_best_loss_z:.6e}")

            if (epoch % log_freq == 0 or epoch == 1) and verbose == 1:
                log_str += f',\t Elapsed time: {elapsed_time} (+{self.get_epoch_duration()})'
                logging.info(log_str)

        if epoch == epochs and x_val is None and y_val is None:
            self.save_weights_z(os.path.join(self.checkpoints_dir, 'easy_checkpoint_model_z'))

    def epoch_callback_u(self, epoch, epoch_loss, epochs, x_val=None, y_val=None, val_freq=1000, log_freq=1000,
                       verbose=1):
        """
        Callback function, which is called after each epoch, to produce proper training logging
        and keep track of training metrics.

        :param int epoch: current epoch
        :param float epoch_loss: current loss value
        :param int epochs: number of training epochs
        :param tf.tensor x_val: input tensor of the test dataset, used to evaluate current accuracy
        :param tf.tensor y_val: output tensor of the test dataset, used to evaluate current accuracy
        :param int val_freq: number of epochs passed before trigger validation
        :param int log_freq: number of epochs passed before each logging
        """
        self.train_loss_results_u[epoch] = epoch_loss
        elapsed_time = self.get_elapsed_time()
        self.train_time_results[epoch] = elapsed_time

        # -------- Early stopping 判断逻辑 --------
        patience = 20
        min_delta = 1e-6

        if epoch % val_freq == 0 or epoch == 1:
            length = len(str(epochs))
            log_str = f'\tEpoch: {str(epoch).zfill(length)}/{epochs},\t' \
                      f'Loss: {epoch_loss:.4e}'

            if x_val is not None and y_val is not None:
                [mean_squared_error_u, errors_u, Y_u_pred] = self.evaluate_u(x_val, y_val)
                self.train_accuracy_results_u[epoch] = mean_squared_error_u
                self.train_pred_results_u[epoch] = Y_u_pred
                log_str += f',\tAccuracy (MSE): {mean_squared_error_u:.4e}'
                if mean_squared_error_u <= min(self.train_accuracy_results_u.values()):
                    self.save_weights_u(os.path.join(self.checkpoints_dir, 'easy_checkpoint_model_u'))

                if mean_squared_error_u < self.early_stop_best_loss_z - min_delta:
                    self.early_stop_best_loss_u = mean_squared_error_u
                    self.early_stop_wait_u = 0
                else:
                    self.early_stop_wait_u += 1
                    if self.early_stop_wait_u >= patience:
                        self.early_stop_triggered_u = True
                        logging.info(
                            f"[EarlyStopping] Stop at epoch {epoch}, best val MSE: {self.early_stop_best_loss_z:.6e}")

            if (epoch % log_freq == 0 or epoch == 1) and verbose == 1:
                log_str += f',\t Elapsed time: {elapsed_time} (+{self.get_epoch_duration()})'
                logging.info(log_str)

        if epoch == epochs and x_val is None and y_val is None:
            self.save_weights_u(os.path.join(self.checkpoints_dir, 'easy_checkpoint_model_u'))

    def evaluate_z(self, x_val, y_val, metric='MSE'):
        """
        Calculates the accuracy on a testing dataset.

        :param tf.tensor x_val: input tensor of the testing dataset
        :param tf.tensor y_val: output tensor of the testing dataset
        :param str metric: name of the error type, choose from 'MSE' or 'MAE'
        :return: tf.tensor mean_error: the mean squared/absolute error value,
        tf.tensor errors: the squared/absolute errors over inputs,
        tf.tensor y_pred: the prediction on the inputs of the testing dataset
        """

        y_pred = self.model_z.predict(x_val)
        errors = None
        if metric == 'MSE':
            errors = tf.square(y_val - y_pred)
        elif metric == 'MAE':
            errors = tf.abs(y_val - y_pred)

        mean_error = tf.reduce_mean(errors)

        return mean_error, errors, y_pred

    def evaluate_u(self, x_val, y_val, metric='MSE'):
        """
        Calculates the accuracy on a testing dataset.

        :param tf.tensor x_val: input tensor of the testing dataset
        :param tf.tensor y_val: output tensor of the testing dataset
        :param str metric: name of the error type, choose from 'MSE' or 'MAE'
        :return: tf.tensor mean_error: the mean squared/absolute error value,
        tf.tensor errors: the squared/absolute errors over inputs,
        tf.tensor y_pred: the prediction on the inputs of the testing dataset
        """

        y_pred = self.model_u.predict(x_val)
        errors = None
        if metric == 'MSE':
            errors = tf.square(y_val - y_pred)
        elif metric == 'MAE':
            errors = tf.abs(y_val - y_pred)

        mean_error = tf.reduce_mean(errors)

        return mean_error, errors, y_pred

    def prediction_time(self, batch_size, executions=1000):
        """
        Helper function to measure the mean prediction time of the neural network.

        :param int batch_size: dummy batch size of the input tensor
        :param int executions: number of performed executions to determine the mean value
        :return: float mean_prediction_time: the mean prediction time of the neural network on all executions
        """
        X = tf.random.uniform(shape=[executions, batch_size, self.input_dim], dtype=self.dtype)

        start_time = time.time()
        for execution in range(executions):
            _ = self.predict(X[execution])
        prediction_time = time.time() - start_time
        mean_prediction_time = prediction_time / executions

        return mean_prediction_time
