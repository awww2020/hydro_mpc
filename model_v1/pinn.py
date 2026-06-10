#增加物理方程的约束
import abc
import tensorflow as tf
from model_v1.nn import NN
import logging
import time
from optimizer.lbfgs import LBFGS
import os
import numpy as np

class CombinedModel(tf.keras.Model):
    def __init__(self, model_z, model_u):
        super().__init__()
        self.model_z = model_z
        self.model_u = model_u

    def call(self, x, training=False):
        z = self.model_z(x, training=training)
        u = self.model_u(x, training=training)
        return tf.concat([z, u], axis=1)  # [N, 2*output_dim]


class PINN(NN, metaclass=abc.ABCMeta):
    """
    Class used to represent a Physics informed Neural Network, children of NN.
    abc.ABCMeta 是 Python 内建模块 abc（Abstract Base Classes）的一部分，用于创建抽象基类。
    """

    def __init__(self, layers, lb, ub):
        """
        Constructor.

        :param list layers: widths of the layers
        :param np.ndarray lb: lower bounds of the inputs of the training data
        :param np.ndarray ub: upper bounds of the inputs of the training data
        """

        super().__init__(layers, lb, ub)
        self.model = CombinedModel(self.model_z, self.model_u)
        self.loss_object = tf.keras.losses.MeanSquaredError()

        self._x_loss = None
        self.w_z = tf.Variable(1.0, dtype=self.dtype, trainable=False)
        self.w_u = tf.Variable(1.0, dtype=self.dtype, trainable=False)
        self.w_phys = tf.Variable(1.0, dtype=self.dtype, trainable=False)
        self.w_ema = self.w_ema = tf.constant(0.9, dtype=self.dtype)
        self.z0 = tf.constant([-1.4, -2.4, -3.4, -4.4, -5.4], dtype=self.dtype)
        self.phys_scale = tf.Variable(1.0, dtype=self.dtype, trainable=False)

        self._x_loss = None

        self.strategy = "grad"
        self.constraint_indices = tf.constant([0, self.output_dim - 1], dtype=tf.int32)


        self.rel_T = tf.constant(1.0, dtype=self.dtype)
        self.rel_alpha = tf.constant(0.9, dtype=self.dtype)
        self.rel_rho = tf.constant(0.99, dtype=self.dtype)
        self.rel_eps = tf.constant(1e-12, dtype=self.dtype)

        self.lam_z = tf.Variable(1.0, dtype=self.dtype, trainable=True)
        self.lam_u = tf.Variable(1.0, dtype=self.dtype, trainable=True)
        self.lam_p = tf.Variable(1.0, dtype=self.dtype, trainable=True)

        self.lprev_z = tf.Variable(1.0, dtype=self.dtype, trainable=False)
        self.lprev_u = tf.Variable(1.0, dtype=self.dtype, trainable=False)
        self.lprev_p = tf.Variable(1.0, dtype=self.dtype, trainable=False)

        self.l0_z = tf.Variable(1.0, dtype=self.dtype, trainable=False)
        self.l0_u = tf.Variable(1.0, dtype=self.dtype, trainable=False)
        self.l0_p = tf.Variable(1.0, dtype=self.dtype, trainable=False)

        self.rel_inited = tf.Variable(False, dtype=tf.bool, trainable=False)

        # init
        self.phys_scale = tf.Variable(1.0, dtype=self.dtype, trainable=False)
        self.phys_scale_inited = tf.Variable(False, dtype=tf.bool, trainable=False)

        self.lra_alpha = tf.constant(0.9, self.dtype)
        self.lra_eps = tf.constant(1e-8, self.dtype)

        self.gn_alpha = tf.constant(1.5, self.dtype)
        self.gn_eps   = tf.constant(1e-12, self.dtype)
        self.gn_inited = tf.Variable(False, trainable=False, dtype=tf.bool)

        self.gn_l0_z = tf.Variable(1.0, trainable=False, dtype=self.dtype)
        self.gn_l0_u = tf.Variable(1.0, trainable=False, dtype=self.dtype)
        self.gn_l0_p = tf.Variable(1.0, trainable=False, dtype=self.dtype)

        self.opt_lam = tf.keras.optimizers.Adam(learning_rate=1e-3)

    def set_constraint_count(self, n_c):
        n_c = int(n_c)
        if n_c == 2:
            indices = [0, self.output_dim - 1]
        elif n_c == 3:
            indices = [0, self.output_dim // 2, self.output_dim - 1]
        elif n_c == self.output_dim:
            indices = list(range(self.output_dim))
        else:
            raise ValueError(f"Unsupported N_c={n_c}; expected 2, 3, or {self.output_dim}")
        self.constraint_indices = tf.constant(indices, dtype=tf.int32)
        self.n_constraints = n_c


    def _mean_abs_grad(self, grads):

        flat = []
        for g in grads:
            if g is None:
                continue
            flat.append(tf.reshape(g, [-1]))
        if not flat:
            return tf.constant(0.0, self.dtype)
        v = tf.concat(flat, axis=0)
        return tf.reduce_mean(tf.abs(v))

    def _zero_if_none(self, g, like):
        return tf.zeros_like(like) if g is None else g


    @tf.function
    def relobralo_update(self, L_z, L_u, L_p):
        Lz = tf.cast(L_z, self.dtype)
        Lu = tf.cast(L_u, self.dtype)
        Lp = tf.cast(L_p, self.dtype)

        def _init():
            self.l0_z.assign(tf.stop_gradient(Lz));
            self.lprev_z.assign(tf.stop_gradient(Lz))
            self.l0_u.assign(tf.stop_gradient(Lu));
            self.lprev_u.assign(tf.stop_gradient(Lu))
            self.l0_p.assign(tf.stop_gradient(Lp));
            self.lprev_p.assign(tf.stop_gradient(Lp))
            self.rel_inited.assign(True)
            return 0

        tf.cond(self.rel_inited, lambda: 0, _init)

        T = self.rel_T
        eps = self.rel_eps
        N = tf.cast(3.0, self.dtype)

        logits_prev = tf.stack([
            Lz / (self.lprev_z * T + eps),
            Lu / (self.lprev_u * T + eps),
            Lp / (self.lprev_p * T + eps),
        ], axis=0)

        logits_0 = tf.stack([
            Lz / (self.l0_z * T + eps),
            Lu / (self.l0_u * T + eps),
            Lp / (self.l0_p * T + eps),
        ], axis=0)

        lam_hat = tf.stop_gradient(tf.nn.softmax(logits_prev) * N)
        lam0_hat = tf.stop_gradient(tf.nn.softmax(logits_0) * N)

        alpha = self.rel_alpha
        rho = self.rel_rho

        new_lam_z = rho * alpha * self.lam_z + (1 - rho) * alpha * lam0_hat[0] + (1 - alpha) * lam_hat[0]
        new_lam_u = rho * alpha * self.lam_u + (1 - rho) * alpha * lam0_hat[1] + (1 - alpha) * lam_hat[1]
        new_lam_p = rho * alpha * self.lam_p + (1 - rho) * alpha * lam0_hat[2] + (1 - alpha) * lam_hat[2]

        self.lam_z.assign(tf.clip_by_value(new_lam_z, 1e-6, 1e6))
        self.lam_u.assign(tf.clip_by_value(new_lam_u, 1e-6, 1e6))
        self.lam_p.assign(tf.clip_by_value(new_lam_p, 1e-6, 1e6))

        self.lprev_z.assign(tf.stop_gradient(Lz))
        self.lprev_u.assign(tf.stop_gradient(Lu))
        self.lprev_p.assign(tf.stop_gradient(Lp))

        return self.lam_z, self.lam_u, self.lam_p

    @tf.function
    def lrannealing_update(self, tape, shared_vars, L_z, L_u, L_p, main='z'):
        Lz = tf.cast(L_z, self.dtype)
        Lu = tf.cast(L_u, self.dtype)
        Lp = tf.cast(L_p, self.dtype)

        gz = tape.gradient(Lz, shared_vars)
        gu = tape.gradient(Lu, shared_vars)
        gp = tape.gradient(Lp, shared_vars)

        mz = self._mean_abs_grad(gz)
        mu = self._mean_abs_grad(gu)
        mp = self._mean_abs_grad(gp)

        eps = self.lra_eps
        alpha = self.lra_alpha

        if main == 'z':
            m_main = mz
        elif main == 'u':
            m_main = mu
        else:
            m_main = mp

        lam_z_hat = m_main / (mz + eps)
        lam_u_hat = m_main / (mu + eps)
        lam_p_hat = m_main / (mp + eps)

        new_lam_z = alpha * self.lam_z + (1.0 - alpha) * lam_z_hat
        new_lam_u = alpha * self.lam_u + (1.0 - alpha) * lam_u_hat
        new_lam_p = alpha * self.lam_p + (1.0 - alpha) * lam_p_hat

        self.lam_z.assign(tf.clip_by_value(new_lam_z, 1e-6, 1e6))
        self.lam_u.assign(tf.clip_by_value(new_lam_u, 1e-6, 1e6))
        self.lam_p.assign(tf.clip_by_value(new_lam_p, 1e-6, 1e6))

        return self.lam_z, self.lam_u, self.lam_p

    @tf.function
    def gradnorm_update(self, tape, shared_W, L_z, L_u, L_p):

        Lz = tf.cast(L_z, self.dtype)
        Lu = tf.cast(L_u, self.dtype)
        Lp = tf.cast(L_p, self.dtype)

        def _init():
            self.gn_l0_z.assign(tf.stop_gradient(Lz))
            self.gn_l0_u.assign(tf.stop_gradient(Lu))
            self.gn_l0_p.assign(tf.stop_gradient(Lp))
            self.gn_inited.assign(True)
            return 0

        tf.cond(self.gn_inited, lambda: 0, _init)

        Lz_w = self.lam_z * Lz
        Lu_w = self.lam_u * Lu
        Lp_w = self.lam_p * Lp

        gz = tape.gradient(Lz_w, shared_W)
        gu = tape.gradient(Lu_w, shared_W)
        gp = tape.gradient(Lp_w, shared_W)

        if gz is None:
            gz = tf.zeros_like(shared_W)
        if gu is None:
            gu = tf.zeros_like(shared_W)
        if gp is None:
            gp = tf.zeros_like(shared_W)

        Gz = tf.norm(gz)
        Gu = tf.norm(gu)
        Gp = tf.norm(gp)

        G_avg = (Gz + Gu + Gp) / tf.cast(3.0, self.dtype)

        eps = self.gn_eps
        rz = Lz / (self.gn_l0_z + eps)
        ru = Lu / (self.gn_l0_u + eps)
        rp = Lp / (self.gn_l0_p + eps)
        r_avg = (rz + ru + rp) / tf.cast(3.0, self.dtype)

        Rz = rz / (r_avg + eps)
        Ru = ru / (r_avg + eps)
        Rp = rp / (r_avg + eps)

        alpha = self.gn_alpha

        Tz = tf.stop_gradient(G_avg * tf.pow(Rz, alpha))
        Tu = tf.stop_gradient(G_avg * tf.pow(Ru, alpha))
        Tp = tf.stop_gradient(G_avg * tf.pow(Rp, alpha))

        L_w = tf.abs(Gz - Tz) + tf.abs(Gu - Tu) + tf.abs(Gp - Tp)

        grads_lam = tape.gradient(L_w, [self.lam_z, self.lam_u, self.lam_p])
        grads_lam = [tf.zeros_like(v) if g is None else g for g, v in zip(grads_lam, [self.lam_z, self.lam_u, self.lam_p])]
        self.opt_lam.apply_gradients(zip(grads_lam, [self.lam_z, self.lam_u, self.lam_p]))

        self.lam_z.assign(tf.clip_by_value(self.lam_z, 1e-6, 1e6))
        self.lam_u.assign(tf.clip_by_value(self.lam_u, 1e-6, 1e6))
        self.lam_p.assign(tf.clip_by_value(self.lam_p, 1e-6, 1e6))

        s = self.lam_z + self.lam_u + self.lam_p + eps
        N = tf.cast(3.0, self.dtype)
        self.lam_z.assign(self.lam_z * (N / s))
        self.lam_u.assign(self.lam_u * (N / s))
        self.lam_p.assign(self.lam_p * (N / s))

        return self.lam_z, self.lam_u, self.lam_p


    def set_x_for_loss(self, x: tf.Tensor) -> None:
        self._x_loss = tf.convert_to_tensor(x, dtype=self.dtype)

    def f_model_from_pred(self, y_z_pred, y_u_pred):



        h = y_z_pred - self.z0
        A = (2.0 + 3.0 * h) * h * 0.5
        Q = A * y_u_pred

        A_i = A[:, :-1]  # [N, 4]
        A_ip1 = A[:, 1:]  # [N, 4]

        Q_i = Q[:, :-1]  # [N, 4]
        Q_ip1 = Q[:, 1:]  # [N, 4]

        dA_i_dt = A_i[1:] - A_i[:-1]  # [N-1, 4]
        dA_ip1_dt = A_ip1[1:] - A_ip1[:-1]  # [N-1, 4]

        Q_ip1_avg = (Q_ip1[1:] + Q_ip1[:-1]) * 0.5  # [N-1, 4]
        Q_i_avg = (Q_i[1:] + Q_i[:-1]) * 0.5  # [N-1, 4]

        f_raw = 500.0 * (dA_i_dt + dA_ip1_dt) - 60.0 * ( Q_i_avg - Q_ip1_avg )

        t = self._x_loss[:, 0]


        dt_steps = t[1:] - t[:-1]  # [N-1]
        mask = tf.cast(dt_steps > 0, dtype=f_raw.dtype)[:, tf.newaxis]
        f = f_raw * mask

        return f

    def compute_phys_loss(self, z_pred, u_pred):
        """Compute physics-informed loss"""
        f_pred = self.f_model_from_pred(z_pred, u_pred)

        loss_phys_raw = tf.reduce_mean(tf.square(f_pred))
        '''
        new_scale = tf.stop_gradient(tf.sqrt(loss_phys_raw) + 1e-12)
        self.phys_scale.assign(0.9 * self.phys_scale + 0.1 * new_scale)
        loss_phys_norm = loss_phys_raw / (self.phys_scale ** 2)
        '''

        def _init():
            self.phys_scale.assign(tf.sqrt(loss_phys_raw) + 1e-12)
            self.phys_scale_inited.assign(True)
            return 0

        tf.cond(self.phys_scale_inited, lambda: 0, _init)
        loss_phys_norm =  loss_phys_raw / (self.phys_scale ** 2)

        return loss_phys_raw, loss_phys_norm

    @tf.function
    def loss(self, y, y_pred):
        """Combined data + physics loss"""
        y_z, y_u = y[:, :self.output_dim], y[:, self.output_dim:]
        z_pred, u_pred = y_pred[:, :self.output_dim], y_pred[:, self.output_dim:]

        L_z = tf.reduce_mean(tf.square(y_z - z_pred))
        L_u = tf.reduce_mean(tf.square(y_u - u_pred))
        '''
        y_z0 = y_z[:1, :]
        z0 = z_pred[:1, :]

        y_u0 = y_u[:1, :]
        u0 = u_pred[:1, :]
        '''
        '''
        y_z_sel = tf.gather(y_z, self.constraint_indices, axis=1)
        z_sel = tf.gather(z_pred, self.constraint_indices, axis=1)

        y_u_sel = tf.gather(y_u, self.constraint_indices, axis=1)
        u_sel = tf.gather(u_pred, self.constraint_indices, axis=1)

        L_z = tf.reduce_mean(tf.square(y_z_sel - z_sel)) + tf.reduce_mean(tf.square(y_z0 - z0))
        L_u = tf.reduce_mean(tf.square(y_u_sel - u_sel)) + tf.reduce_mean(tf.square(y_u0 - u0))
        '''
        L_phys_raw, L_phys = self.compute_phys_loss(z_pred, u_pred)

        # lam_z, lam_u, lam_p = self.relobralo_update(L_z, L_u, L_phys)
        # L  = lam_z * L_z + lam_u * L_u + lam_p * L_phys

        # return L_z, L_u, L_phys_raw, L_phys, L
        return L_z, L_u, L_phys_raw, L_phys

    @tf.function
    def train_step(self, x, y_z, y_u):
        """PINN training step with physics constraint"""
        y = tf.concat([y_z, y_u], axis=1)
        with tf.GradientTape(persistent=True) as tape:
            y_pred = self.model(x, training=True)
            # loss_z, loss_u, L_phys_raw, L_phys, loss_total = self.loss(y, y_pred)
            loss_z, loss_u, L_phys_raw, L_phys = self.loss(y, y_pred)

            if self.strategy == "relobralo":
                lam_z, lam_u, lam_p = self.relobralo_update(loss_z, loss_u, L_phys)

            elif self.strategy == "lra":
                shared_vars = self.model.trainable_variables  # LRA 用全部共享参数
                lam_z, lam_u, lam_p = self.lrannealing_update(
                    tape, shared_vars, loss_z, loss_u, L_phys, main='z'
                )
            elif self.strategy == "gradnorm":
                shared_W = self.model.trainable_variables[-2]
                lam_z, lam_u, lam_p = self.gradnorm_update(
                tape, shared_W, loss_z, loss_u, L_phys
            )
            else:
                lam_z, lam_u, lam_p = self.lam_z, self.lam_u, self.lam_p
            loss_total = (tf.stop_gradient(lam_z) * loss_z +
                          tf.stop_gradient(lam_u) * loss_u +
                          tf.stop_gradient(lam_p) * L_phys)

        gradients = tape.gradient(loss_total, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        del tape

        return loss_z, loss_u, L_phys_raw, L_phys, loss_total

    def fit(self, x, y_z, y_u, epochs=2000, x_test=None, y_z_test=None, y_u_test=None,
            optimizer='adam', learning_rate=0.001, val_freq=100, log_freq=100, verbose=1):
        """PINN training with physics constraints"""
        epochs_lbfgs = 10000
        x = self.tensor(x)
        y_z = self.tensor(y_z)
        y_u = self.tensor(y_u)

        self.start_time = time.time()
        self.prev_time = self.start_time

        # Set x for physics loss computation
        self.set_x_for_loss(x)
        '''
        logging.info(f'PINN Optimizer: {optimizer}')
        if optimizer == 'adam':
            self.train_adam(x, y_z, y_u, epochs, x_test, y_z_test, y_u_test,
                             learning_rate, val_freq, log_freq, verbose)
        elif optimizer == 'lbfgs':
            self.train_lbfgs(x, y_z, y_u, epochs, x_test, y_z_test, y_u_test,
                              learning_rate, val_freq, log_freq, verbose)
        '''
        self.train_adam(x, y_z, y_u, epochs, x_test, y_z_test, y_u_test,
                        learning_rate, val_freq, log_freq, verbose)
        self.train_lbfgs(x, y_z, y_u, epochs_lbfgs, x_test, y_z_test, y_u_test,
                         learning_rate, val_freq, log_freq, verbose)

    def train_adam(self, x, y_z, y_u, epochs, x_test, y_z_test, y_u_test,
                    learning_rate, val_freq, log_freq, verbose):
        """Adam training for PINN"""
        '''
        if self.use_lr_schedule:
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=10000,
                decay_rate=0.1,
                staircase=True)
        '''
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        epoch_loss_z = tf.keras.metrics.Mean(name='epoch_loss_z')
        epoch_loss_u = tf.keras.metrics.Mean(name='epoch_loss_u')
        # 新增
        epoch_loss_phys = tf.keras.metrics.Mean(name='epoch_loss_phys')
        epoch_loss_phys_norm = tf.keras.metrics.Mean(name='epoch_loss_phys')
        epoch_loss_total = tf.keras.metrics.Mean(name='epoch_loss_total')

        if verbose:
            logging.info(f'Start ADAM optimization')

        for epoch in range(1, epochs + 1):
            loss_z, loss_u, loss_phys, loss_phys_norm, loss_total = self.train_step(x, y_z, y_u)
            # Track progress
            epoch_loss_z.update_state(loss_z)  # Add current batch loss
            epoch_loss_u.update_state(loss_u)
            epoch_loss_phys.update_state(loss_phys)
            epoch_loss_phys_norm.update_state(loss_phys_norm)
            epoch_loss_total.update_state(loss_total)

            self.epoch_callback_adam(epoch, epoch_loss_z.result(), epoch_loss_u.result(), epoch_loss_phys.result(),
                                     epoch_loss_phys_norm.result(), epoch_loss_total.result(), epochs, x_test, y_z_test, y_u_test, val_freq,
                                     log_freq,
                                     verbose)


    def train_lbfgs(self, x, y_z, y_u, epochs, x_test, y_z_test, y_u_test,
                     learning_rate, val_freq, log_freq, verbose):
        """LBFGS training for PINN"""

        optimizer = LBFGS()
        y = tf.concat([y_z, y_u], axis=1)
        y_test = tf.concat([y_z_test, y_u_test], axis=1)

        self.epoch_loss_z = tf.keras.metrics.Mean(name='epoch_loss_z')
        self.epoch_loss_u = tf.keras.metrics.Mean(name='epoch_loss_u')
        # 新增
        self.epoch_loss_phys = tf.keras.metrics.Mean(name='epoch_loss_phys')
        self.epoch_loss_phys_norm = tf.keras.metrics.Mean(name='epoch_loss_phys')
        self.epoch_loss_total = tf.keras.metrics.Mean(name='epoch_loss_total')

        def loss_fn(y_true, y_pred):
            # loss_z, loss_u, L_phys_raw, L_phys, loss_total = self.loss(y_true, y_pred)
            loss_z, loss_u, L_phys_raw, L_phys = self.loss(y_true, y_pred)
            loss_total = (tf.stop_gradient(self.lam_z) * loss_z +
                          tf.stop_gradient(self.lam_u) * loss_u +
                          tf.stop_gradient(self.lam_p) * L_phys)
            self.epoch_loss_z.update_state(loss_z)  # Add current batch loss
            self.epoch_loss_u.update_state(loss_u)
            self.epoch_loss_phys.update_state(L_phys_raw)
            self.epoch_loss_phys_norm.update_state(L_phys)
            self.epoch_loss_total.update_state(loss_total)
            return loss_total

        if verbose:
            logging.info(f'Start lbfgs optimization')
        try:
            optimizer.minimize(
                self.model,
                loss_fn,
                x, y,
                self.epoch_callback_lbfgs,
                epochs=epochs,
                x_test=x_test,
                y_test=y_test,
                val_freq=val_freq,
                log_freq=log_freq,
                verbose=verbose,
                learning_rate=learning_rate
            )
        except StopIteration as e:
            print(e)

    def epoch_callback_lbfgs(self, epoch, epoch_loss, epochs, x_test=None, y_test=None, val_freq=1000, log_freq=1000,
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

        # 保存到训练结果字典
        epoch  = 20000 + epoch
        self.train_loss_results_z[epoch] = self.epoch_loss_z.result()
        self.train_loss_results_u[epoch] = self.epoch_loss_u.result()
        self.train_loss_results_phys[epoch] = self.epoch_loss_phys.result()
        self.train_loss_results_phys_norm[epoch] = self.epoch_loss_phys_norm.result()
        self.train_loss_results_total[epoch] = self.epoch_loss_total.result()

        elapsed_time = self.get_elapsed_time()
        self.train_time_results[epoch] = elapsed_time

        patience = 20
        min_delta = 1e-6

        if epoch % val_freq == 0 or epoch == 1:
            length = len(str(epochs))
            log_str = (
                f'\tEpoch: {str(epoch).zfill(length)}/{epochs},\t'
                f'Loss(z,u,phys,phys_norm,total): '
                f'{float(self.epoch_loss_z.result()):.4e}, {float(self.epoch_loss_u.result()):.4e}, '
                f'{float(self.epoch_loss_phys.result()):.4e}, {float(self.epoch_loss_phys_norm.result()):.4e}, {float(self.epoch_loss_total.result()):.4e}'
            )

            if x_test is not None and y_test is not None:
                n_z = self.model_z.output_shape[-1]
                y_z_test = y_test[:, :n_z]
                y_u_test = y_test[:, n_z:]

                mse_z, _, pred_z = self.evaluate_z(x_test, y_z_test)
                self.train_accuracy_results_z[epoch] = mse_z
                self.train_pred_results_z[epoch] = pred_z

                mse_u, _, pred_u = self.evaluate_u(x_test, y_u_test)
                self.train_accuracy_results_u[epoch] = mse_u
                self.train_pred_results_u[epoch] = pred_u

                log_str += ', Val_MSE_z: %.4e, Val_MSE_u: %.4e' % (mse_z, mse_u)

                if mse_z <= min(self.train_accuracy_results_z.values()):
                    self.save_weights_z(os.path.join(self.checkpoints_dir, 'easy_checkpoint_model_z'))

                if mse_u <= min(self.train_accuracy_results_u.values()):
                    self.save_weights_u(os.path.join(self.checkpoints_dir, 'easy_checkpoint_model_u'))

                if mse_z < self.early_stop_best_loss_z - min_delta:
                    self.early_stop_best_loss_z = mse_z
                    self.early_stop_wait_z = 0
                else:
                    self.early_stop_wait_z += 1
                    if self.early_stop_wait_z >= patience:
                        self.early_stop_triggered_z = True
                        logging.info('[EarlyStopping] z stopped at epoch %d, best MSE: %.6e' % (
                        epoch, self.early_stop_best_loss_z))
                        raise StopIteration("Early stopping for z")

                if mse_u < self.early_stop_best_loss_u - min_delta:
                    self.early_stop_best_loss_u = mse_u
                    self.early_stop_wait_u = 0
                else:
                    self.early_stop_wait_u += 1
                    if self.early_stop_wait_u >= patience:
                        self.early_stop_triggered_u = True
                        logging.info('[EarlyStopping] u stopped at epoch %d, best MSE: %.6e' % (
                        epoch, self.early_stop_best_loss_u))
                        raise StopIteration("Early stopping for u")

            if (epoch % log_freq == 0 or epoch == 1) and verbose == 1:
                log_str += ', Time: %s' % elapsed_time
                logging.info(log_str)

        if epoch == epochs:
            self.save_weights_z(os.path.join(self.checkpoints_dir, 'easy_checkpoint_model_z'))
            self.save_weights_u(os.path.join(self.checkpoints_dir, 'easy_checkpoint_model_u'))

    def predict(self, x):
        """
        Calls the model prediction function and returns the prediction on an input tensor.

        :param tf.tensor x: input tensor
        :return: tf.tensor: output tensor
        """
        y_z_pred = self.model_z.predict(x)
        y_u_pred = self.model_u.predict(x)
        return y_z_pred, y_u_pred
