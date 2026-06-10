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

        self._x_loss = None  # 训练前由 train_lbfgs 填
        self.w_z = tf.Variable(1.0, dtype=self.dtype, trainable=False)
        self.w_u = tf.Variable(1.0, dtype=self.dtype, trainable=False)
        self.w_phys = tf.Variable(1.0, dtype=self.dtype, trainable=False)
        self.w_ema = self.w_ema = tf.constant(0.9, dtype=self.dtype)
        self.z0 = tf.constant([-1.4, -2.4, -3.4, -4.4, -5.4], dtype=self.dtype)
        # 物理损失自适应缩放因子 (必须在 init 中定义)
        self.phys_scale = tf.Variable(1.0, dtype=self.dtype, trainable=False)

        # 占位符
        self._x_loss = None

        self.strategy = "grad"
        self.constraint_indices = tf.constant([0, self.output_dim - 1], dtype=tf.int32)

        # ---- ReLoBRaLo 超参 ----
        self.rel_T = tf.constant(1.0, dtype=self.dtype)
        self.rel_alpha = tf.constant(0.9, dtype=self.dtype)
        self.rel_rho = tf.constant(0.99, dtype=self.dtype)
        self.rel_eps = tf.constant(1e-12, dtype=self.dtype)

        # ---- 三个权重 lambda ----
        self.lam_z = tf.Variable(1.0, dtype=self.dtype, trainable=True)
        self.lam_u = tf.Variable(1.0, dtype=self.dtype, trainable=True)
        self.lam_p = tf.Variable(1.0, dtype=self.dtype, trainable=True)

        # ---- 上一次 loss 和初始 loss（lookback 用）----
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

        # ====== LRA 超参 ======
        self.lra_alpha = tf.constant(0.9, self.dtype)  # EMA 平滑
        self.lra_eps = tf.constant(1e-8, self.dtype)  # 防 0

        # ====== GradNorm 超参/状态 ======
        self.gn_alpha = tf.constant(1.5, self.dtype)      # 论文里的 alpha（常用 0.5~2）
        self.gn_eps   = tf.constant(1e-12, self.dtype)
        self.gn_inited = tf.Variable(False, trainable=False, dtype=tf.bool)

        self.gn_l0_z = tf.Variable(1.0, trainable=False, dtype=self.dtype)
        self.gn_l0_u = tf.Variable(1.0, trainable=False, dtype=self.dtype)
        self.gn_l0_p = tf.Variable(1.0, trainable=False, dtype=self.dtype)

        # GradNorm 需要一个专门更新 lambda 的优化器（你也可以外部传入）
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
        # grads: list of Tensor or None
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
        # 建议 L_p 用你归一化后的 L_phys_norm（训练稳定）
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

        # 两组 logits：相对上一次 / 相对初始
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

        # 更新 lambda（平滑 + lookback 混合）
        new_lam_z = rho * alpha * self.lam_z + (1 - rho) * alpha * lam0_hat[0] + (1 - alpha) * lam_hat[0]
        new_lam_u = rho * alpha * self.lam_u + (1 - rho) * alpha * lam0_hat[1] + (1 - alpha) * lam_hat[1]
        new_lam_p = rho * alpha * self.lam_p + (1 - rho) * alpha * lam0_hat[2] + (1 - alpha) * lam_hat[2]

        # 防止极端（强烈建议）
        self.lam_z.assign(tf.clip_by_value(new_lam_z, 1e-6, 1e6))
        self.lam_u.assign(tf.clip_by_value(new_lam_u, 1e-6, 1e6))
        self.lam_p.assign(tf.clip_by_value(new_lam_p, 1e-6, 1e6))

        # 更新上一轮 loss
        self.lprev_z.assign(tf.stop_gradient(Lz))
        self.lprev_u.assign(tf.stop_gradient(Lu))
        self.lprev_p.assign(tf.stop_gradient(Lp))

        return self.lam_z, self.lam_u, self.lam_p

    @tf.function
    def lrannealing_update(self, tape, shared_vars, L_z, L_u, L_p, main='z'):
        # 已经梯度损失的相对梯度比较，看梯度规模
        Lz = tf.cast(L_z, self.dtype)
        Lu = tf.cast(L_u, self.dtype)
        Lp = tf.cast(L_p, self.dtype)

        # 计算各自对共享参数的梯度
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

        # lambdas_hat：让每项梯度尺度接近主项
        lam_z_hat = m_main / (mz + eps)
        lam_u_hat = m_main / (mu + eps)
        lam_p_hat = m_main / (mp + eps)

        # EMA 平滑更新
        new_lam_z = alpha * self.lam_z + (1.0 - alpha) * lam_z_hat
        new_lam_u = alpha * self.lam_u + (1.0 - alpha) * lam_u_hat
        new_lam_p = alpha * self.lam_p + (1.0 - alpha) * lam_p_hat

        # clip 防极端
        self.lam_z.assign(tf.clip_by_value(new_lam_z, 1e-6, 1e6))
        self.lam_u.assign(tf.clip_by_value(new_lam_u, 1e-6, 1e6))
        self.lam_p.assign(tf.clip_by_value(new_lam_p, 1e-6, 1e6))

        return self.lam_z, self.lam_u, self.lam_p

    @tf.function
    def gradnorm_update(self, tape, shared_W, L_z, L_u, L_p):
        # 通过比较当前损失值和损失初始值的比值来决定权重，看相对初始loss的下降速度
        # 但是计算完比值后，实际上是依据这个比值来调整梯度的
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

        # 每个任务的加权损失
        Lz_w = self.lam_z * Lz
        Lu_w = self.lam_u * Lu
        Lp_w = self.lam_p * Lp

        # 计算每个任务对共享参数 W 的梯度范数 Gi
        gz = tape.gradient(Lz_w, shared_W)
        gu = tape.gradient(Lu_w, shared_W)
        gp = tape.gradient(Lp_w, shared_W)

        # 处理 None
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

        # 相对训练速度（用当前损失与初始损失比）
        eps = self.gn_eps
        rz = Lz / (self.gn_l0_z + eps)
        ru = Lu / (self.gn_l0_u + eps)
        rp = Lp / (self.gn_l0_p + eps)
        r_avg = (rz + ru + rp) / tf.cast(3.0, self.dtype)

        Rz = rz / (r_avg + eps)
        Ru = ru / (r_avg + eps)
        Rp = rp / (r_avg + eps)

        alpha = self.gn_alpha

        # 目标梯度范数
        Tz = tf.stop_gradient(G_avg * tf.pow(Rz, alpha))
        Tu = tf.stop_gradient(G_avg * tf.pow(Ru, alpha))
        Tp = tf.stop_gradient(G_avg * tf.pow(Rp, alpha))

        # lambda 的优化目标：让 Gi 接近 Ti
        L_w = tf.abs(Gz - Tz) + tf.abs(Gu - Tu) + tf.abs(Gp - Tp)

        # 对 lambda 求梯度并更新
        grads_lam = tape.gradient(L_w, [self.lam_z, self.lam_u, self.lam_p])
        # 保险：None -> 0
        grads_lam = [tf.zeros_like(v) if g is None else g for g, v in zip(grads_lam, [self.lam_z, self.lam_u, self.lam_p])]
        self.opt_lam.apply_gradients(zip(grads_lam, [self.lam_z, self.lam_u, self.lam_p]))

        # 防止负数/极端，并把 sum 归一到 N=3（常见做法，保持整体尺度稳定）
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
        """固定用于计算物理损失的 Collocation Points"""
        self._x_loss = tf.convert_to_tensor(x, dtype=self.dtype)
    '''
    def f_model_from_pred(self, y_z_pred, y_u_pred):
        """
        Declaration of the function for the implementation of the f_model for a specific differential equation.
        """
        # 这边是为了方便继承的子类使用
        """
        The actual Physics Informed Neural Network for the approximation of the equation.
        :return: tf.Tensor: the prediction of the PINN 预测值
        返回物理方程项的预测值
        """

        f_pred_list = []

        for i in range(4):
            # 提取第 i 和第 i+1 个断面的预测值
            # 断面水深 h_i和h_ip1 单位 m
            h_i = y_z_pred[:, i:i + 1] - self.z0[i]
            h_ip1 = y_z_pred[:, i + 1:i + 2] - self.z0[i + 1]

            # 断面河宽 单位 m

            # B_i = 1.0 + 3.0 * h_i
            # B_ip1 = 1.0 + 3.0 * h_ip1

            # 断面面积 单位 m^2
            A_i = (2.0 + 3.0 * h_i) * h_i / 2
            A_ip1 = (2.0 + 3.0 * h_ip1) * h_ip1 / 2

            Q_i = A_i * y_u_pred[:, i:i + 1]
            Q_ip1 = A_ip1 * y_u_pred[:, i + 1:i + 2]

            # 计算物理项 f_pred (T-1, 1)
            f = 500.0 * ( A_i[1:] - A_i[:-1] + A_ip1[1:] - A_ip1[:-1] ) \
                - 240.0 * ( (Q_ip1[1:]+Q_ip1[:-1])/2 - (Q_i[1:]+Q_i[:-1])/2 )
            # print('f',  f)
            f_pred_list.append(f)

        f_pred = tf.concat(f_pred_list, axis=1)
        return f_pred

    '''
    def f_model_from_pred(self, y_z_pred, y_u_pred):
        """
        【优化版】完全向量化的物理方程计算
        移除了 for 循环，利用矩阵切片并行计算所有断面
        """
        # 1. 广播减法：一次性计算所有断面的 h
        # y_z_pred: [N, 5], self.z0: [5] -> h: [N, 5]

        h = y_z_pred - self.z0

        # 2. 一次性计算几何属性 A 和流量 Q
        # A: [N, 5], Q: [N, 5]
        A = (2.0 + 3.0 * h) * h * 0.5
        Q = A * y_u_pred

        # 3. 利用切片提取相邻断面 (空间差分)
        # [:, :-1] 代表索引 i (0,1,2,3)
        # [:, 1:]  代表索引 i+1 (1,2,3,4)
        A_i = A[:, :-1]  # [N, 4]
        A_ip1 = A[:, 1:]  # [N, 4]

        Q_i = Q[:, :-1]  # [N, 4]
        Q_ip1 = Q[:, 1:]  # [N, 4]

        # 4. 利用切片提取相邻时间步 (时间差分)
        # [1:, :] 代表 t+1,[:-1, :] 代表 t
        dA_i_dt = A_i[1:] - A_i[:-1]  # [N-1, 4]
        dA_ip1_dt = A_ip1[1:] - A_ip1[:-1]  # [N-1, 4]

        # 对应原代码: (Q_ip1[1:]+Q_ip1[:-1])/2
        Q_ip1_avg = (Q_ip1[1:] + Q_ip1[:-1]) * 0.5  # [N-1, 4]
        Q_i_avg = (Q_i[1:] + Q_i[:-1]) * 0.5  # [N-1, 4]

        # 5. 计算残差 f (保持你的公式结构)
        # 形状: [N-1, 4]
        f_raw = 500.0 * (dA_i_dt + dA_ip1_dt) - 60.0 * ( Q_i_avg - Q_ip1_avg )

        t = self._x_loss[:, 0]  # [N]

        # 计算相邻样本的时间差
        # 正常情况: 60 - 0 = 60 (大于0) 异常情况: 0 - 600 = -600 (小于0)
        dt_steps = t[1:] - t[:-1]  # [N-1]
        # tf.print('dt_steps', dt_steps[:12])
        # 创建掩码: 只有当 dt > 0 (时间增加) 时，计算才有效
        # mask shape: [N-1, 1] (扩展维度以便与 f_raw 广播相乘)
        mask = tf.cast(dt_steps > 0, dtype=f_raw.dtype)[:, tf.newaxis]

        # 强制将跨序列的那一行的残差置为 0
        f = f_raw * mask
        # tf.print('f',f[:12])
        return f

    def compute_phys_loss(self, z_pred, u_pred):
        """Compute physics-informed loss"""
        f_pred = self.f_model_from_pred(z_pred, u_pred)
        # 平方差
        loss_phys_raw = tf.reduce_mean(tf.square(f_pred))
        '''
        new_scale = tf.stop_gradient(tf.sqrt(loss_phys_raw) + 1e-12)
        self.phys_scale.assign(0.9 * self.phys_scale + 0.1 * new_scale)
        loss_phys_norm = loss_phys_raw / (self.phys_scale ** 2)
        '''
        # 只初始化一次（或只在前 N 步更新）
        def _init():
            # 第一次开根号作为除数，后续都缩写这么多倍作为
            self.phys_scale.assign(tf.sqrt(loss_phys_raw) + 1e-12)
            self.phys_scale_inited.assign(True)
            return 0

        tf.cond(self.phys_scale_inited, lambda: 0, _init)
        loss_phys_norm =  loss_phys_raw / (self.phys_scale ** 2)
        # loss_phys_norm = loss_phys_raw

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

            # === 选择策略，更新 lambda ===
            if self.strategy == "relobralo":
                lam_z, lam_u, lam_p = self.relobralo_update(loss_z, loss_u, L_phys)

            elif self.strategy == "lra":
                shared_vars = self.model.trainable_variables  # LRA 用全部共享参数
                lam_z, lam_u, lam_p = self.lrannealing_update(
                    tape, shared_vars, loss_z, loss_u, L_phys, main='z'
                )
            elif self.strategy == "gradnorm":
            # GradNorm 最好选 trunk 的 kernel，不要选最后 bias
                shared_W = self.model.trainable_variables[-2]
                lam_z, lam_u, lam_p = self.gradnorm_update(
                tape, shared_W, loss_z, loss_u, L_phys
            )
            else:
                lam_z, lam_u, lam_p = self.lam_z, self.lam_u, self.lam_p
            # === 组总 loss（建议 stop_gradient，避免 lambda 进入主网络反传）===
            loss_total = (tf.stop_gradient(lam_z) * loss_z +
                          tf.stop_gradient(lam_u) * loss_u +
                          tf.stop_gradient(lam_p) * L_phys)
        # Apply gradients to both models
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
            """损失函数 - 返回总损失"""
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

        # ========== 早停逻辑 ==========
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

            # 验证集评估
            if x_test is not None and y_test is not None:
                # 拆分 y_test 为 z 和 u
                n_z = self.model_z.output_shape[-1]
                y_z_test = y_test[:, :n_z]
                y_u_test = y_test[:, n_z:]

                # 评估 z
                mse_z, _, pred_z = self.evaluate_z(x_test, y_z_test)
                self.train_accuracy_results_z[epoch] = mse_z
                self.train_pred_results_z[epoch] = pred_z

                # 评估 u
                mse_u, _, pred_u = self.evaluate_u(x_test, y_u_test)
                self.train_accuracy_results_u[epoch] = mse_u
                self.train_pred_results_u[epoch] = pred_u

                log_str += ', Val_MSE_z: %.4e, Val_MSE_u: %.4e' % (mse_z, mse_u)

                # 保存最佳权重
                if mse_z <= min(self.train_accuracy_results_z.values()):
                    self.save_weights_z(os.path.join(self.checkpoints_dir, 'easy_checkpoint_model_z'))

                if mse_u <= min(self.train_accuracy_results_u.values()):
                    self.save_weights_u(os.path.join(self.checkpoints_dir, 'easy_checkpoint_model_u'))

                # 早停检查 - z
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

                # 早停检查 - u
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

            # 日志输出
            if (epoch % log_freq == 0 or epoch == 1) and verbose == 1:
                log_str += ', Time: %s' % elapsed_time
                logging.info(log_str)

        # 训练结束时保存权重
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
