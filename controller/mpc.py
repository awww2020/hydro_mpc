import logging
import time

import numpy as np
import torch

from scipy.integrate import solve_ivp

class MPC:
    """
    Class used to represent a Model Predictive Controller.
    """

    def __init__(self, plant, model, u_ub, u_lb, t_sample=0.1, H=10,
                 Q=torch.eye(1, dtype=torch.float64), R=torch.eye(1, dtype=torch.float64)):
        """
        Initializes the MPC with required models, constraints, and parameters.

        :param plant: Function representing the physical system to be controlled.
        :param model: Predictive model for the system, used in simulation.
        :param u_ub: Upper bounds of the control inputs.
        :param u_lb: Lower bounds of the control inputs.
        :param t_sample: Sampling time (time step size).
        :param H: Prediction horizon (number of steps to look ahead).
        :param Q: State weighting matrix in the cost function.
        :param R: Control input weighting matrix in the cost function.
        """
        self.plant = plant
        self.model = model
        self.H = H
        self.t_sample = t_sample

        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=0.01)
        # 控制输入的上下界，对应维数为控制输入的个数
        self.u_ub = torch.tensor(u_ub, dtype=torch.float64)
        self.u_lb = torch.tensor(u_lb, dtype=torch.float64)
        self.input_dim = len(self.u_ub) # Dimension of control inputs

        self.u = torch.nn.Parameter(torch.zeros((self.H, self.input_dim), dtype=torch.float64))
        # np.zeros((self.H, self.input_dim))使用NumPy创建一个形状为(self.H, self.input_dim)的数组，所有元素初始化为0
        # 每行表示一个时间步的控制输入

        self.Q = torch.tensor(Q, dtype=torch.float64) if not isinstance(Q, torch.Tensor) else Q
        self.R = torch.tensor(R, dtype=torch.float64) if not isinstance(R, torch.Tensor) else R


        self.solving_times = {}

    def costs(self, x_ref, x_pred):
        """
        Represents the MPC cost function, which is composed of the step cost and the final cost.
        计算损失函数
        :param x_ref: reference states
        :param x_pred: predicted states
        :return: J: cost value
        """

        J = torch.sum( (x_ref - x_pred) ** 2 @ self.Q) \
            + torch.sum( (self.u) ** 2 @ self.R)

        return J

    def solve_ocp(self, x0, x_ref, iterations=1000, tol=1e-8):
        """
        Solves the optimal control problem (OCP) using iterative optimization.
        使用迭代优化方法解决最优控制问题（OCP），更新控制输入以最小化总成本。如果成本变化小于给定的容忍度（tol），则停止迭代。
        :param x0: Initial state.
        :param x_ref: Reference trajectory. 未来一段时间内的目标状态
        :param iterations: Maximum number of iterations.
        :param tol: Tolerance for convergence.
        :return: Optimal cost and predicted state trajectory.
        """
        J_prev = -1
        for epoch in range(iterations):
            J, x_pred = self.optimization_step(x0, x_ref)
            if np.abs(J - J_prev) < tol:
                return J, x_pred
            J_prev = J

        return J, x_pred  # J：优化后的总成本 x_pred：优化后的预测状态轨迹

    def optimization_step(self, x0, x_ref):
        """
        Performs one step of gradient-based optimization to update the control inputs.
        执行一步基于梯度的优化来更新控制输入，使用梯度下降法调整控制变量以减少总成本，并确保控制输入在规定的约束内。
        :param x0: Current state.
        :param x_ref: Reference trajectory.
        :return: Current cost and predicted states after the control update.
        """

        # 确保梯度是清空的
        self.optimizer.zero_grad()
        # 开环预测
        x_pred = self.sim_open_loop(x0, self.u, t_sample=self.t_sample, H=self.H)
        # 依据该时刻的初值x0预测在H预测区间内的预测值，t_sample 采样时间间隔，在优化开始时，控制输入是零
        # 这边比较有趣的是由于是滚动预测的，self.u不需要重新赋值改长度
        # 计算损失函数
        cost = self.costs(x_ref, x_pred)
        # 反向传播来计算关于u的梯度
        cost.backward()
        # 应用优化器来更新控制变量self.u
        self.optimizer.step()
        # 约束处理
        self.ensure_constraints()

        return cost, x_pred # 返回计算得到的成本 cost 和预测的状态轨迹 x_pred

    def ensure_constraints(self):
        """
        Ensures that the control inputs remain within their specified bounds after each optimization step.
        """
        for k in range(self.H):
            for i, u_ub_i in enumerate(self.u_ub):
                if self.u[k, i] > u_ub_i:
                    self.u.data[k, i] = u_ub_i

            for i, u_lb_i in enumerate(self.u_lb):
                if self.u[k, i] < u_lb_i:
                    self.u.data[k, i] = u_lb_i

    def sim(self, x0, X_ref, T_ref):
        """
        Simulates the system response over a specified time using the MPC control.
        模拟整个系统在给定的时间范围内的响应，使用MPC控制来调整系统状态，以便跟踪给定的参考轨迹。
        :param x0: Initial state.初始状态
        :param X_ref: Reference trajectory.
        :param T_ref: Time instances corresponding to the reference trajectory.
        :return: Simulated state and control input trajectories.
        """
        N = len(T_ref) # 预测时步

        # 初始化
        X_mpc = np.zeros((N, len(x0)))  # 实际状态序列 创建一个二维的 NumPy数组，其形状为 (N, len(x0))。这意味着数组有N行和 len(x0) 列
        X_pred = np.zeros((N, len(x0)))  # 预测状态序列
        U_mpc = np.zeros((N, self.u.shape[1]))  #u.shape 返回元组：各个维度的大小 这边是列数

        X_mpc[0] = x0  # 第0行
        X_pred[0] = x0
        U_mpc[0] = self.u[0].numpy()

        for i, t in enumerate(T_ref[:-1]): # 从下标为0的取到倒数最后一个
            start_time = time.time()
            J, x_pred = self.solve_ocp(X_mpc[i], X_ref[i:i + self.H + 1])
            # 以当前的系统状态 X_mpc[i] (相当于这个时刻的初值)和接下来的状态参考 X_ref[i:i + self.H + 1] 为输入，解决最优控制问题。返回最优成本J和预测状态x_pred
            ocp_solving_time = time.time() - start_time
            self.solving_times[i] = ocp_solving_time

            u_k = self.u[0]  # 提取当前最优控制输入

            # x_true = self.sim_plant_system(X_mpc[i], u_k, self.t_sample) # 模拟实际系统
            # todo: 增加真实值的设置和读取功能，这里先用预测值代替
            x_true = self.sim_open_loop(X_mpc[i], u_k, self.t_sample, self.H)

            # 更新预测和实际状态:下一个预测状态、下一个实际状态、记录使用的控制输入
            X_pred[i + 1] = x_pred[1]
            X_mpc[i + 1] = x_true
            U_mpc[i + 1] = u_k.numpy()

            # 生成一个日志字符串，记录迭代信息，最优成本J，下一个时间点 t + self.t_sample 和控制输入值。

            log_str = f'\tIter: {str(i + 1).zfill(len(str(N - 1)))}/{N - 1},\tJ: {J:.2e},' \
                      f'\tt: {t + self.t_sample:.2f} s,'

            # 记录每个控制输入 u_k 和状态 x_true 的值，并附加解决 OCP 的时间。
            for i in range(len(u_k)):
                log_str = log_str + f'\tu{i + 1}: {u_k.numpy()[i]:.2f},'

            for i in range(int(len(x_true) / 2)):
                log_str = log_str + f'\tx{i + 1}(t, u): {x_true[i]:.2f},'

            log_str = log_str + f'\tOCP-solving-time: {ocp_solving_time:.2e} s'
            logging.info(log_str)

        return X_mpc, U_mpc, X_pred

    def sim_open_loop(self, x0, u_array, t_sample, H):
        """
        Simulates the system's open-loop response over the prediction horizon using the predictive model.
        使用预测模型模拟系统的开环响应，给定初始状态和控制序列，预测系统未来的状态。
        :param x0: Current state.
        :param u_array: Array of control inputs for each step in the horizon.
        :param t_sample: Sampling time.
        :param H: Prediction horizon.
        :return: Predicted states over the horizon.
        """
        # t 表示预测的时间间隔，和其他值拼成一个后作为模型输入

        t = torch.tensor([[t_sample]], dtype=torch.float64) # 创建时间常量(1，1) 若为[t_sample]，则是一维向量(1,)
        x_i = x0.unsqueeze(0)
        # 初始化状态向量，将初始状态向量x0通过扩展维度转换为一个二维张量，新的维度被添加在索引0的位置，使其成为一个形状为 (1,n) 的张量（一行n列的矩阵）
        # 这样做的目的是与控制输入（u_array[i:i + 1]）进行拼接时，维度对齐
        X_pred = x_i # 初始预测状态
        # u_array 为self.u 为一个self.H行，self.input_dim的二维向量
        # u_array[i:i + 1]这样的切片方式可以选择一个范围内的行，这里是从i到i+1，但不包括 i+1。这种切片会保持原数组的维度。返回向量为（1，m）,m为控制变量维度

        # 每次迭代模拟系统的下一个状态，并将其添加到 X_pred 中
        for i in range(H):  # H为预测的步数
            x = torch.cat((t, x_i, u_array[i:i + 1]), dim=1) # 沿列方向拼接向量 (1，1+n+m)  拼接在一起输入了
            x_pred = self.model(x) # 输出形状为（1,n），表示下一时刻
            X_pred = torch.cat((X_pred, x_pred), dim=0) # 沿行方向拼接向量，每次增加一行，终为 (H+1, n)，包含从初始状态到当前迭代步的所有预测状态
            x_i = x_pred # 状态更新

        # 假设，x0是一个包含三个状态变量的系统的状态向量，H为10的情况下，最终X_pred形状为[11,3]
        return X_pred

    def sim_open_loop_plant(self, x0, u_array, t_sample, H):
        """
        Simulates the system's open-loop response over the prediction horizon using the physical plant.
        使用具体物理意义的系统模型模拟整个预测区间内系统动态，使用实际的系统动态和给定的控制序列在预测范围内系统的实际响应。
        :param x0: Current state.
        :param u_array: Array of control inputs for each step in the horizon.
        :param t_sample: Sampling time.
        :param H: Prediction horizon.
        :return: True states over the horizon.
        """
        x_i = x0
        X = x_i

        for i in range(H):
            x = self.sim_plant_system(x_i, u_array[i], t_sample)
            X = np.vstack((X, x))
            x_i = x

        return X

    def sim_plant_system(self, x0, u, tau):
        """
        Simulates the physical plant for a given control input over a single time step.
        使用具体物理意义的系统模拟单个时间步内的系统动态，使用实际的系统动态和控制输入在给定的时间步长内模拟系统的行为
        :param x0: Current state.
        :param u: Control input to be applied.
        :param tau: Time step duration.
        :return: New state after applying the control input.这一时间步结束时的系统状态
        """
        ivp_solution = solve_ivp(self.plant, [0, tau], x0, args=[u])
        # SciPy库中用于解决初始值问题的函数，即给定一个微分方程，初始条件（x0），以及时间区间（0到tau），求解该微分方程。
        z_true = np.moveaxis(ivp_solution.y[:, -1], -1, 0)
        '''
        ivp_solution.y是一个数组，其列包含了在模拟时间点的解向量。ivp_solution.y[:, -1]选取了最后一个时间点（即时间tau）的状态。
        np.moveaxis(ivp_solution.y[:, -1], -1, 0)这一步的作用是将返回的解向量的轴向调整为合适的形式，使其可以直接作为状态向量使用。
        这里假设solve_ivp返回的向量需要调整轴的顺序以匹配函数外部的期望格式。
        ！所以返回的是最后一个时刻的解，而不是这个时间区间里面的解
        '''
        return z_true