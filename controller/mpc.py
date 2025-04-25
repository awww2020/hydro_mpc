import logging
import time

import matplotlib.pyplot as plt
import win32com.client as win32

import numpy as np
import torch
import tensorflow as tf
from utils.hecras_control import modify_unsteady_file,create_plan_file
import pygad

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
        self.U_dis = None

        #self.optimizer = torch.optim.RMSprop(self.parameters(), lr=0.01)
        self.optimizer = tf.keras.optimizers.RMSprop()
        # 控制输入的上下界，对应维数为控制输入的个数
        self.u_ub = u_ub
        self.u_lb = u_lb
        self.input_dim = len(self.u_ub) # Dimension of control inputs

        # 创建初始值为零的 NumPy 数组
        initial_value = np.zeros((self.H, self.input_dim), dtype=np.float64)
        # 将第三列（索引为2）设置为4
        initial_value[:, 2] = 4.0
        # self.u = torch.nn.Parameter(torch.zeros((self.H, self.input_dim), dtype=torch.float64))
        self.u = tf.Variable(initial_value=initial_value, name='u', trainable=True,
                             dtype=tf.float64)

        # np.zeros((self.H, self.input_dim))使用NumPy创建一个形状为(self.H, self.input_dim)的数组，所有元素初始化为0
        # 每行表示一个时间步的控制输入

        # self.Q = torch.tensor(Q, dtype=torch.float64) if not isinstance(Q, torch.Tensor) else Q
        # self.R = torch.tensor(R, dtype=torch.float64) if not isinstance(R, torch.Tensor) else R
        self.Q = tf.convert_to_tensor(Q, dtype=tf.float64)
        self.R = tf.convert_to_tensor(R, dtype=tf.float64)

        self.solving_times = {}
        self.compute_time = {}
        #todo 这个后续可以传入
        self.q_0 = np.array([
            4.00, 4.00, 4.00, 4.00, 4.00,
            4.00, 4.00, 4.00, 4.00, 4.00,
            4.00, 4.00, 4.00, 4.00, 4.00,
            4.00, 4.00, 4.00, 4.00, 4.00,
            4.00
        ])

        # z_0: 20 个值
        self.z_0 = np.array([
            -0.187073, -0.387217, -0.587177, -0.787133, -0.987089,
            -1.187056, -1.387032, -1.587137, -1.786896, -1.986127,
            -2.183517, -2.378099, -2.565130, -2.738064, -2.885815,
            -2.999806, -3.079509, -3.131713, -3.165020, -3.186296
        ])

        self.u_0 = np.zeros([5*1])

        self.errors_pct = []  # 用来存放每个时刻的水位百分比误差
        self.u_prev = None  # 记录上一步施加的闸门流量
        self.iaq_accum = 0.0  # IAQ 累计量
        self.u_history = []  # （可选）记录所有施加流量以备后续分析

        self.t_list = []
        self.z_0_list = []
        self.z_list = []
        self.u_list = []
        self.v_list = []

    def create_fixed_u(self):
        u_fixed = np.zeros_like(self.u)
        u_fixed[:, 0:2] = self.U_dis
        return u_fixed


    def costs(self, x_ref, x_pred):
        """
        Represents the MPC cost function, which is composed of the step cost and the final cost.
        计算损失函数
        :param x_ref: reference states
        :param x_pred: predicted states
        :return: J: cost value
        """
        # print(x_ref.shape, x_pred.shape) # (7,1) 7表示时间，1表示状态个数
        # print(tf.square(x_ref - x_pred).shape)
        # print(self.Q.shape)
        # print(self.u.shape)
        # print(self.R.shape)
        delta_u = self.u[1:,2:3] - self.u[:-1,2:3]
        J = tf.reduce_sum(tf.square(x_ref/100 - x_pred/100) @ self.Q) \
            + tf.reduce_sum(tf.square(delta_u) @ self.R)

        return J

    def solve_ocp(self, x0, x_ref, iter, iterations=5000, tol=1e-9):
        """
        Solves the optimal control problem (OCP) using iterative optimization.
        使用迭代优化方法解决最优控制问题（OCP），更新控制输入以最小化总成本。如果成本变化小于给定的容忍度（tol），则停止迭代。
        :param x0: Initial state.
        :param x_ref: Reference trajectory. 未来一段时间内的目标状态
        :param iterations: Maximum number of iterations.
        :param tol: Tolerance for convergence.
        :return: Optimal cost and predicted state trajectory.
        """
        optimization_method = 'Gradient Descent'

        if optimization_method == 'Gradient Descent':
            J_prev = -1
            for epoch in range(iterations):
                J, x_pred = self.optimization_step_gradient(x0, x_ref)
                if np.abs(J - J_prev) < tol:
                    return J, x_pred
                J_prev = J
        else: # Default to GA
            J, x_pred = self.optimization_step_GA(x0, x_ref,iter)

        return J, x_pred  # J：优化后的总成本 x_pred：优化后的预测状态轨迹

    @tf.function
    def optimization_step_gradient(self, x0, x_ref):
        """
        Performs one step of gradient-based optimization to update the control inputs.
        执行一步基于梯度的优化来更新控制输入，使用梯度下降法调整控制变量以减少总成本，并确保控制输入在规定的约束内。
        :param x0: Current state.
        :param x_ref: Reference trajectory.
        :return: Current cost and predicted states after the control update.
        """

        with tf.GradientTape() as tape:
            x_pred = self.sim_open_loop(x0, self.u, t_sample=self.t_sample, H=self.H) # 开环预测
            cost = self.costs(x_ref, x_pred[:,4:5]) # 计算损失函数

        # 计算损失关于u的梯度
        gradients = tape.gradient(cost, self.u)

        # 创建一个 mask，只允许对[:, 2:] 有梯度
        mask = tf.concat([
            tf.zeros_like(self.u[:, :2]),  # 前两列不优化
            tf.ones_like(self.u[:, 2:])  # 后面列保留梯度
        ], axis=1)

        # 屏蔽前两列的梯度
        masked_gradients = gradients * mask
        # 应用优化器来更新控制变量
        self.optimizer.apply_gradients(zip([masked_gradients], [self.u]))
        # 约束处理
        self.ensure_constraints()
        return cost, x_pred # 返回计算得到的成本 cost 和预测的状态轨迹 x_pred

    def optimization_step_GA(self, x0, x_ref,iter):
        """
        执行一步基于梯度的优化来更新控制输入，使用遗传法调整控制变量以减少总成本，并确保控制输入在规定的约束内。
        :param x0: Current state.
        :param x_ref: Reference trajectory.
        :return: Current cost and predicted states after the control update.
        """
        u_shape = self.u.shape  # (H, 3)，其中 H 为预测步长
        num_steps = u_shape[0]  # 优化变量的维度应与预测步长一致

        # 用当前 self.u 的第三列作为初始值，并添加小扰动生成初始种群
        initial_u3 = self.u[:, 2].numpy()  # shape: (H,)
        sol_per_pop = 50
        initial_population = []
        for _ in range(sol_per_pop):
            # 在初始值周围扰动 ±0.1
            candidate = initial_u3 + np.random.uniform(-0.1, 0.1, size=initial_u3.shape)
            initial_population.append(candidate.tolist())

        def fitness_func(ga_instance, solution, solution_idx):
            # 将当前控制输入复制出来，并只替换第三列
            u_candidate = self.u.numpy().copy()
            u_candidate[:, 2] = solution  # solution 是一个 shape=(num_steps,) 的数组
            # 转换为 tf.Tensor，并确保数据类型为 float64
            u_candidate = tf.convert_to_tensor(u_candidate, dtype=tf.float64)
            # 用候选控制输入进行开环预测
            x_pred = self.sim_open_loop(x0, u_candidate, t_sample=self.t_sample, H=self.H)
            cost = self.costs(x_ref, x_pred[:, 4:5])
            fitness = -cost.numpy().item()  # GA 期望 fitness 越高越好，因此返回 -cost
            return fitness

        # 设置 GA 参数
        ga_instance = pygad.GA(
            num_generations=500,
            num_parents_mating=20,
            fitness_func=fitness_func,
            sol_per_pop=sol_per_pop,
            num_genes=num_steps,
            initial_population=initial_population,
            init_range_low=-1.0,  # 若未提供 initial_population，则每个基因在此范围内随机初始化
            init_range_high=1.0,
            mutation_percent_genes=10,
            mutation_type="random",
            crossover_type="two_points",
            parent_selection_type="rank",
            gene_type=np.float64,
            stop_criteria="saturate_10"
        )

        # 运行遗传算法
        ga_instance.run()

        # 调试：输出 GA 运行信息
        best_solution, best_fitness, best_solution_idx = ga_instance.best_solution()
        print(f"GA best fitness: {best_fitness}, found at candidate index: {best_solution_idx}")
        print(f"GA best solution (first 5 genes): {best_solution[:5]}")

        # 可选：画出收敛曲线
        '''
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 5))
            plt.plot(ga_instance.best_solutions_fitness, label='Best Fitness (=-Cost)', marker='o')
            plt.xlabel("Generation")
            plt.ylabel("Fitness")
            plt.title("GA Convergence Curve")
            plt.legend()
            plt.grid(True)
            # plt.savefig(f"ga_convergence_{iter}.png", dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print("Error in plotting GA convergence:", e)
        '''
        # 更新 self.u 的第三列为 GA 找到的最优解
        new_u = self.u.numpy()
        new_u[:, 2] = best_solution
        self.u.assign(new_u)
        self.ensure_constraints()

        # 用更新后的 self.u 进行一次预测，计算最新成本
        x_pred = self.sim_open_loop(x0, self.u, t_sample=self.t_sample, H=self.H)
        cost = self.costs(x_ref, x_pred[:, 4:5])

        return cost, x_pred # 返回计算得到的成本 cost 和预测的状态轨迹 x_pred

    @tf.function
    def ensure_constraints(self):
        """
        Ensures that the control inputs remain within their specified bounds after each optimization step.
        """
        for k in range(self.H):
            for i, u_ub_i in enumerate(self.u_ub):
                if i < 2:
                    continue
                if self.u[k, i] > u_ub_i:
                    # self.u.data[k, i] = u_ub_i
                    self.u[k, i].assign(u_ub_i)

            for i, u_lb_i in enumerate(self.u_lb):
                if i < 2:
                    continue
                if self.u[k, i] < u_lb_i:
                    # self.u.data[k, i] = u_lb_i
                    self.u[k, i].assign(u_lb_i)


    def sim(self, x0,  X_ref, T_ref, U_dis=None):
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
        if U_dis is not None:
            self.U_dis = tf.constant(U_dis, dtype=tf.float64)
            self.u.assign(tf.concat([self.U_dis[:self.H], self.u[:, 2:]], axis=1))

        X_mpc[0] = x0  # 第0行
        X_pred[0] = x0
        U_mpc[0] = self.u[0].numpy()

        for i, t in enumerate(T_ref[:-self.H]): # 从下标为0的取到倒数最后一个
            start_time = time.time()
            # print(X_ref[i:i + self.H + 1].shape)
            print('i',i,' t',t)
            '''
            # warm-start：将当前 self.u 的[:, 2:] 平移一位,仅动第三列
            shifted_u_opt = tf.concat([
                self.u[1:, 2:],  # u_1 到 u_{H-1}
                tf.expand_dims(self.u[-1, 2:], axis=0)  # 最后一行复制填补
            ], axis=0)
            '''
            # print('优化前的(前两列应该是变化的)', self.u[0])
            J, x_pred = self.solve_ocp(X_mpc[i], X_ref[i:i + self.H + 1],i)

            # print('优化后的', self.u)
            # 以当前的系统状态 X_mpc[i] (相当于这个时刻的初值)和接下来的状态参考 X_ref[i:i + self.H + 1] 为输入，解决最优控制问题。返回最优成本J和预测状态x_pred
            # self.H 预测时步

            u_k = self.u[0]  # 提取当前最优控制输入
            # print('当前最优u_k', u_k, u_k.shape)
            ocp_solving_time = time.time() - start_time
            self.solving_times[i] = ocp_solving_time

            start_time_0 = time.time()
            x_true = self.sim_plant_system(X_mpc[i], u_k, self.t_sample) # 模拟实际系统

            compute_time = time.time() - start_time_0
            self.compute_time[i] = compute_time
            # todo: 增加真实值的设置和读取功能，这里先用预测值代替
            # print(X_mpc[i])
            # x_true = self.sim_nn(X_mpc[i], u_k, self.t_sample)
            # print(x_true)
            # 更新预测和实际状态:下一个预测状态、下一个实际状态、记录使用的控制输入
            X_pred[i + 1] = x_pred[1]
            X_mpc[i + 1] = x_true
            U_mpc[i + 1] = u_k.numpy()

            # 更新 self.u 的下一个时刻的前两个维度
            # '''
            if i + 1 < N:
                # 滑动窗口平移：把这一轮的 u[1:] 向前滚动
                shifted_u = tf.concat([
                    self.u[1:],  # u_1 到 u_{H-1}
                    tf.expand_dims(self.u[-1], axis=0)  # 最后一行复制
                ], axis=0)
            # '''
            new_u_dis = tf.constant(self.U_dis[i + 1:i + 1 + self.H], dtype=tf.float64)
            new_u = tf.concat([new_u_dis, shifted_u[:, 2:]], axis=1)
            self.u.assign(new_u)

            # print('i,t',i,t)
            # 生成一个日志字符串，记录迭代信息，最优成本J，下一个时间点 t + self.t_sample 和控制输入值。
            log_str = f'\tIter: {str(i + 1).zfill(len(str(N - 1)))}/{N - 1},\tJ: {J.numpy():.2e},' \
                      f'\tt: {t.item() + self.t_sample:.2f} s,'

            err_pct = float(abs((x_true[4] - X_ref[i]) / abs(X_ref[i])) * 100)

            self.u_curr = float(u_k.numpy()[2])

            self.errors_pct.append(err_pct)
            if self.u_prev is not None:
                self.iaq_accum += abs(self.u_curr - self.u_prev)
            self.u_prev = self.u_curr
            self.u_history.append(self.u_curr)

            # 记录每个控制输入 u_k 和状态 x_true 的值，并附加解决 OCP 的时间。
            for i in range(len(u_k)):
                log_str = log_str + f'\tu{i + 1}: {u_k.numpy()[i]:.2f},'

            log_str = log_str + f'\目标处水位: {x_true[4]:.3f}, 误差百分比: {err_pct:.2f}%, IAQ累计: {self.iaq_accum:.2f}'

            log_str = log_str + f'\tOCP-solving-time: {ocp_solving_time:.2e} s'
            log_str = log_str + f'\tcompute_time: {compute_time:.2e} s'
            logging.info(log_str)

        return X_mpc, U_mpc, X_pred

    @tf.function
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

        # t = torch.tensor([[t_sample]], dtype=torch.float64) # 创建时间常量(1，1) 若为[t_sample]，则是一维向量(1,)
        # x_i = x0.unsqueeze(0)
        t = tf.constant(t_sample, dtype=tf.float64, shape=(1, 1)) # 创建时间常量(1，1)
        x_i = tf.expand_dims(x0, 0)
        # 初始化状态向量，将初始状态向量x0通过扩展维度转换为一个二维张量，新的维度被添加在索引0的位置，使其成为一个形状为 (1,n) 的张量（一行n列的矩阵）
        # 这样做的目的是与控制输入（u_array[i:i + 1]）进行拼接时，维度对齐
        X_pred = x_i # 初始预测状态
        # u_array 为self.u 为一个self.H行，self.input_dim的二维向量
        # u_array[i:i + 1]这样的切片方式可以选择一个范围内的行，这里是从i到i+1，但不包括 i+1。这种切片会保持原数组的维度。返回向量为（1，m）,m为控制变量维度

        # 每次迭代模拟系统的下一个状态，并将其添加到 X_pred 中

        for i in range(H):  # H为预测的步数

            x = tf.concat((t, x_i, u_array[i:i + 1]), 1)  # 沿列方向拼接向量 (1，1+n+m)  拼接在一起输入了
            x_pred = self.model(x) # 输出形状为（1,n），表示下一时刻
            X_pred = tf.concat((X_pred, x_pred), 0) # 沿行方向拼接向量，每次增加一行，终为 (H+1, n)，包含从初始状态到当前迭代步的所有预测状态
            x_i = x_pred # 状态更新
            # 假设，x0是一个包含三个状态变量的系统的状态向量，H为10的情况下，最终X_pred形状为[11,3]
        return X_pred

    def sim_nn(self, x0, u_array, t_sample):
        # 用神经网络计算的值作为真实值近似
        t = tf.constant(t_sample, dtype=tf.float64, shape=(1, 1)) # 创建时间常量(1，1)
        x_i = tf.expand_dims(x0, 0)
        u_array = tf.expand_dims(u_array, 0)
        x = tf.concat((t, x_i, u_array), 1)  # 沿列方向拼接向量 (1，1+n+m)  拼接在一起输入了
        x_pred = self.model(x) # 输出形状为（1,n），表示下一时刻
        return x_pred

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
        x_i = x0/100
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

        ras = win32.Dispatch("RAS641.HECRASController")  # HEC-RAS 6.41版本COM接口
        '''
        
        思路：
        修改流速文件，1. 打开流速文件 2.修改初值 3. 修改边界条件 4. prj文件里面增加 如果没有的话
        修改.p 文件 1. 指定流速文件 2.prj文件里面增加 如果没有的话
        修改.prj文件，指定.p 文件  这边u和p 可以不变，是循环调用的
        '''
        # 这边 tau = 600，
        x0 = x0/100
        print('X0',x0)
        time_steps = int(tau/60 +1)
        # 初始化q_upstream和wl_downstream的数组
        q_upstream = np.zeros(time_steps)
        q_downstream = np.zeros(time_steps)
        # print('计算初值self.q_0', self.q_0)
        # print('计算初值self.z_0',self.z_0)
        # 使用循环生成q_upstream和wl_downstream的值
        for t in range(time_steps):
            q_upstream[t] = u[0:1] + u[1:2] * t
            q_downstream[t] = u[2:3]
        # print('q_upstream',q_upstream)
        # print('q_downstream',q_downstream)
        PROJECT_PATH = r"E:\program\hec_ras_project\mpc_test_1\mpc_test.prj"
        ras.Project_Open(PROJECT_PATH)
        # ras.ShowRAS()

        file_path = r'E:\program\hec_ras_project\mpc_test_1\mpc_test.prj'
        # 这边可以随便选，因为已经有了
        sim_id = 3
        u_filename = modify_unsteady_file(sim_id, q_upstream, q_downstream, self.q_0, self.z_0, x0)
        ras.Project_Save

        new_p_filename, plan_title = create_plan_file(sim_id, 'E:\program\hec_ras_project\mpc_test_1\mpc_test.p03')
        ras.Project_Save

        # 保障.prj有Current Plan
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()  # 按行读取

        # **检查第二行是否是 `Current Plan=` 开头**
        if lines[1].strip().startswith("Current Plan="):
            # print("替换 `Current Plan=`")
            lines[1] = f"Current Plan=p{sim_id:02d}\n"  # **替换第二行**
        else:
            # print("增加 `Current Plan=` 到第二行")
            lines.insert(1, f"Current Plan=p{sim_id:02d}\n")  # **在第二行插入**

        # **保存文件**
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        ras.Project_Save
        ras.Project_Open(PROJECT_PATH)

        current_plan = ras.CurrentPlanFile()
        # print("current_plan",current_plan)

        ras.Compute_HideComputationWindow()
        success = ras.Compute_CurrentPlan(None, None, True)
        if not success:
            raise RuntimeError(f"Compute_CurrentPlan 失败！code={code}, messages={messages}")
            print('X0', x0)
            print('计算初值self.q_0', self.q_0)
            print('计算初值self.z_0', self.z_0)

        print(success)

        z_true = np.zeros((len(x0), ))
        v_true = np.zeros((len(x0),))
        # 用 enumerate 一一对应索引，不再用双重循环
        for idx, j in enumerate(range(1, 22, 5)):
            # 取出第 idx 行 这边第5列代表时间
            z_true[idx] = ras.Output_NodeOutput(1, 1, j, 0, 6, 2)[0]
            v_true[idx] = ras.Output_NodeOutput(1, 1, j, 0, 6, 23)[0]

        for j in range(1, 21):  # j = 1,2,...,20
            self.z_0[j-1] = ras.Output_NodeOutput(1, 1, j, 0, 6, 2)[0]

        for j in range(1, 22):  # j = 1,2,...,21
            self.q_0[j-1] = ras.Output_NodeOutput(1, 1, j, 0, 6, 9)[0]

        ras.QuitRAS()

        print(z_true)
        z_true = z_true *100

        self.t_list.append(tau)
        self.z_0_list.append(x0)
        self.u_list.append(np.array(u))
        self.z_list.append(z_true)
        self.v_list.append(v_true)

        print('下一阶段的初值',self.q_0)
        print('下一阶段初值',self.z_0)


        return z_true