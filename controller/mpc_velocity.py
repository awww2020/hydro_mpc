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

    def __init__(self, plant, model, model_u, u_ub, u_lb, t_sample=0.1, H=10,
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
        # 这边默认不加就是z,model_v是为了区别跟控制变量u
        self.model = model
        self.model_v = model_u
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


    def costs(self, x_ref, x_pred, v_pred):
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
        penalty = tf.reduce_sum( tf.square(tf.nn.relu(x_ref[:,1:2] - v_pred))@ self.Q )

        J = tf.reduce_sum(tf.square(x_ref[:,0:1]/100 - x_pred/100) @ self.Q) \
            + 10.0*penalty  \
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
            u_pred = self.sim_open_loop_v(x0, self.u, t_sample=self.t_sample, H=self.H)  # 开环预测
            cost = self.costs(x_ref, x_pred[:,4:5],u_pred[:,4:5]) # 计算损失函数

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
        # todo 兼顾水位和流速控制的时候这边没有改
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
        :param x0: Initial state.
        :param X_ref: Reference trajectory.
        :param T_ref: Time instances corresponding to the reference trajectory.
        :return: Simulated state and control input trajectories.
        """
        N = len(T_ref)
        X_mpc = np.zeros((N, len(x0)))
        X_pred = np.zeros((N, len(x0)))
        U_mpc = np.zeros((N, self.u.shape[1]))
        if U_dis is not None:
            self.U_dis = tf.constant(U_dis, dtype=tf.float64)
            self.u.assign(tf.concat([self.U_dis[:self.H], self.u[:, 2:]], axis=1))

        X_mpc[0] = x0
        X_pred[0] = x0
        U_mpc[0] = self.u[0].numpy()

        for i, t in enumerate(T_ref[:-self.H]):
            start_time = time.time()
            print('i',i,' t',t)
            J, x_pred = self.solve_ocp(X_mpc[i], X_ref[i:i + self.H + 1],i)
            u_k = self.u[0]
            ocp_solving_time = time.time() - start_time
            self.solving_times[i] = ocp_solving_time

            start_time_0 = time.time()
            x_true,v_true = self.sim_plant_system(X_mpc[i], u_k, self.t_sample) # 模拟实际系统

            compute_time = time.time() - start_time_0
            self.compute_time[i] = compute_time

            X_pred[i + 1] = x_pred[1]
            X_mpc[i + 1] = x_true
            U_mpc[i + 1] = u_k.numpy()


            if i + 1 < N:
                shifted_u = tf.concat([
                    self.u[1:],
                    tf.expand_dims(self.u[-1], axis=0)
                ], axis=0)

            new_u_dis = tf.constant(self.U_dis[i + 1:i + 1 + self.H], dtype=tf.float64)
            new_u = tf.concat([new_u_dis, shifted_u[:, 2:]], axis=1)
            self.u.assign(new_u)

            log_str = f'\tIter: {str(i + 1).zfill(len(str(N - 1)))}/{N - 1},\tJ: {J.numpy():.2e},' \
                      f'\tt: {t.item() + self.t_sample:.2f} s,'

            err_pct = float(abs((x_true[4] - X_ref[i,0:1]) / abs(X_ref[i,0:1])) * 100)

            self.u_curr = float(u_k.numpy()[2])

            self.errors_pct.append(err_pct)
            if self.u_prev is not None:
                self.iaq_accum += abs(self.u_curr - self.u_prev)
            self.u_prev = self.u_curr
            self.u_history.append(self.u_curr)

            for i in range(len(u_k)):
                log_str = log_str + f'\tu{i + 1}: {u_k.numpy()[i]:.2f},'

            log_str = log_str + f'\目标处水位: {x_true[4]:.3f}, 误差百分比: {err_pct:.2f}%, IAQ累计: {self.iaq_accum:.2f}'
            log_str = log_str + f'\目标处流速: {v_true[4]:.2f}'
            log_str = log_str + f'\tOCP-solving-time: {ocp_solving_time:.2e} s'
            log_str = log_str + f'\tcompute_time: {compute_time:.2e} s'
            logging.info(log_str)

        return X_mpc, U_mpc, X_pred

    @tf.function
    def sim_open_loop(self, x0, u_array, t_sample, H):
        """
        Simulates the system's open-loop response over the prediction horizon using the predictive model.
        :param x0: Current state.
        :param u_array: Array of control inputs for each step in the horizon.
        :param t_sample: Sampling time.
        :param H: Prediction horizon.
        :return: Predicted states over the horizon.
        """

        t = tf.constant(t_sample, dtype=tf.float64, shape=(1, 1)) # 创建时间常量(1，1)
        x_i = tf.expand_dims(x0, 0)
        X_pred = x_i


        for i in range(H):

            x = tf.concat((t, x_i, u_array[i:i + 1]), 1)
            x_pred = self.model(x)
            X_pred = tf.concat((X_pred, x_pred), 0)
            x_i = x_pred

        return X_pred

    @tf.function
    def sim_open_loop_v(self, x0, u_array, t_sample, H):
        """
        Simulates the system's open-loop response over the prediction horizon using the predictive model.
        :param x0: Current state.
        :param u_array: Array of control inputs for each step in the horizon.
        :param t_sample: Sampling time.
        :param H: Prediction horizon.
        :return: Predicted states over the horizon.
        """

        t = tf.constant(t_sample, dtype=tf.float64, shape=(1, 1))
        x_i = tf.expand_dims(x0, 0)

        V_pred = x_i

        for i in range(H):
            x = tf.concat((t, x_i, u_array[i:i + 1]), 1)
            v_pred = self.model_v(x)
            V_pred = tf.concat((V_pred, v_pred), 0)
            x_i = v_pred

        return V_pred

    def sim_nn(self, x0, u_array, t_sample):

        t = tf.constant(t_sample, dtype=tf.float64, shape=(1, 1))
        x_i = tf.expand_dims(x0, 0)
        u_array = tf.expand_dims(u_array, 0)
        x = tf.concat((t, x_i, u_array), 1)
        x_pred = self.model(x)
        return x_pred

    def sim_open_loop_plant(self, x0, u_array, t_sample, H):
        """
        Simulates the system's open-loop response over the prediction horizon using the physical plant.
        :param x0: Current state.
        :param u_array: Array of control inputs for each step in the horizon.
        :param t_sample: Sampling time.
        :param H: Prediction horizon.
        :return: True states over the horizon.
        """
        x_i = x0/100
        X = x_i

        for i in range(H):
            x,u = self.sim_plant_system(x_i, u_array[i], t_sample)
            X = np.vstack((X, x))
            x_i = x

        return X

    def sim_plant_system(self, x0, u, tau):
        """
        Simulates the physical plant for a given control input over a single time step.
        :param x0: Current state.
        :param u: Control input to be applied.
        :param tau: Time step duration.
        :return: New state after applying the control input.这一时间步结束时的系统状态
        """

        ras = win32.Dispatch("RAS641.HECRASController")
        # 这边 tau = 600，
        x0 = x0/100
        print('X0',x0)
        time_steps = int(tau/60 +1)

        q_upstream = np.zeros(time_steps)
        q_downstream = np.zeros(time_steps)

        for t in range(time_steps):
            q_upstream[t] = u[0:1] + u[1:2] * t
            q_downstream[t] = u[2:3]

        PROJECT_PATH = r"E:\program\hec_ras_project\mpc_test_1\mpc_test.prj"
        ras.Project_Open(PROJECT_PATH)


        file_path = r'E:\program\hec_ras_project\mpc_test_1\mpc_test.prj'

        sim_id = 3
        u_filename = modify_unsteady_file(sim_id, q_upstream, q_downstream, self.q_0, self.z_0, x0)
        ras.Project_Save

        new_p_filename, plan_title = create_plan_file(sim_id, 'E:\program\hec_ras_project\mpc_test_1\mpc_test.p03')
        ras.Project_Save

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if lines[1].strip().startswith("Current Plan="):

            lines[1] = f"Current Plan=p{sim_id:02d}\n"
        else:
            lines.insert(1, f"Current Plan=p{sim_id:02d}\n")

        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        ras.Project_Save
        ras.Project_Open(PROJECT_PATH)

        current_plan = ras.CurrentPlanFile()

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

        for idx, j in enumerate(range(1, 22, 5)):
            z_true[idx] = ras.Output_NodeOutput(1, 1, j, 0, 6, 2)[0]
            v_true[idx] = ras.Output_NodeOutput(1, 1, j, 0, 6, 23)[0]

        for j in range(1, 21):
            self.z_0[j-1] = ras.Output_NodeOutput(1, 1, j, 0, 6, 2)[0]

        for j in range(1, 22):
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


        return z_true, v_true