import numpy as np

#### 3. Initial Values #########################################################################
# Let's set up some initial values for our ES-EKF solver.
num_data_points = 1000  # 假设我们知道将会处理1000个数据点
U_est = np.zeros([num_data_points, 3])  # position estimates
p_cov = np.zeros((num_data_points, 9, 9))  # 假设状态向量为9维

# Set initial values.
U_est[0] = np.array([0, 0, 0])
p_cov[0] = np.eye(9)  # covariance of estimate


def measurement_update(mea_var, p_cov_check, y_k, p_check):
    '''
    :param mea_var: 测量噪声的方差
    :param p_cov_check: 误差协方差
    :param y_k: 当前的测量值
    :param p_check:先验位置预测值
    :return:
    '''
    # 3.1 Compute Kalman Gain 计算卡尔曼增益
    R = mea_var * np.eye(3)  # mea_var: 这是一个标量，代表测量噪声的方差。
    K_k = p_cov_check @ H.T @ np.linalg.inv(H @ p_cov_check @ h.T + R)

    # 3.2 Compute error state 误差状态
    delta_xk = K_k @ (y_k - p_check)

    # 3.3 Correct predicted state 后验预测值
    p_hat = p_check + delta_xk[:3]

    # 3.4 Compute corrected covariance 更新误差协方差
    p_cov_hat = (np.eye(9) - K_k @ h) @ p_cov_check

    return p_hat, p_cov_hat


def numerical_jacobian(f, state, control, dt, epsilon=1e-5):
    """
    Numerically approximate the Jacobian of a function f.
    """
    jacobian = np.zeros((len(state), len(state)))
    for i in range(len(state)):
        perturbed_state = np.copy(state)
        perturbed_state[i] += epsilon
        f_plus = f(perturbed_state, control, dt)

        perturbed_state[i] -= 2 * epsilon
        f_minus = f(perturbed_state, control, dt)

        jacobian[:, i] = (f_plus - f_minus) / (2 * epsilon)

    return jacobian


#### 5. Main Filter Loop #########################################################################

for k in range(1, ?):
    # 1.1 Linearize the motion model and compute Jacobians
    # 先验预测值
    # Predict state using the NN-based motion model
    U_est[k] = PINN.forward_U( U_est[k-1], X_tar, u[k], t )
    # U_est[k] 是K时刻目标位置的流速预测值，U_est[k-1] 是上一时刻k时刻目标位置的流速预测值，X_tar 是目标位置，u[k] 是k时刻的控制输入， t是时间

    # Compute the Jacobian using numerical differentiation
    F_k = numerical_jacobian(lambda x, u, t: PINN.forward_U(x, u, t), U_est[k-1], X_tar, u[k], t)

    # Propagate uncertainty
    p_cov[k] = F_k @ p_cov[k - 1] @ F_k.T + Q_k

    # 3. Check availability of real-time measurements
    #Check if any real-time data are available
    if gnss_i < len(gnss.t) and abs(gnss.t[gnss_i] - t[k-1]) < 0.001:
        U_est[k], p_cov[k] = measurement_update(var_mea, p_cov[k], gnss.data[gnss_i], U_est[k])
        gnss_i += 1
