# 参考：https://github.com/NekSfyris/ESEKF_IMU_GNSS_Lidar/blob/master/es_ekf.py

import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rotations import angle_normalize, rpy_jacobian_axis_angle, skew_symmetric, Quaternion


l_jac = np.zeros([9, 6])
l_jac[3:, :] = np.eye(6)  # motion model noise jacobian
h_jac = np.zeros([3, 9])
h_jac[:, :3] = np.eye(3)  # measurement model jacobian

#### 3. Initial Values #########################################################################
# Let's set up some initial values for our ES-EKF solver.

p_est = np.zeros([imu_f.data.shape[0], 3])  # position estimates
v_est = np.zeros([imu_f.data.shape[0], 3])  # velocity estimates
q_est = np.zeros([imu_f.data.shape[0], 4])  # orientation estimates as quaternions
p_cov = np.zeros([imu_f.data.shape[0], 9, 9])  # covariance matrices at each timestep

# Set initial values.
p_est[0] = gt.p[0]
v_est[0] = gt.v[0]
p_cov[0] = np.zeros(9)  # covariance of estimate
gnss_i  = 0
lidar_i = 0


def measurement_update(sensor_var, p_cov_check, y_k, p_check):
    '''

    :param sensor_var: 测量噪声的方差
    :param p_cov_check: 误差协方差
    :param y_k: 当前的测量值
    :param p_check:先验位置预测值
    :return:
    '''
    # 3.1 Compute Kalman Gain 计算卡尔曼增益
    R = sensor_var * np.eye(3)  # sensor_var: 这是一个标量，代表测量噪声的方差。
    K_k = p_cov_check @ H.T @ np.linalg.inv(H @ p_cov_check @ h.T + R)

    # 3.2 Compute error state 误差状态
    delta_xk = K_k @ (y_k - p_check)

    # 3.3 Correct predicted state 后验预测值
    p_hat = p_check + delta_xk[:3]

    # 3.4 Compute corrected covariance 更新误差协方差
    p_cov_hat = (np.eye(9) - K_k @ h) @ p_cov_check

    return p_hat, p_cov_hat

#### 5. Main Filter Loop #######################################################################
# start taking in the sensor data and creating estimates for our state in a loop.

for k in range(1, imu_f.data.shape[0]):
    delta_t = imu_f.t[k] - imu_f.t[k - 1]

    # 1. Update state with IMU inputs
    C_ns = Quaternion(*q_est[k-1]).to_mat()
    #print("C_ns = ",C_ns)

    # 1.1 Linearize the motion model and compute Jacobians
    # 先验预测值
    #Update state estimate
    p_est[k] = p_est[k - 1] + delta_t * v_est[k - 1] + ((delta_t ** 2) / 2) * (C_ns @ imu_f.data[k - 1] + g)

    F_k = np.eye(9) # motion model jacobian

    F_k[0:3, 3:6] = delta_t * np.eye(3)

    Q_k[0:3, 0:3] = var_imu_f * Q_k[0:3, 0:3]  # 测量噪声方差
    Q_k[3:6, 3:6] = var_imu_w * Q_k[3:6, 3:6] # can also be accessed with Q[:, -3:]
    Q_k *= delta_t**2

    # 2. Propagate uncertainty 先验误差协方差
    p_cov[k] = F_k @ p_cov[k-1] @ F_k.T + l_jac @ Q_k @ l_jac.T

    # 3. Check availability of GNSS and LIDAR measurements
    #Check if any GNSS data are available
    if gnss_i < gnss.t.shape[0] and abs(gnss.t[gnss_i] - imu_f.t[k-1]) < 0.001:
        p_est[k], p_cov[k] = measurement_update(var_gnss, p_cov[k], gnss.data[gnss_i].T, p_est[k])
        gnss_i += 1
