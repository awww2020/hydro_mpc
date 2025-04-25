import numpy as np
from scipy.interpolate import interp1d


# 读取 txt 文件
file_path = '原始数据.txt'  # 替换为你的文件路径
data = np.loadtxt(file_path)

# 提取时间点和对应的值
time_points = data[:, 0]
values = data[:, 1]

# 检查数据
print("Time Points:", time_points)
print("Values:", values)

# 创建线性插值函数
f_interp = interp1d(time_points, values, kind='linear', fill_value="extrapolate")

# 插值出 0 到 1200 之间的数据，步长为 1
new_time_points = np.arange(0, 1201, 1)
interpolated_values = f_interp(new_time_points)

# 提取 0, 1, 2, 3 的数据
interpolated_values_linear = interpolated_values[0:1201]
print(interpolated_values_linear.shape)
# 函数：滚动计算每组10行的平均值，并赋值给这两行
def rolling_average_and_assign(arr, group_size=4):
    n_rows = len(arr)
    if n_rows < group_size:
        raise ValueError("数组的行数必须大于或等于组大小")

    for start in range(0, n_rows, group_size):
        end = start + group_size
        if end > n_rows:
            end = n_rows  # 如果超出了数组的范围，则截断到最后一行
        # 计算当前组行的平均值
        group_average = np.mean(arr[start:end], axis=0)
        # 将平均值赋值给当前组行
        arr[start:end] = group_average
    return arr
# 调用函数进行处理
values_rolling = rolling_average_and_assign(interpolated_values_linear, group_size=4)

output_file_path = 'interpolated_values_linear.txt'
output_file_path_1 = 'interpolated_values_linear_处理.txt'

np.savetxt(output_file_path_1, values_rolling, fmt='%f', delimiter="\t", header="Time\tInterpolated_Value", comments='')









