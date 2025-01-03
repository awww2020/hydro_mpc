import numpy as np

# 加载 NPZ 文件
data = np.load('data.npz')

# 列出 NPZ 文件中存储的所有数组/项目名称
print(data.files)

# 访问 NPZ 文件中特定数组的数据，例如：
array_example = data['X']
print(array_example)

# Lower and upper bound
lb = data['lb']
ub = data['ub']

# All data
X_star = data['X']
Y_star = data['Y']

U = data['U']
T = data['T']
X0 = data['X0']

X_test = data['X_test']
Y_test = data['Y_test']


print('lb.shape',lb.shape)
print('ub.shape',ub.shape)
print('U.shape',lb.shape)
print('T.shape',lb.shape)
print('X0.shape',lb.shape)
print('X.shape',X_star.shape)
print('Y.shape',Y_star.shape)
print('X_test.shape',X_test.shape)
print('Y_test.shape',Y_test.shape)

'''
多特征输入（7个特征）和多目标输出（4个输出）
lb.shape (7,)
ub.shape (7,)
U.shape (7,)
T.shape (7,)
X0.shape (7,)
X.shape (20480, 7)
Y.shape (20480, 4)
X_test.shape (800, 7)
Y_test.shape (800, 4)
'''


