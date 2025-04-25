import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 假设Excel文件名为 'data.xlsx'，并且数据在第一个工作表中
# df = pd.read_excel()

df = pd.read_excel(
    '验证数据_4分钟间隔.xlsx',
    header=None,    # 第一行也当数据读入
)

# 确保 DataFrame 有 19 列
assert df.shape[1] == 19, "DataFrame should have 19 columns"

X = df.iloc[:, :9].values
X[:, 1:6] *= 100
Y_z = df.iloc[:, 9:14].values*100
Y_u = df.iloc[:, 14:19].values

print(X)
print(X.shape)

print(Y_z)
print(Y_z.shape)

print(Y_u)
print(Y_u.shape)

# 制作U
U = df.iloc[:, 6:9].values
print('U',U.shape)

# 制作X0
X0 = df.iloc[:, 1:6].values
print('XO',X0.shape)

T = df.iloc[:, 0:1].values.flatten()
print('T',T.shape)

X = np.array(X, dtype=np.float64)
Y_z = np.array(Y_z, dtype=np.float64)
Y_u = np.array(Y_u, dtype=np.float64)
U = np.array(U, dtype=np.float64)
T = np.array(T, dtype=np.float64)
X0 = np.array(X0, dtype=np.float64)

# 确保所有数组的维度一致
assert X.ndim == 2, "X should be a 2D array"
assert Y_z.ndim == 2, "Y should be a 2D array"
assert Y_u.ndim == 2, "Y should be a 2D array"
assert U.ndim == 2, "U should be a 2D array"
assert T.ndim == 1, "T should be a 1D array"
assert X0.ndim == 2, "X0 should be a 2D array"


# 保存到npz文件
np.savez('data_sc_4min.npz', X=X, Y_z=Y_z,Y_u=Y_u, U=U, T=T, X0=X0)
# 加载npz文件以验证保存是否正确
data = np.load('data_sc_4min.npz')
print(data.files)
