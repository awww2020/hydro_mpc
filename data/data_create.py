import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_excel('训练数据含流速_梯形断面_剔除.xlsx')

assert df.shape[1] == 19, "DataFrame should have 19 columns"

rows = []
start_row = 0
while start_row < df.shape[0]:
    rows.extend(range(start_row, min(start_row + 11, df.shape[0])))
    start_row += 13

print(df.shape[0])

X = df.iloc[rows, :9].values
X[:, 1:6] *= 100
Y_z = df.iloc[rows, 9:14].values*100
Y_u = df.iloc[rows,14:].values

print('X',X.shape)
print('Y_z',Y_z.shape)
print('Y_u',Y_u.shape)

rows = []
start_row = 0
while start_row < df.shape[0]:
    rows.extend(range(start_row, min(start_row + 1, df.shape[0])))
    start_row += 13
U = df.iloc[rows, 6:9].values

print('U',U.shape)

rows = []
start_row = 0
while start_row < df.shape[0]:
    rows.extend(range(start_row, min(start_row + 1, df.shape[0])))
    start_row += 13

X0 = df.iloc[rows, 1:6].values*100
print('XO',X0)
print('XO',X0.shape)

# T = [0, 60, 120, ..., 600]
T = np.arange(0, 660, 60)

num_groups = 913
group_size = 11

group_indices = np.arange(num_groups)
train_indices, test_indices = train_test_split(group_indices, test_size=90/913, random_state=42)
# print('train_indices',train_indices)

X_train = np.concatenate([X[i * group_size:(i + 1) * group_size] for i in train_indices])
X_test = np.concatenate([X[i * group_size:(i + 1) * group_size] for i in test_indices])

Y_z_train = np.concatenate([Y_z[i * group_size:(i + 1) * group_size] for i in train_indices])
Y_u_train = np.concatenate([Y_u[i * group_size:(i + 1) * group_size] for i in train_indices])

Y_z_test = np.concatenate([Y_z[i * group_size:(i + 1) * group_size] for i in test_indices])
Y_u_test = np.concatenate([Y_u[i * group_size:(i + 1) * group_size] for i in test_indices])

print(X_train.shape)
print(X_test.shape)

ub = X.max(axis=0)
lb = X.min(axis=0)
print(ub)
print(lb)

X_train = np.array(X_train, dtype=np.float64)
Y_z = np.array(Y_z, dtype=np.float64)
Y_u = np.array(Y_u, dtype=np.float64)
U = np.array(U, dtype=np.float64)
T = np.array(T, dtype=np.float64)
X0 = np.array(X0, dtype=np.float64)
X_test = np.array(X_test, dtype=np.float64)
Y_z_test = np.array(Y_z_test, dtype=np.float64)
Y_u_test = np.array(Y_u_test, dtype=np.float64)
ub = np.array(ub, dtype=np.float64)
lb = np.array(lb, dtype=np.float64)
assert X.ndim == 2, "X should be a 2D array"
assert Y_z.ndim == 2, "Y should be a 2D array"
assert U.ndim == 2, "U should be a 2D array"
assert T.ndim == 1, "T should be a 1D array"
assert X0.ndim == 2, "X0 should be a 2D array"
assert X_test.ndim == 2, "X_test should be a 2D array"
assert Y_z_test.ndim == 2, "Y_test should be a 2D array"
assert ub.ndim == 1, "ub should be a 1D array"
assert lb.ndim == 1, "lb should be a 1D array"
#'''
df_1 = pd.read_excel(
    '验证数据_4分钟间隔_真实.xlsx',
    header=None,
)

assert df_1.shape[1] == 19, "DataFrame should have 19 columns"

X_1 = df_1.iloc[:, :9].values
X_1[:, 1:6] *= 100
Y_z_1 = df_1.iloc[:, 9:14].values*100
Y_u_1 = df_1.iloc[:, 14:19].values
U_1 = df.iloc[:, 6:9].values
X0_1 = df.iloc[:, 1:6].values


X_train = np.concatenate((X_train, X_1), axis=0)
Y_z_train = np.concatenate((Y_z_train, Y_z_1), axis=0)
Y_u_train = np.concatenate((Y_u_train, Y_u_1 ), axis=0)
U = np.concatenate((U, U_1), axis=0)
X0 = np.concatenate((X0, X0_1), axis=0)
#'''

np.savez('data_0.npz', X=X_train, Y_z=Y_z_train, Y_u=Y_u_train, U=U, T=T, X0=X0,
         X_test=X_test, Y_z_test=Y_z_test, Y_u_test=Y_u_test, ub=ub, lb=lb)
data = np.load('data_0.npz')
print(data.files)
