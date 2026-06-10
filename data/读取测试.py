import numpy as np

data = np.load('data.npz')

print(data.files)

array_example = data['X']
print(array_example)
print('-------------')
array_example = data['Y']
print(array_example)
print('-------------')
array_example = data['U']
print(array_example)
print('-------------')
array_example = data['T']
print(array_example)
print('-------------')
array_example = data['X0']
print(array_example)

# Lower and upper bound
lb = data['lb']
ub = data['ub']
print(lb)
print(ub)

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
print('U.shape', U.shape)
print('T.shape',T.shape)
print('X0.shape',X0.shape)
print('X.shape',X_star.shape)
print('Y.shape',Y_star.shape)
print('X_test.shape',X_test.shape)
print('Y_test.shape',Y_test.shape)

