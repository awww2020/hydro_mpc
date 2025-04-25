from utils.data import load_ref_trajectory, load_data

X_ref, T_ref = load_ref_trajectory('./data')
print(X_ref)
print(X_ref.shape)
print(T_ref)
print(T_ref.shape)