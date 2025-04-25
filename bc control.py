import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM
from keras.layers import Dense,Dropout
from keras.models import Sequential
from scikeras.wrappers import KerasRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, LSTM, Dense, concatenate
import numpy as np
import joblib
import pywt
import keras_tuner as kt
import tensorflow as tf
import pickle
import time
from sklearn.model_selection import GridSearchCV
import math
from math import sqrt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#加载数据
data1=pd.read_excel(r'D:\pycharm\project\LSTMpj\code\3.xlsx',sheet_name=1)
print(data1.head())
target_columns = ['东茭泾内河水位', '西泗塘内河水位', '郝桥港内河水位']  # 替换为实际的目标列名,'虹口港内河水位','苏州河内河水位'
feature_columns = [col for col in data1.columns if col not in target_columns][:16]
original_targets = data1[target_columns].values
# 对特征列进行归一化
scaler_features = MinMaxScaler(feature_range=(0, 1))
data1_features_scaled = scaler_features.fit_transform(data1[feature_columns])
# 对目标列进行归一化
scaler_targets = MinMaxScaler(feature_range=(0, 1))
data1_targets_scaled = scaler_targets.fit_transform(data1[target_columns])
# 保存归一化参数
scaler_features_filename = r'D:\pycharm\project\LSTMpj\加载参数\scaler_features_water4.pkl'
joblib.dump(scaler_features, scaler_features_filename)
# 保存目标列的归一化参数
scaler_targets_filename = r'D:\pycharm\project\LSTMpj\加载参数\scaler_targets_water4.pkl'
joblib.dump(scaler_targets, scaler_targets_filename)

# 将归一化后的特征和目标列合并
data1_scaled = np.hstack((data1_features_scaled, data1_targets_scaled))

# 数据拆分训练集和测试集
data1_for_training = data1_scaled[:-532]
data1_for_testing = data1_scaled[-532:]

#输入和输出
def createXY(dataset,n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past:i, :len(feature_columns)])
            dataY.append(dataset[i,len(feature_columns):])
    return np.array(dataX),np.array(dataY)

trainX,trainY=createXY(data1_for_training,508)
testX,testY=createXY(data1_for_testing,508)
print(testY)
print(trainY)
print(trainX)
print(testX)
# 输入序列的维度
input_shape = (508, len(feature_columns))  # 每个序列有508个时间步，每个时间步有16个特征
# 输入层
inputs = Input(shape=input_shape)
# LSTM层
lstm1 = LSTM(128, return_sequences=False)(inputs)
# 分叉到两个独立的输出层
output1 = Dense(1, activation='tanh', name='output1')(lstm1)  # 第一个输出层，线性激活通常用于回归
output2 = Dense(1, activation='tanh', name='output2')(lstm1)  # 第二个输出层，同样使用线性激活
output3= Dense(1, activation='tanh', name='output3')(lstm1)
# output4= Dense(1, activation='tanh', name='output4')(lstm1)
# output5= Dense(1, activation='tanh', name='output5')(lstm1)
model = Model(inputs=inputs, outputs=[output1, output2,output3])#,output4,output5
learning_rate = 0.0001
optimizer = Adam(lr=learning_rate)
# 编译模型
model.compile(optimizer=optimizer, loss=['mae','mae', 'mae'], metrics=['mse'])#,'mae','mae'

# 打印模型结构
model.summary()
# 训练模型
history = model.fit(trainX, [trainY[:, 0], trainY[:, 1],trainY[:, 2]], epochs=130, batch_size=128, validation_split=0.2, verbose=1)#,trainY[:, 3],trainY[:, 4]

# 获取最佳模型
predictions = model.predict(testX)
pred1 = predictions[0]  # 第一个输出的预测
pred2 = predictions[1]  # 第二个输出的预测
pred3 = predictions[2]
# pred4= predictions[3]
# pred5= predictions[4]
model.save(r'D:\pycharm\project\LSTMpj\加载参数\lstm_best_model_water4.h5')
# 反归一化预测结果
def inverse_transform_predictions(predictions, scaler):
    predictions_array = np.hstack(predictions)
    return scaler.inverse_transform(predictions_array)

water_predictions = inverse_transform_predictions(predictions, scaler_targets)

# 提取各个预测值
pred_s1 = water_predictions[:, 0]
pred_s2 = water_predictions[:, 1]
pred_s3 = water_predictions[:, 2]
# pred_s4 = water_predictions[:, 3]
# pred_s5 = water_predictions[:, 4]
# 将一维数组转换为二维数组
pred_s1 = pred_s1[:, np.newaxis]
pred_s2 = pred_s2[:, np.newaxis]
pred_s3 = pred_s3[:, np.newaxis]
# pred_s4 = pred_s4[:, np.newaxis]
# pred_s5 = pred_s5[:, np.newaxis]
S=original_targets[-24:]
original_s1 =S[:,0]
original_s2 =S[:, 1]
original_s3 =S[:, 2]
# original_s4 =S[:, 3]
# original_s5 =S[:, 4]
# original_s2 = original_s2.reshape(-1, 1)
# original_s3 = original_s3.reshape(-1, 1)
combined_array = np.concatenate((pred_s1, pred_s2,pred_s3), axis=1)
df_combined = pd.DataFrame(combined_array, columns=['Column1', 'Column2', 'Column3'])
output_file = "D:\\pycharm\\project\\LSTMpj\\code\\shuiweiyuceshuchu.xlsx"
# # 如果需要将结果保存为Excel文件
df_combined.to_excel(output_file, index=False)
#绘图
plt.figure(figsize=(12, 6))

plt.subplot(2, 3, 1)
plt.plot(original_s1, color='red', label='东茭泾water level')
plt.plot(pred_s1, color='blue', label='Predicted water level 1')
plt.title('东茭泾闸内水位')
plt.xlabel('Time Step')
plt.ylabel('Flow')
plt.legend()

plt.subplot(2, 3, 2)
plt.plot(original_s2, color='red', label='西泗塘waterlevel')
plt.plot(pred_s2, color='blue', label='Predicted water level 2')
plt.title('西泗塘闸内水位')
plt.xlabel('Time Step')
plt.ylabel('Flow')
plt.legend()

plt.subplot(2, 3,3)
plt.plot(original_s3, color='red', label='郝桥港water level')
plt.plot(pred_s3, color='blue', label='Predicted water level3')
plt.title('郝桥港闸内水位')
plt.xlabel('Time Step')
plt.ylabel('Flow')
plt.legend()

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体，或其他支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
plt.show(block=True)
# 计算误差
mse1 = mean_squared_error(original_s1,pred_s1)
mse2 = mean_squared_error(original_s2,pred_s2)
mse3 = mean_squared_error(original_s3,pred_s3)
# mse4 = mean_squared_error(original_s4,pred_s4)
# mse5 = mean_squared_error(original_s5,pred_s5)
r2_1 = r2_score(original_s1, pred_s1)
r2_2= r2_score(original_s2, pred_s2)
r2_3= r2_score(original_s3, pred_s3)
# r2_4= r2_score(original_s4, pred_s4)
# r2_5= r2_score(original_s5, pred_s5)，{r2_4}，{r2_5}'，{mse4}，{mse5}'
print(f'R² score: {r2_1},{r2_2},{r2_3}')
print(f'MSE score: {mse1},{mse2},{mse3}')
#print("mean_squared_error:", mean_squared_error(original, pred))
#print("rmse:", sqrt(mean_squared_error(original, pred)))
# 绘制训练和验证loss图
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim([0, max(plt.ylim())])  # 可选：调整y轴范围以更好地显示
plt.legend(loc='upper right')
plt.show()