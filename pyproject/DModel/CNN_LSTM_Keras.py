from keras import Input
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers import LSTM
#模型参数
time_step=100
# Convolution  卷积
filter_length = 5    # 滤波器长度
nb_filter = 64       # 滤波器个数
pool_length = 4      # 池化长度
# LSTM
lstm_output_size = 70   # LSTM 层输出尺寸
# Training   训练参数
batch_size = 30   # 批数据量大小
nb_epoch = 2      # 迭代次数
# 构建模型
model = Sequential()
model.add(Input(shape=(time_step, 128))) # 输入特征接收维度)  # 词嵌入层
model.add(Dropout(0.25)) # Dropout层
# 1D 卷积层，对词嵌入层输出做卷积操作
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
# 池化层
model.add(MaxPooling1D(pool_length=pool_length))
# LSTM 循环层
model.add(LSTM(lstm_output_size))
# 全连接层，只有一个神经元，输入是否为正面情感值
model.add(Dense(1))
model.add(Activation('sigmoid'))  # sigmoid判断情感（此处来做文本的情感分类问题）
model.summary()   # 模型概述
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
X_train = [0, 1, 2, 3]
y_train = [0, 1, 2, 3]
X_test = [0, 2, 4, 6]
y_test = [0, 2, 4, 6]
# 训练
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          validation_data=(X_test, y_test))
