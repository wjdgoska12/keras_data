# 0. 사용할 패키지 불러오기
import numpy as np
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation, Dropout
#from keras import optimizers


# 1. 데이터 준비하기
#dataset = np.genfromtxt("testdata.csv", delimiter=",",dtype=float)
dataset = np.genfromtxt("C:\\Users\\user\\Desktop\\dataset17462.csv", delimiter=",",dtype=float) #    #event 15 - BW - cpi - ipc - AI - Thread 
#dataset = np.genfromtxt("C:\\Users\\user\\Desktop\\dataset17462_V1.csv", delimiter=",",dtype=float) #    #event 15 - BW - cpi - ipc - AI 
#dataset = np.genfromtxt("C:\\Users\\user\\Desktop\\dataset17462_V2.csv", delimiter=",",dtype=float) #    #event 3 - BW - cpi - ipc - AI 

#dataset = np.genfromtxt("ta.csv", delimiter=",",dtype=float)
#encoding='UTF-8,'




# 2. 데이터셋 생성하기
x_train = dataset[:12223,0:19]
y_train = dataset[:12223,20:]

#print(x_train)
#print(y_train)
print(x_train.shape)
print(y_train.shape)


x_val = dataset[12223:13969,0:19]
y_val = dataset[12223:13969,20:]
print(x_val.shape)
print(y_val.shape)


x_test = dataset[13969:,0:19]
y_test = dataset[13969:,20:]
print(x_test.shape)
print(y_test.shape)

#print(y_test[3492])

# 3. 모델 구성하기

model = Sequential()
#model.add(Dense(64, input_dim=2, activation='linear'))
model.add(Dense(64, input_dim=19, activation='relu'))
model.add(Dropout(0.2))   
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))   
#model.add(Dense(5, activation='relu'))
model.add(Dense(1))


# 4. 모델 학습과정 설정하기

#model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])
#model.compile(loss='mse', optimizer='Adadelta', metrics=['mae'])
#model.compile(loss='mse', optimizer='Adagrad', metrics=['mae'])
#model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.compile(loss='mse', optimizer='sgd', metrics=['mae'])

#model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
#model.compile(loss='mse', optimizer='Adadelta', metrics=['accuracy'])
#model.compile(loss='mse', optimizer='Adagrad', metrics=['accuracy'])
#model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
#model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

# 5. 모델 학습시키기
history = model.fit(x_train, y_train, epochs=500, batch_size=10,verbose=1,validation_data=(x_val, y_val))
#

# 6. 모델 평가하기
print(model.predict(x_test,batch_size=10))
scores = model.evaluate(x_test, y_test,batch_size=10)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()