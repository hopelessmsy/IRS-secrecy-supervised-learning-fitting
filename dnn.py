# -*- coding: utf-8 -*-
epochs =200
import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf

data_ae = sio.loadmat("data_AE_4000.mat")
data_ai = sio.loadmat("data_AI_4000.mat")
data_au = sio.loadmat("data_AU_4000.mat")
data_ie = sio.loadmat("data_IE_4000.mat")
data_iu = sio.loadmat("data_IU_4000.mat")

f_data = sio.loadmat("data_beamforming_vector_4000.mat")

target = sio.loadmat("phase_4000.mat")

data_dict = {0: "TE", 1: "TI", 2: "TU", 3: "IE", 4: "IU", 5: "phase", 6: "beamforming_vector", 7: "Secrecy"}

p = np.ones([4000, 1, 1])


def get_new_data(data, q):
    new_data1 = []
    for i in range(len(data[data_dict[q]])):
        dataTT = []
        for j in data[data_dict[q]][i]:
            j = np.array(j).T
            dataT = []
            for k in range(len(j)):
                # Am_value=np.absolute(j[k])
                Phase_value = np.angle(j[k])
                dataT.extend( Phase_value)
            dataT = np.array(dataT).T
            dataTT.append(dataT)
        new_data1.append(dataTT)
    new_data1 = np.array(new_data1)
    return new_data1

def get_new_data1(data, q):
    data_real = []
    data_img = []
    for i in range(len(data[data_dict[q]])):
        real_temp = []
        img_temp = []
        for j in data[data_dict[q]][i]:
            j = np.array(j).T
            dataT = []
            real = []
            img = []
            for k in range(len(j)):
                real.extend(np.real(j[k]))
                img.extend(np.imag(j[k]))
            real = np.array(real).T
            img = np.array(img).T
            real_temp.append(real)
            img_temp.append(img)
        data_real.append(real_temp)
        data_img.append(img_temp)
    real = np.array(data_real)
    img = np.array(data_img)
    data = np.concatenate([real, img], axis=2)
    return data



def get_new_data2(data, q):
    new_data1 = []
    for i in range(len(data[data_dict[q]])):
        dataTT = []
        for j in data[data_dict[q]][i]:
            j = np.array(j).T
            dataT = []
            for k in range(len(j)):
                dataT.extend(j[k])
            dataT = np.array(dataT).T
            dataTT.append(dataT)
        new_data1.append(dataTT)
    new_data1 = np.array(new_data1)
    return new_data1

new_data_ae = get_new_data1(data_ae, 0)
new_data_ai = get_new_data1(data_ai, 1)
new_data_au = get_new_data1(data_au, 2)
new_data_ie = get_new_data1(data_ie, 3)
new_data_iu = get_new_data1(data_iu, 4)
new_target = get_new_data(target, 5)
f_data = get_new_data2(f_data, 6)
new_target = np.reshape(new_target, (4000, 16))

data_all = np.concatenate((new_data_ae, new_data_ai, new_data_au, new_data_ie, new_data_iu, p), axis=2)

data_train, data_test, train_label, test_label = train_test_split(data_all, new_target, train_size=0.8,
                                                                  random_state=22)

data_train = np.reshape(data_train, (len(data_train), 1, 209))
data_test = np.reshape(data_test, (len(data_test), 1, 209))

test_index_list = []
for i in data_test:
    k = 0
    for j in range(len(data_all)):
        if (i == data_all[j]).all():
            test_index_list.append(k)
            break
        k += 1

def dnn():
    model = models.Sequential()
    model.add(layers.Dense(units=512, activation='tanh'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(units=64, activation='tanh'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(units=32, activation='tanh'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(16, activation='tanh'))
    adam = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.9, epsilon=1e-08)
    model.compile(optimizer=adam, loss='mse', metrics=['mae'])

    return model


model = dnn()
history = model.fit(data_train, train_label, verbose=1, batch_size=16, epochs=epochs,
                    validation_data=(data_test, test_label))  # 用history接收
history_dict = history.history
train_loss = history_dict["loss"]
val_loss = history_dict["val_loss"]
plt.figure()
plt.plot(range(epochs), train_loss_4, label='train_loss')
plt.plot(range(epochs), val_loss_4, label='test_loss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()
