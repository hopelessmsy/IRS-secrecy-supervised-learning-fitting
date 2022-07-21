# -*- coding: utf-8 -*-

import math
import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf

epochs =1000

data_ae = sio.loadmat("data_AE.mat")
data_ai = sio.loadmat("data_AI.mat")
data_au = sio.loadmat("data_AU.mat")
data_ie = sio.loadmat("data_IE.mat")
data_iu = sio.loadmat("data_IU.mat")

f_data = sio.loadmat("data_beamforming_vector_1000.mat")

target = sio.loadmat("data_IRS_phase（target）.mat")

data_dict = {0: "TE", 1: "TI", 2: "TU", 3: "IE", 4: "IU", 5: "IRS_phase", 6: "beamforming_vector", 7: "Secrecy"}

p = np.ones([1000, 1, 1])


def get_new_data(data, q):
    new_data1 = []
    for i in range(len(data[data_dict[q]])):
        dataTT = []
        for j in data[data_dict[q]][i]:
            j = np.array(j).T
            dataT = []
            for k in range(len(j)):
                Magn_value = np.absolute(j[k])
                Phase_value = np.angle(j[k])
                dataT.extend(Phase_value)
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

f_data = get_new_data(f_data, 6)

new_target = np.reshape(new_target, (1000, 256))

data_all = np.concatenate((new_data_ae, new_data_ai, new_data_au, new_data_ie, new_data_iu, p), axis=2)

data_all = np.reshape(data_all, (1000, 209))

data_train = data_all[:800]
data_test = data_all[800:]
train_label = new_target[:800]
test_label = new_target[800:]

# data_train, data_test, train_label, test_label = train_test_split(data_all, new_target, train_size=0.8,
#                                                                   random_state=22)

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

data_train = np.reshape(data_train, (len(data_train), 11, 19))
data_test = np.reshape(data_test, (len(data_test), 11, 19))


def cnn(l1=1024, l2=2048, gamma=1e-2, lr=1e-4, w1=2, w2=1):
    model = models.Sequential()
    model.add(layers.Conv1D(209, w1, activation='relu', kernel_regularizer=regularizers.l2(gamma),
                            input_shape=(11, 19), padding='same'))
    model.add(layers.MaxPooling1D(w2))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(512, activation='tanh', kernel_regularizer=regularizers.l2(gamma)))
    model.add(layers.Dense(512, activation='tanh', kernel_regularizer=regularizers.l2(gamma)))
    model.add(layers.Dense(256, activation='tanh', kernel_regularizer=regularizers.l2(gamma)))
    adam = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam, loss='mse')

    return model


model = cnn()
history = model.fit(data_train, train_label, verbose=1, batch_size=32, epochs=epochs,
                    validation_data=(data_test, test_label))  # 用history接收
φ_list = model.predict(data_test)

def phi_to_complex(phi_list):
    phi_list = np.reshape(phi_list, (len(phi_list), 16, 16))

    mask = np.zeros([len(phi_list), 16, 16])
    for i in range(len(phi_list)):
        for j in range(16):
            mask[i][j][j] = 1
        phi_list = np.cos(phi_list)+1j * np.sin(phi_list)
    phi_list = phi_list * mask
    return phi_list

φ_list = phi_to_complex(φ_list)


rate_list = []
new_data_ae = get_new_data2(data_ae, 0)
new_data_ai = get_new_data2(data_ai, 1)
new_data_au = get_new_data2(data_au, 2)
new_data_ie = get_new_data2(data_ie, 3)
new_data_iu = get_new_data2(data_iu, 4)
for i in range(len(data_test)):
    k = test_index_list[i]

    φ = np.reshape(φ_list[i], (16, 16))
    T1 = np.dot(new_data_iu[k], φ)
    T1 = np.dot(T1, np.reshape(new_data_ai[k], (16, 4)))
    T1 = T1 + new_data_au[k]
    T1 = np.dot(T1, f_data[k].T)
    r1 = (abs(T1)) ** 2 + 1

    T2 = np.dot(new_data_ie[k], φ)
    T2 = np.dot(T2, np.reshape(new_data_ai[k], (16, 4)))
    T2 = T2 + new_data_ae[k]
    T2 = np.dot(T2, f_data[k].T)
    r2 = (abs(T2)) ** 2 + 1

    rate = r1 / r2
    rate_list.append(float(rate))
print(φ_list)
print(rate_list)
history_dict = history.history
train_loss = history_dict["loss"]
val_loss = history_dict["val_loss"]

train_loss_4 = (16*np.array(train_loss)).tolist()
val_loss_4 = (16*np.array(val_loss)).tolist()


sio.savemat('φ_list_cnn_1000.mat', {'rate': np.array(φ_list) })
# 绘制损失值
plt.figure()
plt.plot(range(epochs), train_loss_4, label='train_loss')
plt.plot(range(epochs), val_loss_4, label='test_loss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

