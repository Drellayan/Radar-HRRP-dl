from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import scipy.io as scio
import numpy as np

n_input = 256 # 输入的维度
n_classes = 3 # 标签的维度
n_samples = 200 # 样本数量

log_dir = './simple_cnn_log'#可视化数据保存地址

# 导入HRRP数据
file_name = './Train_hrrp.mat'
traindata_base =scio.loadmat(file_name)['aa']


#hrrp = traindata_base[:,3:]
#labels = traindata_base[:,:3]

inver_train = np.copy(traindata_base)
for i in range(256):
    inver_train[:,i+3] = traindata_base[:,258-i]

traindata_ex = np.concatenate([traindata_base, inver_train],axis=0)

hrrp = traindata_ex[:,3:]
labels = traindata_ex[:,:3]

# read data for TEST
file_name2 = './Test_hrrp.mat'
testdata_base = scio.loadmat(file_name2)['bb']
test_hrrp = testdata_base[:,3:3+n_input]
test_labels =testdata_base[:,0:3]

import numpy as np
import os
import scipy.io.wavfile

import matplotlib.pyplot as plt

from kymatio.numpy import Scattering1D
from kymatio.datasets import fetch_fsdd

T = hrrp.shape[-1]
J = 5
Q = 8
scattering = Scattering1D(J, T, Q)
Sx = scattering(hrrp)
hrrp = Sx.mean(axis=-1)

T = test_hrrp.shape[-1]
scattering = Scattering1D(J, T, Q)
Sxt = scattering(test_hrrp)
test_hrrp = Sxt.mean(axis=-1)

# grant train class labels from one-hot index to [0; 1; 2]
la = [i for x in labels for i in range(n_classes) if x[i] == 1]
labels_vector = np.array(la, dtype = np.float64)

# initialise ovr-svm with standard preprocessing
clf = make_pipeline(StandardScaler(), SVC(decision_function_shape='ovr'))
# train svm model
clf.fit(hrrp, labels_vector)

pla = []
err = 0
for j in range(200):
    dec = clf.decision_function([hrrp[j]])
    i = np.argmax(dec)
    pla.append(i)
    if i != la[j]:
        err = err + 1
print(f"Accuracy on train: {1 - (err / 200)}")



# grant test class labels from one-hot index to [0; 1; 2]
tla = [i for x in test_labels for i in range(n_classes) if x[i] == 1]
test_labels_vector = np.array(tla, dtype = np.float64)

pla_ = []
err_ = 0
ind = 0
for x, y in zip(test_hrrp, tla):
    dec = clf.decision_function([x])
    i = np.argmax(dec)
    pla_.append(i)
    if i != y:
        print(f"index: {ind}, real: {y}, machine: {i}")
        err_ = err_ + 1
    ind = ind + 1
print(f"Accuracy on test: {1 - (err_ / n_samples)}")