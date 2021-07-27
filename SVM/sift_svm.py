from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import scipy.io as scio
import numpy as np

n_input = 256 # 输入的维度
n_classes = 3 # 标签的维度
n_samples = 200 # 检测样本数量

hrrp = np.load('../HRRP_data/sift/hrrp_sift.npz')['hrrp']
labels =scio.loadmat('../HRRP_data/sift/train_label.mat')['Trf']
labels_vector = np.array(labels, dtype = np.float64).reshape(-1)

# initialise ovr-svm with standard preprocessing
clf = make_pipeline(StandardScaler(), SVC(decision_function_shape='ovr'))
# train svm model
clf.fit(hrrp, labels_vector)

test_hrrp = np.load('../HRRP_data/sift/test_sift.npz')['hrrp']
test_labels =scio.loadmat('../HRRP_data/sift/test_label.mat')['Tef']
test_labels_vector = np.array(test_labels, dtype = np.float64).reshape(-1)


pla = []
err = 0
for j in range(200):
    dec = clf.decision_function([hrrp[j]])
    i = np.argmax(dec)
    pla.append(i)
    if i != labels_vector[j]:
        err = err + 1
print(f"Accuracy on train: {1 - (err / 200)}")

pla_ = []
err_ = 0
ind = 0
for x, y in zip(test_hrrp, test_labels_vector):
    dec = clf.decision_function([x])
    i = np.argmax(dec)
    pla_.append(i)
    if i != y:
        print(f"index: {ind}, real: {y}, machine: {i}")
        err_ = err_ + 1
    ind = ind + 1
print(f"Accuracy on test: {1 - (err_ / 200)}")