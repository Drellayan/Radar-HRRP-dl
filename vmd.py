
import scipy.io as scio
import numpy as np

n_input = 256 # 输入的维度
n_categories = 3 # 标签的维度
n_samples = 400 # 样本数量
file_name = './HRRP_data/Train_hrrp.mat'
traindata_base =scio.loadmat(file_name)['aa']

inver_train = np.copy(traindata_base)
for i in range(256):
    inver_train[:,i+3] = traindata_base[:,258-i]

traindata_ex = np.concatenate([traindata_base, inver_train],axis=0)

hrrp = traindata_ex[:,3:]
labels = traindata_ex[:,:3]
#hrrp = traindata_base[:,3:]
#labels = traindata_base[:,:3]

# grant train class labels from one-hot index to [0; 1; 2]
la = [i for x in labels for i in range(n_categories) if x[i] == 1]
labels_vector = np.array(la, dtype = np.int64)

# read data for TEST
file_name2 = './HRRP_data/Test_hrrp.mat'
#file_name2 = './Test_hrrp.mat'
testdata_base = scio.loadmat(file_name2)['bb']
test_hrrp = testdata_base[:,3:3+n_input]
test_labels =testdata_base[:,0:3]
# grant test class labels from one-hot index to [0; 1; 2]
tla = [i for x in test_labels for i in range(n_categories) if x[i] == 1]
test_labels_vector = np.array(tla, dtype = np.int64)

import numpy as np
from scipy.signal import hilbert


class VMD:
    def __init__(self, K, alpha, tau, tol=1e-7, maxIters=200, eps=1e-9):
        """
        :param K: 模态数
        :param alpha: 每个模态初始中心约束强度
        :param tau: 对偶项的梯度下降学习率
        :param tol: 终止阈值
        :param maxIters: 最大迭代次数
        :param eps: eps
        """
        self.K =K
        self.alpha = alpha
        self.tau = tau
        self.tol = tol
        self.maxIters = maxIters
        self.eps = eps

    def __call__(self, f):
        T = f.shape[0]
        t = np.linspace(1, T, T) / T
        omega = t - 1. / T
        # 转换为解析信号
        f = hilbert(f)
        f_hat = np.fft.fft(f)
        u_hat = np.zeros((self.K, T), dtype=np.complex_)
        omega_K = np.zeros((self.K,))
        lambda_hat = np.zeros((T,), dtype=np.complex_)
        # 用以判断
        u_hat_pre = np.zeros((self.K, T), dtype=np.complex_)
        u_D = self.tol + self.eps

        # 迭代
        n = 0
        while n < self.maxIters and u_D > self.tol:
            for k in range(self.K):
                # u_hat
                sum_u_hat = np.sum(u_hat, axis=0) - u_hat[k, :]
                res = f_hat - sum_u_hat
                u_hat[k, :] = (res + lambda_hat / 2) / (1 + self.alpha * (omega - omega_K[k]) ** 2)

                # omega
                u_hat_k_2 = np.abs(u_hat[k, :]) ** 2
                omega_K[k] = np.sum(omega * u_hat_k_2) / np.sum(u_hat_k_2)

            # lambda_hat
            sum_u_hat = np.sum(u_hat, axis=0)
            res = f_hat - sum_u_hat
            lambda_hat -= self.tau * res

            n += 1
            u_D = np.sum(np.abs(u_hat - u_hat_pre) ** 2)
            u_hat_pre[::] = u_hat[::]

        # 重构，反傅立叶之后取实部
        u = np.real(np.fft.ifft(u_hat, axis=-1))

        omega_K = omega_K * T
        idx = np.argsort(omega_K)
        omega_K = omega_K[idx]
        u = u[idx, :]
        return u, omega_K

K = 6
alpha = 2000
tau = 1e-6
vmd = VMD(K, alpha, tau)
hrrp_vmd = np.zeros((n_samples,K,n_input))
for j in range(n_samples):
    u, _ = vmd(hrrp[j])
    hrrp_vmd[j] = u

test_vmd = np.zeros((200,K,n_input))
for j in range(200):
    u, _ = vmd(test_hrrp[j])
    test_vmd[j] = u

np.savez('./HRRP_data/vmd_train.npz', hrrp=hrrp_vmd, labels=labels_vector)
np.savez('./HRRP_data/vmd_test.npz', hrrp=test_vmd, labels=test_labels_vector)

print("Done!")