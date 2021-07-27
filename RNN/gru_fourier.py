import torch
from torch import nn
from d2l import torch as d2l
import math

import scipy.io as scio
import numpy as np

n_input = 256 # 输入的维度
n_categories = 3 # 标签的维度
n_samples = 400 # 样本数量

# read data for TRAIN
#file_name = './Train_hrrp.mat'
file_name = '../HRRP_data/Train_fourier.mat'
traindata_base =scio.loadmat(file_name)['Trf']

inver_train = np.copy(traindata_base)
for i in range(256):
    inver_train[:,i+3] = traindata_base[:,258-i]

traindata_ex = np.concatenate([traindata_base, inver_train],axis=0)

hrrp = traindata_ex[:,3:]
print(hrrp.shape)
labels = traindata_ex[:,:3]
#hrrp = traindata_base[:,3:]
#labels = traindata_base[:,:3]

# grant train class labels from one-hot index to [0; 1; 2]
la = [i for x in labels for i in range(n_categories) if x[i] == 1]
labels_vector = np.array(la, dtype = np.int64)

#hrrp = torch.from_numpy(hrrp)
hrrp = torch.Tensor(hrrp)
#hrrp = torch.reshape(hrrp, (n_samples,1,-1))
labels = torch.from_numpy(labels_vector)
#labels = torch.reshape(labels, (n_samples,1,-1))

class SeqDataLoader:
    """加载序列数据的迭代器。"""

    def __init__(self, batch_size, num_steps, use_random_iter, corpus):
        # self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
        self.corpus = corpus
        self.batch_size, self.num_steps = batch_size, num_steps

    def data_iter_fn(self):
        Xs = self.corpus[0]
        Ys = self.corpus[1]
        ns = self.num_steps
        for i in range(0, Xs.size()[0], ns):
            X = Xs[i:(i + ns), :]
            Y = Ys[i:(i + ns)]
            yield X, Y

    def __iter__(self):
        return self.data_iter_fn()


def load_data_time_machine(batch_size, num_steps, corpus,
                           use_random_iter=False):
    """返回时光机器数据集的迭代器和词汇表。"""
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter,
                              corpus)
    return data_iter,

def get_params(vocab_size, num_hiddens, device):
    num_inputs = n_input
    num_outputs = n_categories

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    def three():
        return (normal(
            (num_inputs, num_hiddens)), normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    W_xz, W_hz, b_z = three()  # 更新门参数
    W_xr, W_hr, b_r = three()  # 重置门参数
    W_xh, W_hh, b_h = three()  # 候选隐藏状态参数
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),)

def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = H @ W_hq + b_q
        #Y = nn.Softmax(Y,dim=1)
        outputs.append(Y)
    #outputs = nn.Softmax(outputs,dim=1)
    return torch.cat(outputs, dim=0), (H,)

def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练模型一个迭代周期（定义见第8章）。"""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # 训练损失之和, 标记数量
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化`state`
            state = net.begin_state(batch_size=batch_size, device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # `state`对于`nn.GRU`是个张量
                state.detach_()
            else:
                # `state`对于`nn.LSTM`或对于我们从零开始实现的模型是个张量
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        # y = Y
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)

        m = nn.Softmax(dim=1)
        y_hat_ = m(y_hat)
        #print(y_hat_)
        # print(y)
        # print(X.size())
        # print(y.long().size())
        l = loss(y_hat, y.long()).mean()
        #print(l)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # 因为已经调用了`mean`函数
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

def train_ch8(net, train_iter, lr, num_epochs, device,
              use_random_iter=False):
    """训练模型（定义见第8章）。"""
    loss = nn.CrossEntropyLoss()

    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    # predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter)
    #ppl, speed = train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter)


class RNNModelScratch:

    def __init__(self, vocab_size, num_hiddens, device, get_params,
                 init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)

def grad_clipping(net, theta):
    """裁剪梯度。"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


batch_size, num_steps = 1, 10
corpus=[hrrp, labels]
train_iter, = load_data_time_machine(batch_size, num_steps, corpus)

vocab_size, num_hiddens, device = n_input, 128, d2l.try_gpu()
num_epochs, lr = 500, 0.1
model = RNNModelScratch(vocab_size, num_hiddens, device, get_params,
                            init_gru_state, gru)
train_ch8(model, train_iter, lr, num_epochs, device)






# read data for TEST
file_name2 = '../HRRP_data/Test_fourier.mat'
testdata_base = scio.loadmat(file_name2)['Tef']
test_hrrp = testdata_base[:,3:3+n_input]
test_labels =testdata_base[:,0:3]
# grant test class labels from one-hot index to [0; 1; 2]
tla = [i for x in test_labels for i in range(n_categories) if x[i] == 1]
test_labels_vector = np.array(tla, dtype = np.int64)

test_hrrp = torch.Tensor(test_hrrp)
test_labels_vector = torch.from_numpy(test_labels_vector)


corpus_test = [test_hrrp, test_labels_vector]
test_iter, = load_data_time_machine(batch_size, num_steps, corpus_test)
loss = nn.CrossEntropyLoss()

def predict(net, test_iter, loss, device, use_random_iter=False):
    #net.eval()
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # 训练损失之和, 标记数量
    count = 0
    for X, Y in test_iter:
        state = net.begin_state(batch_size=batch_size, device=device)
        """
                if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化`state`
            state = net.begin_state(batch_size=batch_size, device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # `state`对于`nn.GRU`是个张量
                state.detach_()
            else:
                # `state`对于`nn.LSTM`或对于我们从零开始实现的模型是个张量
                for s in state:
                    s.detach_()
        """

        y = Y.T.reshape(-1)
        # y = Y
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)

        m = nn.Softmax(dim=1)
        y_hat_ = m(y_hat)
        pre_y = y_hat_.argmax(axis=1)
        #print("predict:")
        #print(pre_y)
        #print("actual:")
        #print(y)
        c = pre_y.eq(y)
        count = count + c.sum()

        # print(X.size())
        # print(y.long().size())
        l = loss(y_hat, y.long()).mean()
        print(l)
    return count

total_correct = predict(model, test_iter, loss, device)

print(f"predict rate: {total_correct/200}")