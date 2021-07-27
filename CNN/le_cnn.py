import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv1d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv1d(6, 16, 5, padding=2)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 16, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool1d(F.relu(self.conv1(x)), 4)
        # If the size is a square, you can specify with a single number
        x = F.max_pool1d(F.relu(self.conv2(x)), 4)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


lenet = Net()
print(lenet)

learning_rate = 5e-3
batch_size = 10
epochs = 100

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(lenet.parameters(), lr=learning_rate)

import torch
from torch import nn

import scipy.io as scio
import numpy as np

n_input = 256 # 输入的维度
n_categories = 3 # 标签的维度
n_samples = 400 # 样本数量

# read data for TRAIN
#file_name = './Train_hrrp.mat'
file_name = '../HRRP_data/Train_hrrp.mat'
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

#hrrp = torch.from_numpy(hrrp)
hrrp = torch.Tensor(hrrp)
#hrrp = torch.reshape(hrrp, (n_samples,1,-1))
labels = torch.from_numpy(labels_vector)
#labels = torch.reshape(labels, (n_samples,1,-1))

# read data for TEST
file_name2 = '../HRRP_data/Test_hrrp.mat'
#file_name2 = './Test_hrrp.mat'
testdata_base = scio.loadmat(file_name2)['bb']
test_hrrp = testdata_base[:,3:3+n_input]
test_labels =testdata_base[:,0:3]
# grant test class labels from one-hot index to [0; 1; 2]
tla = [i for x in test_labels for i in range(n_categories) if x[i] == 1]
test_labels_vector = np.array(tla, dtype = np.int64)

test_hrrp = torch.Tensor(test_hrrp)
test_labels_vector = torch.from_numpy(test_labels_vector)

from torch.utils.data import Dataset, DataLoader, TensorDataset
train_dataset = TensorDataset(hrrp, labels)
train_dataloader = DataLoader(train_dataset, batch_size)

test_dataset = TensorDataset(test_hrrp, test_labels_vector)
test_dataloader = DataLoader(test_dataset, batch_size)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.reshape(10,1,-1)
        #print(X.shape)
        #print(y.shape)
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.reshape(10,1,-1)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, lenet, loss_fn, optimizer)
    test_loop(test_dataloader, lenet, loss_fn)

print("Done!")