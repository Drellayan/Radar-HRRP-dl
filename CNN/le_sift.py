import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as scio
import numpy as np
import cv2


def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()。"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=2)
        #self.conv3 = nn.Conv2d(16, 16, 3, padding=1)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 16 * 18, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), 4)
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 8)
        #x = F.max_pool2d(F.relu(self.conv3(x)), 4)

        #print(x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


lenet = Net()
lenet = lenet.to(device=try_gpu())
print(lenet)

learning_rate = 5e-3
batch_size = 10
epochs = 100

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(lenet.parameters(), lr=learning_rate)





n_input = 256 # 输入的维度
n_categories = 3 # 标签的维度
n_samples = 400 # 样本数量
file_name = '../HRRP_data/sift/label.mat'
hrrpdata_base = scio.loadmat(file_name)
labels_vector = hrrpdata_base['Trf']
test_labels =hrrpdata_base['Tef']
labels = torch.from_numpy(labels_vector)
labels = labels.cuda()
test_labels_vector = torch.from_numpy(test_labels)
test_labels_vector = test_labels_vector.cuda()
"""
# read data for TRAIN
#file_name = './Train_hrrp.mat'
file_name = '../HRRP_data/sift/train_label.mat'
traindata_base = np.load(file_name)
#hrrp = traindata_base['Trf']
labels_vector = traindata_base['Trf']

#hrrp = torch.from_numpy(hrrp)
#hrrp = torch.Tensor(hrrp)
#hrrp = hrrp.cuda()
#hrrp = torch.reshape(hrrp, (n_samples,1,-1))
labels = torch.from_numpy(labels_vector)
labels = labels.cuda()
#labels = torch.reshape(labels, (n_samples,1,-1))

# read data for TEST
file_name2 = '../HRRP_data/sift/test_label.mat'
testdata_base = np.load(file_name2)
test_hrrp = testdata_base['hrrp']
test_labels =testdata_base['labels']

test_hrrp = torch.Tensor(test_hrrp)
test_hrrp = test_hrrp.cuda()
test_labels_vector = torch.from_numpy(test_labels)
test_labels_vector = test_labels_vector.cuda()
"""


from torch.utils.data import Dataset, DataLoader, TensorDataset

class SiftDataset(Dataset):
    def __init__(self, path, labels):
        ds = []

        for i in range(200):
            img = cv2.imread(f"../HRRP_data/sift/sift_{path}/{i + 1}_gai.jpg")
            #img = cv2.imread(f"../HRRP_data/sift/{path}/{i + 1}.jpg")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img.transpose((2, 0, 1)))
            #print(img.dtype)
            img = img.float().div(255)
            #print(img.dtype)

            ds.append(img.cuda())
        #print(ds[0].dtype)
        self.image = ds
        self.label = labels.long().reshape(-1)
    def __len__(self):
        return len(self.label)
    def __getitem__(self, idx):
        img = self.image[idx]
        label = self.label[idx]
        #print(label.shape)
        sample = {'image': img, 'label': label}
        return sample

train_dataset = SiftDataset("train", labels)
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)

test_dataset = SiftDataset("test", test_labels_vector)
test_dataloader = DataLoader(test_dataset, batch_size)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, batch_data in enumerate(dataloader):
        # Compute prediction and loss
        #X = X.reshape(10,1,-1)

        X = batch_data['image']
        y = batch_data['label']
        #print(y)
        #print(y.dtype)
        #print(X.shape)
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            #print(pred.device)
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def tst_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for batch_data in dataloader:
            #X = X.reshape(10,1,-1)
            X = batch_data['image']
            y = batch_data['label']
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, lenet, loss_fn, optimizer)
    tst_loop(test_dataloader, lenet, loss_fn)

print("Done!")

#torch.save(lenet.state_dict(), 'le6.params')

#print(*[(name, param.device) for name, param in lenet.named_parameters()])