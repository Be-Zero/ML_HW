# coding=utf-8
import numpy as np
import scipy.io as scio
from torch import nn, optim
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_data(path):
    data = scio.loadmat(path)
    data_x = torch.from_numpy(np.array(data['X']))
    data_y = torch.from_numpy(np.array(data['Y'][:, 0] - 1))
    data = TensorDataset(data_x, data_y)
    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size
    train_data, test_data = random_split(data, [train_size, test_size], generator=torch.manual_seed(0))
    return train_data, test_data


class MyNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.to(torch.float32)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def train(train_loader, model):
    optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.8)
    closs = nn.CrossEntropyLoss()
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        pred = model(inputs)
        loss = closs(pred, labels)
        loss.backward()
        optimizer.step()


def test(test_loader, model):
    correct = 0
    total = 0
    for i, data in enumerate(test_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        pred = model(inputs)
        _, predicted = torch.max(pred.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print(' 准确率：{}%'.format(correct * 100 / total))
    return correct * 100 / total


if __name__ == '__main__':
    mnist_path = 'datasets/MNIST.mat'
    lung_path = 'datasets/lung.mat'
    yale_path = 'datasets/Yale.mat'

    print('mnist: ')
    batch_size = 50
    input_size = 784
    hidden_size = 500
    num_classes = 10
    epoch = 30  # 0.5 0.8
    train_data, test_data = load_data(mnist_path)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    model = MyNet(input_size, hidden_size, num_classes).to(device)
    best = 0
    for i in range(epoch):
        print('epoch:', i, end='')
        train(train_loader, model)
        best = max(test(test_loader, model), best)
    print('最高准确率：{}%'.format(best))
    torch.cuda.empty_cache()

    print('lung: ')
    batch_size = 20
    input_size = 3312
    hidden_size = 500
    num_classes = 5
    epoch = 100  # 0.0005 0.9
    train_data, test_data = load_data(lung_path)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    model = MyNet(input_size, hidden_size, num_classes).to(device)
    best = 0
    for i in range(epoch):
        print('epoch:', i, end='')
        train(train_loader, model)
        best = max(test(test_loader, model), best)
    print('最高准确率：{}%'.format(best))
    torch.cuda.empty_cache()

    print('yale: ')
    batch_size = 10
    input_size = 1024
    hidden_size = 300
    num_classes = 15
    epoch = 50  # 0.000082 0.86
    train_data, test_data = load_data(yale_path)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    model = MyNet(input_size, hidden_size, num_classes).to(device)
    best = 0
    for i in range(epoch):
        print('epoch:', i, end='')
        train(train_loader, model)
        best = max(test(test_loader, model), best)
    print('最高准确率：{}%'.format(best))
    torch.cuda.empty_cache()