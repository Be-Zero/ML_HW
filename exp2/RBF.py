# coding=utf-8
import numpy as np
import scipy.io as scio
from torch import nn, optim
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_data(path):
    data = scio.loadmat(path)
    data_x = torch.from_numpy(np.array(data['X']).astype(np.float32))
    data_y = torch.from_numpy(np.array(data['Y'][:, 0] - 1))
    data = TensorDataset(data_x, data_y)
    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size
    train_data, test_data = random_split(data, [train_size, test_size], generator=torch.manual_seed(0))
    return data_x, train_data, test_data


class RBFN(nn.Module):
    """
    以高斯核作为径向基函数
    """
    def __init__(self, centers, n_out=3):
        """
        :param centers: shape=[center_num,data_dim]
        :param n_out: num_classes
        """
        super(RBFN, self).__init__()
        self.n_out = n_out
        self.num_centers = centers.size(0)  # 隐层节点的个数
        self.dim_centure = centers.size(1)
        self.centers = nn.Parameter(centers)
        self.beta = (torch.ones(1, self.num_centers) * 10).to(device)
        # 对线性层的输入节点数目进行了修改
        self.linear = nn.Linear(self.num_centers + self.dim_centure, self.n_out, bias=True)
        self.initialize_weights()  # 创建对象时自动执行

    def kernel_fun(self, batches):
        n_input = batches.size(0)  # number of inputs
        A = self.centers.view(self.num_centers, -1).repeat(n_input, 1, 1)
        B = batches.view(n_input, -1).unsqueeze(1).repeat(1, self.num_centers, 1)
        C = torch.exp(-self.beta.mul((A - B).pow(2).sum(2, keepdim=False)))
        return C

    def forward(self, batches):
        radial_val = self.kernel_fun(batches)
        class_score = self.linear(torch.cat([batches, radial_val], dim=1))
        return class_score

    def initialize_weights(self, ):
        """
        网络权重初始化
        :return:
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()

    def print_network(self):
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(self)
        print('Total number of parameters: %d' % num_params)


def train(train_loader, rbf):
    optimizer = optim.SGD(rbf.parameters(), lr=0.000088, momentum=0.88)
    closs = nn.CrossEntropyLoss()
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        pred = rbf(inputs)
        loss = closs(pred, labels)
        loss.backward()
        optimizer.step()


def test(test_loader, rbf):
    correct = 0
    total = 0
    for i, data in enumerate(test_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        pred = rbf(inputs)
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
    num_classes = 10
    epoch = 20  # 0.5 0.3
    data_x, train_data, test_data = load_data(mnist_path)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    rbf = RBFN(data_x, num_classes).to(device)
    best = 0
    for i in range(epoch):
        print('epoch:', i, end='')
        train(train_loader, rbf)
        best = max(test(test_loader, rbf), best)
    print('最高准确率：{}%'.format(best))
    torch.cuda.empty_cache()

    print('lung: ')
    batch_size = 10
    input_size = 3312
    num_classes = 5
    epoch = 20  # 0.5 0.3
    data_x, train_data, test_data = load_data(lung_path)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    rbf = RBFN(data_x, num_classes).to(device)
    best = 0
    for i in range(epoch):
        print('epoch:', i, end='')
        train(train_loader, rbf)
        best = max(test(test_loader, rbf), best)
    print('最高准确率：{}%'.format(best))
    torch.cuda.empty_cache()

    print('yale: ')
    batch_size = 10
    input_size = 1024
    num_classes = 15
    epoch = 20  # 0.5 0.3
    data_x, train_data, test_data = load_data(yale_path)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    rbf = RBFN(data_x, num_classes).to(device)
    best = 0
    for i in range(epoch):
        print('epoch:', i, end='')
        train(train_loader, rbf)
        best = max(test(test_loader, rbf), best)
    print('最高准确率：{}%'.format(best))
    torch.cuda.empty_cache()
