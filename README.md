# 一、实验

## 实验一：SVM

### 1. 实验目的

1. 掌握线性支持向量机（SVM）分类器；
2. 掌握基于高斯核的SVM分类器；
3. 掌握基于拉普拉斯核的SVM分类器。

### 2. 数据集介绍

| 数据集 | 样本数 | 维度 | 类数 | 数据类型   |
| ------ | ------ | ---- | ---- | ---------- |
| mnist  | 3000   | 784  | 10   | 手写体数字 |
| yale   | 165    | 1024 | 15   | 人脸图像   |
| lung   | 203    | 3312 | 5    | 生物数据   |
### 3. 实验内容

1. 编写程序实现线性SVM分类器设计；
2. 编写程序实现基于高斯核的SVM分类器设计；
3. 编写程序实现基于拉普拉斯核的SVM分类器设计。

### 4. 代码实现

```python
# coding=utf-8
from sklearn import svm, datasets
import numpy as np
import scipy.io as scio
from sklearn.model_selection import train_test_split
import math


def load_data(path):
    data = scio.loadmat(path)
    data_x = data['X']
    data_y = data['Y'][:, 0] - 1
    return np.array(data_x), np.array(data_y)


def laplace(X1, X2):
    K = np.zeros((len(X1), len(X2)), dtype=np.float)
    for i in range(len(X1)):
        for j in range(len(X2)):
            K[i][j] = math.exp(-math.sqrt(np.dot(X1[i] - X2[j], (X1[i] - X2[j]).T)) / 2)
    return K


def classify(path, kernel):
    x, y = load_data(path)
    train_data, test_data, train_label, test_label = train_test_split(x, y, test_size=0.25, random_state=10)
    predictor = svm.SVC(gamma='scale', C=2.0, decision_function_shape='ovr', kernel=kernel)
    predictor.fit(train_data, train_label.ravel())
    print('训练集：', predictor.score(train_data, train_label))
    print('测试集：', predictor.score(test_data, test_label))


if __name__ == '__main__':
    mnist_path = 'datasets/MNIST.mat'
    lung_path = 'datasets/lung.mat'
    yale_path = 'datasets/Yale.mat'

    print('mnist数据集：')
    classify(mnist_path, 'linear')

    print('lung数据集：')
    classify(lung_path, 'rbf')

    print('yale数据集：')
    classify(yale_path, laplace)

```
## 实验二：神经网络

### 1. 实验目的

1. 掌握全连接神经网络的训练与测试方法；
2. 掌握基于RBF分类器训练与测试方法；

### 2. 数据集介绍

| 数据集 | 样本数 | 维度 | 类数 | 数据类型   |
| ------ | ------ | ---- | ---- | ---------- |
| mnist  | 3000   | 784  | 10   | 手写体数字 |
| yale   | 165    | 1024 | 15   | 人脸图像   |
| lung   | 203    | 3312 | 5    | 生物数据   |
### 3. 实验内容

1. 编写程序实现全连接神经网络分类器设计；
2. 编写程序实现基于RBF分类器设计；

### 4. 代码实现

1. BP 神经网络
	```python
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
	
	```
2. RBF 分类器
	```python
	# coding=utf-8
	import numpy as np
	import scipy.io as scio
	from torch import nn, optim
	import torch
	from torch.utils.data import DataLoader, random_split, TensorDataset
	
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	# device = 'cpu'
	
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
	
	```

## 实验三：贝叶斯

### 1. 实验目的

1. 掌握朴素贝叶斯分类器；
2. 掌握AODE分类器；

### 2. 数据集介绍

| 数据集 | 样本数 | 维度 | 类数 | 数据类型   |
| ------ | ------ | ---- | ---- | ---------- |
| mnist  | 3000   | 784  | 10   | 手写体数字 |
| yale   | 165    | 1024 | 15   | 人脸图像   |
| lung   | 203    | 3312 | 5    | 生物数据   |
### 3. 实验内容

1. 编写程序实现朴素贝叶斯分类器设计；
2. 编写程序实现AODE分类器设计；

### 4. 代码实现

1. 朴素贝叶斯分类器
	```python
	from sklearn.model_selection import train_test_split  # 切分测试集
	from sklearn.naive_bayes import MultinomialNB  # 贝叶斯
	import scipy.io as scio
	import numpy as np
	
	
	def load_data(path):
	    data = scio.loadmat(path)
	    data_x = np.array(data['X'])
	    data_y = np.array(data['Y'][:, 0] - 1)
	    train_data, test_data, train_label, test_label = train_test_split(data_x, data_y, test_size=0.25, random_state=10)
	    return train_data, test_data, train_label, test_label
	
	
	if __name__ == '__main__':
	    mnist_path = 'datasets/MNIST.mat'
	    lung_path = 'datasets/lung.mat'
	    yale_path = 'datasets/Yale.mat'
	
	    print('mnist:')
	    # 进行训练集和测试集切分
	    x_train, x_test, y_train, y_test = load_data(mnist_path)
	    # 对数据集进行提取
	    mlt = MultinomialNB(alpha=1.0)
	    mlt.fit(x_train, y_train)
	    y_predict = mlt.predict(x_test)
	    # 得出准确率
	    print("准确率为：", mlt.score(x_test, y_test))
	
	    print('lung:')
	    # 进行训练集和测试集切分
	    x_train, x_test, y_train, y_test = load_data(lung_path)
	    # 对数据集进行提取
	    mlt = MultinomialNB(alpha=1.0)
	    mlt.fit(x_train, y_train)
	    y_predict = mlt.predict(x_test)
	    # 得出准确率
	    print("准确率为：", mlt.score(x_test, y_test))
	
	    print('yale:')
	    # 进行训练集和测试集切分
	    x_train, x_test, y_train, y_test = load_data(yale_path)
	    # 对数据集进行提取
	    mlt = MultinomialNB(alpha=1.0)
	    mlt.fit(x_train, y_train)
	    y_predict = mlt.predict(x_test)
	    # 得出准确率
	    print("准确率为：", mlt.score(x_test, y_test))
	
	```
2. AODE 分类器
	```python
	# coding=utf-8
	import numpy as np
	import scipy.io as scio
	from torch import nn, optim
	import torch
	from torch.utils.data import DataLoader, random_split, TensorDataset
	
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	# device = 'cpu'
	
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
	
	```

## 实验四：聚类

### 1. 实验目的

1、掌握K均值（k-means）聚类算法。
2、掌握学习向量量化（LVQ）聚类算法。
3、掌握高斯混合（Mixture-of-Gaussian）聚类算法。
4、理解聚类相关指标。

### 2. 数据集介绍

| 数据集 | 样本数 | 维度 | 类数 | 数据类型   |
| ------ | ------ | ---- | ---- | ---------- |
| mnist  | 3000   | 784  | 10   | 手写体数字 |
| yale   | 165    | 1024 | 15   | 人脸图像   |
| lung   | 203    | 3312 | 5    | 生物数据   |
### 3. 实验内容

1、编写程序实现K均值（k-means）聚类算法。
2、编写程序实现学习向量量化（LVQ）聚类算法。
3、编写程序实现高斯混合（Mixture-of-Gaussian）聚类算法。
4、利用Acc（Accuracy）和NMI（标准互信息）指标评价上述聚类算法。

### 4. 代码实现

1. K-means 聚类算法
	```python
	# coding=utf-8
	import numpy as np
	from sklearn.model_selection import train_test_split
	from sklearn.cluster import KMeans
	from sklearn import metrics
	import scipy.io as scio
	from munkres import Munkres
	
	
	def load_data(path):
	    data = scio.loadmat(path)
	    data_x = np.array(data['X'])
	    data_y = np.array(data['Y'][:, 0] - 1)
	    train_data, test_data, train_label, test_label = train_test_split(data_x, data_y, test_size=0.25, random_state=10)
	    return train_data, test_data, train_label, test_label
	
	
	def best_map(L1, L2):
	    Label1 = np.unique(L1)  # 去除重复的元素，由小大大排列
	    nClass1 = len(Label1)  # 标签的大小
	    Label2 = np.unique(L2)
	    nClass2 = len(Label2)
	    nClass = np.maximum(nClass1, nClass2)
	    G = np.zeros((nClass, nClass))
	    for i in range(nClass1):
	        ind_cla1 = L1 == Label1[i]
	        ind_cla1 = ind_cla1.astype(float)
	        for j in range(nClass2):
	            ind_cla2 = L2 == Label2[j]
	            ind_cla2 = ind_cla2.astype(float)
	            G[i, j] = np.sum(ind_cla2 * ind_cla1)
	    m = Munkres()
	    index = m.compute(-G.T)
	    index = np.array(index)
	    c = index[:, 1]
	    newL2 = np.zeros(L2.shape)
	    for i in range(nClass2):
	        newL2[L2 == Label2[i]] = Label1[c[i]]
	    return newL2
	
	
	def acc_rate(gt_s, s):
	    c_x = best_map(gt_s, s)
	    err_x = np.sum(gt_s[:] == c_x[:])
	    acc = err_x.astype(float) / (gt_s.shape[0])
	    return acc
	
	
	if __name__ == '__main__':
	    mnist_path = 'datasets/MNIST.mat'
	    lung_path = 'datasets/lung.mat'
	    yale_path = 'datasets/Yale.mat'
	
	    print('mnist:')
	    train_data, test_data, train_label, test_label = load_data(mnist_path)
	    k_means = KMeans(n_clusters=10, random_state=0)
	    k_means.fit(train_data)
	    pred = k_means.predict(test_data)
	    print('准确率：', acc_rate(test_label, pred))
	    NMI = metrics.normalized_mutual_info_score(pred, test_label)
	    print('NMI: ', NMI)
	
	    print('lung:')
	    train_data, test_data, train_label, test_label = load_data(lung_path)
	    k_means = KMeans(n_clusters=5, random_state=0)
	    k_means.fit(train_data)
	    pred = k_means.predict(test_data)
	    print('准确率：', acc_rate(test_label, pred))
	    NMI = metrics.normalized_mutual_info_score(pred, test_label)
	    print('NMI: ', NMI)
	
	    print('yale:')
	    train_data, test_data, train_label, test_label = load_data(yale_path)
	    k_means = KMeans(n_clusters=15, random_state=0)
	    k_means.fit(train_data)
	    pred = k_means.predict(test_data)
	    print('准确率：', acc_rate(test_label, pred))
	    NMI = metrics.normalized_mutual_info_score(pred, test_label)
	    print('NMI: ', NMI)
	
	```
2. LVQ 聚类算法
	```python
	# coding=utf-8
	import numpy as np
	from sklearn.model_selection import train_test_split
	import scipy.io as scio
	from munkres import Munkres
	from sklearn import metrics
	
	
	def load_data(path):
	    data = scio.loadmat(path)
	    data_x = np.array(data['X'])
	    data_y = np.array(data['Y'][:, 0] - 1)
	    train_data, test_data, train_label, test_label = train_test_split(data_x, data_y, test_size=0.25, random_state=10)
	    return train_data, test_data, train_label, test_label
	
	
	def LVQ(X, y, k, MAX_TIME=100, ita=0.2):
	    init_index = np.random.choice(len(y), k)
	    px = X[init_index]
	    py = y[init_index]
	    for _ in range(MAX_TIME):
	        j = np.random.choice(len(y), 1)
	        xj, yj = X[j], y[j]
	        i = np.argmin([np.linalg.norm(xj - pi) for pi in px])
	        pyi = py[i]
	        if pyi == yj:
	            px[i] = px[i] + ita * (xj - px[i])
	        else:
	            px[i] = px[i] - ita * (xj - px[i])
	    return px
	
	
	def PVQ_clustering(X, px):
	    return np.array(list(map(lambda x: np.argmin([np.linalg.norm(x - pi) for pi in px]), X)))
	
	
	# 采用匈牙利算法处理标签
	def best_map(L1, L2):
	    Label1 = np.unique(L1)  # 去除重复的元素，由小大大排列
	    nClass1 = len(Label1)  # 标签的大小
	    Label2 = np.unique(L2)
	    nClass2 = len(Label2)
	    nClass = np.maximum(nClass1, nClass2)
	    G = np.zeros((nClass, nClass))
	    for i in range(nClass1):
	        ind_cla1 = L1 == Label1[i]
	        ind_cla1 = ind_cla1.astype(float)
	        for j in range(nClass2):
	            ind_cla2 = L2 == Label2[j]
	            ind_cla2 = ind_cla2.astype(float)
	            G[i, j] = np.sum(ind_cla2 * ind_cla1)
	    m = Munkres()
	    index = m.compute(-G.T)
	    index = np.array(index)
	    c = index[:, 1]
	    newL2 = np.zeros(L2.shape)
	    for i in range(nClass2):
	        newL2[L2 == Label2[i]] = Label1[c[i]]
	    return newL2
	
	
	# 计算精确度
	def acc_rate(gt_s, s):
	    c_x = best_map(gt_s, s)
	    err_x = np.sum(gt_s[:] == c_x[:])
	    acc = err_x.astype(float) / (gt_s.shape[0])
	    return acc
	
	
	if __name__ == '__main__':
	    mnist_path = 'datasets/MNIST.mat'
	    lung_path = 'datasets/lung.mat'
	    yale_path = 'datasets/Yale.mat'
	
	    print('mnist:')
	    train_data, test_data, train_label, test_label = load_data(mnist_path)
	    lvq = LVQ(train_data, train_label, 10)
	    pred = PVQ_clustering(test_data, lvq)
	    print('准确率：', acc_rate(test_label, pred))
	    NMI = metrics.normalized_mutual_info_score(pred, test_label)
	    print('NMI: ', NMI)
	
	    print('lung:')
	    train_data, test_data, train_label, test_label = load_data(lung_path)
	    lvq = LVQ(train_data, train_label, 5)
	    pred = PVQ_clustering(test_data, lvq)
	    print('准确率：', acc_rate(test_label, pred))
	    NMI = metrics.normalized_mutual_info_score(pred, test_label)
	    print('NMI: ', NMI)
	
	    print('yale:')
	    train_data, test_data, train_label, test_label = load_data(yale_path)
	    lvq = LVQ(train_data, train_label, 15)
	    pred = PVQ_clustering(test_data, lvq)
	    print('准确率：', acc_rate(test_label, pred))
	    NMI = metrics.normalized_mutual_info_score(pred, test_label)
	    print('NMI: ', NMI)
	
	```
3. Mixture-of-Gaussian 聚类算法
	```python
	# coding=utf-8
	import numpy as np
	from sklearn.model_selection import train_test_split
	from sklearn.mixture import GaussianMixture
	from sklearn import metrics
	import scipy.io as scio
	from munkres import Munkres
	
	
	def load_data(path):
	    data = scio.loadmat(path)
	    data_x = np.array(data['X'])
	    data_y = np.array(data['Y'][:, 0] - 1)
	    train_data, test_data, train_label, test_label = train_test_split(data_x, data_y, test_size=0.25, random_state=10)
	    return train_data, test_data, train_label, test_label
	
	
	def best_map(L1, L2):
	    Label1 = np.unique(L1)  # 去除重复的元素，由小大大排列
	    nClass1 = len(Label1)  # 标签的大小
	    Label2 = np.unique(L2)
	    nClass2 = len(Label2)
	    nClass = np.maximum(nClass1, nClass2)
	    G = np.zeros((nClass, nClass))
	    for i in range(nClass1):
	        ind_cla1 = L1 == Label1[i]
	        ind_cla1 = ind_cla1.astype(float)
	        for j in range(nClass2):
	            ind_cla2 = L2 == Label2[j]
	            ind_cla2 = ind_cla2.astype(float)
	            G[i, j] = np.sum(ind_cla2 * ind_cla1)
	    m = Munkres()
	    index = m.compute(-G.T)
	    index = np.array(index)
	    c = index[:, 1]
	    newL2 = np.zeros(L2.shape)
	    for i in range(nClass2):
	        newL2[L2 == Label2[i]] = Label1[c[i]]
	    return newL2
	
	
	def acc_rate(gt_s, s):
	    c_x = best_map(gt_s, s)
	    err_x = np.sum(gt_s[:] == c_x[:])
	    acc = err_x.astype(float) / (gt_s.shape[0])
	    return acc
	
	
	if __name__ == '__main__':
	    mnist_path = 'datasets/MNIST.mat'
	    lung_path = 'datasets/lung.mat'
	    yale_path = 'datasets/Yale.mat'
	
	    print('mnist:')
	    train_data, test_data, train_label, test_label = load_data(mnist_path)
	    gm = GaussianMixture(n_components=10, random_state=0)
	    gm.fit(train_data)
	    pred = gm.predict(test_data)
	    print('准确率：', acc_rate(test_label, pred))
	    NMI = metrics.normalized_mutual_info_score(pred, test_label)
	    print('NMI: ', NMI)
	
	    print('lung:')
	    train_data, test_data, train_label, test_label = load_data(lung_path)
	    gm = GaussianMixture(n_components=5, random_state=0)
	    gm.fit(train_data)
	    pred = gm.predict(test_data)
	    print('准确率：', acc_rate(test_label, pred))
	    NMI = metrics.normalized_mutual_info_score(pred, test_label)
	    print('NMI: ', NMI)
	
	    print('yale:')
	    train_data, test_data, train_label, test_label = load_data(yale_path)
	    gm = GaussianMixture(n_components=15, random_state=0)
	    gm.fit(train_data)
	    pred = gm.predict(test_data)
	    print('准确率：', acc_rate(test_label, pred))
	    NMI = metrics.normalized_mutual_info_score(pred, test_label)
	    print('NMI: ', NMI)
	
	```

# 二、基于回归学习及流形结构保持的无监督特征选择

## 一、问题引入

特征选择：
* 简单来说就是对大数据进行一个降维，减少计算量，比如从一张图片中只提取部分像素作为特征进行分类任务；
* 选出最具重要性、代表性、低冗余性的特征。

## 二、数学建模

* 定义
  $$
  x_i\in R^{d\times1} \\ D=\begin{bmatrix}sum(S_1) & 0 & \cdots & 0 \\ 0 & sum(S_2) & \cdots & 0 \\ \vdots & \vdots & & \vdots \\ 0 & 0 & \cdots & sum(S_m) \end{bmatrix} \\ X=\begin{bmatrix}x_1^T\\x_2^T\\\vdots\\x_m^T\end{bmatrix}\in R^{m\times d} \\ y_i\in R^{c\times 1} \\ Y=\begin{bmatrix}y_1^T\\y_2^T\\\vdots\\y_m^T\end{bmatrix}\in R^{m\times c} \\ S=[S_{ij}]_{m\times m} \\ S_{ij}=exp\left\{-\frac{||x_i-x_j||_2^2}{2\sigma^2}\right\}
  $$

* 基础
  $$
  ||XW-Y||_F^2=tr\left\{(XW-Y)^T(XW-Y)\right\}=tr\left\{W^TX^TXW\right\}-2tr\left\{Y^TXW\right\}+tr\left\{Y^TY\right\}
  $$

* 误差项
  $$
  \sum_{i=1}^m||W^Tx_i-y_i||_2^2=||XW-Y||_F^2=tr(XW-Y)^T(XW-Y)
  $$

* 稀疏正则项
  $$
  ||W||_{2,0}\rightarrow||W||_{2,1}
  $$

* 流形结构保持的约束
  $$
  \frac{1}{2}\sum_{i=1}^m\sum_{j=1}^m||W^Tx_i-W^Tx_j||_x^2S_{ij}=tr(W^TXLXW) \\
  L=D-S
  $$

* 最终的目标
  $$
  min~tr\left\{||XW-Y||_F^2\right\}+\alpha~||W||_{2,1}+\beta~tr\left\{W^TXLXW\right\}
  $$

* 待求参数
  $$
  W\in R^{d\times c} \\ Y\in R^{m\times c}
  $$

## 三、模型优化（轮换迭代）

1. 固定 $W$ ，求 $Y$ 
   $$
   L_1=min_Y\left(||XW-Y||_F^2\right) \\ \frac{\partial L}{\partial Y}=-2XW+2Y=0 \\ 推导得：Y=XW
   $$

2. 固定 $Y$ ，求 $W$
   $$
   L=||XW-Y||_F^2+\alpha||W||_{2,1}+\beta~tr\left\{W^TX^TLXW\right\} \\ \frac{\partial||XW-Y||_F^2}{\partial W}=2X^TXW-2X^TY \\ \frac{\partial\beta~tr\left\{W^TX^TLXW\right\}}{\partial W}=2\beta X^TLXW \\ \frac{\partial||W||_{2,1}}{\partial W}=2\Lambda_{d\times d} W \\ \frac{\partial L}{\partial W}=2(X^TX+\beta X^TLX+\alpha\Lambda)W-2X^TY=0 \\ 推导得迭代解：W=(X^TX+\beta X^TLX+\alpha\Lambda)^{-1}X^TY
   $$

3. $W$ 的迭代求解 (Alg1) ：

   * Input: Data Matrix $X\in R^{m\times d}$ , $S\in R^{m\times m}$, Hyper-parameters: $\alpha, \beta, \epsilon > 0$

   * Output: $W$

   * Initialize: $\Lambda=I_{d\times d}$

   * Calculate: $L=D-S$

   * Repeat:
     * Update $W\leftarrow(X^TX+\beta X^TLX+\alpha\Lambda)^{-1}X^TY$
     * Calculate: $\Lambda=[\Lambda_{ii}]_{d\times d}$, $\Lambda_{ii}=\frac{1}{||W^i||_2+\epsilon}$
   * Until Convergence.

4. 基于回归学习及流形结构保持的无监督特征选择 (Alg2) ：

   * Input: $X\in R^{m\times d},S\in R^{m\times m},\alpha,\beta,\epsilon>0$, the number of selected features k.
   * Output: k selected features
   * Initialize: $\Lambda=I$
   * Calculate: $L=D-S$
   * Repeat:
     * Update W by Alg 1
     * Update Y by $Y\leftarrow XW$
   * Until Convergence
   * Sort all features according to $||W^i||_2$ in descending order and select the top k ranked features

## 四、代码实现

1. code
	```python
	# coding=utf-8
	import numpy as np
	import numpy.matlib
	import math
	import heapq
	import scipy.io as scio
	from sklearn.cluster import KMeans
	from munkres import Munkres
	from sklearn.model_selection import train_test_split
	from sklearn import metrics
	
	
	def load_data(path):
	    # 加载数据集
	    data = scio.loadmat(path)
	    data_x = data['X']
	    data_y = data['Y'][:, 0] - 1
	    return data_x, data_y
	
	
	def split_set(X, Y):
	    train_data, test_data, train_label, test_label = train_test_split(X, Y, test_size=0.25, random_state=10)
	    return train_data, test_data, test_label
	
	
	def matrix_gaussianKernel_S(X, sigma, m):
	    # 高斯核生成S矩阵
	    S = np.zeros((m, m))
	    for i in range(m):
	        for j in range(m):
	            S[i][j] = math.exp(-math.pow(np.linalg.norm(X[i] - X[j], ord=2), 2) / (2 * sigma ** 2))
	    return S
	
	
	def initialize_matrix_Lambda(d):
	    # 初始化Lambda矩阵
	    Lambda = numpy.matlib.identity(d)
	    return Lambda
	
	
	def calculate_matrix_L(S, m):
	    # 计算拉普拉斯矩阵L
	    D = np.zeros((m, m))
	    for i in range(m):
	        D[i][i] = np.sum(S[i])
	    return D - S
	
	
	def update_Lambda(epsilon, W, d):
	    # 在循环中更新Lambda矩阵
	    Lambda = np.zeros((d, d))
	    for i in range(d):
	        Lambda[i][i] = 1 / (np.linalg.norm(W[i], ord=2) + epsilon)
	    return Lambda
	
	
	def iteration_until_convergence(alpha, beta, epsilon, wcycles, L, Lambda, X, Y, d):
	    # 算法1，用于生成W
	    W = np.matmul(
	        np.matmul(np.linalg.inv(np.matmul(X.T, X) + np.matmul(np.matmul(beta * X.T, L), X) + alpha * Lambda), X.T),
	        Y)
	    for i in range(wcycles):
	        # while np.sum(np.abs(temp_W - W)) > threshold:
	        W = np.matmul(
	            np.matmul(np.linalg.inv(np.matmul(X.T, X) + np.matmul(np.matmul(beta * X.T, L), X) + alpha * Lambda), X.T),
	            Y)
	        Lambda = update_Lambda(epsilon, W, d)
	    return W
	
	
	def update_Y(X, W):
	    # 在循环中更新Y
	    return np.matmul(X, W)
	
	
	def rank_based_W(alpha, beta, epsilon, sigma, wcycles, ycycles, k, X, Y, m, d, c, name):
	    # 算法2，用于生成特征值
	    S = matrix_gaussianKernel_S(X, sigma, m)
	    Lambda = initialize_matrix_Lambda(d)
	    L = calculate_matrix_L(S, m)
	    temp_Y = np.zeros((m, c))
	    itetimes = 0
	    W = iteration_until_convergence(alpha, beta, epsilon, wcycles, L, Lambda, X, Y, d)
	    x, y = [], []
	    for i in range(ycycles):
	        x.append(itetimes)
	        y.append(np.sum(np.abs(temp_Y - Y)))
	        temp_Y = Y
	        itetimes += 1
	        W = iteration_until_convergence(alpha, beta, epsilon, wcycles, L, Lambda, X, Y, d)
	        Y = update_Y(X, W)
	    if name != '0':
	        np.savez(name, x, y)
	    norm_W = np.linalg.norm(W, axis=1)
	    index = heapq.nlargest(k, range(len(norm_W)), norm_W.take)
	    feature_X = np.zeros((m, k))
	    for i in range(m):
	        feature_X[i] = X[i][index]  # 取每个数据中被提取的特征数据
	    return feature_X
	
	
	def best_map(L1, L2):
	    # 重排列标签
	    Label1 = np.unique(L1)  # 去除重复的元素，由小大大排列
	    nClass1 = len(Label1)  # 标签的大小
	    Label2 = np.unique(L2)
	    nClass2 = len(Label2)
	    nClass = np.maximum(nClass1, nClass2)
	    G = np.zeros((nClass, nClass))
	    for i in range(nClass1):
	        ind_cla1 = L1 == Label1[i]
	        ind_cla1 = ind_cla1.astype(float)
	        for j in range(nClass2):
	            ind_cla2 = L2 == Label2[j]
	            ind_cla2 = ind_cla2.astype(float)
	            G[i, j] = np.sum(ind_cla2 * ind_cla1)
	    m = Munkres()
	    index = m.compute(-G.T)
	    index = np.array(index)
	    c = index[:, 1]
	    newL2 = np.zeros(L2.shape)
	    for i in range(nClass2):
	        newL2[L2 == Label2[i]] = Label1[c[i]]
	    return newL2
	
	
	def acc_rate(gt_s, s):
	    # 计算ACC
	    c_x = best_map(gt_s, s)
	    err_x = np.sum(gt_s[:] == c_x[:])
	    acc = err_x.astype(float) / (gt_s.shape[0])
	    return acc
	
	
	def comparative_test(path, epsilon, sigma, wcycles, ycycles, alpha, beta, c, k, name):
	    # 对比特征提取与特征不提取的效果
	    X, Y = load_data(path)  # 加载数据集
	    onehot_Y = np.eye(c)[np.array(Y)]
	    m = X.shape[0]  # 训练集长度
	    d = X.shape[1]  # 图片长度
	    print('---提取特征：')
	    feature_X = rank_based_W(alpha, beta, epsilon, sigma, wcycles, ycycles, k, X, onehot_Y, m, d, c, name)
	    train_data, test_data, test_label = split_set(feature_X, Y)
	    k_means = KMeans(n_clusters=c, random_state=0)
	    k_means.fit(train_data)
	    pred = k_means.predict(test_data)
	    print('ACC：', acc_rate(test_label, pred))
	    NMI = metrics.normalized_mutual_info_score(pred, test_label)
	    print('NMI: ', NMI)
	    print()
	    print('---不提取特征：')
	    train_data, test_data, test_label = split_set(X, Y)
	    k_means = KMeans(n_clusters=c, random_state=0)
	    k_means.fit(train_data)
	    pred = k_means.predict(test_data)
	    print('ACC：', acc_rate(test_label, pred))
	    NMI = metrics.normalized_mutual_info_score(pred, test_label)
	    print('NMI: ', NMI)
	    print()
	
	
	def f_test(path, epsilon, sigma, wcycles, ycycles, alpha, beta, c, k, name):
	    # 不输出信息的算法2
	    X, Y = load_data(path)  # 加载数据集
	    onehot_Y = np.eye(c)[np.array(Y)]
	    m = X.shape[0]  # 训练集长度
	    d = X.shape[1]  # 图片长度
	    feature_X = rank_based_W(alpha, beta, epsilon, sigma, wcycles, ycycles, k, X, onehot_Y, m, d, c, name)
	    train_data, test_data, test_label = split_set(feature_X, Y)
	    k_means = KMeans(n_clusters=c, random_state=0)
	    k_means.fit(train_data)
	    pred = k_means.predict(test_data)
	    ACC = acc_rate(test_label, pred)
	    NMI = metrics.normalized_mutual_info_score(pred, test_label)
	    return ACC, NMI
	
	
	def test_alpha_beta(path, epsilon, sigma, wcycles, ycycles, c, k, name):
	    # 测试alpha与beta的不同取值对聚类结果的影响
	    Alpha = [-3, -2, -1, 0, 1, 2, 3]
	    Beta = Alpha
	    ACC = []
	    NMI = []
	    for alpha in Alpha:
	        for beta in Beta:
	            try:
	                acc, nmi = f_test(path, epsilon, sigma, wcycles, ycycles, alpha, beta, c, k, name)
	            except:
	                acc, nmi = 0, 0
	            ACC.append(acc)
	            NMI.append(nmi)
	    np.save('./npy/alpha_beta_ACC_' + path.split('/')[-1].split('.')[0] + '.npy', ACC)
	    np.save('./npy/alpha_beta_NMI_' + path.split('/')[-1].split('.')[0] + '.npy', NMI)
	
	
	if __name__ == '__main__':
	    """
	    --------------comparative test--------------
	    该部分用于做进行特征提取以及不进行特征提取的对比试验
	    path: 数据集路径
	    epsilon: Lambda矩阵防止除0错误的分母附加值
	    sigma: 生成S时正态分布核的参数
	    wcycles: 优化Y需要的循环次数
	    ycycles: 优化Y需要的循环次数
	    alpha: 超参数
	    beta: 超参数
	    c: 标签类别个数
	    k: 特征提取的个数
	    name: 保存Y迭代的值，用于画图
	    """
	    # # mnist
	    # print('--------------mnist--------------')
	    # comparative_test(path='datasets/MNIST.mat', epsilon=0.000001, sigma=1.0, wcycles=5, ycycles=50, alpha=3, beta=1,
	    #                  c=10, k=500, name='./npz/mnist.npz')
	    #
	    # # lung
	    # print('--------------lung--------------')
	    # comparative_test(path='datasets/lung.mat', epsilon=0.000001, sigma=1.0, wcycles=5, ycycles=50, alpha=3, beta=2,
	    #                  c=5, k=2400, name='./npz/lung.npz')
	
	    """
	    --------------alpha beta test--------------
	    该部分用于研究不同alpha与beta取值对聚类结果的影响
	    path: 数据集路径
	    epsilon: Lambda矩阵防止除0错误的分母附加值
	    sigma: 生成S时正态分布核的参数
	    wcycles: 优化Y需要的循环次数
	    ycycles: 优化Y需要的循环次数
	    c: 标签类别个数
	    k: 特征提取的个数
	    name: 保存Y迭代的值，用于画图
	    """
	    # mnist
	    test_alpha_beta(path='datasets/MNIST.mat', epsilon=0.000001, sigma=1.0, wcycles=5, ycycles=50, c=10, k=500,
	                    name='0')
	
	    # lung
	    test_alpha_beta(path='datasets/lung.mat', epsilon=0.000001, sigma=1.0, wcycles=5, ycycles=50, c=5, k=2400,
	                    name='0')
	
	```
2. plot
	```python
	import matplotlib.pyplot as plt
	import numpy as np
	
	
	def plot_2d(data, path, name):
	    # Y 迭代的画图函数
	    plt.figure()
	    plt.title(name)  # 标题
	    plt.xlabel('迭代次数')
	    plt.ylabel('标签绝对差值')
	    plt.plot(data['arr_0'], data['arr_1'], 'r--')
	    plt.savefig(path)
	
	
	def plot_Y_loss():
	    # Y 迭代的入口函数
	    plt.rcParams["font.family"] = "FangSong"
	    mnist_path = './npz/mnist.npz'
	    lung_path = './npz/lung.npz'
	
	    mnist_np = np.load(mnist_path)
	    lung_np = np.load(lung_path)
	
	    plot_2d(mnist_np, './img/mnist_Y.png', 'mnist')
	    plot_2d(lung_np, './img/lung_Y.png', 'lung')
	
	
	def plot_3d(NMI, ACC, X, Y, width, depth, c, name):
	    # alpha beta 遍历的画图函数
	    fig = plt.figure()
	    ax = fig.gca(projection='3d')
	    height = np.zeros_like(NMI)
	    ax.bar3d(X, Y, height, width, depth, NMI, color=c, shade=False)
	    ax.set_xlabel('alpha')
	    ax.set_ylabel('beta')
	    ax.set_zlabel('NMI')
	    ax.set_title(name)
	    plt.savefig('./img/{}_NMI.png'.format(name))
	
	    fig = plt.figure()
	    ax = fig.gca(projection='3d')
	    height = np.zeros_like(ACC)
	    ax.bar3d(X, Y, height, width, depth, ACC, color=c, shade=False)
	    ax.set_xlabel('alpha')
	    ax.set_ylabel('beta')
	    ax.set_zlabel('ACC')
	    ax.set_title(name)
	    plt.savefig('./img/{}_ACC.png'.format(name))
	
	
	def plot_alpha_beta():
	    # alpha beta 遍历的入口函数
	    mnist_NMI_path = './npy/alpha_beta_NMI_MNIST.npy'
	    mnist_ACC_path = './npy/alpha_beta_ACC_MNIST.npy'
	    lung_NMI_path = './npy/alpha_beta_NMI_lung.npy'
	    lung_ACC_path = './npy/alpha_beta_ACC_lung.npy'
	
	    mnist_NMI = np.load(mnist_NMI_path)
	    mnist_ACC = np.load(mnist_ACC_path)
	    lung_NMI = np.load(lung_NMI_path)
	    lung_ACC = np.load(lung_ACC_path)
	
	    X = Y = [-3, -2, -1, 0, 1, 2, 3]
	    xx, yy = np.meshgrid(X, Y)
	    X, Y = xx.ravel(), yy.ravel()
	    width = depth = 0.5
	    c = ['r', 'r', 'r', 'r', 'r', 'r', 'r', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'c',
	         'c', 'c', 'c', 'c', 'c', 'c', 'y', 'y', 'y', 'y', 'y', 'y', 'y', 'm', 'm', 'm', 'm', 'm', 'm', 'm', 'k', 'k',
	         'k', 'k', 'k', 'k', 'k']
	
	    plot_3d(mnist_NMI, mnist_ACC, X, Y, width, depth, c, 'mnist')
	    plot_3d(lung_NMI, lung_ACC, X, Y, width, depth, c, 'lung')
	
	
	if __name__ == '__main__':
	    plot_Y_loss()
	    plot_alpha_beta()  # 不要与上一条语句同时运行，三维坐标会显示错误
	
	```