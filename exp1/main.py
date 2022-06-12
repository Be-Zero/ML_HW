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
