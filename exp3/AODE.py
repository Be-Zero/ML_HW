import math
import scipy.io as scio
import numpy as np


def divdata(filename):
    data = scio.loadmat(filename)
    dataX = data['X']
    dataY = data['Y'].T[0]

    dataX_train = []
    dataX_predict = []
    dataY_train = []
    dataY_predict = []
    num_Y = np.unique(dataY).astype(int)
    for i in range(len(num_Y)):
        temp = dataY == num_Y[i]
        temp.astype(float)
        num_Y[i] = np.sum(temp)
        flag = 0
        for j in range(len(dataY)):
            if temp[j] == 1:
                if flag < int(round(0.9 * num_Y[i])):
                    dataX_train.append(dataX[j])
                    dataY_train.append(dataY[j])
                    flag += 1
                else:
                    dataX_predict.append(dataX[j])
                    dataY_predict.append(dataY[j])

        dataX_train = np.array(dataX_train)
        dataX_predict = np.array(dataX_predict)
        dataY_train = np.array(dataY_train)
        dataY_predict = np.array(dataY_predict)
        return dataX_train, dataX_predict, dataY_train, dataY_predict
    else:
        return dataX, dataX, dataY, dataY


# 按标签类别生成不同标签样本组成的集合，返回值为每种类别样本的索引
def divSamples(dataY):
    label = np.unique(dataY)
    D = []
    for i in label:
        D.append(np.argwhere(dataY == i).T[0])
    return np.array(D)


# 计算先验概率P(c,xi)
def beforeProb(sample_cIndx, i, xi, D):
    numerator = len(np.argwhere(D[sample_cIndx][:, i] == xi).T[0]) + 1
    denominator = D.shape[0] + 1
    return float(numerator) / denominator


# 计算条件概率P(xj|c,xi)
def condProb(i, xi, j, xj, sample_cIndx, D):
    D_c = D[sample_cIndx]
    D_c_xi = D_c[np.argwhere(D_c[:, i] == xi).T[0]]
    D_c_xi_xj = D_c_xi[np.argwhere(D_c_xi[:, j] == xj).T[0]]
    numerator = len(D_c_xi_xj) + 1
    denominator = len(D_c_xi) + 1
    return float(numerator) / denominator


# 计算后验概率P(c|x)
def afterProb(sample_x, c, dataX, dataY):
    sample_c = divSamples(dataY)
    prob = 0
    for i in range(len(sample_x)):
        p1 = 0
        p = beforeProb(sample_c[c], i, sample_x[i], dataX)
        for j in range(len(sample_x)):
            p1 += math.log10(condProb(i, sample_x[i], j, sample_x[j], sample_c[c], dataX))  # 防止下溢
        prob += p * p1
    return prob


# 计算最大概率对应的类
def argMaxProb_c(sample_x, dataX, dataY):
    label = np.unique(dataY)
    argProb1 = []

    for c in label:
        temp_prob = afterProb(sample_x, c - 1, dataX, dataY)
        argProb1.append(temp_prob)

    argProb = np.array(argProb1)
    return label[np.argmax(argProb)]


def bayesClassifier(dataPredict, dataX, dataY):
    pred = []
    for sample_x in dataPredict:
        label_pred = argMaxProb_c(sample_x, dataX, dataY)
        pred.append(label_pred)
    return pred


def acc(L1, L2):
    sum = np.sum(L1[:] == L2[:])
    return sum / len(L2)


if __name__ == '__main__':
    mnist_path = 'datasets/MNIST.mat'
    lung_path = 'datasets/lung.mat'
    yale_path = 'datasets/Yale.mat'

    print('mnist:')
    dataX_train, dataX_predict, dataY_train, dataY_predict = divdata(mnist_path)
    pred = bayesClassifier(dataX_predict, dataX_train, dataY_train)
    print('正确率为：', acc(pred, dataY_predict))

    print('lung:')
    dataX_train, dataX_predict, dataY_train, dataY_predict = divdata(lung_path)
    pred = bayesClassifier(dataX_predict, dataX_train, dataY_train)
    print('正确率为：', acc(pred, dataY_predict))

    print('yale:')
    dataX_train, dataX_predict, dataY_train, dataY_predict = divdata(yale_path)
    pred = bayesClassifier(dataX_predict, dataX_train, dataY_train)
    print('正确率为：', acc(pred, dataY_predict))
