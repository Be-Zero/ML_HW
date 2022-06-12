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
