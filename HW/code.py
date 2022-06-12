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
