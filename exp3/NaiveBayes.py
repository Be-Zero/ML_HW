from sklearn.model_selection import train_test_split  # 切分测试集
from sklearn.naive_bayes import MultinomialNB  # 贝叶斯
import scipy.io as scio
import numpy as np


def load_data(path):
    data = scio.loadmat(path)
    data_x = np.array(data['X'])
    data_y = np.array(data['Y'][:, 0] - 1)
    train_data, test_data, train_label, test_label = train_test_split(data_x, data_y, test_size=0.25, random_state=0)
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
