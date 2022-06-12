import matplotlib.pyplot as plt
import numpy as np


def plot_2d(data, path, name):
    plt.figure()
    plt.title(name)  # 标题
    plt.xlabel('迭代次数')
    plt.ylabel('标签绝对差值')
    plt.plot(data['arr_0'], data['arr_1'], 'r--')
    plt.savefig(path)


def plot_Y_loss():
    plt.rcParams["font.family"] = "FangSong"
    mnist_path = './npz/mnist.npz'
    lung_path = './npz/lung.npz'

    mnist_np = np.load(mnist_path)
    lung_np = np.load(lung_path)

    plot_2d(mnist_np, './img/mnist_Y.png', 'mnist')
    plot_2d(lung_np, './img/lung_Y.png', 'lung')


def plot_3d(NMI, ACC, X, Y, width, depth, c, name):
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
    # plot_Y_loss()
    # plot_alpha_beta()
    mnist_NMI_path = './npy/alpha_beta_NMI_MNIST.npy'
    mnist_ACC_path = './npy/alpha_beta_ACC_MNIST.npy'
    lung_NMI_path = './npy/alpha_beta_NMI_lung.npy'
    lung_ACC_path = './npy/alpha_beta_ACC_lung.npy'

    mnist_NMI = np.load(mnist_NMI_path)
    mnist_ACC = np.load(mnist_ACC_path)
    lung_NMI = np.load(lung_NMI_path)
    lung_ACC = np.load(lung_ACC_path)

    print(mnist_NMI)
    print(mnist_ACC)
    print(lung_NMI)
    print(lung_ACC)