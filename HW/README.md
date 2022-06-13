# 一、实验环境

1. PC：Windows 10 家庭中文版

2. 系统类型：64 位操作系统，基于 x64 的处理器

3. 处理器：11th Gen Intel(R) Core(TM) i5-11400F @ 2.60GHz 2.59GHz

4. Python: 3.10.4

5. 外部库包含（详见 requirements.txt）：

   ```python
   CacheControl==0.12.11
   cachy==0.3.0
   certifi==2022.5.18.1
   charset-normalizer==2.0.12
   cleo==0.8.1
   clikit==0.6.2
   crashtest==0.3.1
   distlib==0.3.4
   filelock==3.7.0
   html5lib==1.1
   idna==3.3
   importlib-metadata==4.11.4
   keyring==23.5.1
   lockfile==0.12.2
   msgpack==1.0.3
   packaging==20.9
   pastel==0.2.1
   pexpect==4.8.0
   pkginfo==1.8.2
   platformdirs==2.5.2
   poetry==1.1.13
   poetry-core==1.0.8
   ptyprocess==0.7.0
   pylev==1.4.0
   pyparsing==3.0.9
   pywin32-ctypes==0.2.0
   requests==2.27.1
   requests-toolbelt==0.9.1
   shellingham==1.4.0
   six==1.16.0
   tomlkit==0.10.2
   urllib3==1.26.9
   virtualenv==20.14.1
   webencodings==0.5.1
   zipp==3.8.0
   ```

# 二、参数设置

## 1. 参数介绍

```python
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
```

## 2. 参数取值

| epsilon  | sigma | wcycles | ycycles |  alpha  |  beta   |
| :------: | :---: | :-----: | :-----: | :-----: | :-----: |
| 0.000001 |  1.0  |    5    |   50    | [-3, 3] | [-3, 3] |

# 三、代码功能

## 1. code.py

```python
def load_data(path):  # 加载数据集
    
def split_set(X, Y):  # 划分数据集
    
def matrix_gaussianKernel_S(X, sigma, m):  # 高斯核生成S矩阵

def initialize_matrix_Lambda(d):  # 初始化Lambda矩阵
    
def calculate_matrix_L(S, m):  # 计算拉普拉斯矩阵L
    
def update_Lambda(epsilon, W, d):  # 在循环中更新Lambda矩阵
    
def iteration_until_convergence(alpha, beta, epsilon, wcycles, L, Lambda, X, Y, d):  # 算法1，用于生成W

def update_Y(X, W):  # 在循环中更新Y
    
def rank_based_W(alpha, beta, epsilon, sigma, wcycles, ycycles, k, X, Y, m, d, c, name):  # 算法2，用于生成特征值

def best_map(L1, L2):  # 重排列聚类后的标签
    
def acc_rate(gt_s, s):  # 计算ACC
    
def comparative_test(path, epsilon, sigma, wcycles, ycycles, alpha, beta, c, k, name):  # 对比特征提取与特征不提取的效果

def f_test(path, epsilon, sigma, wcycles, ycycles, alpha, beta, c, k, name):  # 不打印信息的算法2
    
def test_alpha_beta(path, epsilon, sigma, wcycles, ycycles, c, k, name):  # 测试alpha与beta的不同取值对聚类结果的影响
    
# 函数调用方法举例：
comparative_test(path='datasets/MNIST.mat', epsilon=0.000001, sigma=1.0, wcycles=5, ycycles=50, alpha=3, beta=1, c=10, k=500, name='./npz/mnist.npz')  # 数据集为mnist ，在给定 alpha 和 beta 值的情况下，评价聚类的效果

test_alpha_beta(path='datasets/MNIST.mat', epsilon=0.000001, sigma=1.0, wcycles=5, ycycles=50, c=10, k=500, name='0')  # 数据集为 mnist ，遍历所有 alpha 和 beta 的组合，评价聚类的效果
```

## 2. plot_loss.py

```python
def plot_2d(data, path, name):  # Y Loss 的画图函数
    
def plot_Y_loss():  # Y Loss 的入口函数

def plot_3d(NMI, ACC, X, Y, width, depth, c, name):  # 遍历 alpha beta 的画图函数
    
def plot_alpha_beta():  # 遍历 alpha beta 的入口函数
    
# 函数调用方法举例：
plot_Y_loss()  # 绘制给定 alpha 和 beta 时 Y 迭代的 Loss
plot_alpha_beta()  # 遍历不同 alpha 和 beta 的组合，绘制其聚类效果的 NMI 和 ACC 指标
```

# 四、代码运行

## 1. 特征选择及聚类：

运行 code.py 即可，运行完毕后会在 `./npy` 和 `./npz` 目录下生成 Loss 数据文件，在控制台打印聚类效果。

## 2. 绘图

运行 plot_loss.py 即可，运行完毕后会在 `./img` 目录下生成图像文件。