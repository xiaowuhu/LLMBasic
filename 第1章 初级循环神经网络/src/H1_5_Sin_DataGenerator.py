import os
import numpy as np
from matplotlib import pyplot as plt

def create_data1(split_count, count, is_train = True):
    X = np.linspace(0, 2*np.pi, split_count)[:, np.newaxis]
    if is_train == True:
        X = np.repeat(X, count).reshape(-1,1)
        Y = np.sin(X) + np.random.randn(count*split_count).reshape(-1,1)/50
    else:
        Y = np.sin(X)
    plt.scatter(X,Y, s=2)
    plt.grid()
    plt.show()
    print(X.shape)
    print(Y.shape)
    return np.hstack((X, Y))

def save_data(data, name):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    filename = os.path.join(current_dir, "data", name)
    np.savetxt(filename, data, fmt="%.6f")

# 在 0-4pi 生成 41 组数据（steps=20）
def create_data2(steps, sample_count, is_train = True):
    split_count = steps * 2 + 1
    # 生成时间步
    t = np.linspace(0, 4 * np.pi, split_count)
    if is_train == True:
        t = np.repeat(t, sample_count).reshape(-1,1)
        # 生成训练数据(sin(t))
        x = np.sin(t) + np.random.randn(sample_count * split_count).reshape(-1,1)/50
    else:
        x = np.sin(t)
    plt.scatter(t,x, s=2)
    plt.grid()
    plt.show()
    x = x.reshape(split_count, -1).T # -> N x T
    X = []
    Y = []
    for i in range(0, steps):
        for j in range(sample_count):
            # 取连续的 20 个时间步做训练样本
            X.append(x[j, i:i+steps])
            # 取第 21 个时间步做标签值
            Y.append(x[j, i+steps])
    X = np.array(X)
    Y = np.array(Y)
    X = np.expand_dims(X, axis=2) # -> N x Timestep x Feature
    Y = np.expand_dims(Y, axis=1) # -> N x Label
    print(X.shape)
    print(Y.shape)
    return X, Y


def create_data3(steps, count, is_train = True):
    # 生成时间步
    t = np.linspace(0, 2 * np.pi, 21)
    # 生成特征值
    x = np.sin(t)
    plt.scatter(t,x)
    plt.grid()
    plt.show()
    # 根据指定的steps截取特征值
    start = 0
    X = []
    Y = []
    while True:
        end = start + steps + 1 # 加一位标签值
        if end > len(t):
            break
        X.append(x[start:end-1])
        Y.append(x[end-1])
        start += 1
    X = np.array(X).reshape(-1, steps)
    Y = np.array(Y).reshape(-1, 1)
    X = np.repeat(X, count, axis=0)
    Y = np.repeat(Y, count, axis=0)
    assert (X.shape[0] == Y.shape[0])
    noise = np.random.randn(*X.shape)/50
    X += noise
    noise = np.random.randn(*Y.shape)/50
    Y += noise
    X = np.expand_dims(X, axis=2) # -> N x Timestep x Feature
    print(X.shape)
    print(Y.shape)
    return X, Y

def save_npz(filename, X, Y):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    filename = os.path.join(current_dir, "data", filename)
    np.savez(filename, data=X, label=Y)    

if __name__=='__main__':
    np.random.seed(17)
    # for FC regression
    # xy = create_data1(21, 25, is_train=True)
    # save_data(xy, "sin_train_linear.txt")
    # xy = create_data1(21, 2, is_train=False)
    # save_data(xy, "sin_test_linear.txt")
    # print("done")
    # # for RNN
    # X, Y = create_data2(20, 25, is_train=True)
    # save_npz("sin_train_20.npz", X, Y)
    # X, Y = create_data2(20, 1, is_train=False)
    # save_npz("sin_test_20.npz", X, Y)
    # print("done")
    # for RNN2
    X, Y = create_data3(7, 20, is_train=True)
    save_npz("sin_train_steps.npz", X, Y)
    X, Y = create_data3(7, 1, is_train=False)
    save_npz("sin_test_steps.npz", X, Y)
    print("done")
