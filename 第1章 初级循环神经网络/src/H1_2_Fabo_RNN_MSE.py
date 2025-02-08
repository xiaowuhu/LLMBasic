
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import os
import common.Layers as layers
from common.DataLoader_14 import DataLoader_14
from common.Module import Sequential

from H17_2_Fabo_RNN_Train import build_model

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=9)


def load_data(train_file_name, test_file_name):
    train_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", train_file_name)
    test_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", test_file_name)
    data_loader = DataLoader_14(train_file_path, test_file_path)
    data_loader.load_txt_data()
    data_loader.shuffle_data()
    return data_loader

def calculate_loss(model:Sequential, X, Y, U, W):
    m = X.shape[0]
    model.operator_seq[0].U.W = U
    model.operator_seq[0].W.W = W
    Z = model.forward(X)
    Loss = (Z - Y) ** 2
    Loss = Loss.sum(axis=0, keepdims=True)/m/2
    #print(U, W, Loss)
    return Loss


def func_UW(U):
    # U + 0.618UW = 1.618
    W = (1.618 - U)/(0.618*U)
    return W

if __name__=="__main__":
    data_loader = load_data("fabo_train.txt", "fabo_test.txt")
    X, Y = data_loader.get_train()
    m = X.shape[0]
    model = build_model(1, 1, 1)
    
    losses = []
    U = np.linspace(1-2,1+2, 11)
    W = func_UW(U)
    print(U)
    print(W)
    for u in U:
        for w in W:
            loss = calculate_loss(model, X, Y, u, w)
            if loss[0,0] > 10:
                losses.append(10)
            else:
                losses.append(loss[0,0])
    np.set_printoptions(precision=3)
    print(np.array(losses).reshape(11,11))

    calculate_loss(model, X, Y, 1, 1)
    calculate_loss(model, X, Y, 1.5366, 0.0857)

    w_len = 101 # 分辨率
    w_width = 10
    U = np.linspace(1 - w_width, 1 + w_width, w_len)
    W = np.linspace(1 - w_width, 1 + w_width, w_len)
    U, W = np.meshgrid(U, W)

    h1 = np.dot(X[:,0:1], U.ravel().reshape(1, w_len*w_len))
    h2 = np.dot(X[:,1:2], U.ravel().reshape(1, w_len*w_len))
    hp1 = np.multiply(h1, W.ravel().reshape(1, w_len*w_len))
    Z = hp1 + h2
    Loss = (Z - Y) ** 2
    Loss = Loss.sum(axis=0, keepdims=True)/m/2
    Loss = Loss.reshape(w_len, w_len)/3e+8

    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(U, W, Loss, norm=LogNorm(), cmap='rainbow')
    ax.set_xlabel("U")
    ax.set_ylabel("W")
    ax.set_zlabel("MSE")

    ax = fig.add_subplot(1, 2, 2)
    ax.contour(U, W, Loss,  norm=LogNorm(), cmap=plt.cm.jet)
    ax.set_xlabel("U")
    ax.set_ylabel("W")
    ax.grid()

    plt.show()

