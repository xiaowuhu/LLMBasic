import os
import math
import numpy as np

from common.DataLoader_14 import DataLoader_14
import common.Layers as layers
from common.Module import Sequential
from common.HyperParameters import HyperParameters
from common.TrainingHistory import TrainingHistory
import matplotlib.pyplot as plt

from H17_5_Sin_FW_Train import load_data, build_model

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=9)


def test_model(data_loader: DataLoader_14, model: Sequential):
    test_x, label = data_loader.get_test()
    test_loss, test_accu = model.compute_loss_accuracy(test_x, label)
    print("测试集: loss %.6f, accu %.4f" %(test_loss, test_accu))



def test_expand(data_loader: DataLoader_14, model: Sequential):
    test_x, test_y = data_loader.get_test()
    plt.scatter(test_x, test_y, marker='o', label="测试样本")

    # 生成 [0,2pi] 之间的任意数据
    test_x = np.array([0.02, 0.23, 0.41, 0.67, 0.92]) * 2 * np.pi
    pred = np.sin(test_x)
    plt.scatter(test_x, pred, marker="x", s=50, label="任意 $x$ 的预测结果")

    # 生成超过 2pi 的数据
    test_x = np.linspace(0, 4*np.pi, 41).reshape(-1, 1)
    pred = model.predict(test_x)
    plt.scatter(test_x, pred, marker='.', label="扩展测试")
    plt.grid()
    plt.legend()
    plt.show()

if __name__=="__main__":
    model = build_model()
    data_loader = load_data("sin_train_linear.txt", "sin_test_linear.txt")
    model.load("model_sin_linear_best")
    test_model(data_loader, model)
    test_expand(data_loader, model)
