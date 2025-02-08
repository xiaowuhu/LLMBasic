import os
import math
import numpy as np

from common.DataLoader_14 import DataLoader_14
import common.Layers as layers
from common.Module import Sequential
from common.HyperParameters import HyperParameters
import common.LearningRateScheduler as LRScheduler
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

from H17_1_Train_Base import train_model, load_data

def test_model(data_loader: DataLoader_14, model: Sequential):
    test_x, test_y = data_loader.get_test()
    pred = model.forward(test_x)
    np.set_printoptions(precision=0, suppress=True)
    for i in range(len(pred)):
        print(test_x[i,0], "-", test_x[i,1], "=", pred[i], " vs ", test_y[i])

def build_model():
    model = Sequential(
        layers.Linear(2, 1, optimizer="Adam", has_bias=False),
    )
    model.set_loss_function(layers.MSE()) 
    return model

if __name__=="__main__":
    model = build_model()
    data_loader = load_data("minus_train_oct.txt", "minus_test_oct.txt")
    params = HyperParameters(max_epoch=50, batch_size=8, learning_rate=0.1)
    lrs = LRScheduler.step_lrs(0.05, 0.95, 10)
    # 保存最佳结果
    training_history = train_model(data_loader, model, params, lrs, checkpoint=1) #, name="model_minus_best")
    training_history.show_loss()
    # test
    model.load("model_minus_best")
    test_model(data_loader, model)
