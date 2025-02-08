import os
import math
import numpy as np

from common.DataLoader_14 import DataLoader_14
import common.Layers as layers
from common.Module import Sequential
from common.HyperParameters import HyperParameters
from common.TrainingHistory import TrainingHistory
import matplotlib.pyplot as plt
import common.LearningRateScheduler as LRScheduler

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

from H17_1_Train_Base import train_model

def load_data(train_file_name, test_file_name):
    train_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", train_file_name)
    test_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", test_file_name)
    data_loader = DataLoader_14(train_file_path, test_file_path)
    data_loader.load_txt_data()
    data_loader.shuffle_data()
    data_loader.split_data(0.8)
    return data_loader



def test_model(data_loader: DataLoader_14, model: Sequential):
    test_x, label = data_loader.get_test()
    test_loss, test_accu = model.compute_loss_accuracy(test_x, label)
    print("测试集: loss %.6f, accu %.4f" %(test_loss, test_accu))

def build_model():
    model = Sequential(
        layers.Linear(1, 4, init_method="normal", optimizer="Adam"),
        layers.Tanh(),
        layers.Linear(4, 1)
    )
    model.set_loss_function(layers.MSE())
    return model

def test_expand(model: Sequential):
    # 生成超过 2pi 的数据
    test_x = np.linspace(0, 4*np.pi, 37).reshape(-1, 1)
    test_y = np.sin(test_x)
    pred = model.predict(test_x)
    plt.scatter(test_x, test_y)
    plt.scatter(test_x, pred)
    plt.show()

if __name__=="__main__":
    model = build_model()
    data_loader = load_data("sin_train_linear.txt", "sin_test_linear.txt")
    params = HyperParameters(max_epoch=50, batch_size=8, learning_rate=0.1)
    lrs = LRScheduler.step_lrs(0.1, 0.95, 100)
    training_history = train_model(data_loader, model, params, lrs, checkpoint=1) #, name="model_sin_linear_best")
    training_history.show_loss()
    