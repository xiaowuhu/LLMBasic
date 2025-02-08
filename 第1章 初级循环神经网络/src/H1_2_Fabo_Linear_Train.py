import os
import numpy as np

from common.DataLoader_14 import DataLoader_14
import common.Layers as layers
from common.Module import Sequential
from common.HyperParameters import HyperParameters
import matplotlib.pyplot as plt
import common.LearningRateScheduler as LRScheduler

from H1_1_Train_Base import train_model

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def load_data(train_file_name, test_file_name):
    train_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", train_file_name)
    test_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", test_file_name)
    data_loader = DataLoader_14(train_file_path, test_file_path)
    data_loader.load_txt_data()
    # data_loader.MinMaxScaler_X()
    # data_loader.MinMaxScaler_Y()
    data_loader.shuffle_data()
    #data_loader.split_data(0.8)
    return data_loader


def test_model(data_loader: DataLoader_14, model: Sequential):
    test_x, test_y = data_loader.get_test()
    #test_x = data_loader.MinMaxScaler_pred_X(test_x)
    predict = model.forward(test_x)
    #y = data_loader.de_MinMaxScaler_Y(predict)
    np.set_printoptions(precision=0, suppress=True)
    for i in range(len(predict)):
        print(test_x[i], "=>", predict[i], " vs ", test_y[i])

def build_model():
    model = Sequential(
        layers.Linear(2, 1, init_method="normal", optimizer="Adam", has_bias=False),
    )
    model.set_loss_function(layers.MSE()) # 多分类函数+交叉熵损失函数
    return model

if __name__=="__main__":
    model = build_model()
    data_loader = load_data("fabo_train.txt", "fabo_test.txt")
    params = HyperParameters(max_epoch=1000, batch_size=8, learning_rate=0.1)
    lrs = LRScheduler.step_lrs(0.1, 0.95, 100)
    training_history = train_model(data_loader, model, params, lrs, checkpoint=1)
    training_history.show_loss()
    model.save("model_fabo_linear_nn")
    
    #show_result(data_loader)
    # test
    # model = build_model()
    #model.load("model_fabo_linear_0")
    test_model(data_loader, model)
