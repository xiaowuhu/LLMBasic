
from matplotlib import pyplot as plt
import numpy as np
import os
import math
import common.Layers as layers
from common.DataLoader_14 import DataLoader_14
from common.HyperParameters import HyperParameters
from common.Module import Sequential
import common.LearningRateScheduler as LRScheduler

from H17_1_Train_Base import train_model, load_data

class RNN_Simple2(layers.Operator):
    def __init__(self, num_input, num_hidden, num_output,
                 init_method: str='normal',
                 optimizer: str="SGD",
                 has_bias=False  # default is False for this simple RNN
    ):
        self.has_bias = has_bias
        self.U = layers.WeightsBias(num_input, num_hidden, init_method, optimizer)        
        self.W = layers.WeightsBias(num_hidden, num_hidden, init_method, optimizer)        

    def get_parameters(self):
        return self.U, self.W

    def forward(self, xt):
        self.x1 = xt[:, 0:1]
        self.x2 = xt[:, 1:2]
        # 式（1.2.5）
        self.h1 = np.dot(self.x1, self.U.Weight)
        self.h2 = np.dot(self.x2, self.U.Weight)
        self.hp1 = np.dot(self.h1, self.W.Weight)
        z = self.hp1 + self.h2
        return z

    def backward(self, dz):
        # 式（1.2.6）
        dW = np.dot(self.h1.T, dz)
        dU = np.dot(self.x1.T, np.dot(dz, self.W.Weight.T)) + np.dot(self.x2.T, dz)
        self.U.dW = dU
        self.W.dW = dW

    def load(self, name):
        U = super().load_from_txt_file(name + "_U")
        if U.ndim == 1:
            U = U.reshape(-1, 1)
        self.U.set_WB_value(U)

        W = super().load_from_txt_file(name + "_W")
        if W.ndim == 1:
            W = W.reshape(-1, 1)
        self.W.set_WB_value(W)
    
    def save(self, name):
        U = self.U.get_WB_value()
        super().save_to_txt_file(name + "_U", U)
        W = self.W.get_WB_value()
        super().save_to_txt_file(name + "_W", W)

def build_model(num_input, num_hidden, num_output):
    model = Sequential(
        RNN_Simple2(
            num_input, num_hidden, num_output,
            init_method="normal", optimizer="Adam", has_bias=False
        )
    )
    model.set_loss_function(layers.MSE())
    return model

def test_model(data_loader: DataLoader_14, model: Sequential):
    test_x, test_y = data_loader.get_test()
    pred = model.forward(test_x)
    np.set_printoptions(precision=0, suppress=True)
    for i in range(len(pred)):
        print(test_x[i], "=>", pred[i], " vs ", test_y[i])

def test_model_n_step(data_loader: DataLoader_14, model: Sequential):
    test_x, test_y = data_loader.get_test()
    np.set_printoptions(precision=0, suppress=True)
    test_x_step = test_x[24:25]
    for i in range(6):
        pred = model.forward(test_x_step)
        print("step", i+1, "=>", pred)
        test_x_step[0,0] = test_x_step[0,1]
        test_x_step[0,1] = pred[0,0]

if __name__=='__main__':
    num_input, num_hidden, num_output = 1, 1, 1
    model = build_model(num_input, num_hidden, num_output)
    data_loader = load_data("fabo_train.txt", "fabo_test.txt")
    params = HyperParameters(max_epoch=150, batch_size=8, learning_rate=0.01)
    lrs = LRScheduler.step_lrs(0.05, 0.95, 10)
    training_history = train_model(data_loader, model, params, lrs, checkpoint=1) #, name="model_fabo_rnn_best")
    training_history.show_loss()
    #model.save("model_fabo_rnn")
    #model.load("model_fabo_rnn_best")
    test_model(data_loader, model)
    test_model_n_step(data_loader, model)
