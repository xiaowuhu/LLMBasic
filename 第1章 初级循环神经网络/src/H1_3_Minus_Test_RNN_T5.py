
from matplotlib import pyplot as plt
import numpy as np
import os
import math
import common.Layers as layers
from common.DataLoader_14 import DataLoader_14
from common.HyperParameters import HyperParameters
from common.Module import Sequential
import common.LearningRateScheduler as LRScheduler
from common.Estimators import tpn2

from H17_1_Train_Base import train_model, load_data

class RNN_Simple5(layers.Operator):
    def __init__(self, num_input, num_hidden, num_output,
                 init_method: str='normal',
                 optimizer: str="SGD",
                 has_bias=False  # default is False for this simple RNN
    ):
        self.has_bias = has_bias
        self.U = layers.WeightsBias(num_input, num_hidden, init_method, optimizer)
        self.W = layers.WeightsBias(num_hidden, num_hidden, init_method, optimizer)
        self.V = layers.WeightsBias(num_hidden, num_output, init_method, optimizer)
        # Tanh的正向计算时记录了结果a在反向时使用，所以分开成4个，而不能共用一个
        self.tanh1 = layers.Tanh() 
        self.tanh2 = layers.Tanh()
        self.tanh3 = layers.Tanh()
        self.tanh4 = layers.Tanh()
        self.tanh5 = layers.Tanh()
        self.logit = layers.Logisitic()
        self.loss_func = layers.BCE()
        self.set_parameters(self.U, self.V, self.W)
        self.dw_norm = []

    def forward(self, x):
        self.x1 = x[:, 0]
        self.x2 = x[:, 1]
        self.x3 = x[:, 2]
        self.x4 = x[:, 3]
        self.x5 = x[:, 4]
        # t1 式（1.3.1）
        self.h1 = np.dot(self.x1, self.U.Weight)
        self.s1 = self.tanh1(self.h1)
        self.z1 = np.dot(self.s1, self.V.Weight)
        self.a1 = self.logit(self.z1)
        # t2 式（1.3.2）
        self.h2 = np.dot(self.x2, self.U.Weight) + np.dot(self.s1, self.W.Weight)
        self.s2 = self.tanh2(self.h2)
        self.z2 = np.dot(self.s2, self.V.Weight)
        self.a2 = self.logit(self.z2)
        # t3 式（1.3.2）
        self.h3 = np.dot(self.x3, self.U.Weight) + np.dot(self.s2, self.W.Weight)
        self.s3 = self.tanh3(self.h3)
        self.z3 = np.dot(self.s3, self.V.Weight)
        self.a3 = self.logit(self.z3)
        # t4 式（1.3.2）
        self.h4 = np.dot(self.x4, self.U.Weight) + np.dot(self.s3, self.W.Weight)
        self.s4 = self.tanh4(self.h4)
        self.z4 = np.dot(self.s4, self.V.Weight)
        self.a4 = self.logit(self.z4)
        # t5 式（1.3.2）
        self.h5 = np.dot(self.x5, self.U.Weight) + np.dot(self.s4, self.W.Weight)
        self.s5 = self.tanh5(self.h5)
        self.z5 = np.dot(self.s5, self.V.Weight)
        self.a5 = self.logit(self.z5)

        return np.hstack((self.a1, self.a2, self.a3, self.a4, self.a5))

    def compute_loss_accuracy(self, x, y):
        predict = self.forward(x)
        loss1 = self.loss_func(predict[:,0:1], y[:,0:1])
        accu1 = tpn2(predict[:,0:1], y[:,0:1])
        loss2 = self.loss_func(predict[:,1:2], y[:,1:2])
        accu2 = tpn2(predict[:,1:2], y[:,1:2])
        loss3 = self.loss_func(predict[:,2:3], y[:,2:3])
        accu3 = tpn2(predict[:,2:3], y[:,2:3])
        loss4 = self.loss_func(predict[:,3:4], y[:,3:4])
        accu4 = tpn2(predict[:,3:4], y[:,3:4])
        loss5 = self.loss_func(predict[:,4:5], y[:,4:5])
        accu5 = tpn2(predict[:,4:5], y[:,4:5])
        loss = (loss1 + loss2 + loss3 + loss4 + loss5)/6
        accu = (accu1 + accu2 + accu3 + accu4 + accu5)/6
        return loss, accu

    def set_parameters(self, *params):
        self.paramters_dict = {}
        unique_id = 0
        for wb in params:
            op_name = self.__class__.__name__ + "_" + str(unique_id)
            if wb is not None:
                self.paramters_dict[op_name] = wb
            unique_id += 1  

    def load(self, name):
        U = super().load_from_txt_file(name + "_U")
        if U.ndim == 1:
            U = U.reshape(-1, 1)
        self.U.set_WB_value(U)

        W = super().load_from_txt_file(name + "_W")
        if W.ndim == 1:
            W = W.reshape(-1, 1)
        self.W.set_WB_value(W)

        V = super().load_from_txt_file(name + "_V")
        if V.ndim == 1:
            V = V.reshape(-1, 1)
        self.V.set_WB_value(V)

    def save(self, name):
        super().save_to_txt_file(name + "_U", self.U.get_WB_value())
        super().save_to_txt_file(name + "_W", self.W.get_WB_value())
        super().save_to_txt_file(name + "_V", self.V.get_WB_value())

def load_data(train_file_name, test_file_name):
    train_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", train_file_name)
    test_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", test_file_name)
    data_loader = DataLoader_14(train_file_path, test_file_path)
    data_loader.load_npz_data()
    data_loader.shuffle_data()
    return data_loader

def reverse(a):
    l = a.tolist()
    l.reverse()
    return l

def test_model(data_loader: DataLoader_14, model):
    test_x, test_y = data_loader.get_test()
    pred = np.round(model.forward(test_x)).astype(np.int64)
    count = 0
    error = 0
    for i in range(len(pred)):
        if np.allclose(pred[i], test_y[i]) == True:
            count += 1
        else:
            error += 1 
            if error < 10:
                print("  x1:", reverse(test_x[i,:,0].astype(np.int64)))
                print("- x2:", reverse(test_x[i,:,1].astype(np.int64)))
                print("------------------")
                print("pred:", reverse(pred[i]))
                print("true:", reverse(test_y[i]))
                print("====================")                
    print("Accuracy: {}/{} ({:.2f}%)".format(count, len(pred), count/len(pred)*100))

def test(model):
    np.set_printoptions(precision=1)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    for i in range(4):
        h = np.dot(X[i], model.U.Weight)
        s = model.tanh1(h)
        z = np.dot(s, model.V.Weight)
        a = model.logit(z)
        t = np.dot(s, model.W.Weight)    
        print(X[i], "h:",h, "\ts:", s, "\tz:", z, "\ta:", a, "\tt:", t)



if __name__=='__main__':
    num_input, num_hidden, num_output = 2, 3, 1
    data_loader = load_data("minus_test_bin_5.npz", "minus_test_bin_5.npz")
    model = RNN_Simple5(num_input, num_hidden, num_output, optimizer="SGD")
    model.load("model_minus_rnn_best")
    test_model(data_loader, model)
    test(model)
