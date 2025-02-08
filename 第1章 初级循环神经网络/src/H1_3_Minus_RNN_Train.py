
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

class RNN_Simple4(layers.Operator):
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
        self.logit = layers.Logisitic()
        self.loss_func = layers.BCE()
        self.set_parameters(self.U, self.V, self.W)
        self.dw_norm = []

    def forward(self, x):
        self.batch_size = x.shape[0]
        self.x1 = x[:, 0]
        self.x2 = x[:, 1]
        self.x3 = x[:, 2]
        self.x4 = x[:, 3]
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

        return np.hstack((self.a1, self.a2, self.a3, self.a4))

    def backward(self, a, y):
        # 式（1.3.4）
        dz = a - y
        dz1 = dz[:,0:1]
        dz2 = dz[:,1:2]
        dz3 = dz[:,2:3]
        dz4 = dz[:,3:4]
        # t4 式（1.3.5）
        ds4 = np.dot(dz4, self.V.Weight.T)
        dh4 = self.tanh4.backward(ds4)
        # t3 式（1.3.6）
        ds3 = np.dot(dz3, self.V.Weight.T) + np.dot(dh4, self.W.Weight.T)
        dh3 = self.tanh3.backward(ds3)
        # t2 式（1.3.6）
        ds2 = np.dot(dz2, self.V.Weight.T) + np.dot(dh3, self.W.Weight.T)
        dh2 = self.tanh2.backward(ds2)
        # t1 式（1.3.6）
        ds1 = np.dot(dz1, self.V.Weight.T) + np.dot(dh2, self.W.Weight.T)
        dh1 = self.tanh1.backward(ds1)
        # 式（1.3.7）
        dV4 = np.dot(self.s4.T, dz4)
        dV3 = np.dot(self.s3.T, dz3)
        dV2 = np.dot(self.s2.T, dz2)
        dV1 = np.dot(self.s1.T, dz1)
        self.V.dW = (dV1 + dV2 + dV3 + dV4) / self.batch_size
        # 式（1.3.8）
        dU4 = np.dot(self.x4.T, dh4)
        dU3 = np.dot(self.x3.T, dh3)
        dU2 = np.dot(self.x2.T, dh2)
        dU1 = np.dot(self.x1.T, dh1)
        self.U.dW = (dU1 + dU2 + dU3 + dU4) / self.batch_size
        # 式（1.3.9）
        dW4 = np.dot(self.s3.T, dh4)
        dW3 = np.dot(self.s2.T, dh3)
        dW2 = np.dot(self.s1.T, dh2)
        self.W.dW = (dW2 + dW3 + dW4) / self.batch_size
        # for testing only
        dw_norm = np.linalg.norm(self.W.dW)
        if dw_norm > 1:  # 梯度剪裁
            self.W.dW = self.W.dW / dw_norm
        self.dw_norm.append(np.linalg.norm(self.W.dW))

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
        loss = (loss1 + loss2 + loss3 + loss4)/4
        accu = (accu1 + accu2 + accu3 + accu4)/4
        return loss, accu

    def set_parameters(self, *params):
        self.paramters_dict = {}
        unique_id = 0
        for wb in params:
            op_name = self.__class__.__name__ + "_" + str(unique_id)
            if wb is not None:
                self.paramters_dict[op_name] = wb
            unique_id += 1
    
    def update(self, lr):
        for _, WB in self.paramters_dict.items():
            if isinstance(WB, tuple):
                for wb in WB:
                    wb.Update(lr)
            else: # WeightsBias object
                WB.Update(lr)        

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
            if error < 2:
                print("  x1:", reverse(test_x[i,:,0].astype(np.int64)))
                print("- x2:", reverse(test_x[i,:,1].astype(np.int64)))
                print("------------------")
                print("pred:", reverse(pred[i]))
                print("true:", reverse(test_y[i]))
                print("====================")                
    print("Accuracy: {}/{} ({:.2f}%)".format(count, len(pred), count/len(pred)*100))

# 12-5=7
def test_model_12_5_7(model: Sequential):
    np.set_printoptions(precision=1)

    x = np.array([[0,1],[0,0],[1,1],[1,0]])
    y = np.array([[1,1,1,0]])
    # t1
    h1 = np.dot(x[0], model.U.Weight)
    s1 = model.tanh1(h1)
    z1 = np.dot(s1, model.V.Weight)
    a1 = model.logit(z1)
    t1 = np.dot(s1, model.W.Weight)    
    print(x[0], "-> h1:",h1, "\ts1:", s1, "\tz1:", z1, "\ta1:", a1, "\tt1:", t1)
    # t2
    h2 = np.dot(x[1], model.U.Weight) + t1
    s2 = model.tanh2(h2)
    z2 = np.dot(s2, model.V.Weight)
    a2 = model.logit(z2)
    t2 = np.dot(s2, model.W.Weight)    
    print(x[1], "-> h2:",h2, "\ts2:", s2, "\tz2:", z2, "\ta2:", a2, "\tt2:", t2)
    # t3
    h3 = np.dot(x[2], model.U.Weight) + t2
    s3 = model.tanh3(h3)
    z3 = np.dot(s3, model.V.Weight)
    a3 = model.logit(z3)
    t3 = np.dot(s3, model.W.Weight)    
    print(x[2], "-> h3:",h3, "\ts3:", s3, "\tz3:", z3, "\ta3:", a3, "\tt3:", t3)
    # t4
    h4 = np.dot(x[3], model.U.Weight) + t3
    s4 = model.tanh4(h4)
    z4 = np.dot(s4, model.V.Weight)
    a4 = model.logit(z4)
    print(x[3], "-> h4:",h4, "\ts4:", s4, "\tz4:", z4, "\ta4:", a4)


if __name__=='__main__':
    num_input, num_hidden, num_output = 2, 3, 1
    model = RNN_Simple4(num_input, num_hidden, num_output, optimizer="SGD")
    data_loader = load_data("minus_train_bin.npz", "minus_train_bin.npz")
    params = HyperParameters(max_epoch=120, batch_size=8)
    lrs = LRScheduler.step_lrs(0.1, 0.95, 1000)
    training_history = train_model(data_loader, model, params, lrs, checkpoint=1) #, name="model_minus_rnn_best")
    training_history.show_loss(yscale="linear")
    #model.load("model_minus_rnn_best")
    test_model(data_loader, model)
    #test_model_n_step(data_loader, model)
    # 绘制 dw_norm 的变化
    # dw_norm = np.array(model.dw_norm)
    # print(np.mean(dw_norm))
    # plt.plot(dw_norm)
    # plt.grid()
    # plt.show()
    # 解析 12-5=7 的计算过程
    test_model_12_5_7(model)
