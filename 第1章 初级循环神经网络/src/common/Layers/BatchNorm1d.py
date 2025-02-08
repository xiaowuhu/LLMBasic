import numpy as np
from .Operator import Operator
from .WeightsBias import WeightsBias

class BatchNorm1d(Operator):
    def __init__(self, input_size, momentum=0.9):
        self.WB = WeightsBias(1, input_size, init_method="kaiming", optimizer="SGD")
        self.WB.Weight.fill(1)  # 初始值设置为 1
        self.WB.Bias.fill(0)  # 初始值设置为 0
        self.eps = 1e-5
        self.input_size = input_size
        self.output_size = input_size
        self.momentum = momentum
        self.running_mean = np.zeros((1,input_size))
        self.running_var = np.zeros((1,input_size))

    def forward(self, input):
        assert(input.ndim == 2)  # fc or cv
        self.x = input
        # 式(12.7.1)
        self.mu = np.mean(self.x, axis=0, keepdims=True)
        # 式(12.7.2)
        self.x_mu  = self.x - self.mu
        self.var = np.mean(self.x_mu**2, axis=0, keepdims=True) + self.eps
        # 式(12.7.3)
        self.std = np.sqrt(self.var)
        self.y = self.x_mu / self.std
        # 式(12.7.4)
        self.z = self.WB.Weight * self.y + self.WB.Bias
        # mean and var history, for test/inference
        self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * self.mu
        self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * self.var
        return self.z

    def predict(self, input):
        y = (input - self.running_mean) / np.sqrt(self.running_var + self.eps)
        z = self.WB.Weight * y + self.WB.Bias
        return z
    
    def backward(self, dZ):
        assert(dZ.ndim == 2)  # fc or cv
        m = self.x.shape[0]
        # 计算参数梯度
        # 式(12.7.5)
        self.WB.dW = np.sum(dZ * self.y, axis=0, keepdims=True)
        # 式(12.7.6)
        self.WB.dB = np.sum(dZ, axis=0, keepdims=True)
        # 计算输出梯度
        # 式（12.7.8）
        d_y = self.WB.Weight * dZ 
        # 式（12.7.10）
        d_var = -0.5 * np.sum(d_y * self.x_mu, axis=0, keepdims=True) / (self.var * self.std) # == self.var ** (-1.5)
        # 式（12.7.12）
        d_mu = -np.sum(d_y / self.std, axis=0, keepdims=True) \
               -2 * d_var * np.sum(self.x_mu, axis=0, keepdims=True) / m
        # 式（12.7.7）
        dX = d_y / self.std + d_var * 2 * self.x_mu / m + d_mu / m
        return dX       

    def save(self, name):
        data = np.vstack((self.WB.Weight, self.WB.Bias, self.running_var, self.running_mean))
        super().save_to_txt_file(name, data)

    def load(self, name):
        data = super().load_from_txt_file(name)
        self.WB.Weight = data[0:1]
        self.WB.Bias = data[1:2]
        self.running_var = data[2:3]
        self.running_mean = data[3:4]

    def get_parameters(self):
        return self.WB
