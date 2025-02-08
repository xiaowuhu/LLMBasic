import numpy as np
from .Operator import Operator
from .WeightsBias import WeightsBias
import common.Optimizers as Optimizers

class BatchNorm2d(Operator):
    def __init__(self, input_size, momentum=0.9):
        self.WB = Bn2dWeightBias(input_size, init_method="kaiming", optimizer="SGD")
        self.WB.W.fill(1) # 初始值设置为 1
        self.WB.B.fill(0) # 初始值设置为 0
        self.eps = 1e-5
        self.input_size = input_size
        self.output_size = input_size
        self.momentum = momentum
        self.running_mean = np.zeros((1,input_size,1,1))
        self.running_var = np.zeros((1,input_size,1,1))

    def forward(self, input):
        assert(input.ndim == 4)  # fc or cv
        self.x = input
        # 式(12.7.1)
        self.mu = np.mean(self.x, axis=(0,2,3), keepdims=True)
        # 式(12.7.2)
        self.x_mu  = self.x - self.mu
        self.var = np.mean(self.x_mu**2, axis=(0,2,3), keepdims=True) + self.eps
        # 式(12.7.3)
        self.std = np.sqrt(self.var)
        self.y = self.x_mu / self.std
        # 式(12.7.4)
        self.z = self.WB.W * self.y + self.WB.B
        # mean and var history, for test/inference
        self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * self.mu
        self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * self.var
        return self.z

    def predict(self, input):
        y = (input - self.running_mean) / np.sqrt(self.running_var + self.eps)
        z = self.WB.W * y + self.WB.B
        return z
    
    def backward(self, dZ):
        assert(dZ.ndim == 4)  # fc or cv
        m = self.x.shape[0] * self.x.shape[2] * self.x.shape[3]
        # 计算参数梯度
        # 式(12.7.5)
        self.WB.dW = np.sum(dZ * self.y, axis=(0,2,3), keepdims=True)
        # 式(12.7.6)
        self.WB.dB = np.sum(dZ, axis=(0,2,3), keepdims=True)
        # 计算输出梯度
        # 式（12.7.8）
        d_y = self.WB.W * dZ 
        # 式（12.7.10）
        d_var = -0.5 * np.sum(d_y * self.x_mu, axis=(0,2,3), keepdims=True) / (self.var * self.std) # == self.var ** (-1.5)
        # 式（12.7.12）
        d_mu = -np.sum(d_y / self.std, axis=(0,2,3), keepdims=True) \
               -2 * d_var * np.sum(self.x_mu, axis=(0,2,3), keepdims=True) / m
        # 式（12.7.7）
        dX = d_y / self.std + d_var * 2 * self.x_mu / m + d_mu / m
        return dX
    
    def load(self, name):
        wb_value = super().load_from_txt_file(name)
        self.WB.set_WB_value(wb_value)
    
    def save(self, name):
        wb_value = self.WB.get_WB_value()
        super().save_to_txt_file(name, wb_value)    

    def get_parameters(self):
        return self.WB

class Bn2dWeightBias(WeightsBias):
    def __init__(self, input_size, init_method="normal", optimizer="SGD"):
        self.input_size = input_size  # scalar
        self.W_shape = (1, input_size, 1, 1)
        self.B_shape = (1, input_size, 1, 1)
        self.W, self.B = self.create(self.W_shape, self.B_shape, init_method)
        self.dW = np.zeros(self.W.shape)
        self.dB = np.zeros(self.B.shape)
        self.opt_W = Optimizers.Optimizer.create_optimizer(optimizer)
        self.opt_B = Optimizers.Optimizer.create_optimizer(optimizer)

    def create(self, w_shape, b_shape, method):
        assert(len(w_shape) == 4)
        num_input = w_shape[3]
        num_output = 1
        
        if method == "zero":
            W = np.zeros(w_shape)
        elif method == "normal":
            W = np.random.normal(0, 1, w_shape)
        elif method == "kaiming":
            W = np.random.normal(0, np.sqrt(2/num_input*num_output), w_shape)
        elif method == "xavier":
            t = np.sqrt(6/(num_output+num_input))
            W = np.random.uniform(-t, t, w_shape)
        
        B = np.zeros(b_shape)
        return W, B

    # 因为w,b 的形状不一样，需要reshape成 n行1列的，便于保存
    def get_WB_value(self):
        value = np.concatenate((self.W.reshape(-1,1), self.B.reshape(-1,1)))
        return value

    def get_dWB_value(self):
        value = np.concatenate((self.dW.reshape(-1,1), self.dB.reshape(-1,1)))
        return value

    # 先把一列数据读入，然后变形成为对应的W,B的形状
    def set_WB_value(self, value):
        self.W = value[0:self.input_size].reshape(self.W_shape)
        self.B = value[self.input_size:].reshape(self.B_shape)

    def set_dWB_value(self, value):        
        self.dW = value[0:self.input_size].reshape(self.W_shape)
        self.dB = value[self.input_size:].reshape(self.B_shape)
