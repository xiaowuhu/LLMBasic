import numpy as np
from .Operator import Operator
from .WeightsBias import WeightsBias


# 线性映射层
class Linear(Operator):
    def __init__(self, 
                 input_size: int, 
                 output_size: int, 
                 init_method: str='normal',
                 optimizer: str="SGD",
                 regularizer: tuple = ("None", 0.1),
                 has_bias=True
    ): # W 是权重，B 是偏移
        self.WB = WeightsBias(input_size, output_size, init_method, optimizer)
        self.regular_name = regularizer[0]
        self.regular_value = regularizer[1]
        self.has_bias = has_bias

    def get_parameters(self):
        return self.WB

    def forward(self, input): # input 是输入
        self.input = input
        if self.has_bias:
            return np.dot(self.input, self.WB.Weight) + self.WB.Bias
        else:
            return np.dot(self.input, self.WB.Weight)

    def backward(self, delta_in): # delta 是反向传播的梯度
        m = self.input.shape[0]
        if self.regular_name == "L2":
            self.WB.dW = (np.dot(self.input.T, delta_in) + self.regular_value * self.WB.Weight) / m
        elif self.regular_name == "L1":
            self.WB.dW = (np.dot(self.input.T, delta_in) + self.regular_value * np.sign(self.WB.Weight))/m
        else:
            self.WB.dW = np.dot(self.input.T, delta_in) / m
    
        if self.has_bias:
            self.WB.dB = np.mean(delta_in, axis=0, keepdims=True)
        delta_out = np.dot(delta_in, self.WB.Weight.T)  # 传到前一层的梯度

        return delta_out
    
    def get_regular_cost(self):
        if self.regular_name == "L1":
            return np.sum(np.abs(self.WB.Weight)) * self.regular_value
        elif self.regular_name == "L2":
            return np.sum(np.square(self.WB.Weight)) * self.regular_value
        else:
            return 0

    def load(self, name):
        WB = super().load_from_txt_file(name)
        if WB.ndim == 1:
            WB = WB.reshape(-1, 1)
        self.WB.set_WB_value(WB)
    
    def save(self, name):
        WB = self.WB.get_WB_value()
        super().save_to_txt_file(name, WB)
