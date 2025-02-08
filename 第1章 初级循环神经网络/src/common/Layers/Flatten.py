import numpy as np
from .Operator import Operator
from .WeightsBias import WeightsBias


# 线性映射层
class Flatten(Operator):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = np.prod(output_shape)

    def forward(self, input):
        self.m = input.shape[0]
        output = np.reshape(input, (self.m, self.output_shape))
        return output
    
    def backward(self, delta_in):
        return np.reshape(delta_in, (self.m, *self.input_shape))
