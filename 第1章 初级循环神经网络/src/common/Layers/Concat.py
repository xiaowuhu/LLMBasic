import numpy as np
from .Operator import Operator

# 把输入的矩阵横向拼接
class Concat(Operator):
    def __init__(self, modules, input_size, output_size):
        self.modules = list(modules)
        self.input_size = list(input_size)
        self.output_size = list(output_size)
        assert(len(self.modules) == len(self.input_size) == len(self.output_size))
        self.slice_input = []
        start_idx = 0
        for i in range(len(self.modules)):
            end_idx = start_idx + self.input_size[i]
            self.slice_input.append((start_idx, end_idx))
            start_idx = end_idx

        self.slice_output = []
        start_idx = 0
        for i in range(len(self.modules)):
            end_idx = start_idx + self.output_size[i]
            self.slice_output.append((start_idx, end_idx))
            start_idx = end_idx


    def forward(self, X, is_debug=False):
        outputs = []
        for i, module in enumerate(self.modules):
            output = module.forward(X[:, self.slice_input[i][0]:self.slice_input[i][1]], is_debug)
            outputs.append(output)
        output = np.hstack(outputs)
        return output

    def backward(self, delta_in):
        for i, module in enumerate(self.modules):
            # 由于前面没有其它网络，所以这个delta_out可丢弃
            delta_out = module.backward(delta_in[:, self.slice_output[i][0]:self.slice_output[i][1]])

    def update(self, lr): # lr = learning rate 学习率
        for module in self.modules:
            module.update(lr)

    def save(self, name):
        for module in self.modules:
            module.save(name)

    def load(self, name):
        for module in self.modules:
            module.load(name)
