import numpy as np
from .Operator import Operator


# 一维池化
class Pool1d(Operator):
    def __init__(self, 
                 input_shape,  # (input_channel, input_w)
                 pool_len,       # pool width
                 stride,
                 padding = 0,
                 pool_type = "max", # or "mean"
    ):
        self.input_channel = input_shape[0]
        self.input_length = input_shape[1]
        self.pool_length = pool_len
        self.stride = stride
        self.padding = padding
        self.output_length = 1 + (self.input_length + 2 * self.padding - self.pool_length) // self.stride

    def forward(self, x):
        self.x = x
        assert(self.input_length == self.x.shape[2])
        assert(self.x.shape[1] == self.input_channel)
        self.m = self.x.shape[0]
        self.z = np.zeros((self.m, self.input_channel, self.output_length))
        self.argmax = np.zeros((self.m, self.input_channel, self.output_length)).astype(np.int64)
        for i in range(self.m):
            for in_c in range(self.input_channel):
                for j in range(self.output_length):
                    start = j * self.stride
                    end = start + self.pool_length
                    data_window = self.x[i, in_c, start:end]
                    self.z[i, in_c, j] = np.max(data_window)
                    self.argmax[i, in_c, j] = np.argmax(data_window)
        return self.z

    def backward(self, delta_in):
        delta_out = np.zeros_like(self.x)
        for i in range(self.m):
            for in_c in range(self.input_channel):
                for j in range(self.output_length):
                    pos = self.argmax[i, in_c, j]
                    delta_out[i, in_c, j * self.stride+pos] = delta_in[i, in_c, j]
        return delta_out
