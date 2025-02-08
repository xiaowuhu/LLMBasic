import numpy as np
from .Operator import Operator

# 丢弃层
class Dropout(Operator):
    def __init__(self, dropout_ratio=0.5):
        assert( 0 < dropout_ratio < 1)
        self.keep_ratio = 1 - dropout_ratio
        self.mask = None

    def forward(self, input):
        assert(input.ndim == 2)
        self.mask = np.random.binomial(n=1, p=self.keep_ratio, size=input.shape)
        self.z = input * self.mask / self.keep_ratio
        return self.z

    def predict(self, input):
        assert(input.ndim == 2)
        self.z = input * self.keep_ratio
        #self.z = input
        return self.z

    def backward(self, delta_in):
        delta_out = self.mask * delta_in / self.keep_ratio
        return delta_out