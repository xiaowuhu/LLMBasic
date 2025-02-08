import numpy as np
from .Operator import Operator

class Tanh(Operator):
    def __call__(self, z):
        return self.forward(z)

    def forward(self, z):
        self.a = 2.0 / (1.0 + np.exp(-2 * z)) - 1.0
        return self.a

    # 用 z 也可以计算，但是不如 a 方便
    def backward(self, delta_in):
        da = 1 - np.multiply(self.a, self.a) # 导数
        delta_out = np.multiply(delta_in, da)
        return delta_out

class Sigmoid(Operator):
    def forward(self, z):
        self.z = z # 因为反向时不需要z，所以这里可以不保存
        self.a = 1.0 / (1.0 + np.exp(-self.z))
        return self.a

    # 用 z 也可以计算，但是不如 a 方便
    def backward(self, delta_in):
        da = np.multiply(self.a, 1 - self.a) # 导数
        delta_out = np.multiply(delta_in, da)
        return delta_out

class Relu(Operator):
    def __call__(self, z):
        return self.forward(z)

    def forward(self, z):
        self.z = z
        a = np.maximum(0, z)
        return a

    # 注意这里判断的是输入时 z 的情况
    def backward(self, delta_in):
        da = np.where(self.z > 0, 1, 0).astype(np.float32)
        delta_out = np.multiply(delta_in, da)
        return delta_out
