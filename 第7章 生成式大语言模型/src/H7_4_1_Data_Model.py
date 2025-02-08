import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.nn.functional as F
import copy
from H7_4_Optimizers import Adam


class DataGenerator(object):
    def __init__(self, k, amp=None, phase=None):
        self.k = k  # 正弦曲线的数据点的个数
        # 振幅分布
        self.amplitude = amp if amp is not None else np.random.uniform(0.1, 5.0)
        # 相位分布
        self.phase = phase if phase is not None else np.random.uniform(0, 2 * np.pi)
        self.train_x = torch.tensor(self._sample_x(), dtype=torch.float32).unsqueeze(1)
        self.train_y = self._sine_function(self.train_x)
        self.test_x = torch.tensor(self._sample_x(True), dtype=torch.float32).unsqueeze(1)
        self.test_y = self._sine_function(self.test_x)
    
    def _sample_x(self, in_seq = False):
        # 取的数据点在[-5, 5]之间
        if not in_seq:
            return np.random.uniform(-5, 5, self.k)  # 用于训练
        else:
            return np.linspace(-5, 5, self.k * 10)  # 用于测试

    # 正弦曲线
    def _sine_function(self, x):
        return self.amplitude * np.sin(x - self.phase)

    def get_train_set(self):
        return self.train_x, self.train_y
    
    def get_test_set(self):
        return self.test_x, self.test_y

    def get_support_set(self):
        return self.train_x, self.train_y
    
    def get_query_set(self):
        return self.test_x, self.test_y

class SineModel(nn.Module):
    def __init__(self):
        super(SineModel, self).__init__()
        self.hidden1 = nn.Linear(1, 64)
        self.hidden2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)
        self.total_grad = None
        self.optimizer = {}  # 自定义模型内置优化器
        for name, param in self.named_parameters():
            self.optimizer[name] = Adam()

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.out(x)
        return x

    # 以下这些帮助函数是因为 Pytorch 的反向传播机制的限制
    # 所以需要手动实现元模型的梯度更新
    
    # 得到模型的参数
    def get_paramaters_copy(self):
        with torch.no_grad():
            param_copy = copy.deepcopy(list(self.parameters())) 
        return param_copy
    # 设置模型的参数
    def set_paramaters(self, param_list):
        with torch.no_grad():
            for i, param in enumerate(self.parameters()):
                param.data = copy.deepcopy(param_list[i].data)
    # 保留梯度
    def keep_grad(self, grad):
        with torch.no_grad():
            if self.total_grad is None:
                self.total_grad = list(grad)
            else:  # 把所有梯度累加
                assert(len(grad) == len(self.total_grad))
                for i in range(len(grad)):
                    self.total_grad[i] += grad[i]
    # 清空梯度
    def zero_grad(self):
        self.total_grad = None
    # 更新参数
    def update_parameters(self, lr, batch):
        with torch.no_grad():
            for i, (name, param) in enumerate(self.named_parameters()):
                # 用 Adam 更新参数
                param.data = self.optimizer[name].update(lr, param.data, self.total_grad[i]/batch)


if __name__=="__main__":
    # 可视化三条正弦曲线，每条正弦曲线上有100个点
    for _ in range(3):
        generator = DataGenerator(k=10)
        x, y = generator.get_test_set()
        plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()
    # 可视化1000条正弦曲线，每条正弦曲线上有100个点
    avg = None
    for _ in range(100):
        generator = DataGenerator(k=10)
        x, y = generator.get_test_set()
        if avg is None:
            avg = y
        else:
            avg = np.add(avg, y)
        plt.plot(x, y, color='lightgray')
    plt.plot(x, avg/1000, color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()
