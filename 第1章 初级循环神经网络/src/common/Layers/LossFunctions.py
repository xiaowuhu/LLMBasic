
import numpy as np
from .Operator import Operator

# 均方差损失函数
class MSE(Operator):
    def __call__(self, predict, label):
        return self.forward(predict, label)

    def forward(self, predict, label):
        return np.mean(np.square(predict - label)) / 2

    def backward(self, predict, label):
        return predict - label

# 二分类交叉熵损失函数
class BCE(Operator):
    def __call__(self, predict, label):
        return self.forward(predict, label)
    
    def forward(self, predict, label):
        predict = predict
        label = label
        p1 = label * np.log(predict)
        p2 = (1-label) * np.log(1 - predict + 1e-5)
        return np.mean(-(p1+p2))

    def backward(self, predict, label):
        p1 = predict - label
        p2 = predict * (1 - predict)
        return p1 / p2

# 多分类交叉熵误差函数
class CrossEntropy3(Operator):
    def forward(self, predict, label):
        self.predict = predict
        self.label = label
        p = self.label * np.log(predict+1e-5)
        sum = np.sum(-p, axis=1) # 按行（在一个样本内）求和
        loss = np.mean(sum) # 按列求所有样本的平均数
        return loss

    def backward(self):
        return -(self.label / self.predict) 

    def __call__(self, predict, label):
        return self.forward(predict, label)

# 二分类函数接二分类交叉熵损失函数
class Logistic_CE2_loss(Operator):
    def forward(self, z, label):  # 接收 linear 层输出的 z
        self.label = label
        # logistic
        self.predict = 1.0 / (1.0 + np.exp(-z)) 
        # cross entropy 2
        p1 = self.label * np.log(self.predict)
        p2 = (1 - self.label) * np.log(1 - self.predict)    
        return np.mean(-(p1 + p2))
    
    # 联合求导
    def backward(self):
        return self.predict - self.label

# 多分类函数接多分类交叉熵损失函数
class Softmax_CE3_loss(Operator):
    def forward(self, z, label):  # 接收 linear 层输出的 z
        self.label = label
        # softmax
        shift_z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(shift_z)
        self.predict = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        # cross entropy 3
        p = self.label * np.log(self.predict)
        sum = np.sum(-p, axis=1) # 按行（在一个样本内）求和
        loss = np.mean(sum) # 按列求所有样本的平均数
        return loss
    
    # 联合求导
    def backward(self):
        return self.predict - self.label

if __name__ == "__main__":
    # test CE3
    predict = np.array([[0.1, 0.2, 0.7], 
                        [0.3, 0.6, 0.1]])
    label = np.array([[0, 0, 1],
                      [0, 1, 0]])
    ce3 = CrossEntropy3()
    #loss = ce3.forward(predict, label)
    loss = ce3(predict, label)
    print(loss)
    print(ce3.backward())
