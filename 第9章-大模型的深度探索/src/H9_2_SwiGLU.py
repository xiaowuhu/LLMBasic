import numpy as np
import torch
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

x = np.linspace(-4, 4, 100)
x = torch.tensor(x)

print(x)
sigmoid = torch.nn.Sigmoid()
y = sigmoid(x)
print(y)
plt.plot(x, y, label="Sigmoid")

silu = torch.nn.SiLU()
y = silu(x)
print(y)
plt.plot(x, y, ":", label="Swish")
plt.grid()
plt.legend()
plt.show()



import torch.nn as nn
 
class SwiGLU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SwiGLU, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(input_dim, hidden_dim)
        self.swish = nn.SiLU()  # 使用内置 Swish 函数
 
    def forward(self, x):
        return self.fc1(x) * self.swish(self.fc2(x))

# input_dim = 128
# hidden_dim = 256
# x = torch.randn(32, input_dim)  # Batch size 32, Input dimension 128
 
# swiglu = SwiGLU(input_dim, hidden_dim)
# output = swiglu(x)
# print(output.shape)  # 输出维度: [32, 256]


x = np.linspace(-2, 2, 100)
x = torch.tensor(x)

def swiglu(w1, w2):
    y1 = x * w1
    swish = nn.SiLU()
    y2 = swish(x * w2)
    y = y1 * y2
    print(y)
    plt.plot(x, y1, ":", label="GLU")
    plt.plot(x, y2, "-.", label="Swish")
    plt.plot(x, y, "-", label="SwiGLU")
    plt.grid()
    plt.legend()
    plt.show()


w1 = 0.5
w2 = 1.5
swiglu(w1, w2)

w1=-0.5
swiglu(w1, w2)
