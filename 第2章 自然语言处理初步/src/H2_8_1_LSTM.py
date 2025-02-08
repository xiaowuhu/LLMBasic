import numpy as np
import torch
from torch import nn

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def tanh(x):
    return 2.0 / (1.0 + np.exp(-2*x)) - 1.0


def compute_forward():
    f_t = sigmoid(0.7 * (-0.1) + (-0.3) * 0.5)
    print("f_t = ", f_t)
    i_t = sigmoid(0.7 * 0.2 + (-0.3) * 0.6)
    print("i_t = ", i_t)
    _c_t = tanh(0.7 * 0.3 + (-0.3) * 0.2)
    print("_c_t = ", _c_t)
    o_t = sigmoid(0.7 * 0.4 + (-0.3) * (-0.7))
    print("o_t = ", o_t)

    fc_t_1 = 0.4 * f_t
    print("fc_t_1 = ", fc_t_1)
    ic_t = i_t * _c_t
    print("ic_t = ", ic_t)
    c_t = fc_t_1 + ic_t
    print("c_t = ", c_t)

    h_t = tanh(c_t) * o_t
    print("h_t = ", h_t)

def LSTM():
    weight_ih = torch.tensor([[0.2],[-0.1],[0.3],[0.4]])
    weight_hh = torch.tensor([[0.6],[0.5],[0.2],[-0.7]])
    lstm = nn.LSTM(1, 1, 1, False, True)
    print("初始化权重:")
    for name, param in lstm.named_parameters():
        print(name, param)

    lstm_weight = lstm.state_dict()
    lstm_weight["weight_ih_l0"].copy_(weight_ih)
    lstm_weight["weight_hh_l0"].copy_(weight_hh)

    print("手动赋值权重:")
    for name, param in lstm.named_parameters():
        print(name, param)

    input = torch.tensor([[0.7]])
    h_0 = torch.tensor([[-0.3]])
    c_0 = torch.tensor([[0.4]])
    z, (h,c) = lstm(input, (h_0, c_0))
    print("z,h,c=", z, h, c)
    v = torch.tensor([[0.5]], requires_grad=True)
    output = torch.matmul(z, v)
    print("output=", output)

    y = torch.tensor([[1.0]])  # 标签值
    loss_func = nn.MSELoss()
    loss = loss_func(output, y)
    print("loss=", loss)
    loss.backward()

    for name, param in lstm.named_parameters():
        print("初始参数:")
        print(name, param)
        print("grad=", param.grad)
        param.data.add_(param.grad.data, alpha=-1)
        print("更新后的参数:")
        print(name, param)
    
    print("初始参数:", v)
    print("v.grad=",v.grad)
    v.data.add_(v.grad.data, alpha=-1)
    print("更新后的参数:", v)

        
    
    

    


if __name__=="__main__":
    LSTM()
    print("更新后的 o_t:", sigmoid(0.7 * 0.4375 +(-0.3)*0.0161))
