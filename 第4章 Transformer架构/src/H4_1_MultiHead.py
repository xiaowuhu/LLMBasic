import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size, k_size, v_size):
        super(SelfAttention,self).__init__()
        self.Wq = nn.Linear(embed_size, k_size, bias=False)
        self.Wk = nn.Linear(embed_size, k_size, bias=False)
        self.Wv = nn.Linear(embed_size, v_size, bias=False)
        self._norm_fact = 1 / math.sqrt(k_size)
        
    def forward(self, x):
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        print("第一种方法的权重")
        print("Q:", Q) #self.Wq.weight)
        print("K:", K) # self.Wk.weight)
        print("V:", V) #self.Wv.weight)
        QKT = torch.mm(Q, K.permute(1,0))# * self._norm_fact
        weight = F.softmax(QKT, dim=-1)
        print("weight:\n", weight)
        A = torch.mm(weight, V)
        print("A:\n", A)
        return A

class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, embed_size, k_size, v_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        # 第一种方法
        self.heads = []
        for i in range(num_heads):
            head = SelfAttention(embed_size, k_size, v_size)
            self.heads.append(head)
        self.Wo = nn.Linear(v_size * num_heads, v_size * num_heads, bias=False)
        
        # 第二种方法,把SelfAttention中的矩阵都拼接在一起
        self.Wq = nn.Linear(embed_size, k_size * num_heads, bias=False)
        self.Wk = nn.Linear(embed_size, k_size * num_heads, bias=False)
        self.Wv = nn.Linear(embed_size, v_size * num_heads, bias=False)
        for i in range(num_heads):
            head = self.heads[i]
            start = i * k_size
            end = start + k_size
            self.Wq.weight.data[start:end, :] = head.Wq.weight.data
            self.Wk.weight.data[start:end, :] = head.Wk.weight.data
            self.Wv.weight.data[i*v_size:(i+1)*v_size, :] = head.Wv.weight.data


    def forward(self, x):
        # 第一种方法
        mhA = []
        for i in range(self.num_heads):
            a = self.heads[i](x)
            mhA.append(a)
        A1 = torch.cat(mhA, dim=-1)
        print("A1:\n", A1)
        A1 = self.Wo(A1)
        #return A1
        # 第二种方法
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        # print("第二种方法的权重")
        # print("Q:\n", Q)
        # print("K:\n", K)
        # print("V:\n", V)
        Q = Q.view(-1, self.num_heads, k_size)#.permute(0, 2, 1)
        K = K.view(-1, self.num_heads, k_size)#.permute(0, 2, 1)
        V = V.view(-1, self.num_heads, v_size)#.permute(0, 2, 1)
        attention = torch.matmul(Q, K.permute(0, 2, 1))# / self.scale
        attention = torch.softmax(attention, dim=-1)
        A2 = torch.matmul(attention, V)
        A2 = A2.reshape(-1, self.num_heads * v_size)
        return A1

if __name__=="__main__":
    n_heads = 2
    embed_size = 4
    k_size = 6 // n_heads
    v_size = 4 // n_heads

    mha = MultiHeadAttention(embed_size, k_size, v_size, n_heads)
    x = torch.randn(3, embed_size)
    print("x:\n", x)
#x = to_positive(x)
#x[:,:,0:4] = 0
#x[:,0:2,:] = 0
    A = mha(x) # q, k, v = x, x, x
    print(A.shape)
    #print(A2.shape)
    print("A:\n", A)
    #print(A2)
