import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v):
        super(SelfAttention, self).__init__()
        self.Wq = nn.Linear(dim_q, dim_k, bias=False)
        self.Wk = nn.Linear(dim_q, dim_k, bias=False)
        self.Wv = nn.Linear(dim_q, dim_v, bias=False)
        self.norm_fact = 1 / math.sqrt(dim_k)

    def forward(self, x):
        batch, n, dim_q = x.shape
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        dist = torch.bmm(Q, K.transpose(1, 2)) * self.norm_fact
        prob = torch.softmax(dist, dim=-1)
        attention = torch.bmm(prob, V)
        return attention


if __name__=="__main__":
    torch.manual_seed(0)
    dim_q, dim_k, dim_v = 4, 4, 4
    sa = SelfAttention(dim_q, dim_k, dim_v)
    print(sa.Wq.weight)
