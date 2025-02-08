import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def PositionEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for pos in range(seq_len):
        for j in np.arange(d):
            if j % 2 == 0:
                denominator = np.power(n, j/d)
                P[pos, j] = np.sin(pos/denominator)
            else:
                denominator = np.power(n, (j-1)/d)
                P[pos, j] = np.cos(pos/denominator)
    return P

def show_PE_vector():
    PE = PositionEncoding(seq_len=20, d=512, n=10000)
    X = PE
    pos = [0,1,2,3] # 要小于 seq_len
    fmts=('-', 'm--', 'g-.', 'r:')
    for id, i in enumerate(pos):
        plt.plot(X[i,0:20], fmts[id], marker='.')
    plt.legend(["pos =%d"% p for p in pos])
    plt.grid()
    plt.xlabel("$j$")
    plt.ylabel("PE")
    plt.xticks([0,2,4,6,8,10,12,14,16,18,20])
    plt.show()

def show_heatmap():
    dim = 60
    PE = PositionEncoding(seq_len=50, d=dim, n=10000)
    cax = plt.matshow(PE)
    plt.colorbar(cax)
    plt.xlabel("dim")
    plt.ylabel("pos")
    plt.show()


def find_identity():
    PE = []
    dim = 128
    j = 1  #  0 <= j < 512, j 越大重复率越低
    L = 1024
    for pos in range(L):
        pe = np.sin(pos / np.power(10000, j/dim))
        PE.append(round(pe, 5)) # 精度=5，可以选 1 到 7
    print(PE)
    for i in range(L):
        for j in range(L):
            if i == j:
                continue
            else:
                if PE[i] == PE[j]:
                    print("相等: pos1=%d, pos2=%d, v=%.5f"%(i, j, PE[i]))


class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, max_len=1000):
        super(PositionalEncoding, self).__init__()
        #self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return X
        #return self.dropout(X)

if __name__=="__main__":

    pe = PositionalEncoding(8)
    X = torch.zeros(2, 3, 8)
    Y = pe(X)
    print(Y)
    exit(0)

    # 位置编码计算
    P = PositionEncoding(seq_len=3, d=4, n=10000)
    print(P)
    show_PE_vector()
    show_heatmap()
    find_identity()

