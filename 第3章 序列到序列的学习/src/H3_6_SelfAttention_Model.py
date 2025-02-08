import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Self_Attention(nn.Module):
    def __init__(self, input_size, embed_size, k_size, v_size):
        super(Self_Attention,self).__init__()
        self.embedding = nn.Embedding(input_size, embed_size)
        self.Wq = nn.Linear(embed_size, k_size, bias=False)
        self.Wk = nn.Linear(embed_size, k_size, bias=False)
        self.Wv = nn.Linear(embed_size, v_size, bias=False)
        self._norm_fact = 1 / math.sqrt(k_size)
        
    def forward(self, input):
        x = self.embedding(input)
        print("输入词向量:\n", x)
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        print("Q 矩阵:\n", Q)
        print("K 矩阵:\n", K)
        print("V 矩阵:\n", V)
        KQT = torch.bmm(Q,K.permute(0,2,1)) * self._norm_fact
        alpha = F.softmax(KQT, dim=-1)
        print("权重矩阵:\n", alpha)
        A = torch.bmm(alpha, V)
        return A

if __name__=="__main__":
    dict_vocab = {"SOS":0, "EOS":1, "PAD":2, "我":3, "弹":4, "琴":5}
    input_size, embed_size, k_size, v_size = len(dict_vocab), 4, 5, 4
    attn = Self_Attention(input_size, embed_size, k_size, v_size)
    sentence = ["我", "弹", "琴"]
    ids = []
    for word in sentence:
        id = dict_vocab[word]
        ids.append(id) # [3，4，5]
    input = torch.tensor(ids).unsqueeze(0) # 在第一维加批量维度=1
    A = attn.forward(input)
    print("结果:\n", A)
