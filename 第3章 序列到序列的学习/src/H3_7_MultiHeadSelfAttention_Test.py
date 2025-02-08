import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=9)

from H3_7_SelfAttention_Train import words_to_tensor, prepareData, training_sentences
from H3_7_MultiHeadSelfAttention_Train import load_model

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
        QKT = torch.mm(Q, K.permute(1,0)) * self._norm_fact
        weight = F.softmax(QKT, dim=-1)
        A = torch.mm(weight, V)
        return A, weight

class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, input_size, embed_size, k_size, v_size, num_heads, max_length):
        super(MultiHeadAttention, self).__init__()
        self.embedding = nn.Embedding(input_size, embed_size)
        self.num_heads = num_heads
        self.heads = []
        for i in range(num_heads):
            head = SelfAttention(embed_size, k_size, v_size)
            self.heads.append(head)
        self.Wo = nn.Linear(v_size * num_heads, v_size * num_heads, bias=False)
        self.fc = nn.Linear(max_length * num_heads, 2)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, input):
        x = self.embedding(input)
        mhA = []
        weights = []
        for i in range(self.num_heads):
            a, w = self.heads[i](x)
            mhA.append(a)
            weights.append(w)
        A = torch.cat(mhA, dim=-1)
        #z = self.Wo(A)
        z = self.fc(A.reshape(1, -1))
        output = self.logsoftmax(z)
        return output, weights

def show_attention(s, weights):
    words = s.split(" ")
    Y = []
    y = 1
    for i in range(len(words)):
        Y.append(y)
        plt.text(0.1, y, words[i], fontsize=12)
        plt.text(0.5, y, words[i], fontsize=12, ha="right")
        y -= 0.15

    c = len(words)
    from_word_idx = 2
    ls = ['r-', 'b:']
    for i in range(len(weights)):
        # 取第 i 个头的权重矩阵
        weight = weights[i][0:c, 0:c]
        # 只取第一个词 "i" 的权重
        weight_i = weight[from_word_idx].topk(3)
        print(weight_i)
        for j in range(3):
            w = weight_i[0][j]
            p = weight_i[1][j]
            # 从第一个词 i 画三条线到右侧
            plt.plot((0.15, 0.45), (Y[from_word_idx]+0.01, Y[p]+0.01), ls[i], linewidth=w*10)
            
    plt.axis('off')
    plt.show()        

def test(s, model, vocab, max_length):
    #for s in test_sentences:
    tensor = words_to_tensor(s, vocab, max_length)
    output, weights = model(tensor)
    label = torch.argmax(output, dim=1)
    print(s, '-> positive' if label == 1 else '-> negative')
    #print("weight:\n", weights)
    show_attention(s, weights)

if __name__=="__main__":
    s = "i think this film is not good"
    vocab, X, max_length = prepareData(training_sentences)
    print(len(vocab))
    print(vocab.index2word)
    input_size, embed_size, k_size, v_size = len(vocab), 10, 4, 1
    num_heads = 2
    model = MultiHeadAttention(input_size, embed_size, k_size, v_size, num_heads, max_length)
    model.eval()
    load_model(model, "MH_SA1.pth", "cpu")
    test(s, model, vocab, max_length)
