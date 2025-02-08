import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=9)

from H3_Helper import load_model
from H3_7_SelfAttention_Train import prepareData, training_sentences

class Self_Attention(nn.Module):
    def __init__(self, input_size, embed_size, d_k, d_v, max_length):
        super(Self_Attention,self).__init__()
        self.embedding = nn.Embedding(input_size, embed_size)
        self.Wq = nn.Linear(embed_size, d_k, bias=False)
        self.Wk = nn.Linear(embed_size, d_k, bias=False)
        self.Wv = nn.Linear(embed_size, d_v, bias=False)
        self.fc = nn.Linear(d_v * max_length, 2)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self._norm_fact = 1 / math.sqrt(d_k)
        
    def forward(self, input):
        x = self.embedding(input)
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        QKT = torch.mm(Q,K.permute(1,0)) * self._norm_fact
        weight = F.softmax(QKT, dim=-1)
        A = torch.mm(weight, V)
        z = self.fc(A.T)
        output = self.logsoftmax(z)
        return output, A, weight


def words_to_tensor(sentense, vocab, max_length):
    tensor = torch.zeros(max_length)
    words = sentense.split(" ")
    for i in range(len(words)):
        id = vocab[words[i]]
        tensor[i] = id
    return tensor.long()

       
def test(s, model, vocab, max_length):
    #for s in test_sentences:
    tensor = words_to_tensor(s, vocab, max_length)
    output, A, weight = model(tensor)
    label = torch.argmax(output, dim=1)
    print(s, '-> positive' if label == 1 else '-> negative')
    print("weight:\n", weight)
    print("A:\n", A)
    show_attention(s, weight)


def show_attention(s, weight):
    words = s.split(" ")
    Y = []
    y = 1
    for i in range(len(words)):
        Y.append(y)
        plt.text(0.1, y, words[i], fontsize=12)
        plt.text(0.5, y, words[i], fontsize=12, ha="right")
        y -= 0.15

    c = len(words)
    weight = weight[0:c, 0:c]
    colors = ["red", "blue", "green", "cyan", "yellow", "magenta", "black"]
    for i in range(c):
        for j in range(c):
            if i == j: continue
            w = weight[i, j]
            plt.plot((0.15, 0.45), (Y[i]+0.01, Y[j]+0.01), linewidth=w*2, color=colors[i])
            
    plt.axis('off')
    plt.show()

if __name__=="__main__":
    s = "i think this film is not good"
    vocab, X, max_length = prepareData(training_sentences)
    print(len(vocab))
    print(vocab.index2word)
    input_size, embed_size, k_size, v_size = len(vocab), 10, 8, 1
    model = Self_Attention(input_size, embed_size, k_size, v_size, max_length)
    model.eval()
    load_model(model, "SelfAttention1.pth", "cpu")
    test(s, model, vocab, max_length)
