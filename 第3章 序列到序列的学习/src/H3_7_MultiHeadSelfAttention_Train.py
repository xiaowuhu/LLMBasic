import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from H3_Helper import load_model
from H3_7_SelfAttention_Train import randomSample, words_to_tensor, prepareData, training_sentences, Y, test_sentences

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
        alpha = F.softmax(QKT, dim=-1)
        A = torch.mm(alpha, V)
        return A

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
        for i in range(self.num_heads):
            a = self.heads[i](x)
            mhA.append(a)
        A = torch.cat(mhA, dim=-1)
        #z = self.Wo(A)
        z = self.fc(A.reshape(1, -1))
        output = self.logsoftmax(z)
        return output

def train(model, max_iter, X, Y, model_name=None):
    loss_func = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-6)
    model.train()
    best_loss = 10
    running_loss = 0
    for iter in range(max_iter):
        x, y = randomSample(X, Y)
        pred = model(x)
        loss = loss_func(pred, y)
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (iter + 1) % 20 == 0:
            print("loss = %.6f"%(running_loss))
            if running_loss < best_loss:
                best_loss = running_loss
                if model_name is not None:
                    save_model(model, model_name)
            running_loss = 0

def save_model(model: nn.Module, name: str):
    print("---- save model... ----")
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    train_pth = os.path.join(current_dir, "model", name)
    torch.save(model.state_dict(), train_pth)
    for i in range(len(model.heads)):
        child_model = model.heads[i]
        child_train_pth = os.path.join(current_dir, "model", str(i) + "_" + name)
        torch.save(child_model.state_dict(), child_train_pth)

def load_model(model:nn.Module, name:str, device):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    model_pth = os.path.join(current_dir, "model", name)
    print("load model ", name)
    model.load_state_dict(torch.load(model_pth, map_location=device, weights_only=True))
    for i in range(len(model.heads)):
        child_model = model.heads[i]
        child_train_pth = os.path.join(current_dir, "model", str(i) + "_" + name)
        child_model.load_state_dict(torch.load(child_train_pth, map_location=device, weights_only=True))


def test(model, vocab, max_length):
    for s in test_sentences:
        tensor = words_to_tensor(s, vocab, max_length)
        output = model(tensor)
        label = torch.argmax(output, dim=1)
        print(s, '-> positive' if label == 1 else '-> negative')

if __name__=="__main__":
    vocab, X, max_length = prepareData(training_sentences)
    print(len(vocab))
    print(vocab.index2word)
    input_size, embed_size, k_size, v_size = len(vocab), 10, 4, 1
    num_heads = 2
    model = MultiHeadAttention(input_size, embed_size, k_size, v_size, num_heads, max_length)
    max_iter = 5000
    #train(model, max_iter, X, Y) #, "MH_SA1.pth")
    load_model(model, "MH_SA1.pth", "cpu")
    test(model, vocab, max_length)
