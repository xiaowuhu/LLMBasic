import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from H3_Helper import save_model

training_sentences = [
    # positive samples
    "i like the white cat",
    "i love to drink tea", 
    "i want to see this film",
    "i love to drink coffee", 
    "i like dog", 
    "i love sugar and sweet food",
    "i don't think dog is bad",
    "i think coffee is good",
    "i don't think this film is not good",
    # negative samples
    "i don't like dog", 
    "i think the coffee is bad", 
    "i hate coffee and sugar",
    "i don't like the white cat",
    "i hate to see this film",
    "i think this film is bad", 
    "i don't think dog is good", 
    "i think this film is not good",
    "i dislike the black cat",
]

Y = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 
                  0, 0, 0, 0, 0, 0, 0, 0, 0])

test_sentences = [
    # positive samples
    "i like the cat",
    "i love to see this film",
    "i don't think this film is not good",
    # negative samples
    "i hate to drink coffee", 
    "i don't think this film is good",
]


class Self_Attention(nn.Module):
    def __init__(self, input_size, embed_size, d_k, d_v, max_length):
        super(Self_Attention,self).__init__()
        self.embedding = nn.Embedding(input_size, embed_size)
        self.Wq = nn.Linear(embed_size, d_k, bias=False)
        self.Wk = nn.Linear(embed_size, d_k, bias=False)
        self.Wv = nn.Linear(embed_size, d_v, bias=False)
        self.fc = nn.Linear(max_length, 2)
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
        return output

class Vocab:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.n_words = 3  # Count PAD, SOS, EOS

    def addWords(self, sentence):
        count = 0
        for word in sentence.split(' '):
            self.addWord(word)
            count += 1
        return count

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.word2index.get(tokens)
        return [self.__getitem__(token) for token in tokens]

    def __len__(self):
        return self.n_words

def prepareData(data):
    max_length = 0
    vocab = Vocab()
    for s in data:
        count = vocab.addWords(s)
        if count > max_length:
            max_length = count
    X = []
    for s in data:
        ids = vocab[s.split(" ")]
        if len(ids) < max_length:
            ids.extend([0] * (max_length-len(ids)))
        X.append(ids)
    X = torch.LongTensor(X)
    return vocab, X, max_length

def words_to_tensor(sentense, vocab, max_length):
    tensor = torch.zeros(max_length)
    words = sentense.split(" ")
    for i in range(len(words)):
        id = vocab[words[i]]
        tensor[i] = id
    return tensor.long()

def randomSample(X, Y):
    idx = random.randint(0, len(training_sentences)-1)
    x = X[idx]
    y = Y[idx:idx+1]
    return x, y

def train(model, max_iter, X, Y, model_name=None):
    loss_func = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= 0.01, weight_decay=1e-6)
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
        if (iter + 1) % 10 == 0:
            print("loss = %.6f"%(running_loss))
            if running_loss < best_loss:
                best_loss = running_loss
                if model_name is not None:
                    save_model(model, model_name)
            running_loss = 0
        
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
    input_size, embed_size, k_size, v_size = len(vocab), 10, 8, 1
    model = Self_Attention(input_size, embed_size, k_size, v_size, max_length)
    max_iter = 1000
    train(model, max_iter, X, Y) #, "SelfAttention1.pth")
    test(model, vocab, max_length)
