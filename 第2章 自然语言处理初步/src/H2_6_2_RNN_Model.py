import torch
from torch import nn

class RNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, num_hiddens, batch_first=True)
        self.V = nn.Linear(num_hiddens, 2)

    def forward(self, inputs):
        # inputs的形状是（批量大小，时间步数）=(64 x 500)
        embeddings = self.embedding(inputs) # 64 x 500 x 100
        self.rnn.flatten_parameters()
        z, hidden = self.rnn(embeddings) # z: 64 x 500 x 128, hidden: 1 x 64 x 128
        # 只取最后一个时间步的输出值
        outputs = self.V(z[:,-1,:]) # 64 x 2
        return outputs


class DeepRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers):
        super(DeepRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, num_hiddens, num_layers=num_layers, batch_first=True)
        self.V = nn.Linear(num_hiddens, 2)

    def forward(self, inputs):
        # inputs的形状是（批量大小，时间步数）=(64 x 500)
        embeddings = self.embedding(inputs) # 64 x 500 x 100
        self.rnn.flatten_parameters()
        z, hidden = self.rnn(embeddings) # z: 64 x 500 x 128, hidden: 1 x 64 x 128
        # 只取最后一个时间步的输出值
        outputs = self.V(z[:,-1,:]) # 64 x 2
        return outputs


class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens):
        super(BiRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, num_hiddens, bidirectional=True, batch_first=True)
        self.V = nn.Linear(num_hiddens * 4, 2) # (32*4) x 2

    def forward(self, inputs):
        # inputs的形状是（批量大小，时间步数）
        embeddings = self.embedding(inputs) # 64 x 50 -> 64 x 50 x 100
        self.rnn.flatten_parameters()
        # outputs的形状是（批量大小，时间步数，2隐藏单元数）
        z, hidden = self.rnn(embeddings) # z: 64 x 50 x 64, hidden: 2 x 64 x 32
        # 取第一个时间步和最后一个时间步的输出值
        z2 = torch.cat((z[:,0,:], z[:,-1,:]), dim=1) # 64 x (64+64)
        outputs = self.V(z2) # 64 x 2
        return outputs

class DeepBiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers):
        super(DeepBiRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, num_hiddens, num_layers, bidirectional=True, batch_first=True)
        self.V = nn.Linear(num_hiddens * 4, 2) # (32*4) x 2

    def forward(self, inputs):
        # inputs的形状是（批量大小，时间步数）
        embeddings = self.embedding(inputs) # 64 x 50 -> 64 x 50 x 100
        self.rnn.flatten_parameters()
        # outputs的形状是（批量大小，时间步数，2隐藏单元数）
        z, hidden = self.rnn(embeddings) # z: 64 x 50 x 64, hidden: 2 x 64 x 32
        # 取第一个时间步和最后一个时间步的输出值
        z2 = torch.cat((z[:,0,:], z[:,-1,:]), dim=1) # 64 x (64+64)
        outputs = self.V(z2) # 64 x 2
        return outputs


def init_weights(layer):
    if type(layer) == nn.Linear:
        nn.init.xavier_uniform_(layer.weight)
    if type(layer) == nn.LSTM:
        for param in layer._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(layer._parameters[param])


if __name__=="__main__":
    embed_size, num_layers, num_steps = 100, 1, 500
    num_epochs, batch_size, num_hiddens = 20, 64, 128
    net = RNN(46158, embed_size, num_hiddens)
    total_size = 0
    for name, param in net.named_parameters():
        if name != "embedding.weight":
            total_size += param.numel()
        print(name, ":", param.size(), "->", param.numel())
    print(total_size)