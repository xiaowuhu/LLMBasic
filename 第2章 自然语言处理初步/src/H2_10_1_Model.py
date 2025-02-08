import torch
from torch import nn

class LSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers=1):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size) # 46158 x 100
        self.lstm = nn.LSTM(embed_size,             # 100
                            num_hiddens,            # 128
                            num_layers=num_layers,  # 1
                            bidirectional=False,    
                            batch_first=True)
        self.V = nn.Linear(num_hiddens, 2)  # 128 x 2

    def forward(self, inputs):
        embeddings = self.embedding(inputs)  # 64 x 500 -> 54 x 500 x 100
        self.lstm.flatten_parameters()
        z, (h, c) = self.lstm(embeddings) # z: 64 x 500 x 128 , h,c: 1 x 64 x 128
        outputs = self.V(z[:, -1, :]) # 64 x 2
        return outputs

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers=1):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size,   # 100
                            num_hiddens,  # 128
                            num_layers=num_layers,  # 1
                            bidirectional=True, 
                            batch_first=True)
        self.V = nn.Linear(4 * num_hiddens, 2) # 512 x 2

    def forward(self, inputs):
        embeddings = self.embedding(inputs) # 64 x 500 -> 64 x 500 x 100
        self.lstm.flatten_parameters()
        z, (h, c) = self.lstm(embeddings) # z: 64 x 500 x 256, h,c: 2 x 64 x 128
        z2 = torch.cat((z[:, 0, :], z[:, -1, :]), dim=1) # 64 x 512
        outputs = self.V(z2) # 64 x 2
        return outputs

class GRU(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers=1):
        super(GRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size,  # 100
                           num_hiddens, # 128
                           num_layers=num_layers, # 1
                           bidirectional=False, 
                           batch_first=True)
        self.V = nn.Linear(num_hiddens, 2) # 128 x 2

    def forward(self, inputs):
        embeddings = self.embedding(inputs) # 64 x 500 -> 64 x 500 x 100
        self.gru.flatten_parameters()
        z, h = self.gru(embeddings) # z: 64 x 500 x 128, h: 1 x 64 x 128
        outputs = self.V(z[:, -1, :]) # 64 x 2
        return outputs

class BiGRU(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers=1):
        super(BiGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, # 100
                           num_hiddens, # 128
                           num_layers=num_layers, # 1
                           bidirectional=True, 
                           batch_first=True)
        self.V = nn.Linear(4 * num_hiddens, 2) # 512 x 2

    def forward(self, inputs):
        embeddings = self.embedding(inputs) # 64 x 500 -> 64 x 500 x 100
        self.gru.flatten_parameters()
        z, h = self.gru(embeddings) # z: 64 x 500 x 256, h: 2 x 64 x 128
        z2 = torch.cat((z[:, 0, :], z[:, -1, :]), dim=1) # 64 x 512
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
    nets = []
    embed_size, num_layers, num_steps = 100, 1, 500
    num_epochs, batch_size, num_hiddens = 20, 64, 128
    nets.append(LSTM(46158, embed_size, num_hiddens, num_layers))
    nets.append(BiLSTM(46158, embed_size, num_hiddens, num_layers))
    nets.append(GRU(46158, embed_size, num_hiddens, num_layers))
    nets.append(BiGRU(46158, embed_size, num_hiddens, num_layers))
    for i in range(4):
        print("----")
        total_size = 0
        for name, param in nets[i].named_parameters():
            if name != "embedding.weight":
                total_size += param.numel()
            print(name, ":", param.size(), "->", param.numel())
        print(total_size)
