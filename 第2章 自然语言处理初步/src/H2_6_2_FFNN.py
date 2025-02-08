import torch
from torch import nn
import torch.nn.functional as F

from H2_6_1_3_IMDb_DataReader import load_data_imdb
import H2_6_Helper as helper
from H2_5_2_Embedding import set_embedding_weights

class FFNN(nn.Module):
    def __init__(self, vocab_size, embed_size, input_length, num_hiddens):
        super(FFNN, self).__init__()
        self.input_size = embed_size * input_length
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.fc1 = nn.Linear(self.input_size, num_hiddens)
        self.fc2 = nn.Linear(num_hiddens, 2)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        x = embeddings.reshape(-1, self.input_size)
        z1 = self.fc1(x)
        a1 = F.tanh(z1)
        z2 = self.fc2(a1)
        return z2

def compute_params(net):
    total_size = 0
    for name, param in net.named_parameters():
        if name != "embedding.weight":
            total_size += param.numel()
        print(name, ":", param.size(), "->", param.numel())
    print(total_size)

def main():
    batch_size, embed_size, num_hiddens, input_length = 64, 100, 128, 50
    train_data, test_data, vocab = load_data_imdb(batch_size, input_length)
    model = FFNN(len(vocab), embed_size, input_length, num_hiddens)
    print(model)
    compute_params(model)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(DEVICE)
    model.apply(helper.init_weights)
    set_embedding_weights(model, vocab)

    num_epochs = 20
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.CrossEntropyLoss(reduction="mean")
    helper.train(model, train_data, test_data, loss_func, optimizer, num_epochs, DEVICE, "IMDB_FFNN.pth")

if __name__=="__main__":
    main()
