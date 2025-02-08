import torch
from torch import nn
import matplotlib.pyplot as plt

from H2_6_1_3_IMDb_DataReader import load_data_imdb
from H2_5_2_Embedding import set_embedding_weights
import H2_6_Helper as helper
from H2_10_1_Model import LSTM, BiLSTM, GRU, BiGRU

if __name__=="__main__":

    num_epochs, batch_size, num_hiddens = 20, 64, 128
    embed_size, num_layers, num_steps = 100, 1, 500
    train_data, test_data, vocab = load_data_imdb(batch_size, num_steps)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nets = []
    nets.append(LSTM(len(vocab), embed_size, num_hiddens, num_layers))
    nets.append(BiLSTM(len(vocab), embed_size, num_hiddens, num_layers))
    nets.append(GRU(len(vocab), embed_size, num_hiddens, num_layers))
    nets.append(BiGRU(len(vocab), embed_size, num_hiddens, num_layers))

    model_names = ["LSTM.pth", "BiLSTM.pth", "GRU.pth", "BiGRU.pth"]

    all_losses = []

    for i in range(len(nets)):
        net = nets[i]
        net.apply(helper.init_weights)
        set_embedding_weights(net, vocab)
        print(net)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        loss_func = nn.CrossEntropyLoss()  # mean
        all_loss = helper.train(net, train_data, test_data, loss_func, optimizer, num_epochs, DEVICE, model_names[i])
        all_losses.append(all_loss)
        helper.predict_sentiment(net, vocab, "this movie is so great and i like it", DEVICE)
        helper.predict_sentiment(net, vocab, "this movie is too bad and i don't like it", DEVICE)

    ls = ["--", "-.", ":", "-"]
    labels = ["LSTM", "BiLSTM", "GRU", "BiGRU"]
    for i in range(len(nets)):
        plt.plot(all_losses[i], linestyle=ls[i], label=labels[i])
    plt.grid()
    plt.legend()
    plt.show()
