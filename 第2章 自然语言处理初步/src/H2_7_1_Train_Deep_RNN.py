import torch
from torch import nn

from H2_6_1_3_IMDb_DataReader import load_data_imdb
from H2_5_2_Embedding import set_embedding_weights
import H2_6_Helper as helper
from H2_6_2_RNN_Model import DeepRNN


if __name__=="__main__":

    num_layers = [2, 4]
    model_names = ["IMDb_RNN_2_50.pth", "IMDb_RNN_4_50.pth"]
    for i in range(2):
        num_epochs, batch_size, num_hiddens = 20, 64, 32
        embed_size, num_steps = 100, 50
        train_data, test_data, vocab = load_data_imdb(batch_size, num_steps)
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = DeepRNN(len(vocab), embed_size, num_hiddens, num_layers[i])
        net.apply(helper.init_weights)
        set_embedding_weights(net, vocab)
        print(net)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
        loss_func = nn.CrossEntropyLoss()  # mean
        helper.train(net, train_data, test_data, loss_func, optimizer, num_epochs, DEVICE, model_names[i])

        helper.predict_sentiment(net, vocab, "this movie is so great and i like it", DEVICE)
        helper.predict_sentiment(net, vocab, "this movie is too bad and i don't like it", DEVICE)
