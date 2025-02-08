import torch
from torch import nn
import os

from H2_6_1_3_IMDb_DataReader import load_data_imdb
from H2_5_2_Embedding import TokenEmbedding
import H2_6_Helper as helper
from H2_6_2_RNN_Model import RNN

def load_model(model:nn.Module, name:str, device):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    model_pth = os.path.join(current_dir, "model", name)
    model.load_state_dict(torch.load(model_pth, map_location=device))

def predict_sentiment(net, vocab, sequence):
    sequence = torch.tensor(vocab[sequence.split()])
    output = net(sequence.reshape(1, -1))
    label = torch.argmax(output, dim=1)
    print('positive' if label == 1 else 'negative')
    return 'positive' if label == 1 else 'negative'

if __name__ == "__main__":
    batch_size, embed_size, num_hiddens, num_layers, num_steps = 64, 100, 100, 1, 500
    train_dataset, test_dataset, vocab = load_data_imdb(64, num_steps)

    DEVICE = torch.device("cpu")
    net = RNN(len(vocab), embed_size, num_hiddens, num_layers)


    glove_embedding = TokenEmbedding('glove.6b.100d')
    embeds = glove_embedding[vocab.idx_to_token]
    print(embeds.shape)

    net.embedding.weight.data.copy_(embeds)
    net.embedding.weight.requires_grad = False

    load_model(net, "IMDb_RNN_1_500.pth", DEVICE)

    test_acc = helper.evaluate_accuracy_gpu(net, test_dataset, DEVICE)
    print("test accu:", test_acc)

    predict_sentiment(net, vocab, "this movie is so great and i like it")
    predict_sentiment(net, vocab, "this movie is too bad and i don't like it")

    load_model(net, "IMDb_RNN_1_50.pth", DEVICE)

    test_acc = helper.evaluate_accuracy_gpu(net, test_dataset, DEVICE)
    print("test accu:", test_acc)

    predict_sentiment(net, vocab, "this movie is so great and i like it")
    predict_sentiment(net, vocab, "this movie is too bad and i don't like it")

    load_model(net, "IMDb_RNN_1_20.pth", DEVICE)

    test_acc = helper.evaluate_accuracy_gpu(net, test_dataset, DEVICE)
    print("test accu:", test_acc)

    predict_sentiment(net, vocab, "this movie is so great and i like it")
    predict_sentiment(net, vocab, "this movie is too bad and i don't like it")
