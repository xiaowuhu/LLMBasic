import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt

from H3_3_Translate_Data import get_dataloader
from H3_5_Attention_Model import EncoderRNN, AttnDecoderRNN
from H3_Helper import train_seq2seq_with_attention

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_size, hidden_size = 100, 128
    num_epochs, batch_size, lr = 20, 32, 0.001
    franch_vocab, english_vocab, train_dataloader = get_dataloader(batch_size)
    encoder = EncoderRNN(franch_vocab.n_words, embed_size, hidden_size)
    decoder = AttnDecoderRNN(embed_size, hidden_size, english_vocab.n_words)
    
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)
    loss_function = nn.NLLLoss()
    all_loss = train_seq2seq_with_attention(encoder, decoder, train_dataloader, None, 
          loss_function, 100, # checkpoint
          encoder_optimizer, decoder_optimizer,
          num_epochs, device,
          save_model_names=("Add_Attention_Encoder.pth", "Add_Attention_Decoder.pth"))

    plt.plot(all_loss)
    plt.show()
