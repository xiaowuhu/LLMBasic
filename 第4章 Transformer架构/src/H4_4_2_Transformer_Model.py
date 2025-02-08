from torch import Tensor
import torch
import torch.nn as nn
import math

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 #dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        denominator = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * denominator)
        pos_embedding[:, 1::2] = torch.cos(pos * denominator)
        pos_embedding = pos_embedding.unsqueeze(0)  # 变成 [1, 5000, 512]
        #self.dropout = nn.Dropout(dropout)
        # 把局部变量 pos_embedding 注册为模型的参数，随着模型一起可以移动的GPU上
        # 但是在训练的时候不会更新
        self.register_buffer('pe', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return token_embedding + self.pe[:,:token_embedding.size(1), :]


# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 embed_size: int,
                 num_head: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model=embed_size,
                                       nhead=num_head,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       batch_first=True,
                                       dropout=dropout)
        self.src_embed = nn.Embedding(src_vocab_size, embed_size)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, embed_size)
        self.pe = PositionalEncoding(embed_size)
        self.fc = nn.Linear(embed_size, tgt_vocab_size)
        self.sqrt_emb_size = math.sqrt(embed_size)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                memory_mask = None,
                src_key_padding_mask = None,
                tgt_key_padding_mask = None,
                memory_key_padding_mask = None):
        
        src_emb = self.pe(self.src_embed(src) * self.sqrt_emb_size) # 64x6 -> 64x6x512
        tgt_emb = self.pe(self.tgt_embed(trg) * self.sqrt_emb_size) # 64x5 -> 64x6x512
        outs = self.transformer(src_emb, tgt_emb, #-> 64x5x512
                                src_mask, tgt_mask, memory_mask, 
                                src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask)
        return self.fc(outs)  # (64x5x512) * (512x28037) -> 64x5x10837

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.pe(self.src_embed(src) * self.sqrt_emb_size), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.pe(self.tgt_embed(tgt) * self.sqrt_emb_size), memory, tgt_mask)

if __name__=="__main__":
    # 位置编码计算
    pe = PositionalEncoding(emb_size=4, maxlen=3)
    t = torch.zeros((1, 3, 4)).long()
    X = pe(t)
    print(X)
    # X.squeeze_(1)
    # pos = [0,1,2,3] # 要小于 seq_len
    # fmts=('-', 'm--', 'g-.', 'r:')
    # for id, i in enumerate(pos):
    #     plt.plot(X[i,0:20], fmts[id], marker='.')
    # plt.legend(["pos =%d"% p for p in pos])
    # plt.grid()
    # plt.xlabel("$j$")
    # plt.ylabel("PE")
    # plt.xticks([0,2,4,6,8,10,12,14,16,18,20])
    # plt.show()
