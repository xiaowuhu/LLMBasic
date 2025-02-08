import torch
import torch.nn as nn

class BERTEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_ffw, num_heads, num_layers, max_len=64):
        super(BERTEncoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.segment_embedding = nn.Embedding(2, embed_size)
        #self.pos_embedding = nn.Parameter(torch.randn(1, max_len, embed_size))
        self.pos_embedding = nn.Embedding(max_len, embed_size)
        self.register_buffer(
            "position_ids", torch.arange(max_len).expand((1, -1)), persistent=False
        )
        encoder_layer = nn.TransformerEncoderLayer(embed_size, num_heads, num_ffw, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

   
    def forward(self, tokens, segments, src_mask, src_key_padding_mask):
        # X = self.token_embedding(tokens) + self.segment_embedding(segments)
        # X += self.pos_embedding.data[:, :X.shape[1], :]
        position_ids = self.position_ids[:, :tokens.shape[1]] # 取出 [0,1,2,...]
        X = self.token_embedding(tokens) \
          + self.segment_embedding(segments) \
          + self.pos_embedding(position_ids)
        output = self.encoder.forward(X, src_mask, src_key_padding_mask)
        return output

class MaskLM(nn.Module):
    def __init__(self, vocab_size, num_hiddens, num_inputs=768):
        super(MaskLM, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions] # gather
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat

class NextSentencePred(nn.Module):
    def __init__(self, num_inputs):
        super(NextSentencePred, self).__init__()
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        return self.output(X)

class BERTModel(nn.Module):
    def __init__(self, vocab_size, embed_size,
                 ffn_num_hiddens, num_heads, num_layers, 
                 max_len=1000):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, embed_size, 
                    ffn_num_hiddens, num_heads, num_layers,
                    max_len=max_len)
        self.pool = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.Tanh()
        )
        self.mlm = MaskLM(vocab_size, embed_size, embed_size)
        self.nsp = NextSentencePred(embed_size)

    def forward(self, tokens, segments, src_mask, src_key_padding_mask, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, src_mask, src_key_padding_mask)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # 用于下一句预测的多层感知机分类器的隐藏层，0是“<cls>”标记的索引
        output = self.pool(encoded_X[:, 0, :])
        nsp_Y_hat = self.nsp(output)
        return encoded_X, mlm_Y_hat, nsp_Y_hat


if __name__=="__main__":
    vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
    num_layers = 2
    encoder = BERTEncoder(vocab_size, num_hiddens, ffn_num_hiddens, num_heads, num_layers)
    print(encoder)
    tokens = torch.randint(0, vocab_size, (2, 8))
    segments = torch.tensor([
        [0, 0, 0, 0, 1, 1, 1, 1], 
        [0, 0, 0, 1, 1, 1, 1, 1]])
    encoded_X = encoder(tokens, segments, None, None)
    print(encoded_X.shape)

# mlm = MaskLM(vocab_size, num_hiddens)
# mlm_positions = torch.tensor([[1, 5, 2], [6, 1, 5]])
# mlm_Y_hat = mlm(encoded_X, mlm_positions)
# print(mlm_Y_hat.shape)

# mlm_Y = torch.tensor([[7, 8, 9], [10, 20, 30]])
# loss = nn.CrossEntropyLoss(reduction='none')
# mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
# print(mlm_l.shape)




# encoded_X = torch.flatten(encoded_X, start_dim=1)
# # NSP的输入形状:(batchsize，num_hiddens)
# nsp = NextSentencePred(encoded_X.shape[-1])
# nsp_Y_hat = nsp(encoded_X)
# print(nsp_Y_hat.shape)

# nsp_y = torch.tensor([0, 1])
# nsp_l = loss(nsp_Y_hat, nsp_y)
# print(nsp_l.shape)
