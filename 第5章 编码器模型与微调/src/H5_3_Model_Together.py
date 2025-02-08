import torch
import torch.nn as nn

class BERTModel(nn.Module):
    def __init__(self, vocab_size, embed_size,
                 ffn_num_hiddens, num_heads, num_layers, 
                 max_len=1000):
        super(BERTModel, self).__init__()
        # embedding and encoder
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.segment_embedding = nn.Embedding(2, embed_size)
        self.pos_embedding = nn.Embedding(max_len, embed_size)
        self.register_buffer(
            "position_ids", torch.arange(max_len).expand((1, -1)), persistent=False
        )
        encoder_layer = nn.TransformerEncoderLayer(embed_size, num_heads, ffn_num_hiddens, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)        
        # MLM
        self.mlm = nn.Sequential(nn.Linear(embed_size, embed_size),
                                 nn.ReLU(),
                                 nn.LayerNorm(embed_size),
                                 nn.Linear(embed_size, vocab_size))        
        # classifier
        self.pool = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.Tanh()
        )
        # NSP
        self.nsp = nn.Linear(embed_size, 2)

    def encoder_forward(self, tokens, segments, src_mask, src_key_padding_mask):
        position_ids = self.position_ids[:, :tokens.shape[1]] # 取出 [0,1,2,...]
        X = self.token_embedding(tokens) \
          + self.segment_embedding(segments) \
          + self.pos_embedding(position_ids)
        output = self.encoder.forward(X, src_mask, src_key_padding_mask)
        return output
    
    def mlm_forward(self, X, mask_pos):
        # 做各种矩阵变换，最终根据 mask_pos 的数据从 X 中获得被掩码的词向量
        num_masked = mask_pos.shape[1]  # 10
        mask_pos = mask_pos.reshape(-1) # 32 x 10 -> 320
        batch_size = X.shape[0] # 32
        batch_idx = torch.arange(0, batch_size) # [0, 1, 2, ..., 31]
        batch_idx = torch.repeat_interleave(batch_idx, num_masked) # [0, 0, ..., 0, 1, 1, ..., 31, 31]
        masked_X = X[batch_idx, mask_pos] # gather
        masked_X = masked_X.reshape((batch_size, num_masked, -1)) # 320x128 -> 320x10x128
        mlm_Y_hat = self.mlm(masked_X)
        return mlm_Y_hat

    def forward(self, tokens, segments, src_mask, src_key_padding_mask, pred_positions=None):
        encoded_X = self.encoder_forward(tokens, segments, src_mask, src_key_padding_mask)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm_forward(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # 用于下一句预测的多层感知机分类器的隐藏层，0是“<cls>”标记的索引
        output = self.pool(encoded_X[:, 0, :])
        nsp_Y_hat = self.nsp(output)
        return encoded_X, mlm_Y_hat, nsp_Y_hat


if __name__=="__main__":
    vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
    num_layers = 2
    encoder = BERTModel(vocab_size, num_hiddens, ffn_num_hiddens, num_heads, num_layers)
    tokens = torch.randint(0, vocab_size, (2, 8))
    segments = torch.tensor([
        [0, 0, 0, 0, 1, 1, 1, 1], 
        [0, 0, 0, 1, 1, 1, 1, 1]])
    encoded_X = encoder(tokens, segments, None, None)
    print(encoded_X[0].shape)

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
