import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from H3_3_Translate_Data import SOS_token, MAX_STEPS

class EncoderRNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.embedding = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)

    def forward(self, input):
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded)
        return output, hidden


class Additive(nn.Module):
    def __init__(self, key_size, query_size, hidden_size):
        super(Additive, self).__init__()
        self.Wa = nn.Linear(query_size, hidden_size, bias=False)# 128 x 128
        self.Ua = nn.Linear(key_size, hidden_size, bias=False)  # 128 x 128
        self.Va = nn.Linear(hidden_size, 1, bias=False)         # 128 x 1

    def forward(self, query, keys, values):# query:32x1x128, keys:32x10x128
        # q = self.Wa(query) # 32 x 1 x 128
        # k = self.Ua(keys) # 32 x 10 x 128
        # qk = q + k # 广播求和 32 x 10 x 128
        # qk_tanh = torch.tanh(qk) # 32 x 10 x 128
        # s = self.Va(qk_tanh) # 32 x 10 x 1
        score = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        score = score.permute(0, 2, 1) # -> 32 x 1 x 10
        alpha = F.softmax(score, dim=-1)  # 注意力权重 -> 32 x 1 x 10
        # (32x1x10) * (32x10x128) -> 32x1x128
        context = torch.bmm(alpha, values) # 上下文向量 -> 32 x 1 x 128
        return context, alpha


class DotProductAttention(nn.Module):
    def __init__(self):
        super(DotProductAttention, self).__init__()

    # 如果改成dot-product attention，那么就是query和key的点积
    # q * k.T = 32 x 1 x 128 * 32 x 128 x 10 = 32 x 1 x 10
    # k.t = keys.permute(0, 2, 1) # 32 x 128 x 10
    def forward(self, query, keys, values):# query:32x1x128, keys:32x10x128
        d = query.size(-1)  # 128
        k = keys.permute(0, 2, 1) # 32 x 128 x 10
        # (32x1x128) * (32x128x10) = 32 x 1 x 10
        s = torch.bmm(query, k) / math.sqrt(d) # -> s: 32 x 1 x 10
        alpha = F.softmax(s, dim=-1)  # 注意力权重 -> 32 x 1 x 10
        # (32x1x10) * (32x10x128) = 32 x 1 x 128
        context = torch.bmm(alpha, values) # 上下文向量 -> 32 x 1 x 128
        return context, alpha

class AttnDecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, embed_size)
        self.attention = Additive(hidden_size, hidden_size, hidden_size)
        #self.attention = DotProductAttention()
        self.gru = nn.GRU(embed_size + hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None, device="cpu"):
        batch_size = encoder_outputs.size(0)
        input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        h = encoder_hidden
        list_outputs = []
        attention_weight = []

        for i in range(MAX_STEPS):
            z_t, h, weights = self.forward_step(input, h, encoder_outputs)
            list_outputs.append(z_t)
            attention_weight.append(weights)

            if target_tensor is not None:
                # 教师强制模式
                input = target_tensor[:, i].unsqueeze(1)
            else:
                # 自由接力模式
                _, topi = z_t.topk(1)
                input = topi.squeeze(-1).detach()

        outputs = torch.cat(list_outputs, dim=1)
        outputs = F.log_softmax(outputs, dim=-1)
        attention_weight = torch.cat(attention_weight, dim=1)
        return outputs, h, attention_weight
    
    # input: 32x1, h_t_1: 1x32x128, encoder_outputs: 32 x 10 x 128
    def forward_step(self, input, h_t_1, encoder_outputs):
        embedded =  self.embedding(input) # 32 x 1 -> 32 x 1 x 100
        query = h_t_1.permute(1, 0, 2) # -> 32 x 1 x 128
        # context_vector:32x1x128, attn_weights:32x1x10
        context_vector, attn_weights = self.attention(query, encoder_outputs, encoder_outputs)
        input_gru = torch.cat((embedded, context_vector), dim=2)
        # output:32x1x228, h_t:1x32x128, input_gru:32x1x228, h_t_1:1x32x128
        output, h_t = self.gru(input_gru, h_t_1)
        z_t = self.fc(output) # 32 x 1 x 128 -> 32 x 1 x 2991
        return z_t, h_t, attn_weights
