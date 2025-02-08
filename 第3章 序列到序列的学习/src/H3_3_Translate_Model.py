import torch
import torch.nn as nn
import torch.nn.functional as F

from H3_3_Translate_Data import SOS_token, MAX_STEPS


class EncoderRNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size): # 4601, 100, 128
        super(EncoderRNN, self).__init__()
        self.embedding = nn.Embedding(input_size, embed_size) # 需要训练
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)

    def forward(self, input):
        embedded = self.embedding(input) # 32x10 -> 32x10x100
        output, hidden = self.gru(embedded) # -> o: 32x10x128, h: 1x32x128
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size): # 100, 128, 2991
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    # batch_size: 32, context_vector: 1x32x128, target_tensor: 32x10
    def forward(self, batch_size, context_vector, target_tensor=None, device="cpu"):
        # 第一个词是 <SOS>
        input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        h = context_vector  # 上下文向量 c
        list_outputs = []

        for i in range(MAX_STEPS):  # 由于目标是英文，所以最长为 10
            z_t, h  = self.forward_step(input, h)
            list_outputs.append(z_t)

            if target_tensor is not None:
                # 训练时，教师强制模式，用标签当作下一个输入
                input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # 预测时，自由接力模式 Free-run 用输出当作下一个输入
                _, topi = z_t.topk(1)
                input = topi.squeeze(-1).detach()  # detach from history as input

        outputs = torch.cat(list_outputs, dim=1)
        outputs = F.log_softmax(outputs, dim=-1)
        return outputs, h, None # We return `None` for consistency in the training loop
    # 每个时间步训练
    def forward_step(self, input, h_t_1): # input:32x1, h_t-1:1x32x128
        output = self.embedding(input)  # 32 x 1 -> 32 x 1 x 100
        output, h_t = self.gru(output, h_t_1) # o:32x1x128, h:1x32x128
        z_t = self.fc(output) # -> 3x1x2991
        return z_t, h_t
