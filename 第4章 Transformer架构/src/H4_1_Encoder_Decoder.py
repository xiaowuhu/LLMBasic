import torch.nn as nn
from H4_1_MultiHead import MultiHeadAttention

class EncoderCell(nn.Module):
    def __init__(self, key_size, value_size, num_hiddens, norm_shape, num_heads):
        super(EncoderCell, self).__init__()
        # 第一层
        self.attention = MultiHeadAttention(key_size, value_size, num_heads)
        self.addnorm1 = nn.LayerNorm(norm_shape)
        # 第二层
        self.addnorm2 = nn.LayerNorm(norm_shape)
        self.ffn = nn.Sequential(
            nn.Linear(value_size, num_hiddens),
            nn.ReLU(),
            nn.Linear(num_hiddens, num_hiddens),
        )

    def forward(self, H0):
        H1 = self.addnorm1(H0, self.attention(H0))
        H2 = self.addnorm2(H1, self.ffn(H1))
        return H2


class DecoderCell(nn.Module):
    """解码器中第i个块"""
    def __init__(self, key_size, value_size, num_hiddens, norm_shape,  num_heads):
        super(DecoderCell, self).__init__()
        # 第一层
        self.attention1 = MaskMultiHeadAttention(key_size, value_size, num_heads)
        self.addnorm1 = nn.LayerNorm(norm_shape)
        # 第二层
        self.attention2 = MultiHeadAttention(key_size, value_size, num_heads)
        self.addnorm2 = nn.LayerNorm(norm_shape)
        # 第三层
        self.ffn = nn.Linear(value_size, num_hiddens)
        self.addnorm3 = nn.LayerNorm(norm_shape)

    def forward(self, S0, H):
        S1 = self.addnorm1(S0, self.attention1(S0))
        S2 = self.addnorm2(S1, self.attention2(S1, H, H))
        S3 = self.addnorm3(S2, self.ffn(S2))
        return S3

# 此文件只是示意性，其中的 MaskMultiHeadAttention 没有实现
