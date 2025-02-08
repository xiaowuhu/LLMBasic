import torch
import torch.nn.functional as F
import math


def local_attention(query, key, value, window_size):
    batch_size, seq_len, embed_dim = query.size()
    # 将序列划分为多个窗口
    sub_seq_len = seq_len // window_size
    queries = query.view(batch_size, sub_seq_len, window_size, embed_dim).permute(1, 0, 2, 3)
    keys = key.view(batch_size, sub_seq_len, window_size, embed_dim).permute(1, 0, 2, 3)
    values = value.view(batch_size, sub_seq_len, window_size, embed_dim).permute(1, 0, 2, 3)
    
    # 计算窗口内的注意力
    scores = torch.matmul(queries, keys.transpose(-1, -2))
    scores = scores / math.sqrt(embed_dim)
    attention = F.softmax(scores, dim=-1)
    
    # 应用注意力权重
    context = torch.matmul(attention, values)
    
    # 将窗口内的结果拼接回原始序列
    context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, seq_len, embed_dim)
    return context
