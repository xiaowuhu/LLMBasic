import torch
import torch.nn as nn
import torch.nn.functional as F

def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项

    Defined in :numref:`sec_seq2seq_decoder`"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作

    Defined in :numref:`sec_attention-scoring-functions`"""
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


def tril_mask(X):
    ones = torch.ones_like(X)
    mask = torch.tril(ones)
    X = X.masked_fill(mask == 0, -float("inf"))
    print("上三角掩码:\n", X)
    return F.softmax(X, dim=-1)


X = torch.rand((2, 4, 4))
print("原始数据:\n", X)
X1 = sequence_mask(X, torch.tensor([3, 4]))
print("序列掩码:\n", X1)

print("-------------")

X = torch.rand((1, 3, 3))
print("原始数据:\n", X)
print("softmax:\n", F.softmax(X, dim=-1))
X2 = masked_softmax(X, torch.tensor([2]))
print("掩码softmax:\n", X2)

print("-------------")

X = torch.rand(4,4)
print("原始数据:\n", X)
X3 = tril_mask(X)
print("掩码权重:\n", X3)