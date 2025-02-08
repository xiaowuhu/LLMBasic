import torch
import torch.nn as nn

x1 = torch.tensor([[ 0.7145, -0.5732,  0.7637],
        [-0.4790, -0.8239,  1.0038],
        [ 1.4015,  0.3322,  0.4328],
        [-0.9872, -1.0619, -0.3931],
        [-0.5997,  0.5391,  0.1228],
        [ 0.5246, -1.4017, -0.8023],
        [ 0.6056,  0.2123,  0.9335]])

x2 = torch.tensor([[ 0.6320, -0.9176,  1.2018],  # i
        [-0.8383, -0.6971,  1.4629],  # hate
        [ 1.4015,  0.3322,  0.4328],  # coffee
        [-0.9539, -0.6687, -0.5920],  # like
        [-0.5997,  0.5391,  0.1228],  # cat
        [ 0.3940, -1.2595, -0.3765],  # love
        [ 0.6056,  0.2123,  0.9335]]) # milk


def compute_cosine(x):
    result = torch.zeros((7, 7))
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            xi = x[i]
            xj = x[j]
            s = nn.functional.cosine_similarity(xi, xj, dim=0, eps=1e-8)
            result[i, j] = s

    print(result)

if __name__=="__main__":
    compute_cosine(x1)
    compute_cosine(x2)
