import torch 

x = torch.arange(1.0, 13.0).reshape(3,4)
print(x)

y = torch.layer_norm(x, (4,))
print(y)

rms_norm = torch.nn.RMSNorm((4,))
y = rms_norm(x)
print(y)

