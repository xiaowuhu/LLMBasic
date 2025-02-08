import torch


z32 = torch.tensor(1.00001, dtype=torch.float32)
y32 = torch.tensor(1.0, dtype=torch.float32)
loss = torch.nn.functional.mse_loss(z32, y32)
print(loss)

g32 = 2 * (z32 - y32)
print(g32)

z16 = torch.tensor(1.00001, dtype=torch.float16)
y16 = torch.tensor(1.0, dtype=torch.float16)
g16 = 2 * (z16 - y16)
print(g16)

