import torch
import torch.nn as F

y = torch.tensor([1,2])
z = torch.tensor([[1.,2.,3.,4.,5.],[1,4,3,2,3.]]) # NxC=2,5
print("LogSoftmax + NLLLoss ---------")
logsoftmax = F.LogSoftmax(dim=1)
a = logsoftmax(z)
print("a=",a)
loss_func = F.NLLLoss()
loss = loss_func(a, y)
print("loss=", loss)
print("Softmax + CrossEntropyLoss -------")
softmax = F.Softmax(dim=1)
a = softmax(z)
print("a=",a)
loss_func = F.CrossEntropyLoss()
loss = loss_func(z, y)
print("loss=", loss)

