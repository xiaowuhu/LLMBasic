from torch import nn
import torch

class PositionWiseFN(nn.Module):
    def __init__(self, num_input, num_hiddens, num_outputs):
        super(PositionWiseFN, self).__init__()
        self.fc1 = nn.Linear(num_input, num_hiddens)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(num_hiddens, num_outputs)

    def forward(self, X):
        return self.fc2(self.relu(self.fc1(X)))

pwfn = PositionWiseFN(4, 5, 6)
pwfn.eval()
x = torch.ones((1, 3, 4))
x = torch.tensor([[[1, 2, 3, 4], [1, 2, 3, 4], [0, 2, 3, 4]]], dtype=torch.float32)
print(x)
output = pwfn(x)
print(output)
