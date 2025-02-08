import torch.nn as nn

class model(nn.Module):
    def __init__(self, devices):
        self.fc1 = nn.Linear(1024, 2048).to(devices[0])
        self.fc2 = nn.Linear(2048, 2048).to(devices[1])
        self.fc3 = nn.Linear(2048, 2048).to(devices[2])
        self.fc4 = nn.Linear(2048, 1000).to(devices[3])

    def forward(self, x, devices):
        x = x.to(devices[0])
        x = self.fc1(x).to(devices[1])
        x = self.fc2(x).to(devices[2])
        x = self.fc3(x).to(devices[3])
        y = self.fc4(x)
        return y

