import torch
import torch.nn as nn
import torch.optim as optim
import os

from H8_6_E_Data import generate_data

def save_model(model: nn.Module, name: str):
    print("---- save model... ----")
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    train_pth = os.path.join(current_dir, "model", name)
    torch.save(model.state_dict(), train_pth)

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        #x = torch.tanh(self.layer2(x))
        #x = torch.relu(self.layer3(x))
        #x = torch.relu(self.layer4(x))
        return torch.softmax(self.out(x), dim=1)

class Expert2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):        
        super(Expert2, self).__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, 1, hidden_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input(x)
        x = self.encoder(x)
        x = self.out(x)
        return torch.softmax(x, dim=1)

def train_model(model, x, y, epochs, learning_rate, device):
    model.to(device)
    x = x.to(device)
    y = y.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if (epoch+1)%10 == 0:
            print(f"loss={loss.item()}")
        if (epoch+1)%100 == 0:
            test(model, x_test, y_test, device)
            save_model(model, "Emergent.pth")

# Evaluate all models
def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        outputs = model(x)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == y).sum().item()
        accuracy = correct / len(y)
    return accuracy

def main(x_train, y_train, device):
    model = Expert(input_dim, hidden_dim, output_dim)
    epochs, learning_rate = 1000, 0.001
    train_model(model, x_train, y_train, epochs, learning_rate, device)
    return model

def test(model, x_test, y_test, device):
    x = x_test.to(device)
    y = y_test.to(device)
    accuracy_moe = evaluate(model, x, y)
    print(f"测试集准确率:{accuracy_moe * 100}%")

if __name__=="__main__":
    # 生成数据
    num_samples, input_dim, hidden_dim, output_dim = 6000, 4, 16, 3
    x_train, y_train, x_test, y_test = generate_data(num_samples, input_dim)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")    
    model = main(x_train, y_train, device)
    test(model, x_test, y_test, device)
