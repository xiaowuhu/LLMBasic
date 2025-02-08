
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from H1_7_1_Names_DataReader import getRandomTrainingData, n_languages, n_letters

# 这是 pytorch 中的原版例子
class RNN1(nn.Module):
    def __init__(self, n_categories, input_size, hidden_size, output_size):
        super(RNN1, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

# 这是用标准 RNN 概念建立的模型
class RNN2(nn.Module):
    def __init__(self, n_languages, input_size, hidden_size, output_size):
        super(RNN2, self).__init__()
        self.hidden_size = hidden_size
        self.U = nn.Linear(n_languages + input_size + hidden_size, hidden_size)
        self.W = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, output_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, language, name_char, s_prev):
        h_prev = self.W(s_prev)
        hcx = torch.cat((language, name_char, h_prev), 1)
        s = self.U(hcx)
        h = F.tanh(s)
        z = self.V(h)
        a = self.logsoftmax(z)
        return a, h

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


def save_model(model: nn.Module, name: str):
    print("---- save model... ----")
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    train_pth = os.path.join(current_dir, "model", name)
    torch.save(model.state_dict(), train_pth)


def train(model, loss_func, optimizer, category_tensor, input_line_tensor, target_line_tensor, DEVICE):
    target_line_tensor.unsqueeze_(-1)
    hidden = model.initHidden().to(DEVICE)

    optimizer.zero_grad()

    loss = torch.Tensor([0]).to(DEVICE) # you can also just simply use ``loss = 0``

    for i in range(input_line_tensor.size(0)):
        output, hidden = model(category_tensor, input_line_tensor[i], hidden)
        l = loss_func(output, target_line_tensor[i])
        loss += l

    loss.backward()
    optimizer.step()

    return output, loss.item() / input_line_tensor.size(0)


if __name__ == '__main__':
    #DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    DEVICE = torch.device("cpu")
    rnn_model = RNN2(n_languages, n_letters, 128, n_letters)
    model = rnn_model.to(DEVICE)

    n_iters = 100000 # 100000
    checkpoint = 1000 # 5000
    all_losses = []
    total_loss = 0 # Reset every checkpoint
    loss_func = nn.NLLLoss()
    best_loss = 10

    optimizer = torch.optim.Adam(model.parameters(), lr= 0.001, weight_decay=1e-6)

    for iter in range(1, n_iters + 1):
        language_tensor, input_name_tensor, target_name_tensor = getRandomTrainingData()
        output, loss = train(model, loss_func, optimizer, language_tensor, input_name_tensor, target_name_tensor, DEVICE)
        total_loss += loss

        if iter % checkpoint == 0:
            avg_loss = total_loss / checkpoint
            print('(%d %d%%) avg_loss:%.4f' % (iter, iter / n_iters * 100, avg_loss))
            if avg_loss < best_loss: 
                best_loss = avg_loss
                save_model(model, "H17_7_2_All_model.pth")
            all_losses.append(avg_loss)
            total_loss = 0

    plt.figure()
    plt.plot(all_losses)
    plt.show()
    