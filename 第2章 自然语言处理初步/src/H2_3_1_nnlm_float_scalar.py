import torch
import torch.nn as nn
import torch.optim as optimizer
import torch.utils.data as Data

def make_data_float(sentences):
    input_data = []
    target_data = []
    for sen in sentences:
        sen = sen.split()  # ['i', 'like', 'cat']
        input_tmp = [word2idx[w] for w in sen[:-1]]
        target_tmp = word2idx[sen[-1]]
        c_w = torch.tensor(input_tmp) / (V_count-1)
        input_data.append(c_w)
        target_data.append(target_tmp)
    return input_data, target_data


sentences = ['i like cat', 'i love coffee', 'i hate milk']
sentences_list = " ".join(sentences).split()  # ['i', 'like', 'cat', 'i', 'love'. 'coffee',...]
V = list(set(sentences_list))
word2idx = {w: i for i, w in enumerate(V)}
idx2word = {i: w for i, w in enumerate(V)}
print(word2idx)
V_count = len(V)

input_data, target_data = make_data_float(sentences)
input_data, target_data = (torch.stack(input_data)).float(), torch.LongTensor(target_data)
dataset = Data.TensorDataset(input_data, target_data)
loader = Data.DataLoader(dataset, 3, True)

# parameters
n = 2  # 用 n 个词预测第 n+1 个词(相当于用n-1个词预测第n个词)
h = 10  # 隐层神经元数量

class NNLM(nn.Module):
    def __init__(self):
        super(NNLM, self).__init__()
        self.H = nn.Parameter(torch.randn(n, h), requires_grad=True)
        self.U = nn.Parameter(torch.randn(h, V_count), requires_grad=True)

    def forward(self, input):
        x = input
        hidden_out = torch.tanh(x.matmul(self.H))
        output = hidden_out.matmul(self.U)
        return output


model = NNLM().cuda()

optimizer = optimizer.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(1, 1000 + 1):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        if epoch % 100 == 0:
            print("epoch:{}, loss:{}".format(epoch, loss.item()))
        loss.backward()
        optimizer.step()

pred = model(input_data.cuda()).max(1, keepdim=True)[1]
print([idx2word[idx.item()] for idx in pred])

# print("H:", model.H)
# print("W:", model.W)
# print("U:", model.U)
