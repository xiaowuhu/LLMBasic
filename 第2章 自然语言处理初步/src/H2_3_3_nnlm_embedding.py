import torch
import torch.nn as nn
import torch.optim as optimizer
import torch.utils.data as Data
import matplotlib.pyplot as plt

# 建立 V
sentences = ['i like cat', 'i love coffee', 'i hate milk']
sentences_list = " ".join(sentences).split()  # ['i', 'like', 'cat', 'i', 'love'. 'coffee',...]
V = list(set(sentences_list))
word2idx = {w: i for i, w in enumerate(V)}
idx2word = {i: w for i, w in enumerate(V)}
print(word2idx)
print(idx2word)
V_count = len(V)

# 准备数据，得到 one-hot
def make_data_one_hot(sentences):
    input_data = []
    target_data = []
    for sen in sentences:
        sen = sen.split()  # ['i', 'like', 'cat']
        input_tmp = [word2idx[w] for w in sen[:-1]]
        target_tmp = word2idx[sen[-1]]
        c_w = nn.functional.one_hot(torch.tensor(input_tmp), V_count)
        input_data.append(c_w)
        target_data.append(target_tmp)
    return input_data, target_data

# 训练数据
input_data, target_data = make_data_one_hot(sentences)
input_data, target_data = (torch.stack(input_data)).float(), torch.LongTensor(target_data)
dataset = Data.TensorDataset(input_data, target_data)
loader = Data.DataLoader(dataset, 3, True)

# 网络参数
n = 3
h_n = 1
m = 2
mn = m * (n-1)
# 网络结构
class NNLM(nn.Module):
    def __init__(self):
        super(NNLM, self).__init__()
        self.C = nn.Parameter(torch.randn(V_count, m), requires_grad=True)
        self.H = nn.Parameter(torch.randn(mn, h_n), requires_grad=True)
        self.W = nn.Parameter(torch.randn(mn, V_count), requires_grad=True)
        self.U = nn.Parameter(torch.randn(h_n, V_count), requires_grad=True)

    def forward(self, input):
        c = input.matmul(self.C)
        x = c.view(-1, mn)
        h = torch.tanh(x.matmul(self.H))
        output = x.matmul(self.W) + h.matmul(self.U)
        return output

# 训练
model = NNLM().cuda()
print("初始化 C:", model.C)

optimizer = optimizer.Adam(model.parameters(), lr=0.01)
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

# 预测
pred = model(input_data.cuda()).max(1, keepdim=True)[1]  # B x 1
print([idx2word[idx.item()] for idx in pred])
print("结果 C:", model.C)

# 绘制向量图
def draw_vector(C):
    for id, word in idx2word.items():
        if word in ["milk","cat","coffee"]:
            continue
        vector = C[id]
        plt.plot((0,vector[0]),(0,vector[1]))
        plt.text(vector[0],vector[1], word)
    plt.grid()
    plt.show()

draw_vector(model.C.cpu().detach().numpy())

