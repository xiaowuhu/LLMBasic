import torch
import torch.nn as nn
import torch.optim as optimizer
import torch.utils.data as Data

sentences = ['i like cat', 'i love coffee', 'i hate milk']
sentences_list = " ".join(sentences).split()  # ['i', 'like', 'cat', 'i', 'love'. 'coffee',...]
vocab = list(set(sentences_list))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for i, w in enumerate(vocab)}

V = len(vocab)


def make_data(sentences):
    input_data = []
    target_data = []
    for sen in sentences:
        sen = sen.split()  # ['i', 'like', 'cat']
        input_tmp = [word2idx[w] for w in sen[:-1]]
        target_tmp = word2idx[sen[-1]]

        input_data.append(input_tmp)
        target_data.append(target_tmp)
    return input_data, target_data


input_data, target_data = make_data(sentences)
input_data, target_data = torch.LongTensor(input_data), torch.LongTensor(target_data)
dataset = Data.TensorDataset(input_data, target_data)
loader = Data.DataLoader(dataset, 3, True)

# parameters
m = 2
n_step = 2
h = 1


class NNLM(nn.Module):
    def __init__(self):
        super(NNLM, self).__init__()
        self.C = nn.Embedding(V, m)
        self.H = nn.Parameter(torch.randn(h, n_step * m), requires_grad=True)
        self.d = nn.Parameter(torch.zeros(h), requires_grad=True)
        self.W = nn.Parameter(torch.randn(V, n_step * m), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(V), requires_grad=True)
        self.U = nn.Parameter(torch.randn(V, h), requires_grad=True)

    def forward(self, input):
        # input -> B x n_step

        x = self.C(input).view(-1, n_step * m)  # B x (n_step x m)

        hidden_out = torch.tanh(x.matmul(self.H.transpose(0, 1)) + self.d)  # B x h

        output = x.matmul(self.W.transpose(0, 1)) + self.b + hidden_out.matmul(self.U.transpose(0, 1))  # B x V

        return output


model = NNLM().cuda()

optimizer = optimizer.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


for epoch in range(1, 1000 + 1):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()

        # batch_x -> B x n_step
        # batch_y -> B
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()

        pred = model(batch_x)  # B x V

        loss = criterion(pred, batch_y)

        if epoch % 100 == 0:
            print("epoch:{}, loss:{}".format(epoch, loss.item()))

        loss.backward()
        optimizer.step()


pred = model(input_data.cuda()).max(1, keepdim=True)[1]  # B x 1

print([idx2word[idx.item()] for idx in pred])

print([idx2word[idx.item()] for idx in pred.squeeze()])  # ['cat', 'coffee', 'milk']

print("H:", model.H)
print("d:", model.d)
print("W:", model.W)
print("b:", model.b)
print("U:", model.U)
