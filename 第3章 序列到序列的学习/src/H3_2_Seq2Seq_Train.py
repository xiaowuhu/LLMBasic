import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from H3_Helper import train_seq2seq

NUM_LETTERS = 8 # 一共 8 个字符 ABCDEFGH
NUM_FEATURES = NUM_LETTERS * 2 + 2 # 加上SOS,EOS, one-hot 编码10维
NUM_STEPS = 4 # 三个字母+EOS
SOS_token = 0
EOS_token = NUM_FEATURES - 1
# ABCDEFGH -> one-hot: 1,2,3,4,5,6,7,8
# abcdefgh -> one-hot: 9,10,...,16
# 从 ABC -> abc, 遍历所有三个字符的组合，字符可重复
def generate_data():
    n = NUM_LETTERS * NUM_LETTERS * NUM_LETTERS
    X = np.zeros((n, NUM_STEPS, NUM_FEATURES), dtype=np.int32)
    Y = np.zeros((n, NUM_STEPS), dtype=np.int32)
    id = 0
    # 三个字符组合
    for char1 in range(1, NUM_LETTERS+1): # A=1, B=2, ... 0=SOS
        for char2 in range(1, NUM_LETTERS+1):
            for char3 in range(1, NUM_LETTERS+1):
                # one-hot 编码
                X[id] = np.eye(NUM_FEATURES)[[char1, char2, char3, EOS_token]]
                Y[id] = [char1+NUM_LETTERS, char2+NUM_LETTERS, char3+NUM_LETTERS, EOS_token]
                id += 1
    return X, Y


def create_dataset(batch_size):
    X, Y = generate_data()
    train_data = TensorDataset(torch.FloatTensor(X),
                               torch.LongTensor(Y))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=batch_size,
        pin_memory=True)
    return train_dataloader


class Encoder(nn.Module):
    def __init__(self, feature_size, hidden_size): # 18, 6
        super(Encoder, self).__init__()
        self.gru = nn.GRU(feature_size, hidden_size, batch_first=True)

    def forward(self, input): # 32 x 4 x 18 ->
        output, hidden = self.gru(input)
        return output, hidden # o:32 x 4 x 6, h:1 x 32 x 6


class Decoder(nn.Module):
    def __init__(self, feature_size, hidden_size, output_size): # 18, 6, 18
        super(Decoder, self).__init__()
        self.gru = nn.GRU(feature_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size) # 6 x 18

    def forward(self, batch_size, context_vector, target_tensor=None, device="cpu"):
        input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        hidden = context_vector
        decoder_outputs = []

        for i in range(NUM_STEPS):
            #32x1x8 1x32x6                      32x1   1x32x6
            output, hidden  = self.forward_step(input, hidden, device)
            decoder_outputs.append(output)
            if target_tensor is not None:
                # 训练时，教师强制模式，用标签当作下一个输入
                input = target_tensor[:, i].unsqueeze(1) # 扩充为二维数据 32 x 1
            else:
                # 预测时，自由接力模式 Free-run 用输出当作下一个输入
                _, topi = output.topk(1)
                input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, hidden, None # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden, device): # 32x1, 1x32x6
        # 把 input 变成 one-hot 编码
        onehot = torch.eye(NUM_FEATURES, device=device)[input] # 32 x 1 -> 32 x 1 x 18
        #input = F.relu(onehot)  # 如果有负值，变成 0
        output, hidden = self.gru(onehot, hidden) # -> o:32x1x6, h:1x32x6
        output = self.out(output) # 32x1x6 -> 32x1x18
        return output, hidden

if __name__=="__main__":
    #DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEVICE = torch.device("cpu")
    checkpoint = 8
    hidden_size, num_epochs, batch_size, lr = 4, 200, 32, 0.1
    train_dataloader = create_dataset(batch_size)
    encoder = Encoder(NUM_FEATURES, hidden_size)
    decoder = Decoder(NUM_FEATURES, hidden_size, NUM_FEATURES)
    
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)
    loss_function = nn.NLLLoss()
    all_loss = train_seq2seq(encoder, decoder, 
                     train_dataloader, None, 
                     loss_function, checkpoint,
                     encoder_optimizer, decoder_optimizer,
                     num_epochs, DEVICE,)
                     #save_model_names=("Encoder_ABC.pth", "Decoder_abc.pth"))
