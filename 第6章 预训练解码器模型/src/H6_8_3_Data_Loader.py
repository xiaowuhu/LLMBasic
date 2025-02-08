import torch
import numpy as np
import pickle
from torch.utils.data import TensorDataset, DataLoader
from tokenizers import Tokenizer


def load_train_data(batch_size):
    ids_path = "../data/Tang_poetry/train_ids.pkl"
    ids_file = open(ids_path, "rb")
    list_data = pickle.load(ids_file)
    ids_file.close()
    line_count = len(list_data)
    # 得到所有样本中最长的 token 数量
    max_len = 0
    for i in range(line_count):
        line = list_data[i]
        max_len = max(max_len, len(line))
    # 建立二维数组
    X = np.zeros((line_count, max_len), dtype=np.int32)
    for i in range(line_count):
        line = list_data[i]
        X[i, :len(line)] = line
    
    train_dataset = TensorDataset(torch.LongTensor(X))
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        collate_fn = collate_fn,
        shuffle=True,
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=True)
    
    return train_dataloader
    

def collate_fn(batch_data):
    x_batch = []
    for x, in batch_data:  # batch_data 是一个 tuple，所以要用 x, 来获得，一般情况是 (x,y)
        x_batch.append(x[x>0])
    X = torch.nn.utils.rnn.pad_sequence(x_batch, batch_first=True, padding_value=3)
    return X


if __name__ == '__main__':
    load_train_data(4)
    # 特殊字符的 token id
    file = "../model/ch21/bpe-poetry-7000.json"
    my_tokenizer: Tokenizer = Tokenizer.from_file(file)
    pad_token = my_tokenizer.encode("[PAD]")
    print(pad_token.ids)
