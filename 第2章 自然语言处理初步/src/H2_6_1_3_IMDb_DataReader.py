import torch
import os
import collections
from torch.utils import data
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)


def tokenize(lines, token='word'):
    if token == 'word':
        return [line.split() for line in lines] # 形成二级列表
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)

def count_corpus(tokens):
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

# 文本词表
class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(
        dataset, batch_size, shuffle=is_train, 
        num_workers=4, pin_memory=True, persistent_workers=True)

def truncate_pad(line, num_steps, padding_token):
    if len(line) > num_steps:
        # 从部取 num_steps 个词, 因为评论文字通常在尾部表达意见
        return line[-num_steps:]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充

def load_data_imdb(batch_size, num_steps, show_hist=False):
    data_dir = os.path.join('..', 'data', "aclImdb")
    print("load IMDb train data...")
    train_X_sentense, train_Y = read_imdb(data_dir, True)
    print("load IMDb test data...")
    test_X_sentense, test_Y = read_imdb(data_dir, False)
    print("create vocab...")
    # 把每条评论的句子变成词列表
    train_X_words = tokenize(train_X_sentense, token='word')

    if show_hist:
        plt.hist([len(words) for words in train_X_words], bins=range(0,1000,50))
        plt.xlabel("每条评论中的词数")
        plt.ylabel("词频")
        plt.show()
    
    test_X_words = tokenize(test_X_sentense, token='word')
    # 生成词表
    vocab = Vocab(train_X_words, min_freq=5)
    print("create dataset...")
    train_word_idx = torch.tensor([truncate_pad(
        vocab[words], num_steps, vocab['<pad>']) for words in train_X_words])
    test_word_idx = torch.tensor([truncate_pad(
        vocab[words], num_steps, vocab['<pad>']) for words in test_X_words])
    train_iter = load_array((train_word_idx, torch.tensor(train_Y)),
                                batch_size)
    test_iter = load_array((test_word_idx, torch.tensor(test_Y)),
                               batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab


def read_imdb(data_dir, is_train):
    X_sentense, Y = [], []
    for name in ('pos.txt', 'neg.txt'):
        file_name = os.path.join(data_dir, 'train' if is_train else 'test', name)
        with open(file_name, 'rb') as f:
            text = f.read().decode('utf-8')
            lines = text.split("\n")
            for line in lines:
                if len(line) == 0:
                    continue
                tokens = line.lower().replace("\r", "")
                X_sentense.append(tokens)
                Y.append(1 if name == 'pos.txt' else 0)  # pos.txt 对应的标签为 1
    return X_sentense, Y


if __name__=="__main__":
    load_data_imdb(64, 500, True)
