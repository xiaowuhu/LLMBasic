from io import open
import torch
import numpy as np
import torch.utils
from torchtext.data.utils import get_tokenizer
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


EN, FR = 0, 1  # 英语，法语
PAD, UNK, SOS, EOS  = 0, 1, 2, 3  # 四个特殊字符, PAD 必须是 0，因为后面要用非零条件判断时间步长度
special_symbols = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']

class Vocab:
    def __init__(self, name):
        self.name = name      # EN=0, FR=1
        self.word2index = {}  # 词到 id 的映射
        self.word2count = {}  # 词频
        self.index2word = {}  # id 到词的映射
        self.n_words = 0      # 词表大小
        self.add_special_symbols(special_symbols) # 加入四个特殊字符
        if self.name == FR: # 法语分词器
            self.tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')
        elif self.name == EN: # 英语分词器
            self.tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
        else:
            raise NotImplementedError

    def __getitem__(self, word):
        if word not in self.word2index:
            return UNK
        else:
            return self.word2index[word]

    def get_words_from_tokens(self, token_list):
        words = []
        for token in token_list:
            if token in self.index2word:
                word = self.index2word[token]
            else:
                word = self.index2word[UNK]
            words.append(word)
        return words

    def addWords(self, sentence):
        word_list = self.tokenizer(sentence) # 分词
        for word in word_list:
            if len(word) == 1 and ord(word)>256: # unicode char, such as \2f20
                continue
            self.addWord(word)
        return len(word_list)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    # 加入四个特殊字符
    def add_special_symbols(self, symbols):
        for symbol in symbols:
            self.addWord(symbol)


# 把句子分词再变成 id 串, 加首尾标记
def sentence_to_ids(vocab: Vocab, sentence):
    id_s = [SOS] # 加首标记
    word_list = vocab.tokenizer(sentence)  # 用 spcay 分词
    for word in word_list:
        if len(word) == 1 and ord(word)>256: # unicode char, such as \2f20
            continue
        id_s.append(vocab[word])
    id_s.append(EOS) # 加尾标记
    return id_s


# 从.txt中读取双语对，加入各自的词表，返回词表和语对（先法语后英语）
def load_train_data():
    print("Reading lines...")
    f = open('../data/eng-fra/eng-fra-10.txt', encoding='utf-8')
    #f = open('../data/eng-fra/eng-fra-train-10.txt', encoding='utf-8')
    lines = f.read().strip().split('\n')
    f.close()
    print("Read %s sentence pairs" % len(lines))
    # 建立词表
    fr_vocab = Vocab(FR)
    en_vocab = Vocab(EN)
    max_step_en = 0
    max_step_fr = 0
    # Split every line into pairs and normalize
    pairs = []
    for line in lines:
        pair = line.split('\t')
        len_fr = fr_vocab.addWords(pair[FR]) # 填入词表
        len_en = en_vocab.addWords(pair[EN])
        if len_en > max_step_en:
            max_step_en = len_en
            # print(max_step_en)
            # print(pair[EN])
        if len_fr > max_step_fr:
            max_step_fr = len_fr
        if len_fr <= 12:
            pairs.append([pair[FR], pair[EN]]) # 先法语，后英语                 
    
    print("Counted words ------")
    print("fr words:", fr_vocab.n_words, "max lenngth:", max_step_fr)
    print("en words:", en_vocab.n_words, "max lenngth:", max_step_en)
    return fr_vocab, en_vocab, pairs, max_step_fr, max_step_en


def collate_fn(batch_data):
    fr_batch, en_batch = [], []
    for x, y in batch_data:
        fr_batch.append(x[x>0])
        en_batch.append(y[y>0])
    X = torch.nn.utils.rnn.pad_sequence(fr_batch, batch_first=True, padding_value=PAD)
    Y = torch.nn.utils.rnn.pad_sequence(en_batch, batch_first=True, padding_value=PAD)
    return X, Y

def get_train_dataloader(batch_size):
    fr_vocab, en_vocab, pairs, max_step_fr, max_step_en = load_train_data()
    print("prepare data loader...")
    n = len(pairs)
    # fill with 0 - '<PAD>'
    X = np.zeros((n, max_step_fr + 2), dtype=np.int32)  # 添加首尾标记所以+2
    Y = np.zeros((n, max_step_en + 2), dtype=np.int32)

    for idx, (fr_s, en_s) in enumerate(pairs):
        fr_ids = sentence_to_ids(fr_vocab, fr_s)
        en_ids = sentence_to_ids(en_vocab, en_s)
        X[idx, :len(fr_ids)] = fr_ids
        Y[idx, :len(en_ids)] = en_ids

    train_data = TensorDataset(torch.LongTensor(X), torch.LongTensor(Y))
    train_dataloader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn,
        num_workers=4, pin_memory=True, persistent_workers=True)
    return fr_vocab, en_vocab, train_dataloader

def get_test_dataloader(fr_vocab, en_vocab, batch_size):
    print("Reading lines...")
    #f = open('../data/eng-fra/eng-fra-test.txt', encoding='utf-8')
    f = open('../data/eng-fra/eng-fra-test-10.txt', encoding='utf-8')
    lines = f.read().strip().split('\n')
    f.close()
    print("Read %s sentence pairs" % len(lines))
    # Split every line into pairs and normalize
    pairs = []
    max_step_en = 0
    max_step_fr = 0
    n = len(lines)
    for line in lines:
        pair = line.split('\t')
        pairs.append([pair[FR], pair[EN]]) # 先法语，后英语                 
        if len(pair[EN]) > max_step_en:
            max_step_en = len(pair[EN])
        if len(pair[FR]) > max_step_fr:
            max_step_fr = len(pair[FR])
    
    X = np.zeros((n, max_step_fr + 2), dtype=np.int32)  # 添加首尾标记所以+2
    Y = np.zeros((n, max_step_en + 2), dtype=np.int32)

    for idx, (fr_s, en_s) in enumerate(pairs):
        fr_ids = sentence_to_ids(fr_vocab, fr_s)
        en_ids = sentence_to_ids(en_vocab, en_s)
        X[idx, :len(fr_ids)] = fr_ids
        Y[idx, :len(en_ids)] = en_ids

    test_data = TensorDataset(torch.LongTensor(X), torch.LongTensor(Y))
    test_dataloader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn,
        num_workers=4, pin_memory=True, persistent_workers=True)
    return test_dataloader


# 把数据文件随机提取出 1000 个样本作为测试集，剩下的作为训练集，并保存文件
def split_data(count=1000):
    print("Reading lines...")
    f = open('../data/eng-fra/eng-fra-10.txt', encoding='utf-8')
    lines = f.read().strip().split('\n')
    f.close()
    np.random.seed(5)
    samples = np.random.choice(len(lines), count, replace=False)
    test_file = open('../data/eng-fra/eng-fra-test-10.txt', 'w', encoding='utf-8')
    train_file = open('../data/eng-fra/eng-fra-train-10.txt', 'w', encoding='utf-8')
    for i, line in enumerate(lines):
        if i in samples:
            test_file.write(line + '\n')
        else:
            train_file.write(line + '\n')
    test_file.close()
    train_file.close()
    print("split data done")



if __name__=="__main__":
    pass
    #test_data_loader()
    split_data(1000)
    

    # a = generate_tril_mask(5, torch.device('cpu'))
    # b = generate_square_subsequent_mask2(5, torch.device('cpu'))
    # print(a == b)
