from io import open
import unicodedata
import re
import random
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import numpy as np

SOS_token = 0
EOS_token = 1
MAX_STEPS = 10
eng_prefixes_filter = (
    "i am ", "i m ", "he is", "he s ", "she is", "she s ",
    "you are", "you re ", "we are", "we re ", "they are", "they re "
)

class Vocab:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addWords(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()

def read_eng_fra_file():
    print("Reading lines...")
    f = open('../data/eng-fra/eng-fra.txt', encoding='utf-8')
    lines = f.read().strip().split('\n')
    f.close()
    # Split every line into pairs and normalize
    pairs = []
    for line in lines:
        pair = line.split('\t')
        s0 = normalizeString(pair[0])
        s1 = normalizeString(pair[1])
        pairs.append([s0, s1]) # 先法语，后英语                 
  
    return pairs


def filterPair(pair):
    return len(pair[0].split(' ')) < MAX_STEPS and \
        len(pair[1].split(' ')) < MAX_STEPS and \
        pair[0].startswith(eng_prefixes_filter)


def filterPairs(pairs):
    filter_pairs = []
    for pair in pairs:
        if len(pair[0].split(' ')) < MAX_STEPS and \
           len(pair[1].split(' ')) < MAX_STEPS and \
           pair[0].startswith(eng_prefixes_filter):
            filter_pairs.append(pair)
    return filter_pairs


def loadWholeData():
    raw_pairs = read_eng_fra_file()
    print("Read %s sentence pairs" % len(raw_pairs))
    filter_pairs = filterPairs(raw_pairs)
    print("Trimmed to %s sentence pairs" % len(filter_pairs))
    return filter_pairs


# 把句子分词再变成 id 串
def indexesFromSentence(vocab: Vocab, sentence):
    id_s = []
    for word in sentence.split(' '):
        id_s.append(vocab.word2index[word])
    return id_s


def loadSmallData():
    print("Reading lines...")
    f = open('../data/eng-fra/eng-fra-10.txt', encoding='utf-8')
    lines = f.read().strip().split('\n')
    f.close()
    print("Read %s sentence pairs" % len(lines))
    # 建立词表
    franch_vocab = Vocab("fra")
    english_vocab = Vocab("eng")
    # Split every line into pairs and normalize
    pairs = []
    for line in lines:
        pair = line.split('\t')
        s0 = normalizeString(pair[0])
        s1 = normalizeString(pair[1])
        pairs.append([s1, s0]) # 先法语，后英语                 
        franch_vocab.addWords(s1) # 填入词表
        english_vocab.addWords(s0)
    
    print("Counted words:")
    print(franch_vocab.name, franch_vocab.n_words)
    print(english_vocab.name, english_vocab.n_words)
    return franch_vocab, english_vocab, pairs


def get_dataloader(batch_size):
    franch_vocab, english_vocab, pairs = loadSmallData()
    print("prepare data loader...")
    n = len(pairs)
    X_ids = np.zeros((n, MAX_STEPS), dtype=np.int32)
    Y_ids = np.zeros((n, MAX_STEPS), dtype=np.int32)

    for idx, (french_lang, english_lang) in enumerate(pairs):
        fra_ids = indexesFromSentence(franch_vocab, french_lang)
        eng_ids = indexesFromSentence(english_vocab, english_lang)
        fra_ids.append(EOS_token)
        eng_ids.append(EOS_token)
        X_ids[idx, :len(fra_ids)] = fra_ids
        Y_ids[idx, :len(eng_ids)] = eng_ids

    train_data = TensorDataset(torch.LongTensor(X_ids),
                               torch.LongTensor(Y_ids))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=batch_size,
        num_workers=4, pin_memory=True, persistent_workers=True)
    return franch_vocab, english_vocab, train_dataloader


if __name__=="__main__":
    pairs = loadWholeData()
    print(random.choice(pairs))
    # 把过滤后得到的句子写到文件中
    f = open("../data/eng-fra/eng-fra-10.txt", "w+")
    for pair in pairs:
        s = pair[0] + "\t" + pair[1] + "\n"
        f.write(s)
    f.close()
