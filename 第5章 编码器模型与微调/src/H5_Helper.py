from torchtext.data.utils import get_tokenizer
import random
import torch.nn as nn
import torch
import os
import numpy as np

# 词表类
class Vocab:
    def __init__(self, special_symbols = None):
        self.word2index = {}  # 词到 id 的映射
        self.word2count = {}  # 词频
        self.index2word = {}  # id 到词的映射
        self.n_words = 0      # 词表大小
        self._add_special_symbols(special_symbols) # 加入四个特殊字符

    def __getitem__(self, word):
        if word not in self.word2index:
            return self.word2index['<unk>']
        else:
            return self.word2index[word]

    def get_ids_from_words(self, word_list):
        tokens = []
        for word in word_list:
            token = self[word]
            tokens.append(token)
        return tokens

    def get_words_from_ids(self, token_list):
        words = []
        for token in token_list:
            if token in self.index2word:
                word = self.index2word[token]
            else:
                word = '<unk>'
            words.append(word)
        return words

    def get_word_from_id(self, token_id):
        if token_id in self.index2word:
            return self.index2word[token_id]
        else:
            return '<unk>'

    def addWords(self, word_list):
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

    def randomChoice(self, start=0):
        return random.randint(start, self.n_words - 1)

    def get_total_token(self):
        total = 0
        for k,v in self.word2count.items():
            total += v
        return total

    # 加入特殊字符
    def _add_special_symbols(self, symbols):
        print("特殊字符:", symbols)
        for symbol in symbols:
            self.addWord(symbol)

    def __len__(self):
        return self.n_words

# 分词器类
class Tokenizer:
    # space:按空格分词
    # spacy:用spacy包分词,需要安装
    def __init__(self, tokenizer_name="space"):
        self.tokenizer_name = tokenizer_name
        if self.tokenizer_name == "spacy":
            self.tokenize = get_tokenizer('spacy', language='en_core_web_sm')
        elif self.tokenizer_name == "space":
            self.tokenize = self.tokenize_space
        elif self.tokenizer_name == "char":
            self.tokenize = self.tokenize_char
        else:
            raise NotImplementedError

    def tokenize_space(self, line):
        return line.split() # 用空格分词

    def tokenize_char(self, line):
        return [list(line)]

def save_model(model: nn.Module, name: str):
    print("---- save model... ----")
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    train_pth = os.path.join(current_dir, "model", name)
    torch.save(model.state_dict(), train_pth)

def load_model(model:nn.Module, name:str, device):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    model_pth = os.path.join(current_dir, "model", name)
    print("load model ", name)
    model.load_state_dict(torch.load(model_pth, map_location=device, weights_only=True))

def create_mask(src, DEVICE, PAD):
    src_seq_len = src.shape[1] 
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)
    src_key_padding_mask = (src == PAD)
    return src_mask, src_key_padding_mask

def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
