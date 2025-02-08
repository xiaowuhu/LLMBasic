import os
import collections

def read_ptb(name):
    #current_path = os.path.abspath(__file__)
    #current_dir = os.path.dirname(current_path)
    filename = os.path.join("..", "data", name)
    with open(filename) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]

class Vocab:
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        """Defined in :numref:`sec_text_preprocessing`"""
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

def count_corpus(tokens):
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

def generate_ngrams(ngram_dict, tokens, n):
    # 获取n-gram的起始索引范围
    max_index = len(tokens) - n + 1
    # 遍历文本，生成n-grams
    for i in range(max_index):
        # 提取当前n-gram的单词
        gram = ' '.join(tokens[i:i+n])
        # 将n-gram添加到字典中
        if gram in ngram_dict:
            ngram_dict[gram] += 1
        else:
            ngram_dict[gram] = 1


if __name__=="__main__":
    sentences = read_ptb("ptb.train.txt")
    print(f'sentences size: {len(sentences)}')
    vocab = Vocab(sentences, min_freq=0)
    print(f'vocab size: {len(vocab)}')

    # 生成n-grams
    n = 5
    ngrams= {}
    for sentence in sentences:
        generate_ngrams(ngrams, sentence, n)
    
    # 打印生成的n-grams
    print(f"生成的{n}-grams:", len(ngrams))
    count = 0
    for ngram, freq in ngrams.items():
        if freq > 1000:
            print(ngram, ":", freq)
            count += 1
            if count > 20:
                break
    