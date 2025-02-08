import torch
import random
from H5_Helper import Vocab, Tokenizer

PAD, UNK, CLS, SEP, MASK  = 0, 1, 2, 3, 4  # 五个特殊字符, PAD 必须是 0
special_symbols = ['<pad>', '<unk>', '<cls>', '<sep>', '<mask>']

# 句子，一个句子包含多个词，分词使用指定的分词器
class Sentence():
    def __init__(self, s, tokenizer, vocab, is_train):
        self.word_list = tokenizer.tokenize(s) # 分词
        if is_train:
            vocab.addWords(self.word_list) # 加入词表

    def __len__(self):
        return len(self.word_list)

    def __getitem__(self, idx):
        return self.word_list[idx]

# 段落，一个段落包含多个句子，每个句子以句号做分隔符，但是不包含句号
class Paragraph():
    def __init__(self):
        self.sentence_list: list[Sentence] = []

    def read(self, line, tokenizer, vocab, keep_case, is_train):
        ss = line.split(' . ')  # wiki 中以 ' . ' 为句子分隔符
        for s in ss:
            if keep_case == False: # 不保留大小写
                s = s.lower()
            s = s.strip()
            sentence = Sentence(s, tokenizer, vocab, is_train)
            if len(sentence) < 5: # 句子太短，丢弃
                continue
            self.sentence_list.append(sentence)
        return len(self)

    def __len__(self):
        return len(self.sentence_list)

    def get_sentence_by_idx(self, idx, segment_id, vocab: Vocab):
        token_ids = vocab.get_ids_from_words(self.sentence_list[idx].word_list)
        if segment_id == 0: # 第一个句子要加上 [CLS] 和 [SEP]
            tokens = [CLS] + token_ids + [SEP]
        elif segment_id == 1: # 第二个句子只有 [SEP]
            tokens = token_ids + [SEP]
        else:
            raise ValueError("segment_id must be 0 or 1")
        segment = [segment_id] * len(tokens)
        return tokens, segment


# 文章，一篇文章包含很多小节
class Text():
    def __init__(self):
        self.paragraph_list : list[Paragraph] = []

    def read(self, lines, tokenizer, vocab, keep_case, is_train):
        for line in lines:
            paragraph = Paragraph()
            sentence_len = paragraph.read(line, tokenizer, vocab, keep_case, is_train)
            if sentence_len > 2:
                self.paragraph_list.append(paragraph)

    def __len__(self):
        return len(self.paragraph_list)

    def get_random_sentence(self, current_idx, vocab):
        while True:
            random_idx = random.choice(range(len(self)))
            # 两个段落之间的距离不能太近,尤其是在同一个小节内都是描述同一个对象
            # 在 wikitext 中用一个等号标志  = xxx = 表示一个小节
            if abs(random_idx - current_idx) > 20: 
                break
        return self.paragraph_list[random_idx].get_sentence_by_idx(0, 1, vocab)

    def get_nsp_data(self, vocab, max_len):
        nsp_data = []
        for paragraph_id in range(len(self.paragraph_list)):
            paragraph = self.paragraph_list[paragraph_id]
            for sentence_id in range(len(paragraph) - 1):
                sentence_a, segment_a = paragraph.get_sentence_by_idx(sentence_id, 0, vocab)
                if random.random() < 0.5:
                    is_next = True
                    sentence_b, segment_b = paragraph.get_sentence_by_idx(sentence_id + 1, 1, vocab)
                else:
                    is_next = False
                    sentence_b, segment_b = self.get_random_sentence(paragraph_id, vocab)
                if len(sentence_a) + len(sentence_b) <= max_len:
                    nsp_data.append((sentence_a + sentence_b, segment_a + segment_b, is_next))
        return nsp_data

# 读取以行为单位的文本文件, 
# keep_case: 是否保留大小写
# tokenizer: 分词器
def read_text_file(filename, tokenizer, vocab, keep_case = False, is_train=False):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    text = Text()
    text.read(lines, tokenizer, vocab, keep_case, is_train)
    return text


def get_candadiate(token_ids, ratio=0.15, filter=[CLS, SEP, UNK]):
    candadiate = []
    for i in range(len(token_ids)):
        if token_ids[i] not in filter:
            candadiate.append(i)
    n = max(1, round(len(candadiate) * ratio))
    return random.sample(candadiate, n)

def get_mlm_data(nsp_data, vocab, max_len):
    all_token_ids = []
    all_segments = []
    all_mask_pos = []
    all_mask_weights = []
    all_mask_labels = []
    nsp_labels = []
    max_num_mlm_preds = round(max_len * 0.15)
    for token_ids, segments, is_next in nsp_data:
        mask_pos = get_candadiate(token_ids, ratio=0.15)
        mask_label = []
        for pos in mask_pos:
            mask_label.append(token_ids[pos])
            if random.random() < 0.8: # 80%的时间用于替换为“<mask>”词元
                token_ids[pos] = MASK
            elif random.random() < 0.5: # 10%的时间保持不变
                # 从第5个词开始到最后一个词中随机选一个
                random_token_id = vocab.randomChoice(start=5)
                token_ids[pos] = random_token_id
            else: # 10%的时间用随机词替换
                # do nothing
                pass
        assert len(mask_pos) == len(mask_label)
        all_token_ids.append(
            torch.tensor(token_ids + [PAD] * (max_len - len(token_ids)), dtype=torch.long))
        all_segments.append(
            torch.tensor(segments + [0] * (max_len - len(segments)), dtype=torch.long))        
        all_mask_pos.append(
            torch.tensor(mask_pos + [0] * (max_num_mlm_preds - len(mask_pos)), dtype=torch.long))
        all_mask_weights.append(
            torch.tensor([1.0] * len(mask_pos) + [0.0] * (max_num_mlm_preds - len(mask_pos)), dtype=torch.float))
        all_mask_labels.append(
            torch.tensor(mask_label + [0] * (max_num_mlm_preds - len(mask_label)), dtype=torch.long))
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
    return all_token_ids, all_segments, all_mask_pos, all_mask_weights, all_mask_labels, nsp_labels

class WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self, text:Text, vocab, max_len):
        nsp_data = text.get_nsp_data(vocab, max_len)
        self.token_ids, self.segments, self.mask_pos, self.mask_weights, self.mask_labels, self.nsp_labels = get_mlm_data(nsp_data, vocab, max_len)

    def __getitem__(self, idx):
        return (self.token_ids[idx], self.segments[idx], self.nsp_labels[idx],
                self.mask_pos[idx], self.mask_weights[idx], self.mask_labels[idx])

    def __len__(self):
        return len(self.token_ids)
    
# Train: vocab=None, create vocab in this function
# Valid, test: vocab 传入 train 生成的词表
def load_data(vocab:Vocab, batch_size = 128, max_len = 64, name="train"):
    print("初始化词表")
    tokenizer = Tokenizer() # 分词器,缺省用空格分词
    print("读取文本数据")
    if name == "train":
        vocab = Vocab(special_symbols) # 词表
        text = read_text_file('../data/wikitext/wiki.train.tokens', tokenizer, vocab, is_train=True)
    elif name == "valid":
        text = read_text_file('../data/wikitext/wiki.valid.tokens', tokenizer, vocab, is_train=False)
    elif name == "test":
        text = read_text_file('../data/wikitext/wiki.test.tokens', tokenizer, vocab, is_train=False)
    print("生成数据集")
    data_set = WikiTextDataset(text, vocab, max_len)
    print("生成数据集迭代器")
    data_iter = torch.utils.data.DataLoader(data_set, batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True)
    return data_iter, vocab

if __name__=="__main__":
    tokenizer = Tokenizer() # 分词器,缺省用空格分词
    vocab = Vocab(special_symbols) # 词表
    text = read_text_file('../data/wikitext/wiki.train.tokens', tokenizer, vocab)
    print("词汇数量:", vocab.get_total_token())
    print("词元数量:", len(vocab))
    print("句子数量:", len(text))

