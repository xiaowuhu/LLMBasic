import numpy as np
from tqdm import tqdm
from tokenizers import Tokenizer
import pickle


def format_file():
    # 读取训练原始语料中的诗歌格式和诗句内容
    form, title, content = np.loadtxt("../data/Tang_poetry/train_tang.csv", 
                                      delimiter=',', unpack=True, usecols=(0, 4, 5),
                                      encoding='utf-8', dtype=str)
    # 把每行数据变成一行这样的格式：[BOS]格式[SEP]藏头字[SEP]诗歌内容[EOS]
    lines = []
    for i in tqdm(range(len(form))):
        line = '[BOS]' + form[i] + '[SEP]' + title[i] + '[SEP]' + content[i] + '[EOS]\n'
        lines.append(line)
    
    with open("../data/Tang_poetry/train.txt", "w", encoding='utf-8') as f:
        f.writelines(lines)

# 把 token 变成 token id
def convert_to_ids():
    file = "../model/ch21/bpe-poetry-7000.json"
    my_tokenizer: Tokenizer = Tokenizer.from_file(file)
    print(my_tokenizer.get_vocab_size())
    token_path = "../data/Tang_poetry/train.txt"
    token_file = open(token_path, "r", encoding='utf-8')
    new_lines = token_file.readlines()
    token_file.close()
    lines_ids = []
    ids_path = "../data/Tang_poetry/train_ids.pkl"
    ids_file = open(ids_path, "wb")
    for i in tqdm(range(len(new_lines))):
        line = new_lines[i].strip()
        tokens = my_tokenizer.encode(line) # 分词
        lines_ids.append(tokens.ids)
        for id in tokens.ids:
            if id > 7000:
                print(id)
    pickle.dump(lines_ids, ids_file)
    ids_file.close()

if __name__ == '__main__':
    format_file()
    convert_to_ids()
