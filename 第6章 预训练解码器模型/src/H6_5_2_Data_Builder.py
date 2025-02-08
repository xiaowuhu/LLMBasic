from tqdm import tqdm
import pickle
from tokenizers import Tokenizer

# 把 in.txt 与 out.txt 合并
def merge_files():
    # 上联
    shang_lian_path = "../data/couplet/train_in.txt"    
    file_shang_lian = open(shang_lian_path, "r", encoding='utf-8')
    shang_lians = file_shang_lian.readlines()
    file_shang_lian.close()
    line_count = len(shang_lians)
    # 下联
    xia_lian_path = "../data/couplet/train_out.txt"
    file_xia_lian = open(xia_lian_path, "r", encoding='utf-8')
    xia_lians = file_xia_lian.readlines()
    file_xia_lian.close()
    assert line_count == len(xia_lians)
    # 用 [BOS]...[SEP]...[EOS] 连接上下联
    new_lines = []
    for i in tqdm(range(line_count)):
        shang_lian = ''.join(w.strip() for w in shang_lians[i])
        xia_lian = ''.join(w.strip() for w in xia_lians[i])
        new_line = "[BOS]" + shang_lian + "[SEP]" + xia_lian + "[EOS]" + "\n"
        new_lines.append(new_line)
    
    shang_xia_lian_path = "../data/couplet/train.txt"
    shang_xia_lian_file = open(shang_xia_lian_path, "w", encoding='utf-8')
    shang_xia_lian_file.writelines(new_lines)
    shang_xia_lian_file.close()
    
# 把 token 变成 token id
def convert_to_ids():
    file = "../model/ch21/bpe-couplet-10000.json"
    my_tokenizer: Tokenizer = Tokenizer.from_file(file)
    print(my_tokenizer.get_vocab_size())
    token_path = "../data/couplet/train.txt"
    token_file = open(token_path, "r", encoding='utf-8')
    new_lines = token_file.readlines()
    token_file.close()
    lines_ids = []
    ids_path = "../data/couplet/train_ids.pkl"
    ids_file = open(ids_path, "wb")
    for i in tqdm(range(len(new_lines))):
        line = new_lines[i].strip()
        tokens = my_tokenizer.encode(line) # 分词
        lines_ids.append(tokens.ids)
    pickle.dump(lines_ids, ids_file)
    ids_file.close()

if __name__ == '__main__':
    merge_files()
    convert_to_ids()
