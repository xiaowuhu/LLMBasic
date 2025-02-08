# 把小的文本文件合并为一个大文件，每个文本文件是一行，可以提高读取速度

import os
from tqdm import tqdm

# 首先下载数据集，然后解压到指定目录，如 ../data/aclImdb
URL = "https://ai.stanford.edu/~amaas/data/sentiment/"

def merge_files(input_dir, output_file):
    if os.path.exists(output_file):
        print(f"{output_file} already exists, skipping...")
        return
    with open(output_file, 'w', encoding='utf-8') as output_f:
        for root, dirs, files in os.walk(input_dir):
            for file in tqdm(files):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as input_f:
                    output_f.write(input_f.read())
                    output_f.write('\n')

# 统计平均长度
def compute_avg_length(file_name):
    total = 0
    max_len = 0
    min_len = 1000
    with open(file_name, 'r', encoding='utf-8') as f:
        text = f.read()
        lines = text.split("\n")
        for line in lines:
            words = line.split()
            len_w = len(words)
            if len_w == 0:
                continue
            if len_w > max_len:
                max_len = len_w
            if len_w < min_len:
                min_len = len_w
            total += len_w
    return total / len(lines), max_len, min_len


if __name__ == '__main__':
    input_dir = os.path.join("..", "data", "aclImdb", "train", "pos")
    output_file = os.path.join("..", "data", "aclImdb", "train", "pos.txt")
    merge_files(input_dir, output_file)
    avg_l, max_l, min_l = compute_avg_length(output_file)
    print(output_file, ":", avg_l, max_l, min_l)

    input_dir = os.path.join("..", "data", "aclImdb", "train", "neg")
    output_file = os.path.join("..", "data", "aclImdb", "train", "neg.txt")
    merge_files(input_dir, output_file)
    avg_l, max_l, min_l = compute_avg_length(output_file)
    print(output_file, ":", avg_l, max_l, min_l)

    input_dir = os.path.join("..", "data", "aclImdb", "test", "neg")
    output_file = os.path.join("..", "data", "aclImdb", "test", "neg.txt")
    merge_files(input_dir, output_file)
    avg_l, max_l, min_l = compute_avg_length(output_file)
    print(output_file, ":", avg_l, max_l, min_l)

    input_dir = os.path.join("..", "data", "aclImdb", "test", "pos")
    output_file = os.path.join("..", "data", "aclImdb", "test", "pos.txt")
    merge_files(input_dir, output_file)
    avg_l, max_l, min_l = compute_avg_length(output_file)
    print(output_file, ":", avg_l, max_l, min_l)


