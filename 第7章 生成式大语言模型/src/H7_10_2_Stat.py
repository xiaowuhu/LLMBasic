import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import BertTokenizer

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def get_num_tokens(file_path, tokenizer):
    with open(file_path, "r", encoding="utf-8") as json_file:
        json_data = json.load(json_file)

    input_num_tokens = []
    for chat in tqdm(json_data):
        len_chat = 0
        for i in range(len(chat)):
            sentence = chat[i]
            tokens = tokenizer.encode(sentence)
            len_chat += len(tokens)
        input_num_tokens.append(len_chat)

    return input_num_tokens

def count_intervals(num_tokens, interval):
    max_value = max(num_tokens)
    intervals_count = {}
    for lower_bound in range(0, max_value + 1, interval):
        upper_bound = lower_bound + interval
        count = len([num for num in num_tokens if lower_bound <= num < upper_bound])
        intervals_count[f"{lower_bound}-{upper_bound}"] = count
    return intervals_count

def main():
    my_tokenizer: BertTokenizer = BertTokenizer.from_pretrained("uer/gpt2-distil-chinese-cluecorpussmall")
    print("词表:", len(my_tokenizer))

    file_path = "../data/LCCC-base-split/LCCC-base_train.json"
    input_num_tokens = get_num_tokens(file_path, my_tokenizer)
    intervals_count = count_intervals(input_num_tokens, 20)
    print(intervals_count)
    x = [k for k, v in intervals_count.items()]
    y = [v for k, v in intervals_count.items()]
    plt.figure(figsize=(8, 6))
    bars = plt.bar(x, y)
    #plt.title('训练集Token分布情况')
    plt.ylabel('数量')
    plt.xticks(rotation=45)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), va='bottom')
    plt.show()

if __name__ == '__main__':
    main()
