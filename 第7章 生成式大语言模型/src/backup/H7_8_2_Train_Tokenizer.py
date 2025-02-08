
from transformers import AutoTokenizer
from transformers import AutoTokenizer
from H7_8_1_Data_Download import load_raw_data


example = '''def add_numbers(a, b):
    """Add the two numbers `a` and `b`."""
    return a + b'''

def train_tokenizer(corpus_iterator):
    # 训练分词器
    print("train tokenizer...")
    # 虽然是要训练一个新的分词器，但是可以加载已有的分词器，以避免指定分词算法或者特殊 token
    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokens = gpt2_tokenizer.tokenize(example)
    print("GPT-2分词器的分词效果:")
    print(tokens)
    my_tokenizer = gpt2_tokenizer.train_new_from_iterator(corpus_iterator, 52000)
    tokens = my_tokenizer.tokenize(example)
    print("新分词器的分词效果:")
    print(tokens)
    # my_tokenizer.add_special_tokens(special_tokens)
    my_tokenizer.save_pretrained("../model/ch7/pycpilot/tokenizer/")

# 迭代器, 每次从数据集中取出10000条到内存
def get_training_corpus(raw_datasets):
    dataset = raw_datasets["train"]
    for start_idx in range(0, len(dataset), 10000):
        samples = dataset[start_idx : start_idx + 10000]
        yield samples["content"]

# special_tokens = {
#     'unk_token': '[UNK]',
#     'bos_token': '[BOS]',
#     'eos_token': '[EOS]',
#     'pad_token': '[PAD]',
#     'sep_token': '[SEP]',
# }

if __name__=="__main__":
    # huggingface针对codeparrot训练的分词器（我们需要自己训练）
    # print("load existing tokenizer...")
    # tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")
    # 加载数据集
    raw_datasets = load_raw_data()
    corpus_iterator = get_training_corpus(raw_datasets)
    train_tokenizer(corpus_iterator)
    