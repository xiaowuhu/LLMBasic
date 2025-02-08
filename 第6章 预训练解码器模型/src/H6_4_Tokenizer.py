
from tokenizers import normalizers
from tokenizers.normalizers import Lowercase, StripAccents

def normalization_to_KC():
    from tokenizers.normalizers import NFKC
    normalizer = normalizers.Sequence([NFKC(), Lowercase(), StripAccents()])
    a = "⽣" # 12131
    b = "生" # 29983
    # a = "⾦" # 12198
    # b = "金" # 37329
    a1 = normalizer.normalize_str(a)
    b1 = normalizer.normalize_str(b)
    print(f"{ord(a)} -> {ord(a1)}")
    print(f"{ord(b)} -> {ord(b1)}")
    print("测试英文 The -> ", normalizer.normalize_str("The"))

def bert_tokenizer(text):
    from transformers import BertTokenizer
    print("英文 BERT 分词结果:")
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')  
    tokens = tokenizer.tokenize(text)
    print(tokens)
    print("中文 BERT 分词结果:")
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')  
    tokens = tokenizer.tokenize(text)
    print(tokens)

def gpt_tokenizer(text):
    from transformers import GPT2Tokenizer
    print("GPT2 分词结果:")
    tokenizer =  GPT2Tokenizer.from_pretrained("gpt2")
    tokens = tokenizer.tokenize(text)
    print(tokens)


if __name__=="__main__":
    normalization_to_KC()
    text = "BERT is not the unfriendly ChatGPT!"
    print("===", text)
    bert_tokenizer(text)
    gpt_tokenizer(text)
    text = "我像风一样自由free，你的温柔无法no way挽留。"
    print("===", text)
    bert_tokenizer(text)
    gpt_tokenizer(text)

