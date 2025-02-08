from transformers import  BertTokenizer
from tokenizers import Tokenizer
from tokenizers.normalizers import NFKC, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import normalizers
from tqdm import tqdm
import re

def train_WordPiece_tokenizer(files, target, size):
    from tokenizers.models import WordPiece
    from tokenizers.trainers import WordPieceTrainer

    my_tokenizer = Tokenizer(WordPiece())
    my_tokenizer.normalizer = normalizers.Sequence([NFKC(), StripAccents()])
    my_tokenizer.pre_tokenizer = Whitespace()
    trainer = WordPieceTrainer(vocab_size=size, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[BOS]", "[EOS]"])
    my_tokenizer.train(files, trainer)
    my_tokenizer.save(target)
    return my_tokenizer

raw_file_path = "../data/leetcode/leetcode_instructions.jsonl"
clean_file_path = "../data/leetcode/no_chinese.jsonl"

def remove_chinese_char():
    with open(raw_file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        new_lines = []
        for line in tqdm(lines):
            new_line = re.sub('[\u4e00-\u9fff]', '', line)
            new_lines.append(new_line)
    
    with open(clean_file_path, 'w', encoding='utf-8') as file:
        file.writelines(new_lines)
    

remove_chinese_char()

size = 1000
files = [clean_file_path]
my_tokenizer = train_WordPiece_tokenizer(files, "../model/ch7/code/wordpiece-code-" + str(size) + ".json", size)
dict_vocab = my_tokenizer.get_vocab()
tokenizer: BertTokenizer = BertTokenizer.from_pretrained("uer/gpt2-distil-chinese-cluecorpussmall")
print(len(tokenizer))
for k,v in dict_vocab.items():
    tokenizer.add_tokens([k])

print(len(tokenizer))
tokenizer.save_pretrained("../model/ch7/code/new_vocab")
