
from tokenizers import Tokenizer
from tokenizers import normalizers
from tokenizers.normalizers import NFKC, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace

def train_Bpe_tokenizer(files, target, size):
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer

    my_tokenizer = Tokenizer(BPE())
    my_tokenizer.normalizer = normalizers.Sequence([NFKC(), Lowercase(), StripAccents()])
    my_tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(vocab_size=size, 
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[BOS]", "[EOS]"])
    my_tokenizer.train(files, trainer)
    my_tokenizer.save(target)

def test_tokenizer(file, text):
    from tokenizers import Tokenizer
    my_tokenizer: Tokenizer = Tokenizer.from_file(file)
    output = my_tokenizer.encode(text)
    print(output.tokens)

def train_poetry_bpe(size):
    # 中文唐诗
    files = [f"../data/Tang_poetry/{name}.csv" for name in ["train_tang"]]
    train_Bpe_tokenizer(files, "../model/ch21/poetry/bpe-poetry-" + str(size) + ".json", size)
    # load
    file = "../model/ch21/bpe-poetry-" + str(size) + ".json"
    my_tokenizer: Tokenizer = Tokenizer.from_file(file)
    print(my_tokenizer.get_vocab_size())

                             
if __name__=="__main__":
    size = 7000
    train_poetry_bpe(size)
