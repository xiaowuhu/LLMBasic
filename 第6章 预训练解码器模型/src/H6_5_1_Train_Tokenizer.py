
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


def train_couplet_bpe(size):
    # 中文对联
    files = [f"../data/couplet/{name}.txt" for name in ["train_in", "train_out", "test_in", "test_out"]]
    train_Bpe_tokenizer(files, "../model/ch21/bpe-couplet-" + str(size) + ".json", size)
    # load
    file = "../model/ch21/bpe-couplet-" + str(size) + ".json"
    my_tokenizer: Tokenizer = Tokenizer.from_file(file)
    print(my_tokenizer.get_vocab_size())
    
                             
if __name__=="__main__":
    size = 10000
    train_couplet_bpe(size)

