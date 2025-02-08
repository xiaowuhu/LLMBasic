
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

    # from tokenizers.processors import TemplateProcessing
    # my_tokenizer.post_processor = TemplateProcessing(
    #     single="[CLS] $A [SEP]",
    #     pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    #     special_tokens=[
    #         ("[CLS]", 1),
    #         ("[SEP]", 2),
    #     ],
    # )

    trainer = BpeTrainer(vocab_size=size, 
                         special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[BOS]", "[EOS]"])
    my_tokenizer.train(files, trainer)
    my_tokenizer.save(target)


def train_WordPiece_tokenizer(files, target, size):
    from tokenizers.models import WordPiece
    from tokenizers.trainers import WordPieceTrainer

    my_tokenizer = Tokenizer(WordPiece())
    my_tokenizer.normalizer = normalizers.Sequence([NFKC(), Lowercase(), StripAccents()])
    my_tokenizer.pre_tokenizer = Whitespace()
    trainer = WordPieceTrainer(vocab_size=size, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[BOS]", "[EOS]"])
    my_tokenizer.train(files, trainer)
    my_tokenizer.save(target)


def test_tokenizer(file, text):
    from tokenizers import Tokenizer
    my_tokenizer: Tokenizer = Tokenizer.from_file(file)
    output = my_tokenizer.encode(text)
    print(output.tokens)


def train(size):
    # 中文
    files = [f"../data/Tang_poetry/{name}.csv" for name in ["train_tang"]]
    train_Bpe_tokenizer(files, "../model/ch6/bpe-poetry-" + str(size) + ".json", size)
    train_WordPiece_tokenizer(files, "../model/ch6/wordpiece-poetry-" + str(size) + ".json", size)
    # 英文
    files = [f"../data/wikitext/{name}.tokens" for name in ["wiki.train", "wiki.test", "wiki.valid"]]
    train_Bpe_tokenizer(files, "../model/ch6/bpe-wiki-"  + str(size) + ".json", size)
    train_WordPiece_tokenizer(files, "../model/ch6/wordpiece-wiki-" + str(size) + ".json", size)


def test(size):
    text = "七言绝句,再赠,唐,上元夫人,弄玉有夫皆得道，刘纲兼室尽登仙。君能仔细窥朝露，须逐云车拜洞天。"
    print("输入:", text)
    print("WordPiece 分词结果:")
    file = "../model/ch6/wordpiece-poetry-" + str(size) + ".json"
    test_tokenizer(file, text)
    print("BPE 分词结果:")
    file = "../model/ch6/bpe-poetry-" + str(size) + ".json"
    test_tokenizer(file, text)
                             
if __name__=="__main__":
    size = 13000
    train(size)
    print("词表V=13000")
    test(size)

    size = 6400
    train(size)
    print("词表V=6400")
    test(size)



