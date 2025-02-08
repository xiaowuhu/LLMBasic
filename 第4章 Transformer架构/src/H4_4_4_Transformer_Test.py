import torch
import numpy as np

from H4_Helper import load_model, BLEU, create_mask, generate_tril_mask
from H4_4_1_Transformer_Data import get_train_dataloader, get_test_dataloader, SOS, EOS, PAD, sentence_to_ids
from H4_4_2_Transformer_Model import Seq2SeqTransformer

def evaluate(model, test_dataloader, DEVICE):
    model.eval()
    losses = 0
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD)

    for src, tgt in test_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:, :-1]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, DEVICE, PAD)
        logits = model(src, tgt_input, 
                       src_mask, tgt_mask,
                       src_key_padding_mask = src_padding_mask, 
                       tgt_key_padding_mask = tgt_padding_mask, 
                       memory_key_padding_mask = src_padding_mask)
        
        tgt_out = tgt[:, 1:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    loss = losses / len(test_dataloader)
    print("evaluate loss =", loss)
    return loss


# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, DEVICE):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)
    memory = model.encode(src, src_mask) # [N, Ls, D]
    memory = memory.to(DEVICE)
    tgt = torch.ones(1, 1).fill_(SOS).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        tgt_mask = generate_tril_mask(tgt.size(1), DEVICE)
        out = model.decode(tgt, memory, tgt_mask)
        prob = model.fc(out[:, -1])  # 只计算最有一个字符的概率
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        new_tensor = torch.ones(1, 1).type_as(src.data).fill_(next_word)
        tgt = torch.cat([tgt, new_tensor], dim=1)
        if next_word == EOS:
            break
    return tgt


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src, en_vocab, DEVICE):
    model.eval()
    src.unsqueeze_(0)
    num_tokens = src.shape[1]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(model, src, src_mask, num_tokens + 5, DEVICE).flatten()
    token_list = list(tgt_tokens.cpu().numpy())
    words = en_vocab.get_words_from_tokens(token_list)
    return words

def convert_list_to_sentence(words):
    return " ".join(words).replace("<SOS>", "") \
              .replace("<EOS>", "") \
              .replace("<PAD>", "") \
              .strip()

def test(model, fr_vocab, en_vocab, test_dataloader, count, DEVICE):
    for src_batch, tgt_batch in test_dataloader:
        batch_num = src_batch.shape[0]
        samples = np.random.choice(batch_num, count, replace=False)
        for id in samples:
            src = src_batch[id]
            tgt = tgt_batch[id]
            fr_sentence = fr_vocab.get_words_from_tokens(src.tolist())
            en_sentence = en_vocab.get_words_from_tokens(tgt.tolist())
            en_predict = translate(model, src, en_vocab, DEVICE)
            score = BLEU(convert_list_to_sentence(en_predict), 
                         convert_list_to_sentence(en_sentence), 2)
            print("-----------")
            print(">", fr_sentence)
            print("=", en_sentence)
            print("<", en_predict)
            print("BLEU:", score)
        break


def test_samples(model, fr_vocab, DEVICE):
    print("---- test old sample from 3.3 ----")
    pairs = [
        ["tu es tres effrontee", "you re very forward"],
        ["tu es une sacree menteuse", "you re such a liar"],
        ["nous sommes impuissants", "we re helpless"],
        ["je suis ici pour faire ce que je peux", "i m here to do what i can"],
        ["je suis desole mon pere est sorti", "i m sorry my father is out"],
        ["il est accoutume a voyager", "he is used to traveling"]
    ]
    for pair in pairs:
        tokens = sentence_to_ids(fr_vocab, pair[0])
        tokens = torch.Tensor(tokens).type(torch.long)
        en_predict = translate(model, tokens, en_vocab, DEVICE)
        en_predict = convert_list_to_sentence(en_predict)
        score = BLEU(en_predict, pair[1], 2)
        print("-----------")
        print(">", pair[0])
        print("=", pair[1])
        print("<", en_predict)
        print("BLEU:", score)


if __name__=="__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    BATCH_SIZE = 64
    EMBED_SIZE = 512
    NHEAD = 8
    FC_HIDDEN = 512
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3

    torch.manual_seed(0)
    # 必须先加载训练集以获得词表
    fr_vocab, en_vocab, test_dataloader = get_train_dataloader(BATCH_SIZE)
    #test_dataloader = get_test_dataloader(fr_vocab, en_vocab, BATCH_SIZE)
    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, 
                                     EMBED_SIZE, NHEAD, 
                                     fr_vocab.n_words, en_vocab.n_words, 
                                     FC_HIDDEN)
    load_model(transformer, "Transformer_10.pth", DEVICE)
    transformer = transformer.to(DEVICE)
    evaluate(transformer, test_dataloader, DEVICE)
    test(transformer, fr_vocab, en_vocab, test_dataloader, 10, DEVICE)
    test_samples(transformer, fr_vocab, DEVICE)

