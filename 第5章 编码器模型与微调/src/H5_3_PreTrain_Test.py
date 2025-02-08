import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from tqdm import tqdm

from H5_3_Model_Together import BERTModel
from H5_3_PreTrainData import load_data, Tokenizer, Vocab, CLS, SEP
from H5_Helper import load_model, create_mask

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)


def get_bert_encoding(model, device, tokens_id, segment, mask_pos=None):
    token_tensor = torch.tensor(tokens_id, device=device).unsqueeze(0)
    segment_tensor = torch.tensor(segment, device=device).unsqueeze(0)
    # print(token_tensor)
    # print(segment_tensor)
    src_mask, src_key_padding_mask = create_mask(token_tensor, device, vocab['<pad>'])
    encoded_X, mlm_Z, nsp_Z = model(token_tensor, segment_tensor, src_mask, src_key_padding_mask, mask_pos)

    return encoded_X, mlm_Z, nsp_Z

def get_token_id_and_segment(vocab: Vocab, tokenizer: Tokenizer, sentence_a, sentence_b = None, target_token=None):
    tokens_a = tokenizer.tokenize_space(sentence_a.lower())
    if target_token is not None:
        pos_of_bank = tokens_a.index(target_token)
    else:
        pos_of_bank = None
    tokens_id = [CLS] + vocab.get_ids_from_words(tokens_a) + [SEP]
    segments = [0] * len(tokens_id)

    if sentence_b is not None:
        tokens_b = tokenizer.tokenize_space(sentence_b.lower())
        tokens_b_id = vocab.get_ids_from_words(tokens_b) + [SEP]
        segments_b = [1] * len(tokens_b_id)
        tokens_id = tokens_id + tokens_b_id
        segments = segments + segments_b

    return tokens_id, segments, pos_of_bank


def bank_test(model, device, vocab: Vocab):
    tokenizer = Tokenizer()

    bank_1 = [
        "The university is on the left bank of the river",
        "Sarnia is located on the eastern bank of the junction",
        "St Nazaire is on the north bank of the Loire",
    ]
    encoded_1 = []
    for i in range(3):
        tokens_id, segment, pos_of_bank = get_token_id_and_segment(vocab, tokenizer, bank_1[i], target_token="bank")
        encoded, _, _ = get_bert_encoding(model, device, tokens_id, segment, None)
        encoded_bank = encoded[:, pos_of_bank, :]
        encoded_1.append(encoded_bank.detach())

    bank_2 = [
        "Small Bank is always happy to loan money to small businesses",
        "The bank depreciates PCs over a period of five years",
        "That bank stepped in to save the company from financial ruin",
    ]
    encoded_2 = []
    for i in range(3):
        tokens_id, segment, pos_of_bank = get_token_id_and_segment(vocab, tokenizer, bank_2[i], target_token="bank")
        encoded, _, _ = get_bert_encoding(model, device, tokens_id, segment, None)
        encoded_bank = encoded[:, pos_of_bank, :]
        encoded_2.append(encoded_bank.detach())

    cos_value = np.zeros((6,6))
    # bank_1 vs. bank_1
    for i in range(3):
        for j in range(3):
            cos_value[i, j] = torch.cosine_similarity(encoded_1[i], encoded_1[j])
    # bank_1 vs. bank_2            
    for i in range(3):
        for j in range(3):
            cos_value[i, j + 3] = torch.cosine_similarity(encoded_1[i], encoded_2[j])
    # bank_2 vs. bank_1
    for i in range(3):
        for j in range(3):
            cos_value[i + 3, j] = torch.cosine_similarity(encoded_2[i], encoded_1[j])
    # bank_2 vs. bank_2
    for i in range(3):
        for j in range(3):
            cos_value[i + 3, j + 3] = torch.cosine_similarity(encoded_2[i], encoded_2[j])

    cos_value = (cos_value - np.min(cos_value)) / (np.max(cos_value) - np.min(cos_value))

    plt.matshow(cos_value, cmap='bone')    
    plt.xticks(np.arange(6), ["河岸","河岸","河岸","银行","银行","银行"])
    plt.yticks(np.arange(6), ["河岸","河岸","河岸","银行","银行","银行"])
    plt.colorbar()
    plt.show()


def nsp_test(model, device, vocab: Vocab):
    tokenizer = Tokenizer()
    # completion, are
    sentence_b = "After the game 's <mask> , additional episodes <mask> unlocked , some of them having a higher difficulty than those found in the rest of the game"
    sentence_c = "There are also love simulation elements related to the game 's two main <unk> , although they take a very minor role"
    tokens_b_id, segment_b, _ = get_token_id_and_segment(vocab, tokenizer, sentence_b, sentence_c)
    mask_pos = torch.tensor([5, 9], dtype=torch.long).unsqueeze(0)
    encoded_b, mlm_z, nsp_z = get_bert_encoding(model, device, tokens_b_id, segment_b, mask_pos)
    print(nsp_z)
    print(torch.argmax(torch.softmax(nsp_z, dim=1)))
    token_id = torch.argmax(torch.softmax(mlm_z, dim=2), dim=2)
    print(vocab.get_words_from_ids(token_id.squeeze().tolist()))


def test_accuracy(model, device, vocab: Vocab, data_loader):
    total_loss, total_nsp_loss, total_mlm_loss = 0, 0, 0
    for tokens_X, segments_X, nsp_Y, mask_pos_X, mask_weight, mask_Y in data_loader:
        tokens_X = tokens_X.to(device)  #[N, L]
        segments_X = segments_X.to(device) #[N, L]
        mask_pos_X = mask_pos_X.to(device) #[N, 10] 10=64*0.15
        mask_weight_X = mask_weight.to(device) # 掩码 111110000
        mask_Y, nsp_Y = mask_Y.to(device), nsp_Y.to(device) # 预测token in mask
        src_mask, src_key_padding_mask = create_mask(tokens_X, device, vocab['<pad>'])
        hidden, mlm_Y_hat, nsp_Y_hat = model(tokens_X, segments_X,
                                  src_mask, src_key_padding_mask, mask_pos_X)
        mlm_loss = loss_func(mlm_Y_hat.reshape(-1, len(vocab)), mask_Y.reshape(-1)) * mask_weight_X.reshape(-1)
        mlm_loss = mlm_loss.sum() / mask_weight_X.sum()
        # 计算下一句子预测任务的损失
        nsp_loss = loss_func(nsp_Y_hat, nsp_Y).mean()
        total_mlm_loss += mlm_loss.item()
        total_nsp_loss += nsp_loss.item()
        total_loss += mlm_loss.item() + nsp_loss.item()
    print("mlm loss:", total_mlm_loss / len(data_loader))
    print("nsp loss:", total_nsp_loss / len(data_loader))
    print("total loss:", total_loss / len(data_loader))

if __name__=="__main__":
    batch_size, max_len = 32, 64
    data_loader, vocab = load_data(None, batch_size, max_len, name="train")
    test_loader, _ = load_data(vocab, batch_size, max_len, name="test")
    model = BERTModel(len(vocab), embed_size=128,
                    ffn_num_hiddens=256, 
                    num_heads=4, num_layers=2, 
                    max_len=max_len)
    device = torch.device('cpu')
    load_model(model, "Bert_Pretrain.pth", device)
    model.to(device)
    model.eval()
    loss_func = nn.CrossEntropyLoss(reduction='none')
    bank_test(model, device, vocab)
    nsp_test(model, device, vocab)
    test_accuracy(model, device, vocab, test_loader)
