import numpy as np
import json
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset
import pickle
from transformers import BertTokenizer


def build_dataset(my_tokenizer, max_step, file_path):
    with open(file_path, "r", encoding="utf-8") as json_file:
        json_data = json.load(json_file)

    num_sample = int(len(json_data))
    i = 0
    for row in range(num_sample):
        chat = json_data[row]
        if len(chat) >= 4:
            i += 1
    print(f"total {i}")

    input_ids_array = np.zeros((i, max_step), dtype=np.int32)
    # 第一句话前面填充 CLS
    input_ids_array[:, 0] = my_tokenizer.cls_token_id
    # 后面都填充 PAD
    input_ids_array[:, 1:] = my_tokenizer.pad_token_id
    # type_ids_array = np.zeros((num_sample, max_step), dtype=np.int32)
    # type_ids_array[:, 0] = my_tokenizer.bos_token_id  # no speaker

    idx = 0
    for row in tqdm(range(num_sample)):
        chat = json_data[row]
        if len(chat) < 4: 
            continue
        input_ids = []
        #type_ids = []
        for i in range(len(chat)):
            sentence = chat[i]
            tokens = my_tokenizer.encode(sentence, add_special_tokens=False)
            # len_sentence = len(tokens)
            input_ids.extend(tokens)
            # 在两个 speaker 之间添加 SEP
            input_ids.append(my_tokenizer.sep_token_id)
        
        len_chat = len(input_ids)
        len_chat = min(len_chat, max_step-1)
        input_ids_array[idx, 1:len_chat+1] = input_ids[:len_chat]
        idx += 1

    dataset = TensorDataset(torch.LongTensor(input_ids_array))
    return dataset


if __name__=="__main__":
    MAX_STEP = 128
    my_tokenizer: BertTokenizer = BertTokenizer.from_pretrained("uer/gpt2-distil-chinese-cluecorpussmall")

    file_path = "../data/LCCC-base-split/LCCC-base_valid.json"
    dataset = build_dataset(my_tokenizer, MAX_STEP, file_path)
    with open("../model/ch7/chat/valid_dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)

    file_path = "../data/LCCC-base-split/LCCC-base_test.json"
    dataset = build_dataset(my_tokenizer, MAX_STEP, file_path)
    with open("../model/ch7/chat/test_dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)

    file_path = "../data/LCCC-base-split/LCCC-base_train.json"
    dataset = build_dataset(my_tokenizer, MAX_STEP, file_path)
    with open("../model/ch7/chat/train_dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)

