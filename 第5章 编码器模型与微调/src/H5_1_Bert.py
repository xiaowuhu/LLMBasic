import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
# 分词
from transformers import BertTokenizer
from transformers import BertTokenizer, BertModel
from transformers import BertForMaskedLM, BertTokenizer

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')  
text = "BERT is not the unfriendly ChatGPT!"
tokens = tokenizer.tokenize(text)
print(tokens)

# 词向量
model = BertModel.from_pretrained('bert-large-uncased')
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, add_special_tokens=True)
outputs = model(**inputs)
word_embeddings = outputs.last_hidden_state  
print("hidden state:", word_embeddings.shape)
classifier = outputs.pooler_output
print("pooled output:", classifier.shape)

model.forward(inputs['input_ids'], inputs['attention_mask'])


def bank_test(model):

    bank_1 = [
        "The new university is on the left bank of the river",
        "Sarnia is located on the eastern bank of the junction",
        "St Nazaire is on the north bank of the Loire",
    ]
    encoded_1 = []
    for i in range(3):
        tokens = tokenizer.tokenize(bank_1[i])
        print(tokens)
        pos = tokens.index("bank")
        inputs = tokenizer(bank_1[i], return_tensors='pt', padding=True, truncation=True, add_special_tokens=True)
        outputs = model(**inputs)
        encoded = outputs.last_hidden_state  
        encoded_bank = encoded[:, pos, :]
        encoded_1.append(encoded_bank.detach())

    bank_2 = [
        "Bank is always happy to loan money to small businesses",
        "The bank depreciates PCs over a period of five years",
        "Some small bank stepped in to save the company from financial ruin",
    ]
    encoded_2 = []
    for i in range(3):
        tokens = tokenizer.tokenize(bank_2[i])
        print(tokens)
        pos = tokens.index("bank")
        inputs = tokenizer(bank_2[i], return_tensors='pt', padding=True, truncation=True, add_special_tokens=True)
        outputs = model(**inputs)
        encoded = outputs.last_hidden_state  
        encoded_bank = encoded[:, pos, :]
        encoded_2.append(encoded_bank.detach())

    cos_value = np.zeros((6,6))
    # apple_1 vs. apple_1
    for i in range(3):
        for j in range(3):
            cos_value[i, j] = torch.cosine_similarity(encoded_1[i], encoded_1[j])
    # apple_1 vs. apple_2            
    for i in range(3):
        for j in range(3):
            cos_value[i, j + 3] = torch.cosine_similarity(encoded_1[i], encoded_2[j])
    # apple_2 vs. apple_1
    for i in range(3):
        for j in range(3):
            cos_value[i + 3, j] = torch.cosine_similarity(encoded_2[i], encoded_1[j])
    # apple_2 vs. apple_2
    for i in range(3):
        for j in range(3):
            cos_value[i + 3, j + 3] = torch.cosine_similarity(encoded_2[i], encoded_2[j])

    cos_value = (cos_value - np.min(cos_value)) / (np.max(cos_value) - np.min(cos_value))

    plt.matshow(cos_value, cmap='bone')    
    plt.xticks(np.arange(6), ["河岸","河岸","河岸","银行","银行","银行"])
    plt.yticks(np.arange(6), ["河岸","河岸","河岸","银行","银行","银行"])
    plt.colorbar()
    plt.show()

# def mlm_test(model):
#     # completion, are
#     sentence_b = "After the game 's <mask> , additional episodes <mask> unlocked , some of them having a higher difficulty than those found in the rest of the game"

#     inputs = tokenizer(sentence_b, return_tensors='pt', padding=True, truncation=True, add_special_tokens=True)
#     outputs = model(**inputs)
#     encoded = outputs.last_hidden_state  
#     token_id = torch.argmax(torch.softmax(mlm_z, dim=2), dim=2)
#     print(vocab.get_words_from_ids(token_id.squeeze().tolist()))


def mlm():
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    text1 = "BERT is a powerful language model."
    text2 = "BERT is a powerful <mask> model."
    inputs_1 = tokenizer(text1, return_tensors='pt', padding=True, truncation=True, add_special_tokens=True)
    print(inputs_1)
    inputs_2 = tokenizer(text2, return_tensors='pt', padding=True, truncation=True, add_special_tokens=True)
    print(inputs_2)
    outputs = model(**inputs_2, labels=inputs_1['input_ids'])
    print(outputs)
    loss = outputs.loss
    print(loss)
    print(torch.argmax(outputs.logits, dim=2))


if __name__=="__main__":
    #bank_test(model)
    mlm()