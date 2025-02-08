import os
import torch
import torch.nn as nn
import random

from H1_7_2_Names_Train import RNN2
from H1_7_1_Names_DataReader import n_languages, n_letters, languageTensor, inputTensor, all_letters

def load_model(model:nn.Module, name:str, device):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    model_pth = os.path.join(current_dir, "model", name)
    print("load model ", name)
    model.load_state_dict(torch.load(model_pth, map_location=device))

rnn = RNN2(n_languages, n_letters, 128, n_letters)
load_model(rnn, "H1_7_2_All_model.pth", "cpu")
# load_model(rnn, "H17_7_2_Japanese_model.pth", "cpu")

max_length = 20

# Sample from a category and starting letter
def sample(language, start_letter, is_random=False):
    with torch.no_grad():  # no need to track history in sampling
        language_tensor = languageTensor(language) # 语种的 one-hot 编码
        input = inputTensor(start_letter) # 名字字符的 one-hot 编码
        hidden = rnn.initHidden()  # 初始化 h 为 0 
        output_name = start_letter # 名字的第一个字符
        for i in range(max_length):
            output, hidden = rnn(language_tensor, input[0], hidden)
            topv, topi = output.topk(3)  # 获得输出中的最大位置及值
            if is_random == True:
                id = random.choice([0,0,0,0,0,1,1,1,2])
            else:
                id = 0
            topi = topi[0][id] # 最大位置
            if topi == n_letters - 1: # <EOS>
                break
            else:
                letter = all_letters[topi] # 正常字符
                output_name += letter # 拼接名字
            input = inputTensor(letter) # 接力预测
        return output_name # 完整名字

# Get multiple samples from one category and multiple starting letters
def samples(language, start_letters, is_random=False):
    print(f"---- {language} ----")
    for start_letter in start_letters:
        print(sample(language, start_letter, is_random))

samples('Russian', 'XYZ')
samples('Japanese', 'XYZ')
samples('Chinese', 'XYZ')

samples('Russian', 'XYZ', True)
samples('Japanese', 'XYZ', True)
samples('Chinese', 'XYZ', True)
