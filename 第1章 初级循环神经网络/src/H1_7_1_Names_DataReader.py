
from io import open
import glob
import os
import unicodedata
import string
import torch
import random


#all_letters = string.ascii_letters + " .,;'-"
all_letters = string.ascii_letters + " '-"
n_letters = len(all_letters) + 1 # Plus EOS marker

def findFiles(): 
    file_path = "../data/names/*.txt"
    allfiles = glob.glob(file_path)
    return allfiles

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Read a file and split into lines
def readLines(filename):
    with open(filename, encoding='utf-8') as some_file:
        return [unicodeToAscii(line.strip()) for line in some_file]

# Build the category_lines dictionary, a list of lines per category
language_to_names = {}
all_languages = []
for filename in findFiles():
    language = os.path.splitext(os.path.basename(filename))[0]
    all_languages.append(language)
    names = readLines(filename)
    names = list(set(names))  # 去重 dedup
    language_to_names[language] = names

n_languages = len(all_languages)

if n_languages == 0:
    raise RuntimeError('Data not found. Make sure that you downloaded data '
        'from https://download.pytorch.org/tutorial/data.zip and extract it to '
        'the current directory.')

print('# languages:', n_languages, all_languages)
print(unicodeToAscii("O'Néàl"))

# Random item from a list
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

# Get a random category and random line from that category
def getRandomLanguageAndName(language):
    if language == -1:
        language = randomChoice(all_languages)
    name = randomChoice(language_to_names[language])
    return language, name

# One-hot vector for category
def languageTensor(category):
    li = all_languages.index(category)
    tensor = torch.zeros(1, n_languages)
    tensor[0][li] = 1
    return tensor

# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

# ``LongTensor`` of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)

# Make category, input, and target tensors from a random category, line pair
def getRandomTrainingData(language=-1):
    language, name = getRandomLanguageAndName(language)
    language_tensor = languageTensor(language)
    input_name_tensor = inputTensor(name)
    target_name_tensor = targetTensor(name)
    return language_tensor, input_name_tensor, target_name_tensor
