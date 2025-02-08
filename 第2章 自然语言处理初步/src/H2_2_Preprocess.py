
import torch
import torch.nn as nn
import jieba

def word2index(wordList, word2indexDict):
    indexList = list()
    for word in wordList:
        indexList.append(word2indexDict[word])
    return indexList

def index2word(indexList, index2wordDict):
    wordList = list()
    for index in indexList:
        wordList.append(index2wordDict[index])
    return wordList

sentence = "Embedding层是深度学习中的一种重要技术，它可以有效地处理高维、离散和非线性的数据，使得神经网络能够更好地理解和处理复杂的问题。"
wordList = list(set(" ".join(jieba.cut(sentence)).split()))
word2indexDict = dict()
index2wordDict = dict()

for index, word in enumerate(wordList):
    word2indexDict[word] = index + 1
    index2wordDict[index + 1] = word

print(index2wordDict)

def paddingIndexList(indexList, maxLength):
    if len(indexList) > maxLength:
        return indexList[:maxLength]
    else:
        for i in range(maxLength - len(indexList)):
            indexList.append(0)
        return indexList
    
def paddingIndex(sentenceList, word2indexDict, maxLength):
    sentenceIndexList = list()
    for sentence in sentenceList:
        # 分词
        sentenceList = " ".join(jieba.cut(sentence)).split()
        indexList = word2index(sentenceList, word2indexDict)
        indexList = paddingIndexList(indexList, maxLength)
        sentenceIndexList.append(indexList)
    return sentenceIndexList

# 创建最大词个数为词字典长度，每个词用维度为3表示
# index从1开始的所以len(word2indexDict) + 1
embedding = nn.Embedding(len(word2indexDict) + 1, 3)
x = [1]
# 转换为tensor
x = torch.LongTensor(x)
out = embedding(x)
# 输入的形状
print(x.shape)
# 词嵌入矩阵形状
print(out.shape)
# 词嵌入矩阵
print(out)
# 词嵌入权重
print(embedding.weight.shape)
print(embedding.weight)

