#from nltk.translate.bleu_score import sentence_bleu
import math
import collections

# label = [["the", "cat", "is", "sitting", "on", "the", "mat"]]
# pred = ["the", "cat", "is", "on", "the", "mat"]

# print(sentence_bleu(label, pred, weights=(1,0,0,0)))
# print(sentence_bleu(label, pred, weights=(0,1,0,0)))
# print(sentence_bleu(label, pred, weights=(0,0,0,0)))
# print(sentence_bleu(label, pred, weights=(0,0,0,1)))

def bleu(pred_seq, label_seq, k):
    """计算BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

label = "the cat is sitting on the mat"
pred = "the cat is on the mat"

for k in [1,2,3,4]:
    s = bleu(pred, label, k=k)
    print("%d-gram: %f"%(k, s))
