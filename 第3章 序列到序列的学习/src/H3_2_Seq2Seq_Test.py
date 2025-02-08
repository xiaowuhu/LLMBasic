import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

from H3_Helper import load_model
from H3_2_Seq2Seq_Train import Encoder, Decoder

NUM_LETTERS = 8 # 一共 8 个字符 ABCDEFGH
NUM_FEATURES = NUM_LETTERS * 2 + 2 # 加上SOS,EOS, one-hot 编码10维
NUM_STEPS = 4 # 三个字母+EOS
SOS_token = 0
EOS_token = NUM_FEATURES - 1
# ABCDEFGH -> one-hot: 1,2,3,4,5,6,7,8
# abcdefgh -> one-hot: 9,10,...,16
# 从 ABC -> abc, 遍历所有三个字符的组合，字符可重复
def generate_data():
    n = NUM_LETTERS * NUM_LETTERS * NUM_LETTERS
    X = np.zeros((n, NUM_STEPS, NUM_FEATURES), dtype=np.int32)
    Y = np.zeros((n, NUM_STEPS), dtype=np.int32)
    id = 0
    # 三个字符组合
    for char1 in range(1, NUM_LETTERS+1): # A=1, B=2, ... 0=SOS
        for char2 in range(1, NUM_LETTERS+1):
            for char3 in range(1, NUM_LETTERS+1):
                # one-hot 编码
                X[id] = np.eye(NUM_FEATURES)[[char1, char2, char3, EOS_token]]
                Y[id] = [char1+NUM_LETTERS, char2+NUM_LETTERS, char3+NUM_LETTERS, EOS_token]
                id += 1
    return X, Y

def tensorFromSentence(sentence):
    ids = []
    for i in range(len(sentence)):
        id = ord(sentence[i]) - 65 + 1
        ids.append(id)
    ids.append(EOS_token)
    x = torch.eye(NUM_FEATURES)[ids]
    return x

def evaluate(encoder, decoder, input):
    with torch.no_grad():

        input_tensor = tensorFromSentence(input).unsqueeze(0)

        _, context_vector = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(1, context_vector)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(chr(idx+97-9)) # 9 == 'a'(ascii 97)
    return decoded_words, context_vector

def draw3d_vector(ax, pairs):
    vecs = []
    for i in range(len(pairs)):
        pair = pairs[i]
        print('>', pair[0])  # ABC
        print('=', pair[1])  # abc
        output_words, context_vector = evaluate(encoder, decoder, pair[0])
        output_sentence = "".join(output_words)
        print('<', output_sentence)
        print("c =", context_vector)
        vecs.append(context_vector[0,0].numpy())

    pca = PCA(n_components=3)
    np_vecs = np.array(vecs)
    pca.fit(np_vecs.T)
    x = pca.components_[0]
    y = pca.components_[1]
    z = pca.components_[2]

    for i in range(len(pairs)):
        pair = pairs[i]
        ax.plot((0,x[i]),(0,y[i]),(0,z[i]), linestyle=ls[i], marker='.', label=pair[0])

    ax.legend()


def draw2d_vector(pairs):
    vecs = []
    for i in range(len(pairs)):
        pair = pairs[i]
        print('>', pair[0])  # ABC
        print('=', pair[1])  # abc
        output_words, context_vector = evaluate(encoder, decoder, pair[0])
        output_sentence = "".join(output_words)
        print('<', output_sentence)
        print("c =", context_vector)
        vecs.append(context_vector[0,0].numpy())

    pca = PCA(n_components=2)
    np_vecs = np.array(vecs)
    pca.fit(np_vecs.T)
    x = pca.components_[0]
    y = pca.components_[1]

    for i, pair in enumerate(pairs):
        plt.scatter(x[i], y[i])
        plt.text(x[i], y[i], pair[0])
    ls = ["--", "-.", ":", "-"]
    for i in range(0, len(pairs), 2):
        plt.plot((x[i],x[i+1]), (y[i],y[i+1]), linestyle=ls[i//4])
    plt.show()



if __name__=="__main__":
    #DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_printoptions(sci_mode=False)
    DEVICE = torch.device("cpu")
    hidden_size = 4
    encoder = Encoder(NUM_FEATURES, hidden_size)
    decoder = Decoder(NUM_FEATURES, hidden_size, NUM_FEATURES)
    load_model(encoder, "Encoder_ABC.pth", DEVICE)    
    load_model(decoder, "Decoder_abc.pth", DEVICE)    
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ls = ["--", "-.", ":", "-"]
    pairs = [["AAA","aaa"],["BBB","bbb"],["CCC","ccc"],["DDD","ddd"]]
    draw3d_vector(ax, pairs)
    ax = fig.add_subplot(122, projection='3d')
    pairs = [["AAA","aaa"],["AAB","aab"],["AAC","aac"],["AAD","aad"]]
    draw3d_vector(ax, pairs)    
    plt.show()        

    pairs = [["AAA","aaa"],["AAB","aab"],["CCA","cca"],["CCB","ccb"],
             ["DDA","dda"],["DDB","ddb"],["EEA","eea"],["EEB","eeb"]]
    draw2d_vector(pairs)

    pairs = [["ABC","abc"],["ABD","abd"],["CDC","cdc"],["CDD","cdd"],
             ["EFC","efc"],["EFD","efd"],["GHC","ghc"],["GHD","ghd"]]
    draw2d_vector(pairs)
