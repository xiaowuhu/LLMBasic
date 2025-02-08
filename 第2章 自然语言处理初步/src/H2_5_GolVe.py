import argparse
import numpy as np
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def generate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vectors_file', default='d:/Gitee/data/glove.6B/glove.6B.100d.txt', type=str)
    args = parser.parse_args()

    with open(args.vectors_file, 'r', encoding="utf-8") as f:
        vectors = {}
        for line in tqdm(f):
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = [float(x) for x in vals[1:]]

    words = []
    for key in vectors:
        words.append(key)

    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        if word in vocab:
            W[vocab[word], :] = v

    # normalize each word vector to unit variance
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    W_norm = (W.T / d).T
    return (W_norm, vocab, ivocab)

def compute_pca(words, W, vocab):
    vecs = []
    for word in words:
        vecs.append(W[vocab[word], :])
    pca = PCA(n_components=2)
    np_vecs = np.array(vecs)
    pca.fit(np_vecs.T)
    print(pca.components_)
    x = pca.components_[0]
    y = pca.components_[1]
    for i, word in enumerate(words):
        plt.scatter(x[i], y[i])
        #plt.text(x[i], y[i], word)
    ls = ["--", "-.", ":", "-"]
    for i in range(0, len(words), 2):
        plt.plot((x[i],x[i+1]), (y[i],y[i+1]), linestyle=ls[i//4])
    plt.show()


def show_heatmap(words, W, vocab):
    vecs = []
    for word in words:
        vecs.append(W[vocab[word], :])
    np_vecs = np.array(vecs)

    fig, axes = plt.subplots(nrows=len(words), ncols=1, figsize=(6,6)) 
    for i in range(len(words)):
        ax = axes[i]
        ax.axis("off")
        data = np.tile(np_vecs[i:i+1,:], 10).reshape(10, 100)
        ax.imshow(data, cmap='coolwarm')
        ax.text(-1, 5, words[i], horizontalalignment='right')
    plt.show()


if __name__ == "__main__":
    words1 = [
        "queen", "woman",
        "king","man",
        "china", "beijing",
        "japan", "tokyo",
        "boy", "brother",
        "girl", "sister",
        "worse", "bad",
        "better", "good",
    ]

    words2 = [
        "pear", "peach", "mango",
        "netscape", "google", "microsoft",
        "china", "beijing", "taiwan",
    ]

    W, vocab, ivocab = generate()
    compute_pca(words1, W, vocab)
    show_heatmap(words2, W, vocab)
