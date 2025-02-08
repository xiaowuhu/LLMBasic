import torch
import torch.nn as nn
import os
from sklearn.decomposition import PCA
from tqdm import tqdm
import matplotlib.pyplot as plt


class TokenEmbedding:
    """GloVe嵌入"""
    def __init__(self, embedding_name):
        """Defined in :numref:`sec_synonyms`"""
        self.idx_to_token, self.idx_to_vec = self._load_embedding(
            embedding_name)
        self.unknown_idx = 0
        self.token_to_idx = {token: idx for idx, token in
                             enumerate(self.idx_to_token)}

    def _load_embedding(self, embedding_name):
        print("load Embedding data...")
        idx_to_token, idx_to_vec = ['<unk>'], []
        with open(os.path.join("..", "data", embedding_name, 'vec.txt'), 'r', encoding="utf-8") as f:
            for line in tqdm(f):
                elems = line.rstrip().split(' ')
                token, vector = elems[0], [float(elem) for elem in elems[1:]]
                if len(vector) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(vector)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec  # 加 <unk> 全 0
        return idx_to_token, torch.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx)
                   for token in tokens]
        vecs = self.idx_to_vec[torch.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)


def set_embedding_weights(net, vocab):
    glove_embedding = TokenEmbedding('glove.6b.100d')
    embeds = glove_embedding[vocab.idx_to_token]
    print(embeds.shape)
    #print("init values:")
    #print(net.embedding.weight.data)
    net.embedding.weight.data.copy_(embeds)
    net.embedding.weight.requires_grad = False
    #print("embed values:")
    #print(net.embedding.weight.data)

def draw(tokens, glove_embedding):
    vectors = glove_embedding[tokens]
    pca = PCA(n_components=2)
    pca.fit(vectors.T)
    print(pca.components_)
    x = pca.components_[0]
    y = pca.components_[1]
    for i in range(len(tokens)):
        plt.scatter(x[i], y[i])
        plt.text(x[i]+0.001, y[i]+0.01, tokens[i])
    plt.show()


if __name__=="__main__":
    glove_embedding = TokenEmbedding('glove.6b.100d')

    # 用减法

    tokens = ["man", "woman", "girl", "boy", "queen", "king", "brother", "sister", "aunt", "uncle"]
    #draw(tokens, glove_embedding)

    a = glove_embedding["king", "man", "queen", "woman"]
    b = a[0] - a[1] # king - man
    c = a[2] - a[3] # queen - woman
    pca = PCA(n_components=2)
    pca.fit(torch.vstack((b,c)).T)
    x = pca.components_[0]
    y = pca.components_[1]
    for i in range(2):
        plt.scatter(x[i], y[i])
        plt.plot((0, x[i]), (0, y[i]))
        plt.text(x[i]+0.001, y[i]+0.01, tokens[i])
    plt.show()

    nn.functional.cosine_similarity(a, b, dim=0, eps=1e-8)

    tokens = ["china", "beijing", "japan", "tokyo", "london", "britain", "france", "paris"]
    draw(tokens, glove_embedding)
