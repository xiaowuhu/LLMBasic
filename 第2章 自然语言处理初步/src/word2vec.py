import string
import nltk
from nltk.corpus import brown
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
 
nltk.download("brown")
 
# Preprocessing data to lowercase all words and remove single punctuation words
document = brown.sents()
data = []
for sent in document:
  new_sent = []
  for word in sent:
    new_word = word.lower()
    if new_word[0] not in string.punctuation:
      new_sent.append(new_word)
  if len(new_sent) > 0:
    data.append(new_sent)
 
# Creating Word2Vec
model = Word2Vec(
    sentences = data,
    vector_size = 50,
    window = 10,
    #iter = 20,
)
 
# Vector for word love
print("Vector for love:")
print(model.wv["love"])
print()
 
# Finding most similar words
print("3 words similar to 'car':")
words = model.wv.similar_by_word("car", topn=3)
#words = model.most_similar("car", topn=3)
for word in words:
  print(word)
print()
 
#Visualizing data
words = ["france", "germany", "india", "truck", "boat", "road", "teacher", "student"]
 
X = model.wv[words]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
 
pyplot.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()