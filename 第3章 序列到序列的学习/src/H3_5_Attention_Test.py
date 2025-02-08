import random
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.decomposition import PCA
import numpy as np

from H3_3_Translate_Data import loadSmallData, EOS_token
from H3_5_Attention_Model import EncoderRNN, AttnDecoderRNN
from H3_Helper import load_model, BLEU


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long).view(1, -1)

def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn, encoder_hidden, decoder_hidden

def evaluateRandomly(encoder, decoder, franch_vocab, english_vocab, pairs, n=10):
    vec = []
    for i in range(n):
        pair = random.choice(pairs)
        print("--------")
        print('>', pair[0])
        print('=', pair[1])
        output_words, _, eh, dh = evaluate(encoder, decoder, pair[0], franch_vocab, english_vocab)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        score = BLEU(output_sentence, pair[1] + " <EOS>", 2)
        print("BLUE =", score)
        print("cosine:", torch.cosine_similarity(eh, dh, dim=2))
        if score >= 0.9:
            vec.append(eh)
            vec.append(dh)
    return vec

def evaluatePairs(encoder, decoder, franch_vocab, english_vocab, pairs):
    for i in range(len(pairs)):
        pair = pairs[i]
        print('>', pair[0])
        print('=', pair[1])
        output_words, _, _, _ = evaluate(encoder, decoder, pair[0], franch_vocab, english_vocab)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        score = BLEU(output_sentence, pair[1] + " <EOS>", 2)
        print("BLUE =", score)

def showAttention(ax, fig, input_sentence, output_words, weight):
    cax = ax.matshow(weight.cpu().numpy(), cmap='bone')
    fig.colorbar(cax)
    # Set up axes
    ticks = [''] + input_sentence.split(' ') + ['<EOS>']
    ax.set_xticklabels(ticks, rotation=45)
    ax.set_yticklabels([''] + output_words)
    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

def evaluateAndShowAttention(list_sentences):
    fig = plt.figure()
    axes = fig.subplots(2, 2)
    for i in range(len(list_sentences)):
        ax = axes[i // 2, i % 2]
        pair = list_sentences[i]
        output_words, att_weight, _, _ = evaluate(encoder, decoder, pair[0], franch_vocab, english_vocab)
        print('>', pair[0])
        print('=', pair[1])
        print('<', ' '.join(output_words))
        showAttention(ax, fig, pair[0], output_words, att_weight[0, :len(output_words), :])
    plt.show()

def check_embedding(encoder, decoder, franch_vocab, english_vocab):
    pair = ["je vous nous il elle ils", "i you we he she they"]
    fra = pair[0].split(' ')
    eng = pair[1].split(' ')
    vecs = []
    for i in range(6):
        fra_id = franch_vocab.word2index[fra[i]]
        eng_id = english_vocab.word2index[eng[i]]
        v = encoder.embedding.weight.data[fra_id]
        vecs.append(v.numpy())
        v = decoder.embedding.weight.data[eng_id]
        vecs.append(v.numpy())
    pca = PCA(n_components=2)
    np_vecs = np.array(vecs)
    pca.fit(np_vecs.T)
    x = pca.components_[0]
    y = pca.components_[1]
    for i, w in enumerate(fra):
        plt.scatter(x[i], y[i])
        plt.text(x[i], y[i], w)
    for i, w in enumerate(eng):
        plt.scatter(x[i+6], y[i+6])
        plt.text(x[i+6], y[i+6], w)
    plt.show()


if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_size, hidden_size, batch_size = 100, 128, 32
    franch_vocab, english_vocab, pairs = loadSmallData()
    encoder = EncoderRNN(franch_vocab.n_words, embed_size, hidden_size)
    decoder = AttnDecoderRNN(embed_size, hidden_size, english_vocab.n_words)
    load_model(encoder, "Add_Attention_Encoder.pth", device)
    load_model(decoder, "Add_Attention_Decoder.pth", device)
    encoder.eval()
    decoder.eval()

    #check_embedding(encoder, decoder, franch_vocab, english_vocab)
    
    vecs = evaluateRandomly(encoder, decoder, franch_vocab, english_vocab, pairs)
    
    sentences = [
        ["tu es tres effrontee", "you re very forward"],
        ["tu es une sacree menteuse", "you re such a liar"],
        ["nous sommes impuissants", "we re helpless"],
        ["je suis ici pour faire ce que je peux", "i m here to do what i can"],
        ["je suis desole mon pere est sorti", "i m sorry my father is out"],
        ["il est accoutume a voyager", "he is used to traveling"],
    ]
    evaluatePairs(encoder, decoder, franch_vocab, english_vocab, sentences)

    print("-------------")

    list_sentences = [
        ["il n est pas aussi grand que son pere", "he is not as tall as his father"],
        ["je suis trop fatigue pour conduire", "i m too tired to drive"],
        ["je suis desole si c est une question idiote", "i m sorry if this is a stupid question"],
        ["je suis reellement fiere de vous","i m really proud of you"]
    ]
    evaluateAndShowAttention(list_sentences)



