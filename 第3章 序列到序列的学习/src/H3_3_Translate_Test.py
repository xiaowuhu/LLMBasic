import random
import torch

from H3_3_Translate_Data import loadSmallData, EOS_token
from H3_3_Translate_Model import EncoderRNN, DecoderRNN
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

        encoder_outputs, context_vector = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs.size(0), context_vector)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn

def evaluateRandomly(encoder, decoder, franch_vocab, english_vocab, pairs, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, _ = evaluate(encoder, decoder, pair[0], franch_vocab, english_vocab)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        score = BLEU(output_sentence, pair[1] + " <EOS>", 2)
        print("BLUE =", score)


if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_size, hidden_size, batch_size = 100, 128, 32
    franch_vocab, english_vocab, pairs = loadSmallData()
    encoder = EncoderRNN(franch_vocab.n_words, embed_size, hidden_size)
    decoder = DecoderRNN(embed_size, hidden_size, english_vocab.n_words)
    load_model(encoder, "Translate_Encoder.pth", device)
    load_model(decoder, "Translate_Decoder.pth", device)
    encoder.eval()
    decoder.eval()
    evaluateRandomly(encoder, decoder, franch_vocab, english_vocab, pairs)
