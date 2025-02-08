import torchtext; torchtext.disable_torchtext_deprecation_warning()
from torchtext.data.utils import get_tokenizer
import unicodedata
import re

en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
s1 = "Help!"
print(s1, "->", en_tokenizer(s1))

fr_tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')
s2 = "À l'aide !"
print(s2, "->", fr_tokenizer(s2))

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()

print(normalizeString(s1))
print(normalizeString(s2))
