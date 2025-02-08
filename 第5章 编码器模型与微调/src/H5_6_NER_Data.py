from torch.utils.data import Dataset, DataLoader
import numpy as np
from H5_6_NER_Arg import parse_args
from transformers import AutoTokenizer

CATEGORIES = ['LOC', 'ORG', 'PER']

class EntityCategory():
    def __init__(self):
        self.id2label = {0:'O'}
        for c in CATEGORIES:
            self.id2label[len(self.id2label)] = f"B-{c}"
            self.id2label[len(self.id2label)] = f"I-{c}"
        self.label2id = {v: k for k, v in self.id2label.items()}

    def __len__(self):
        return len(self.id2label)

class PeopleDaily(Dataset):
    def __init__(self, data_file):
        self.data = {}
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f.read().split('\n\n')):
                if not line:
                    break
                sentence, labels = '', []
                for i, item in enumerate(line.split('\n')):
                    char, tag = item.split(' ')
                    sentence += char
                    if tag.startswith('B'):
                        labels.append([i, i, char, tag[2:]]) # Remove the B- or I-
                    elif tag.startswith('I'):
                        labels[-1][1] = i
                        labels[-1][2] += char
                self.data[idx] = {
                    'sentence': sentence, 
                    'labels': labels
                }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_dataLoader(args, entity: EntityCategory, dataset:Dataset, tokenizer, batch_size=None, shuffle=False):
    
    def collote_fn(batch_samples):
        batch_sentence, batch_labels  = [], []
        for sample in batch_samples:
            batch_sentence.append(sample['sentence'])
            batch_labels.append(sample['labels'])
        batch_inputs = tokenizer(
            batch_sentence, 
            max_length=args.max_seq_length, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        batch_label = np.zeros(batch_inputs['input_ids'].shape, dtype=np.int64)
        for s_idx, sentence in enumerate(batch_sentence):
            encoding = tokenizer(sentence, max_length=args.max_seq_length, truncation=True)
            for char_start, char_end, word, tag in batch_labels[s_idx]:
                token_start = encoding.char_to_token(char_start)
                token_end = encoding.char_to_token(char_end)
                if not token_start or not token_end:
                    continue
                batch_label[s_idx][token_start] = entity.label2id[f"B-{tag}"]
                batch_label[s_idx][token_start+1:token_end+1] = entity.label2id[f"I-{tag}"]
        return {
            'batch_inputs': batch_inputs, 
            'labels': batch_label
        }
    
    return DataLoader(dataset, batch_size=(batch_size if batch_size else args.batch_size), shuffle=shuffle, 
                      collate_fn=collote_fn)


if __name__=="__main__":
    entity = EntityCategory()
    print(entity.id2label)
    print(entity.label2id)

    valid_data = PeopleDaily('../data/china-people-daily-ner/valid.txt')
    print(valid_data[0])
    print(valid_data[1])

    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    data_loader = get_dataLoader(args, entity, valid_data, tokenizer, batch_size=2, shuffle=False)
    for i, data in enumerate(data_loader):
        if i == 0:
            print("X:\n",data['batch_inputs'])
            print("Y:\n", data['labels'])
            break
