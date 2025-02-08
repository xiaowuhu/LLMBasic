from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


def get_prompt(x):
    prompt = f'感觉很[MASK]。{x}'
    return {
        'prompt': prompt, 
        'mask_offset': prompt.find('[MASK]')
    }

def get_verbalizer(tokenizer, vtype):
    assert vtype in ['base', 'virtual']
    return {
        'pos': {'token': '好', 'id': tokenizer.convert_tokens_to_ids("好")}, 
        'neg': {'token': '差', 'id': tokenizer.convert_tokens_to_ids("差")}
    } if vtype == 'base' else {
        'pos': {
            'token': '[pos]', 'id': tokenizer.convert_tokens_to_ids("[pos]"), 
            'description': '好、棒、赞、酷、美'
        }, 
        'neg': {
            'token': '[neg]', 'id': tokenizer.convert_tokens_to_ids("[neg]"), 
            'description': '差、糟、丑、脏、孬'
        }
    }

class ChnSentiCorp(Dataset):
    def __init__(self, data_file):
        self.data = {} # 建立字典
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                items = line.strip().split('\t')
                assert len(items) == 2 # 只有两个字段
                prompt_data = get_prompt(items[0]) # 添加模板
                self.data[idx] = {
                    'comment': items[0], # 原始评论
                    'prompt': prompt_data['prompt'],  # 添加模板后的文本
                    'mask_offset': prompt_data['mask_offset'],  # mask 所在位置
                    'label': items[1] # 标签
                }
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_dataLoader(args, dataset, tokenizer, verbalizer, batch_size=None, shuffle=False):
    pos_id, neg_id = verbalizer['pos']['id'], verbalizer['neg']['id']
    
    def collote_fn(batch_samples):
        batch_sentences, batch_mask_idxs, batch_labels  = [], [], []
        for sample in batch_samples:
            batch_sentences.append(sample['prompt'])
            encoding = tokenizer(sample['prompt'], truncation=True)
            mask_idx = encoding.char_to_token(sample['mask_offset'])
            assert mask_idx is not None
            batch_mask_idxs.append(mask_idx)
            batch_labels.append(int(sample['label']))
        batch_inputs = tokenizer(
            batch_sentences, 
            max_length=args.max_length, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        label_word_id = [neg_id, pos_id]
        return {
            'batch_inputs': batch_inputs, 
            'batch_mask_idxs': batch_mask_idxs, 
            'label_word_id': label_word_id, 
            'labels': batch_labels
        }
    
    return DataLoader(
        dataset, 
        batch_size=(batch_size if batch_size else args.batch_size), 
        shuffle=shuffle, 
        collate_fn=collote_fn
    )

if __name__=="__main__":
    train_file = '../data/ChnSentiCorp/train.txt'
    pre_trained_model = "bert-base-chinese"
    train_dataset = ChnSentiCorp(train_file)
    tokenizer = AutoTokenizer.from_pretrained(pre_trained_model)
    data = next(iter(train_dataset))
    print("单条数据:\n", data)
    print("分词结果:\n", tokenizer.tokenize(data["prompt"]))
    print("[MASK] ID:", tokenizer.convert_tokens_to_ids('[MASK]'))

    v = ["好","棒","赞","酷","美","差","糟","丑","脏","孬","[unk]"]
    print(v)
    print("对应的 token ID:")
    print(tokenizer.convert_tokens_to_ids(v))
