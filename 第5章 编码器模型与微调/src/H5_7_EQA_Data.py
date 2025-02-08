from torch.utils.data import Dataset, DataLoader
import json
from transformers import AutoTokenizer

from H5_7_EQA_Arg import parse_args

class CMRC_QA(Dataset):
    def __init__(self, data_file):
        self.data = {}
        with open(data_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            idx = 0
            for article in json_data['data']:
                title = article['title']
                context = article['paragraphs'][0]['context']
                for question in article['paragraphs'][0]['qas']:
                    q_id = question['id']
                    ques = question['question']
                    text = [ans['text'] for ans in question['answers']]
                    answer_start = [ans['answer_start'] for ans in question['answers']]
                    self.data[idx] = {
                        'id': q_id,
                        'title': title,
                        'context': context, 
                        'question': ques,
                        'answers': {
                            'text': text,
                            'answer_start': answer_start
                        }
                    }
                    idx += 1
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_dataLoader(args, dataset, tokenizer, mode='train', batch_size=None, shuffle=False):
    
    assert mode in ['train', 'valid', 'test']

    def train_collote_fn(batch_samples):
        batch_question, batch_context, batch_answers = [], [], []
        for sample in batch_samples:
            batch_question.append(sample['question'])
            batch_context.append(sample['context'])
            batch_answers.append(sample['answers'])
        batch_inputs = tokenizer(
            batch_question,
            batch_context,
            max_length=args.max_length,
            truncation="only_second",
            stride=args.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding='max_length',
            return_tensors="pt"
        )
        
        offset_mapping = batch_inputs.pop('offset_mapping')
        sample_mapping = batch_inputs.pop('overflow_to_sample_mapping')

        start_positions = []
        end_positions = []
        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_mapping[i]
            answer = batch_answers[sample_idx]
            start_char = answer['answer_start'][0]
            end_char = answer['answer_start'][0] + len(answer['text'][0])
            sequence_ids = batch_inputs.sequence_ids(i)
            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1
            # If the answer is not fully inside the context, label is (0, 0)
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)
        return {
            'batch_inputs': batch_inputs, 
            'start_positions': start_positions, 
            'end_positions': end_positions
        }

    def test_collote_fn(batch_samples):
        batch_id, batch_question, batch_context = [], [], []
        for sample in batch_samples:
            batch_id.append(sample['id'])
            batch_question.append(sample['question'])
            batch_context.append(sample['context'])
        batch_inputs = tokenizer(
            batch_question,
            batch_context,
            max_length=args.max_length,
            truncation="only_second",
            stride=args.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length", 
            return_tensors="pt"
        )
        offset_mapping = batch_inputs.pop('offset_mapping').numpy().tolist()
        sample_mapping = batch_inputs.pop('overflow_to_sample_mapping')
        example_ids = []
        for i in range(len(batch_inputs['input_ids'])):
            sample_idx = sample_mapping[i]
            example_ids.append(batch_id[sample_idx])

            sequence_ids = batch_inputs.sequence_ids(i)
            offset = offset_mapping[i]
            offset_mapping[i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]
        return {
            'batch_inputs': batch_inputs, 
            'offset_mapping': offset_mapping, 
            'example_ids': example_ids
        }
    
    if mode == 'train':
        collote_fn = train_collote_fn
    else:
        collote_fn = test_collote_fn

    return DataLoader(dataset, batch_size=(batch_size if batch_size else args.batch_size), shuffle=shuffle, 
                      collate_fn=collote_fn)

if __name__=="__main__":
    args = parse_args()
    # train_set = CMRC_QA(args.train_file)
    valid_set = CMRC_QA(args.valid_file)
    # test_set = CMRC_QA(args.test_file)
    # print(f'train set size: {len(train_set)}')
    # print(f'valid set size: {len(valid_set)}')
    # print(f'test set size: {len(test_set)}')
    # print(next(iter(valid_set)))


    checkpoint = 'bert-base-chinese'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    for i in range(1000):
        if valid_set[i]["id"] == "DEV_5_QUERY_0":
            break
    context = valid_set[i]["context"]
    question = valid_set[i]["question"]

    inputs = tokenizer(question, context, max_length=40, truncation="only_second", 
                       stride=4, return_overflowing_tokens=True)

    for ids in inputs["input_ids"]:
        print(tokenizer.decode(ids))