import os
import json
import numpy as np
from tqdm.auto import tqdm
import collections
import torch
from transformers import AutoConfig, AutoTokenizer
from H5_7_EQA_Data import CMRC_QA, get_dataLoader
from H5_7_EQA_Model import BertForExtractiveQA
from H5_7_EQA_Arg import parse_args
from H5_7_EQA_Eva import evaluate
from H5_7_EQA_Train import to_device
from H5_Helper import seed_everything

def test_loop(args, dataloader, dataset, model):
    all_example_ids = []
    all_offset_mapping = []
    for batch_data in dataloader:
        all_example_ids += batch_data['example_ids']
        all_offset_mapping += batch_data['offset_mapping']
    example_to_features = collections.defaultdict(list)
    for idx, feature_id in enumerate(all_example_ids):
        example_to_features[feature_id].append(idx)

    start_logits = []
    end_logits = []
    model.eval()
    with torch.no_grad():
        for batch_data in tqdm(dataloader):
            batch_data.pop('example_ids')
            batch_data.pop('offset_mapping')
            batch_data = to_device(args, batch_data)
            outputs = model(**batch_data)
            pred_start_logits, pred_end_logit = outputs[1], outputs[2]
            start_logits.append(pred_start_logits.cpu().numpy())
            end_logits.append(pred_end_logit.cpu().numpy())
    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)
    
    theoretical_answers = [
        {"id": dataset[s_idx]["id"], "answers": dataset[s_idx]["answers"]} for s_idx in range(len(dataset))
    ]
    predicted_answers = []
    for s_idx in tqdm(range(len(dataset))):
        example_id = dataset[s_idx]["id"]
        context = dataset[s_idx]["context"]
        answers = []
        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = all_offset_mapping[feature_index]

            start_indexes = np.argsort(start_logit)[-1 : -args.n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -args.n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    if (end_index < start_index or end_index-start_index+1 > args.max_answer_length):
                        continue
                    answers.append({
                        "start": offsets[start_index][0], 
                        "text": context[offsets[start_index][0] : offsets[end_index][1]], 
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    })
        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append({
                "id": example_id, 
                "prediction_text": best_answer["text"], 
                "answer_start": best_answer["start"]
            })
        else:
            predicted_answers.append({
                "id": example_id, 
                "prediction_text": "", 
                "answer_start": 0
            })
    return evaluate(predicted_answers, theoretical_answers)

def test(args, test_dataset, model, tokenizer, save_weights:list):
    test_dataloader = get_dataLoader(args, test_dataset, tokenizer, mode='test', shuffle=False)
    for save_weight in save_weights:
        model.load_state_dict(torch.load(os.path.join(args.output_dir, save_weight)))
        metrics = test_loop(args, test_dataloader, test_dataset, model)
        F1_score, EM_score, avg_score = metrics['f1'], metrics['em'], metrics['avg']

def predict(args, device, context:str, question:str, model, tokenizer):
    inputs = tokenizer(
        question,
        context,
        max_length=args.max_length,
        truncation="only_second",
        stride=args.stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length", 
        return_tensors="pt"
    )
    chunk_num = inputs['input_ids'].shape[0]
    inputs.pop('overflow_to_sample_mapping')
    offset_mapping = inputs.pop('offset_mapping').numpy().tolist()
    for i in range(chunk_num):
        sequence_ids = inputs.sequence_ids(i)
        offset = offset_mapping[i]
        offset_mapping[i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]
    
    inputs = {
        'batch_inputs': inputs
    }
    inputs = to_device(device, inputs)
    with torch.no_grad():
        _, pred_start_logits, pred_end_logit = model(**inputs)
        start_logits = pred_start_logits.cpu().numpy()
        end_logits = pred_end_logit.cpu().numpy()
    answers = []
    for feature_index in range(chunk_num):
        start_logit = start_logits[feature_index]
        end_logit = end_logits[feature_index]
        offsets = offset_mapping[feature_index]

        start_indexes = np.argsort(start_logit)[-1 : -args.n_best - 1 : -1].tolist()
        end_indexes = np.argsort(end_logit)[-1 : -args.n_best - 1 : -1].tolist()
        for start_index in start_indexes:
            for end_index in end_indexes:
                if offsets[start_index] is None or offsets[end_index] is None:
                    continue
                if (end_index < start_index or end_index-start_index+1 > args.max_answer_length):
                    continue
                answers.append({
                    "start": offsets[start_index][0], 
                    "text": context[offsets[start_index][0] : offsets[end_index][1]], 
                    "logit_score": start_logit[start_index] + end_logit[end_index],
                })
    if len(answers) > 0:
        best_answer = max(answers, key=lambda x: x["logit_score"])
        return {
            "prediction_text": best_answer["text"], 
            "answer_start": best_answer["start"]
        }
    else:
        return {
            "prediction_text": "", 
            "answer_start": 0
        }
    
if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Set seed
    seed_everything(args.seed)
    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(args.model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = BertForExtractiveQA.from_pretrained(
        args.model_checkpoint,
        config=config,
        num_labels=2
    ).to(device)
    save_weights = [file for file in os.listdir(args.output_dir) if file.endswith('.bin')]
    # Predicting
    test_dataset = CMRC_QA(args.test_file)
    for save_weight in save_weights:
        model.load_state_dict(torch.load(os.path.join(args.output_dir, save_weight)))
        results = []
        model.eval()
        for s_idx in tqdm(range(len(test_dataset))):
            sample = test_dataset[s_idx]
            pred_answer = predict(args, device, sample['context'], sample['question'], model, tokenizer)
            results.append({
                "id": sample['id'], 
                "title": sample['title'], 
                "context": sample['context'], 
                "question": sample['question'], 
                "answers": sample['answers'], 
                "prediction_text": pred_answer['prediction_text'], 
                "answer_start": pred_answer['answer_start']
            })
        with open(os.path.join(args.output_dir, save_weight + '_test_data_pred.json'), 'wt', encoding='utf-8') as f:
            for exapmle_result in results:
                f.write(json.dumps(exapmle_result, ensure_ascii=False) + '\n')
        