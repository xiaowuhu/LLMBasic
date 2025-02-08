import os
import json
import numpy as np
from tqdm.auto import tqdm
import torch
from transformers import AutoConfig, AutoTokenizer

from H5_6_NER_Data import PeopleDaily, get_dataLoader, CATEGORIES, EntityCategory
from H5_6_NER_Model import BertForNER
from H5_6_NER_Arg import parse_args
from H5_Helper import seed_everything
from H5_6_NER_Train import to_device

def predict(device, entity: EntityCategory, sentence:str, model, tokenizer):
    inputs = tokenizer(
        sentence, 
        max_length=args.max_seq_length, 
        truncation=True, 
        return_tensors="pt", 
        return_offsets_mapping=True
    )
    offsets = inputs.pop('offset_mapping').squeeze(0)
    inputs = {
        'batch_inputs': inputs
    }
    inputs = to_device(device, inputs)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs[1]
    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy().tolist()
    predictions = logits.argmax(dim=-1)[0].cpu().numpy().tolist()

    pred_label = []
    idx = 1
    while idx < len(predictions) - 1:
        pred = predictions[idx]
        label = entity.id2label[pred]
        if label != "O":
            label = label[2:] # Remove the B- or I-
            start, end = offsets[idx]
            all_scores = [probabilities[idx][pred]]
            # Grab all the tokens labeled with I-label
            while (
                idx + 1 < len(predictions) - 1 and 
                entity.id2label[predictions[idx + 1]] == f"I-{label}"
            ):
                all_scores.append(probabilities[idx + 1][predictions[idx + 1]])
                _, end = offsets[idx + 1]
                idx += 1

            score = np.mean(all_scores).item()
            start, end = start.item(), end.item()
            word = sentence[start:end]
            pred_label.append({
                "entity_group": label,
                "score": score,
                "word": word,
                "start": start,
                "end": end
            })
        idx += 1
    return pred_label

if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_everything(args.seed)
    entity = EntityCategory()
    config = AutoConfig.from_pretrained(args.model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = BertForNER.from_pretrained(
        args.model_checkpoint,
        config=config,
        num_labels=len(entity)
    ).to(device)
    save_weights = [file for file in os.listdir(args.output_dir) if file.endswith('.bin')]
    # Predicting
    test_dataset = PeopleDaily(args.test_file)
    for save_weight in save_weights:
        model.load_state_dict(torch.load(os.path.join(args.output_dir, save_weight)))
        results = []
        model.eval()
        for s_idx in tqdm(range(len(test_dataset))):
            sample = test_dataset[s_idx]
            pred_label = predict(device, entity, sample['sentence'], model, tokenizer)
            results.append({
                "sentence": sample['sentence'], 
                "pred_label": pred_label, 
                "true_label": sample['labels']
            })
        with open(os.path.join(args.output_dir, save_weight + '_test_data_pred.json'), 'wt', encoding='utf-8') as f:
            for exapmle_result in results:
                f.write(json.dumps(exapmle_result, ensure_ascii=False) + '\n')
