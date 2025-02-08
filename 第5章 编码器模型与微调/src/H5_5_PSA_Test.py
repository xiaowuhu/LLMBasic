import os
import json
from tqdm.auto import tqdm
import torch
from transformers import AutoConfig, AutoTokenizer

from H5_5_PSA_Data import ChnSentiCorp, get_prompt, get_verbalizer
from H5_5_PSA_Model import BertForPrompt
from H5_5_PSA_Train import to_device
from H5_5_PSA_Arg import parse_args
from H5_Helper import seed_everything


def predict(args, comment:str, model, tokenizer, verbalizer):
    prompt_data = get_prompt(comment)
    prompt = prompt_data['prompt']
    encoding = tokenizer(prompt, truncation=True)
    mask_idx = encoding.char_to_token(prompt_data['mask_offset'])
    assert mask_idx is not None
    inputs = tokenizer(
        prompt, 
        max_length=args.max_length, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    inputs = {
        'batch_inputs': inputs, 
        'batch_mask_idxs': [mask_idx], 
        'label_word_id': [verbalizer['neg']['id'], verbalizer['pos']['id']] 
    }
    inputs = to_device(args, inputs)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs[1]
        prob = torch.nn.functional.softmax(logits, dim=-1)
    pred = logits.argmax(dim=-1)[0].item()
    prob = prob[0][pred].item()
    return {
        "pred": pred, 
        "prob": prob
    }

if __name__ == '__main__':
    args = parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()
    # Set seed
    seed_everything(args.seed)
    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(args.model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = BertForPrompt.from_pretrained(
        args.model_checkpoint,
        config=config
    ).to(args.device)
    if args.vtype == 'virtual': # virtual label words
        sp_tokens = ['[pos]', '[neg]']
        tokenizer.add_special_tokens({'additional_special_tokens': sp_tokens})
        model.resize_token_embeddings(len(tokenizer))
        verbalizer = get_verbalizer(tokenizer, vtype=args.vtype)
        # with torch.no_grad():
        #     pos_id, neg_id = verbalizer['pos']['id'], verbalizer['neg']['id']
        #     pos_tokenized = tokenizer.tokenize(verbalizer['pos']['description'])
        #     pos_tokenized_ids = tokenizer.convert_tokens_to_ids(pos_tokenized)
        #     neg_tokenized = tokenizer.tokenize(verbalizer['neg']['description'])
        #     neg_tokenized_ids = tokenizer.convert_tokens_to_ids(neg_tokenized)
        #     new_embedding = model.bert.embeddings.word_embeddings.weight[pos_tokenized_ids].mean(axis=0)
        #     model.bert.embeddings.word_embeddings.weight[pos_id, :] = new_embedding.clone().detach().requires_grad_(True)
        #     new_embedding = model.bert.embeddings.word_embeddings.weight[neg_tokenized_ids].mean(axis=0)
        #     model.bert.embeddings.word_embeddings.weight[neg_id, :] = new_embedding.clone().detach().requires_grad_(True)
    else: # base label words
        verbalizer = get_verbalizer(tokenizer, vtype=args.vtype)


    save_weights = [file for file in os.listdir(args.output_dir) if file.endswith('.bin')]
    # Predicting
    test_dataset = ChnSentiCorp(args.test_file)
    for save_weight in save_weights:
        model.load_state_dict(torch.load(os.path.join(args.output_dir, save_weight)))
        
        results = []
        for s_idx in tqdm(range(len(test_dataset))):
            sample = test_dataset[s_idx]
            pred_res = predict(args, sample['comment'], model, tokenizer, verbalizer)
            results.append({
                "comment": sample['comment'], 
                "label": int(sample['label']), # 转成64位，CrossEntropyLoss 需要
                "pred": pred_res['pred'], 
                "prob": pred_res['prob']
            })
        with open(os.path.join(args.output_dir, save_weight + '_test_data_pred.json'), 'wt', encoding='utf-8') as f:
            for exapmle_result in results:
                f.write(json.dumps(exapmle_result, ensure_ascii=False) + '\n')
