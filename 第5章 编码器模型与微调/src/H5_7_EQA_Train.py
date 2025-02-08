import os
import logging
import numpy as np
from tqdm.auto import tqdm
import collections
import torch
from transformers import AutoConfig, AutoTokenizer
from transformers import AdamW, get_scheduler

from H5_7_EQA_Data import CMRC_QA, get_dataLoader
from H5_7_EQA_Model import BertForExtractiveQA
from H5_7_EQA_Arg import parse_args
from H5_7_EQA_Eva import evaluate
from H5_Helper import seed_everything

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger("Model")

def to_device(device, batch_data):
    new_batch_data = {}
    for k, v in batch_data.items():
        if k == 'batch_inputs':
            new_batch_data[k] = {
                k_: v_.to(device) for k_, v_ in v.items()
            }
        else:
            new_batch_data[k] = torch.tensor(v).to(device)
    return new_batch_data

def train_loop(device, dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_batch_num = epoch * len(dataloader)
    
    model.train()
    for step, batch_data in enumerate(dataloader, start=1):
        batch_data = to_device(device, batch_data)
        outputs = model(**batch_data)
        loss = outputs[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss/(finish_batch_num + step):>7f}')
        progress_bar.update(1)
    return total_loss

def test_loop(device, dataloader, dataset, model):
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
            batch_data = to_device(device, batch_data)
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

def train(args, device, train_dataset, dev_dataset, model, tokenizer):
    """ Train the model """
    train_dataloader = get_dataLoader(args, train_dataset, tokenizer, mode='train', shuffle=True)
    dev_dataloader = get_dataLoader(args, dev_dataset, tokenizer, mode='valid', shuffle=False)
    t_total = len(train_dataloader) * args.num_train_epochs
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    args.warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(
        optimizer_grouped_parameters, 
        lr=args.learning_rate, 
        betas=(args.adam_beta1, args.adam_beta2), 
        eps=args.adam_epsilon
    )
    lr_scheduler = get_scheduler(
        'linear',
        optimizer, 
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total
    )
    # Train!
    logger.info("***** Running training *****")
    logger.info(f"Num examples - {len(train_dataset)}")
    logger.info(f"Num Epochs - {args.num_train_epochs}")
    logger.info(f"Total optimization steps - {t_total}")
    with open(os.path.join(args.output_dir, 'args.txt'), 'wt') as f:
        f.write(str(args))
    
    total_loss = 0.
    best_avg_score = 0.
    for epoch in range(args.num_train_epochs):
        print(f"Epoch {epoch+1}/{args.num_train_epochs}\n" + 30 * "-")
        total_loss = train_loop(device, train_dataloader, model, optimizer, lr_scheduler, epoch, total_loss)
        metrics = test_loop(device, dev_dataloader, dev_dataset, model)
        F1_score, EM_score, avg_score = metrics['f1'], metrics['em'], metrics['avg']
        logger.info(f'Dev: F1 - {F1_score:0.4f} EM - {EM_score:0.4f} AVG - {avg_score:0.4f}')
        if avg_score > best_avg_score:
            best_avg_score = avg_score
            logger.info(f'saving new weights to {args.output_dir}...\n')
            save_weight = f'epoch_{epoch+1}_dev_f1_{F1_score:0.4f}_em_{EM_score:0.4f}_weights.bin'
            torch.save(model.state_dict(), os.path.join(args.output_dir, save_weight))
    logger.info("Done!")

if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Set seed
    seed_everything(args.seed)
    # Load pretrained model and tokenizer
    logger.info(f'loading pretrained model and tokenizer of {args.model_type} ...')
    config = AutoConfig.from_pretrained(args.model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = BertForExtractiveQA.from_pretrained(
        args.model_checkpoint,
        config=config,
        num_labels=2
    ).to(device)
    # Training
    train_dataset = CMRC_QA(args.train_file)
    dev_dataset = CMRC_QA(args.valid_file)
    train(args, device, train_dataset, dev_dataset, model, tokenizer)
