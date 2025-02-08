import os
import logging
import numpy as np
from tqdm.auto import tqdm
import torch
from transformers import AutoConfig, AutoTokenizer
from transformers import AdamW, get_scheduler
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

from H5_6_NER_Data import PeopleDaily, get_dataLoader, CATEGORIES, EntityCategory
from H5_6_NER_Model import BertForNER
from H5_6_NER_Arg import parse_args
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

def train_loop(device, entity, dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_step_num = epoch * len(dataloader)
    
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
        progress_bar.set_description(f'loss: {total_loss/(finish_step_num + step):>7f}')
        progress_bar.update(1)
    return total_loss

def test_loop(args, dataloader, model):
    true_labels, true_predictions = [], []

    model.eval()
    with torch.no_grad():
        for batch_data in tqdm(dataloader):
            batch_data = to_device(args, batch_data)
            outputs = model(**batch_data)
            logits = outputs[1]
            predictions = logits.argmax(dim=-1).cpu().numpy() # [batch, seq]
            labels = batch_data['labels'].cpu().numpy()
            lens = np.sum(batch_data['batch_inputs']['attention_mask'].cpu().numpy(), axis=-1)
            true_labels += [
                [entity.id2label[int(l)] for idx, l in enumerate(label) if idx > 0 and idx < seq_len - 1] 
                for label, seq_len in zip(labels, lens)
            ]
            true_predictions += [
                [entity.id2label[int(p)] for idx, p in enumerate(prediction) if idx > 0 and idx < seq_len - 1]
                for prediction, seq_len in zip(predictions, lens)
            ]
    return classification_report(true_labels, true_predictions, mode='strict', scheme=IOB2, output_dict=True)

def train(device, entity: EntityCategory, train_dataset, dev_dataset, model, tokenizer):
    """ Train the model """
    train_dataloader = get_dataLoader(args, entity, train_dataset, tokenizer, shuffle=True)
    dev_dataloader = get_dataLoader(args, entity, dev_dataset, tokenizer, shuffle=False)
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
    best_f1 = 0.
    for epoch in range(args.num_train_epochs):
        print(f"Epoch {epoch+1}/{args.num_train_epochs}\n-------------------------------")
        total_loss = train_loop(device, entity, train_dataloader, model, optimizer, lr_scheduler, epoch, total_loss)
        metrics = test_loop(args, dev_dataloader, model)
        micro_f1, macro_f1 = metrics['micro avg']['f1-score'], metrics['macro avg']['f1-score']
        dev_f1 = metrics['weighted avg']['f1-score']
        logger.info(f'Dev: micro_F1 - {(100*micro_f1):0.4f} macro_f1 - {(100*macro_f1):0.4f} weighted_f1 - {(100*dev_f1):0.4f}')
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            logger.info(f'saving new weights to {args.output_dir}...\n')
            save_weight = f'epoch_{epoch+1}_dev_macrof1_{(100*macro_f1):0.4f}_microf1_{(100*micro_f1):0.4f}_weights.bin'
            torch.save(model.state_dict(), os.path.join(args.output_dir, save_weight))
    logger.info("Done!")

def test(args, test_dataset, model, tokenizer, save_weights:list):
    test_dataloader = get_dataLoader(args, test_dataset, tokenizer, batch_size=1, shuffle=False)
    logger.info('***** Running testing *****')
    for save_weight in save_weights:
        logger.info(f'loading weights from {save_weight}...')
        model.load_state_dict(torch.load(os.path.join(args.output_dir, save_weight)))
        metrics = test_loop(args, test_dataloader, model)
        micro_f1, macro_f1, dev_f1 = metrics['micro avg']['f1-score'], metrics['macro avg']['f1-score'], metrics['weighted avg']['f1-score']
        logger.info(f'Test: micro_F1 - {(100*micro_f1):0.4f} macro_f1 - {(100*macro_f1):0.4f} weighted_f1 - {(100*dev_f1):0.4f}')


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_gpu = torch.cuda.device_count()
    logger.warning(f'Using {device} device, n_gpu: {n_gpu}')
    # Set seed
    seed_everything(args.seed)
    # Prepare task
    entity = EntityCategory()
    # Load pretrained model and tokenizer
    logger.info(f'loading pretrained model and tokenizer of {args.model_type} ...')
    config = AutoConfig.from_pretrained(args.model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = BertForNER.from_pretrained(
        args.model_checkpoint,
        config=config,
        num_labels=len(entity)
    ).to(device)
    # Training
    train_dataset = PeopleDaily(args.train_file)
    dev_dataset = PeopleDaily(args.valid_file)
    train(device, entity, train_dataset, dev_dataset, model, tokenizer)
