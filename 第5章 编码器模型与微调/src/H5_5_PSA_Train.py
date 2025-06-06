import os
import logging
from tqdm.auto import tqdm
import torch
from transformers import AutoConfig, AutoTokenizer
from transformers import AdamW, get_scheduler
from sklearn.metrics import classification_report
import sys
sys.path.append('../../')


from H5_5_PSA_Data import ChnSentiCorp, get_dataLoader, get_prompt, get_verbalizer
from H5_5_PSA_Model import BertForPrompt
from H5_5_PSA_Arg import parse_args
from H5_Helper import seed_everything

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger("Model")

def to_device(args, batch_data):
    new_batch_data = {}
    for k, v in batch_data.items():
        if k == 'batch_inputs':
            new_batch_data[k] = {
                k_: v_.to(args.device) for k_, v_ in v.items()
            }
        elif k == 'label_word_id':
            new_batch_data[k] = v
        else:
            new_batch_data[k] = torch.tensor(v).to(args.device)
    return new_batch_data

def train_loop(args, dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_step_num = epoch * len(dataloader)

    model.train()
    for step, batch_data in enumerate(dataloader, start=1):
        batch_data = to_device(args, batch_data)
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
    true_labels, predictions = [], []
    model.eval()
    with torch.no_grad():
        for batch_data in tqdm(dataloader):
            true_labels += batch_data['labels']
            batch_data = to_device(args, batch_data)
            outputs = model(**batch_data)
            logits = outputs[1]
            predictions += logits.argmax(dim=-1).cpu().numpy().tolist()
    return classification_report(true_labels, predictions, output_dict=True)

def train(args, train_dataset, dev_dataset, model, tokenizer, verbalizer):
    """ Train the model """
    train_dataloader = get_dataLoader(args, train_dataset, tokenizer, verbalizer, shuffle=True)
    valid_dataloader = get_dataLoader(args, dev_dataset, tokenizer, verbalizer, shuffle=False)
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
    best_f1_score = 0.
    for epoch in range(args.num_train_epochs):
        print(f"Epoch {epoch+1}/{args.num_train_epochs}\n" + 30 * "-")
        total_loss = train_loop(args, train_dataloader, model, optimizer, lr_scheduler, epoch, total_loss)
        metrics = test_loop(args, valid_dataloader, model)
        macro_f1, micro_f1 = metrics['macro avg']['f1-score'], metrics['weighted avg']['f1-score']
        dev_f1_score = (macro_f1 + micro_f1) / 2
        logger.info(f'Dev: micro_F1 - {(100*micro_f1):0.4f} macro_f1 - {(100*macro_f1):0.4f}')
        if dev_f1_score > best_f1_score:
            best_f1_score = dev_f1_score
            logger.info(f'saving new weights to {args.output_dir}...\n')
            save_weight = f'epoch_{epoch+1}_dev_macrof1_{(100*macro_f1):0.4f}_microf1_{(100*micro_f1):0.4f}_weights.bin'
            torch.save(model.state_dict(), os.path.join(args.output_dir, save_weight))
    logger.info("Done!")

if __name__ == '__main__':
    args = parse_args()
    # if args.do_train and os.path.exists(args.output_dir) and os.listdir(args.output_dir):
    #     raise ValueError(f'Output directory ({args.output_dir}) already exists and is not empty.')
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()
    logger.warning(f'Using {args.device} device, n_gpu: {args.n_gpu}')
    # Set seed
    seed_everything(args.seed)
    # Load pretrained model and tokenizer
    logger.info(f'loading pretrained model and tokenizer of {args.model_type} ...')
    config = AutoConfig.from_pretrained(args.model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = BertForPrompt.from_pretrained(
        args.model_checkpoint,
        config=config
    ).to(args.device)
    if args.vtype == 'virtual': # virtual label words
        sp_tokens = ['[pos]', '[neg]']
        logger.info(f'adding special tokens {sp_tokens} to tokenizer...')
        tokenizer.add_special_tokens({'additional_special_tokens': sp_tokens})
        model.resize_token_embeddings(len(tokenizer))
        verbalizer = get_verbalizer(tokenizer, vtype=args.vtype)
        logger.info(f"initializing embeddings of {verbalizer['pos']['token']} and {verbalizer['neg']['token']}...")
        with torch.no_grad():
            pos_id, neg_id = verbalizer['pos']['id'], verbalizer['neg']['id']
            pos_tokenized = tokenizer.tokenize(verbalizer['pos']['description'])
            pos_tokenized_ids = tokenizer.convert_tokens_to_ids(pos_tokenized)
            neg_tokenized = tokenizer.tokenize(verbalizer['neg']['description'])
            neg_tokenized_ids = tokenizer.convert_tokens_to_ids(neg_tokenized)
            new_embedding = model.bert.embeddings.word_embeddings.weight[pos_tokenized_ids].mean(axis=0)
            model.bert.embeddings.word_embeddings.weight[pos_id, :] = new_embedding.clone().detach().requires_grad_(True)
            new_embedding = model.bert.embeddings.word_embeddings.weight[neg_tokenized_ids].mean(axis=0)
            model.bert.embeddings.word_embeddings.weight[neg_id, :] = new_embedding.clone().detach().requires_grad_(True)
    else: # base label words
        verbalizer = get_verbalizer(tokenizer, vtype=args.vtype)
    # Training
    if args.do_train:
        train_dataset = ChnSentiCorp(args.train_file)
        dev_dataset = ChnSentiCorp(args.dev_file)
        train(args, train_dataset, dev_dataset, model, tokenizer, verbalizer)
 