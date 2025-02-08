
import datasets
from transformers import AutoTokenizer
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

def collate_fn(batch_data):
    x_batch = []
    for x in batch_data:
        x_batch.append(x['input_ids'])
    X = torch.nn.utils.rnn.pad_sequence(x_batch, batch_first=True, padding_value=0)
    return X


def data_loader(batch_size):
    tokenized_datasets = datasets.load_from_disk("../data/ch7/pycpilot/tokenized_datasets")
    train_dataloader = DataLoader(tokenized_datasets["train"], 
                                  batch_size=batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn,
                                  num_workers=4, 
                                  pin_memory=True, 
                                  persistent_workers=True)
    valid_dataloader = DataLoader(tokenized_datasets["valid"], batch_size=batch_size)
    return train_dataloader, valid_dataloader


# def evaluate():
#     import evaluate
#     metric = evaluate.load("accuracy")
#     model.eval()
#     for batch in valid_dataloader:
#         batch = {k: v.to(device) for k, v in batch.items()}
#         with torch.no_grad():
#             outputs = model(**batch)

#         logits = outputs.logits
#         predictions = torch.argmax(logits, dim=-1)
#         metric.add_batch(predictions=predictions, references=batch["labels"])

#     metric.compute()

if __name__=="__main__":
    batch_size = 16
    context_length = 256  # 设置序列最大长度, 与数据处理时保持一致
    num_epochs = 3
    log_step = 100
    output_dir = "../model/ch7/pycpilot/"

    print("load tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("../model/ch7/pycpilot/tokenizer/")
    train_dataloader, valid_dataloader = data_loader(batch_size)
    print(len(train_dataloader))
    print(len(valid_dataloader))
    print("load data done")
    

    config = AutoConfig.from_pretrained(
        "gpt2",                     # 继承 GPT-2 模型参数
        vocab_size=len(tokenizer),  # 设置为分词器词表的大小
        n_ctx=context_length,       # 设置为数据集序列最大长度
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = GPT2LMHeadModel(config)
    model_size = sum(t.numel() for t in model.parameters())
    # 计算并显示模型参数
    print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="cosine", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.train()

    tb_writer = SummaryWriter(log_dir="tensorboard_summary/")   
    now = datetime.now()

    overall_step = 0 # 整体的迭代计数器
    running_loss = 0
    for epoch in range(num_epochs):
        internal_step = 0  # 一个 epoch 内部的迭代计数器
        for batch in train_dataloader:
            batch = batch.to(device)
            outputs = model.forward(input_ids=batch, labels=batch)
            loss, logits = outputs[:2]
            running_loss += loss.item() # 记录平均 loss
            loss.backward()         # 反向传播
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()        # 更新参数
            lr_scheduler.step()     # 更新学习率
            optimizer.zero_grad()   # 清空梯度
            if (internal_step + 1) % log_step == 0:
                running_loss = running_loss / log_step
                tb_writer.add_scalar('loss', running_loss, overall_step)
                print('now time: {}:{}. Step {} of epoch {}, loss {}'.format(
                    datetime.now().hour, datetime.now().minute, 
                    internal_step + 1, epoch + 1, running_loss
                ))
                running_loss = 0

            overall_step += 1
            internal_step += 1

        print('saving model for epoch {}'.format(epoch + 1))
        if not os.path.exists(output_dir + 'model_epoch{}'.format(epoch + 1)):
            os.mkdir(output_dir + 'model_epoch{}'.format(epoch + 1))
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(output_dir + 'model_epoch{}'.format(epoch + 1))
        print('epoch {} finished'.format(epoch + 1))

        then = datetime.now()
        print('time: {}'.format(then))
        print('time for this one epoch: {}'.format(then - now))

    print('training finished')
    if not os.path.exists(output_dir + 'final_model'):
        os.mkdir(output_dir + 'final_model')
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir + 'final_model')
    
