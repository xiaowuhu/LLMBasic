'''
读取 jsonl 文件，把 instruction/input/output 变成以下形式:

### instruction
xxx

### input
xxx

### output
xxx
'''
import os
import torch
from transformers import GPT2LMHeadModel, BertTokenizer
from torch.utils.data import DataLoader
from functools import partial
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from transformers import get_scheduler

from LeetCode_Data import read_jsonl_file, custom_collate_fn, InstructionDataset


def save_model(model_output_dir, model, epoch):
    path = os.path.join(model_output_dir, 'model_epoch{}'.format(epoch + 1))
    if not os.path.exists(path):
        os.makedirs(path)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(path)


def sft_model(model, train_loader, valid_loader, device, model_output_dir):
    num_epochs = 5
    lr = 1.5e-4
    max_grad_norm = 1.0
    tb_writer = SummaryWriter(log_dir="tensorboard_summary/")   
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)
    lr_scheduler = get_scheduler(
        name="cosine", 
        optimizer=optimizer, 
        num_warmup_steps=500, 
        num_training_steps=num_epochs * len(train_loader)
    )

    log_step = 50
    valid_step = 500
    overall_step = 0 # 整体的迭代计数器
    running_loss = 0
    now = datetime.now()

    for epoch in range(num_epochs):
        internal_step = 0  # 一个 epoch 内部的迭代计数器
        for input, label in train_loader:
            model.train()
            input = input.to(device)
            outputs = model.forward(input_ids=input, labels=label)
            loss, logits = outputs[:2]
            running_loss += loss.item() # 记录平均 loss
            # 反向传播
            accu_loss = loss
            accu_loss.backward()  # 梯度存入模型参数中，但还没有更新参数
            # 梯度截断，避免梯度过大
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()  # 更新参数
            optimizer.zero_grad() # 清空梯度
            lr_scheduler.step() # 调整下一步的学习率
            # 记录 log
            if (internal_step + 1) % log_step == 0:
                running_loss = running_loss / log_step
                tb_writer.add_scalar('loss', running_loss, overall_step)
                print('now time: {}:{}. Step {} of epoch {}, loss={}, lr={}'.format(
                        datetime.now().hour, 
                        datetime.now().minute, 
                        internal_step + 1, 
                        epoch + 1, 
                        running_loss,
                        lr_scheduler.get_lr()[0],
                    )
                )
                running_loss = 0

            overall_step += 1
            internal_step += 1

            # 验证
            if internal_step % valid_step == 0:
                val_loss = validate_model(model, device, valid_loader)
                tb_writer.add_scalar('Loss/val', val_loss, overall_step)
                print(f"val loss: {val_loss} , step: {overall_step}")
                
        # 保存当前模型
        print('saving model for epoch {}'.format(epoch + 1))
        save_model(model_output_dir, model, epoch)
        print('epoch {} finished'.format(epoch + 1))

        then = datetime.now()
        print('time: {}'.format(then))
        print('time for this one epoch: {}'.format(then - now))

        
    tb_writer.close()


def validate_model(model, device, valid_loader):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for input, label in valid_loader:
            input = input.to(device, dtype=torch.long)
            outputs = model(input_ids=input, labels=label)
            loss, logits = outputs[:2]
            running_loss += loss
    return running_loss / len(valid_loader)


if __name__=="__main__":
    print("read data file...")
    file_path = "../data/leetcode/leetcode_instructions.jsonl"
    data = read_jsonl_file(file_path)
    print(f"数据集长度:{len(data)}")
    # str = format_input(data[10])
    # print(str)

    valid_portion = 200
    train_portion = 200 # len(data) - valid_portion

    train_data = data[:train_portion]
    valid_data = data[-valid_portion:]

    print("Training set length:", len(train_data))
    print("Test set length:", len(valid_data))

    print("load pre-trained tokenizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained("../model/ch7/code/new_vocab")
    print(f"词表大小:{len(tokenizer)}")
    print("load pre-trained model...")
    model_path = "../model/ch7/code/model_epoch1/"
    model_path = "uer/gpt2-distil-chinese-cluecorpussmall"
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)    
    #print(model.config)

    customized_collate_fn = partial(
        custom_collate_fn,
        pad_token_id = tokenizer.pad_token_id, 
        device=device,
        allowed_max_length=model.config.n_ctx  # 1024
    )

    num_workers = 0
    batch_size = 2
    torch.manual_seed(123)

    print("create train dataset and dataloader...")

    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        #persistent_workers=True
    )

    print("create test dataset and dataloader...")

    valid_dataset = InstructionDataset(valid_data, tokenizer)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=1,
        #persistent_workers=True
    )
    output_dir = "../model/ch7/code/"
    sft_model(model, train_loader, valid_loader, device, output_dir)
