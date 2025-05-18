
from datasets import load_dataset
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from functools import partial
from torch.utils.data import DataLoader
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from transformers import get_scheduler
import os

def save_model(model_output_dir, model, epoch):
    path = os.path.join(model_output_dir, 'model_epoch{}'.format(epoch + 1))
    if not os.path.exists(path):
        os.makedirs(path)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(path)


class GSMDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer: GPT2Tokenizer, ds):
        self.tokenized = []
        for data in tqdm(ds):
            new_data = data['instruction']  + "\\n" + data['output'] + "<|endoftext|>"
            token_ids = tokenizer.encode(new_data)
            self.tokenized.append(token_ids)

    def __len__(self):
        return len(self.tokenized)

    def __getitem__(self, idx):
        return self.tokenized[idx]


def custom_collate_fn(
    batch,
    pad_token_id = 0,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    # Find the longest sequence in the batch
    batch_max_length = max(len(item) for item in batch)
    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # Pad sequences to max_length
        padded = (new_item + [pad_token_id] * (batch_max_length - len(new_item)))
        # inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        # targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets
        inputs = torch.tensor(padded)
        targets = torch.tensor(padded)

        # New: Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices] = ignore_index

        # New: Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs and targets to tensors and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

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

    log_step = 100
    valid_step = 5000
    overall_step = 0 # 整体的迭代计数器
    running_loss = 0
    now = datetime.now()

    for epoch in range(num_epochs):
        internal_step = 0  # 一个 epoch 内部的迭代计数器
        for input, label in train_loader:
            model.train()
            input = input.to(device)
            label = label.to(device)
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

if __name__=="__main__":
    model_path = "gpt2-medium"
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    print("特殊 token:", tokenizer.special_tokens_map)
    batch_size = 2
    epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel.from_pretrained(model_path)
    print(f"模型参数: {model.num_parameters():,}")    
    model.to(device)
    customized_collate_fn = partial(
        custom_collate_fn,
        pad_token_id = 0, 
        device=device,
        allowed_max_length=model.config.n_ctx  # 1024
    )
    ds_train = load_dataset("causal-lm/cot_flan", split="train")
    train_dataset = GSMDataset(tokenizer, ds_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=False,
        num_workers=2,
    )

    ds_valid = load_dataset("causal-lm/cot_flan", split="validation")
    valid_dataset = GSMDataset(tokenizer, ds_valid)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    output_dir = "../model/ch8/cot_flan/"
    sft_model(model, train_loader, valid_loader, device, output_dir)
