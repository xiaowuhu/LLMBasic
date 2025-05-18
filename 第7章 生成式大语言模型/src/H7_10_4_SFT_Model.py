import torch
from torch.utils.tensorboard import SummaryWriter
from H7_10_4_Data_Loader import load_dataset
import os
from transformers import GPT2LMHeadModel
from datetime import datetime
from transformers import get_scheduler


def save_model(model_output_dir, model, epoch):
    path = os.path.join(model_output_dir, 'model_epoch{}'.format(epoch + 1))
    if not os.path.exists(path):
        os.makedirs(path)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(path)


def save_best_model(model_output_dir, model):
    path = os.path.join(model_output_dir, 'best')
    if not os.path.exists(path):
        os.makedirs(path)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(path)


def sft_model(model, train_loader, valid_loader, device, model_output_dir):
    num_epochs = 5
    lr = 1.5e-4
    max_grad_norm = 1.0
    gradient_accumulation = 2
    tb_writer = SummaryWriter(log_dir="tensorboard_summary/")   
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)
    lr_scheduler = get_scheduler(
        name="cosine", 
        optimizer=optimizer, 
        num_warmup_steps=500, 
        num_training_steps=num_epochs * len(train_loader)
    )

    log_step = 50
    valid_step = 2000
    overall_step = 0 # 整体的迭代计数器
    running_loss = 0
    best_val_loss = 10
    now = datetime.now()

    for epoch in range(num_epochs):
        internal_step = 0  # 一个 epoch 内部的迭代计数器
        for input, in train_loader:
            model.train()
            input = input.to(device)
            outputs = model.forward(input_ids=input, labels=input)
            loss, logits = outputs[:2]
            running_loss += loss.item() # 记录平均 loss
            # 反向传播
            if gradient_accumulation > 1: # 如果使用了梯度累加
                accu_loss = loss / gradient_accumulation
            else:
                accu_loss = loss
            accu_loss.backward()  # 梯度存入模型参数中，但还没有更新参数
            # 梯度截断，避免梯度过大
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if (overall_step + 1) % gradient_accumulation == 0:
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
                model.eval()
                val_loss = validate_model(model, device, valid_loader)
                tb_writer.add_scalar('Loss/val', val_loss, overall_step)
                print(f"val loss: {val_loss} , step: {overall_step}")
                # 保存最优模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_best_model(model_output_dir, model)
                
        # 保存当前模型
        print('saving model for epoch {}'.format(epoch + 1))
        save_model(model_output_dir, model, epoch)
        print('epoch {} finished'.format(epoch + 1))

        then = datetime.now()
        print('time: {}'.format(then))
        print('time for this one epoch: {}'.format(then - now))

        # 验证
        model.eval()
        val_loss = validate_model(model, device, valid_loader)
        tb_writer.add_scalar('Loss/val', val_loss, overall_step)
        print(f"val loss: {val_loss} , step: {overall_step}")
        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_best_model(model_output_dir, model)
        
    tb_writer.close()


def validate_model(model, device, valid_loader):
    running_loss = 0.0
    with torch.no_grad():
        counter = 0
        for input_ids, in valid_loader:
            input_ids = input_ids.to(device, dtype=torch.long)
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss, logits = outputs[:2]
            running_loss += loss
            counter += 1
            if counter >= 100:
                break
    return running_loss / counter


def main():
    BACTH_SIZE = 64  # 训练一个批次的大小
    model_output_dir = "../model/ch7/chat/"  # 模型保存目录
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel.from_pretrained("uer/gpt2-distil-chinese-cluecorpussmall")
    print(f"模型参数: {model.num_parameters():,}")    
    model = model.to(device)    
    print("Start Load Train Data...")
    train_dataset_path = "../model/ch7/chat/train_dataset.pkl"
    train_loader = load_dataset(BACTH_SIZE, train_dataset_path)
    valid_dataset_path = "../model/ch7/chat/valid_dataset.pkl"
    valid_loader = load_dataset(BACTH_SIZE, valid_dataset_path)
    # 开始训练
    print("Start Training...")
    sft_model(model, train_loader, valid_loader, device, model_output_dir)


if __name__ == '__main__':
    main()
