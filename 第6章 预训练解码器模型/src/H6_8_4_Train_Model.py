import argparse
import os.path
from datetime import datetime
import torch
import transformers
from torch.utils.tensorboard import SummaryWriter
from H6_8_3_Data_Loader import load_train_data
from tokenizers import Tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', default='config/model_config_small.json', type=str, required=False, help='选择模型参数')
    parser.add_argument('--tokenized_data_path', default='../model/ch6/poetry/bpe-poetry-7000.json', type=str, required=False, help='tokenized语料存放位置')
    parser.add_argument('--epochs', default=50, type=int, required=False, help='训练循环次数')
    parser.add_argument('--batch_size', default=48, type=int, required=False, help='训练batch size')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
    parser.add_argument('--gradient_accumulation', default=2, type=int, required=False, help='梯度积累')
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up步数')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--log_step', default=100, type=int, required=False, help='多少步汇报一次loss，必须是gradient accumulation的整数倍')
    parser.add_argument('--output_dir', default='../model/ch6/poetry/', type=str, required=False, help='模型输出路径')
    parser.add_argument('--writer_dir', default='tensorboard_summary/', type=str, required=False, help='TensorBoard路径')

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device:', device)

    # 这里用的是transformers最新的gpt2包
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    config_filename = os.path.join(current_dir, args.model_config)

    my_tokenizer: Tokenizer = Tokenizer.from_file(args.tokenized_data_path)
    vocab_size = my_tokenizer.get_vocab_size()

    # 加载配置文件
    model_config = transformers.models.gpt2.GPT2Config.from_json_file(config_filename)
    # 词表长度设置
    model_config.vocab_size = vocab_size
    model_config.bos_token_id = my_tokenizer.token_to_id("[BOS]")
    model_config.eos_token_id = my_tokenizer.token_to_id("[EOS]")
    print('config:\n' + model_config.to_json_string())

    epochs = args.epochs
    batch_size = args.batch_size
    gradient_accumulation = args.gradient_accumulation
    lr = args.lr
    max_grad_norm = args.max_grad_norm
    warmup_steps = args.warmup_steps
    log_step = args.log_step
    output_dir = args.output_dir
    tb_writer = SummaryWriter(log_dir=args.writer_dir)
    # 检查一下汇报步数是否是gradient accumulation的整数倍
    assert log_step % gradient_accumulation == 0

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # 如果设置了预训练模型就读取，如果没有就按照模型初始默认设置
    model = transformers.models.gpt2.GPT2LMHeadModel(config=model_config)
    # 设置为训练模式
    model.train()
    model.to(device)

    # 打印参数数量
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    print('number of parameters: {}'.format(num_parameters))

    train_dataloader = load_train_data(batch_size) # 加载数据
    num_training_steps = len(train_dataloader) * epochs # 总迭代数
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

    print('start training')
    overall_step = 0 # 整体的迭代计数器
    running_loss = 0
    for epoch in range(epochs):
        print('epoch {}'.format(epoch + 1))
        now = datetime.now()
        print('time: {}'.format(now))
        internal_step = 0  # 一个 epoch 内部的迭代计数器
        for batch_inputs in train_dataloader:
            batch_inputs = batch_inputs.to(device)
            # 前向传播
            outputs = model.forward(input_ids=batch_inputs, labels=batch_inputs)
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
            # 梯度累积到指定步数
            if (overall_step + 1) % gradient_accumulation == 0:
                optimizer.step()  # 更新参数
                optimizer.zero_grad() # 清空梯度
                scheduler.step() # 调整下一步的学习率
            # 记录 log
            if (overall_step + 1) % log_step == 0:
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


if __name__ == '__main__':
    main()
