import os
from transformers import(
    TrainingArguments, 
    TrainerState,
    TrainerControl,
    TrainerCallback,
)
import logging
from trl import SFTTrainer
import torch

from H9_6_1_Datasets import load_bespoke_data, load_gsm8k_data
from H9_6_2_Model import load_model, load_tokenizer

class LoggingCallback(TrainerCallback):
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % args.logging_steps == 0:
            if len(state.log_history) > 0:
                logger.info(state.log_history[-1])

def get_callbacks(): #training_args, model_args, script_args):
    callbacks = [LoggingCallback()] # Instantiate our LoggingCallback
    return callbacks

def set_log(logger):
    logger.setLevel(logging.DEBUG)  # 设置日志级别
    # 创建一个 file handler，用于将日志写入文件
    file_handler = logging.FileHandler("../model/ch9/6/sft.log")  # 指定日志文件名
    file_handler.setLevel(logging.DEBUG)  # 设置文件处理器的日志级别
    # 创建一个 formatter，定义日志格式
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)  # 将 formatter 添加到 file handler
    # 将 file handler 添加到 logger
    logger.addHandler(file_handler)


if __name__=="__main__":
    # Model and Output Configuration (same as before, or adjust as needed)
    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    OUTPUT_MODEL = "../model/ch9/6/SFT0.5B/model/" # New output directory for SFT model
    OUTPUT_TOKENIZER = "../model/ch9/6/SFT0.5B/tokenizer/" # New output directory for SFT model
    os.makedirs(OUTPUT_MODEL, exist_ok=True)
    os.makedirs(OUTPUT_TOKENIZER, exist_ok=True)
    logger = logging.getLogger(__name__)
    set_log(logger)
    # Training Arguments - similar to GRPO, but adjust for SFT
    training_args = TrainingArguments(
        output_dir=OUTPUT_MODEL,        # 保存模型的目录
        overwrite_output_dir=True,      # 是否覆盖输出目录
        num_train_epochs=5,             # 训练5轮
        per_device_train_batch_size=4,  # 训练时的批大小
        per_device_eval_batch_size=4,   # 验证时的批大小
        learning_rate=2e-5,             # SFT 所需的学习率
        warmup_ratio=0.1,               # 线性预热步数（比例）
        weight_decay=0.01,              # 权重衰减系数
        logging_steps=10,               # 日志记录频率
        eval_strategy="steps",          # 验证计量单位（no|steps|epoch）
        eval_steps=50,                  # 验证频率
        save_strategy="epoch",          # 保存计量单位（no|steps|epoch|best）
        save_total_limit=3,             # 最大保存数量（覆盖式）
        dataloader_num_workers=1,       # 数据读取进程
        seed=42,                        # 随机种子
        bf16=True,                      # 混合精度训练BFP16
        #gradient_checkpointing=True,    # 梯度检查
    )

    # 确定训练设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    # 加载数据
    datasets = load_gsm8k_data() # load_bespoke_data()
    tokenizer = load_tokenizer(MODEL_NAME, add_special_token=True)
    # 加载模型
    model_sft = load_model(MODEL_NAME, device, dtype=torch.bfloat16)  # 缺省为 bfloat16
    # 初始化SFT Trainer
    callbacks = get_callbacks()
    sft_trainer = SFTTrainer(
        model=model_sft,
        train_dataset=datasets['train'],  
        eval_dataset=datasets['test'],
        callbacks=callbacks,
        processing_class=tokenizer, # 如果分词器需要定制，则需要在此指定，否则会使用 model 自带的分词器
        args=training_args,               
    )
    # 开始训练
    sft_train_result = sft_trainer.train()
    # 保存分词器
    tokenizer.save_pretrained(OUTPUT_TOKENIZER)
    # 保存模型
    sft_trainer.save_model(OUTPUT_MODEL)
    print(f"训练结束，结果保存在 {OUTPUT_MODEL} 和 {OUTPUT_TOKENIZER}")
