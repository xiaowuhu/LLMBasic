
# https://huggingface.co/docs/trl/main/en/grpo_trainer

import logging
from trl import GRPOConfig, GRPOTrainer
import os
from transformers import(
    AutoTokenizer,
    TrainingArguments,
    TrainerState,
    TrainerControl,
    TrainerCallback,
)
import torch
from transformers import(
    AutoTokenizer,
    AutoModelForCausalLM,
)

from H9_7_1_Datasets import load_gsm8k_data
from H9_7_2_Rewards import reward_functions

def load_tokenizer(MODEL_NAME):
    print(f"加载分词器:{MODEL_NAME}...")
    # Initialize tokenizer with chat template
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right"
    )
    print(f"词表大小: {len(tokenizer)}")
    print(f"最大输入长度: {tokenizer.model_max_length}")
    print(f"特殊 token: {tokenizer.special_tokens_map}")
    return tokenizer


def load_model(MODEL_NAME, device, dtype=torch.bfloat16):
    print(f"加载模型 {MODEL_NAME}...")
    # Initialize base model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=dtype
    )
    print(f"最大输出长度: {model.config.max_position_embeddings}")
    print(f"模型参数量: {model.num_parameters():,}")
    # Check CUDA availability
    print(f"Using device: {device}")
    # Move model to the appropriate device
    model.to(device)
    return model

class LoggingCallback(TrainerCallback):
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % args.logging_steps == 0:
            if len(state.log_history) > 0:
                logger.info(state.log_history[-1])
                #logger.info(f"Step {state.global_step}: Loss = {state.log_history[-1].get('loss', None)}, Learning Rate = {state.log_history[-1].get('learning_rate', None)}")

        # if state.global_step % args.save_steps == 0:
        #     tokenizer.save_pretrained(TEMP_PATH)
        #     grpo_trainer.save_model(TEMP_PATH)
        #     print("model saved")

def get_callbacks(): #training_args, model_args, script_args):
    """
    Returns a list of callbacks to be used during training.
    For now, it includes only the LoggingCallback. You can extend this to add more callbacks.
    """
    callbacks = [LoggingCallback()] # Instantiate our LoggingCallback
    return callbacks


def set_log(logger):
    logger.setLevel(logging.DEBUG)  # 设置日志级别
    # 创建一个 file handler，用于将日志写入文件
    file_handler = logging.FileHandler("../model/ch9/7/RFT.log")  # 指定日志文件名
    file_handler.setLevel(logging.DEBUG)  # 设置文件处理器的日志级别
    # 创建一个 formatter，定义日志格式
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)  # 将 formatter 添加到 file handler
    # 将 file handler 添加到 logger
    logger.addHandler(file_handler)
   

if __name__=="__main__":
    # 使用冷启动微调的结果
    MODEL_NAME = "../model/ch9/6/SFT0.5B/model/"
    TOKENIZER_DIR = "../model/ch9/6/SFT0.5B/tokenizer/"
    OUTPUT_DIR = "../model/ch9/7/RFT/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger = logging.getLogger(__name__)
    set_log(logger)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets = load_gsm8k_data()
    tokenizer = load_tokenizer(MODEL_NAME)
    model = load_model(MODEL_NAME, device)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,          # 训练模型参数保存目录
        overwrite_output_dir=True,      # 覆盖输出目录
        num_train_epochs=3,             # 训练轮数
        per_device_train_batch_size=12, # 训练时的批大小
        per_device_eval_batch_size=4,   # 验证时的批大小
        learning_rate=1e-5,             # 初始学习率 5e-5
        warmup_ratio=0.1,               # 线性预热步数（比例）
        weight_decay=0.01,              # 权重衰减系数
        logging_steps=2,                # 日志记录频率
        eval_steps=50,                  # 验证频率
        save_strategy="steps",          # 保存计量单位（no|steps|epoch|best）
        save_steps=200,                 # 保存频率  
        save_total_limit=5,             # 最大保存数量（覆盖式）
        dataloader_num_workers=1,       # 数据读取进程
        seed=42,                        # 随机种子
        bf16=True,                      # 混合精度训练BFP16
        gradient_checkpointing=True,    # 梯度检查
    )

    # Create GRPOConfig from TrainingArguments
    grpo_config = GRPOConfig(
        num_generations=4,
        max_completion_length=256,
        #use_vllm=True,
        #vllm_mode="colocate",
        **training_args.to_dict(), # Convert TrainingArguments to dictionary and unpack
        **{ 
        # REMOVED model_init_kwargs here 
        # We are passing the instantiated 'model' object, so GRPOTrainer doesn't need model_init_kwargs
        }
    )

    grpo_trainer = GRPOTrainer(
        model=model,                    # 指定模型
        reward_funcs=reward_functions,  # 指定奖励函数列表
        args=grpo_config,               # GRPOConfig (TrainingArguments)
        train_dataset=datasets['train'],
        eval_dataset=datasets['test'],
        callbacks=get_callbacks(),            # 指定回调函数
        processing_class=tokenizer,     # 指定分词器以便使用其内置的模板
    )

    # Start the GRPO Training Loop
    train_result = grpo_trainer.train()

    # Save the tokenizer
    tokenizer.save_pretrained(OUTPUT_DIR)
    # Save the trained model
    grpo_trainer.save_model(OUTPUT_DIR)
    print(f"GRPO Trained model saved to {OUTPUT_DIR}")    
