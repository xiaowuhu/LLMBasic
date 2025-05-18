
# https://huggingface.co/docs/trl/main/en/grpo_trainer

import logging
from datasets import load_dataset
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


def load_tokenizer(MODEL_NAME):
    print(f"加载分词器:{MODEL_NAME}...")
    # Initialize tokenizer with chat template
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right"
    )
    print(f"词表大小: {len(tokenizer)}")
    print(f"最大输出长度: {tokenizer.model_max_length}")
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
    print(f"Model parameters: {model.num_parameters():,}")
    # Check CUDA availability
    print(f"Using device: {device}")
    # Move model to the appropriate device
    model.to(device)
    return model

class LoggingCallback(TrainerCallback):
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % args.logging_steps == 0:
            if len(state.log_history) > 0:
                logger.info(f"Step {state.global_step}: Loss = {state.log_history[-1].get('loss', None)}, Learning Rate = {state.log_history[-1].get('learning_rate', None)}")

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


# def count_english_letters(input_string):
#     count = 0
#     for ch in input_string:
#         if 'a' <= ch <= 'z' or 'A' <= ch <= 'Z':
#             count += 1
#     return count

# 用中文回答，有一个中文逗号(，)和一个中文句号(。)，总长度40个字。对字的内容没有限制。
# def reward_func(completions, **kwargs):
#     rewards = []
#     for completion in completions:
#         c = completion[0]['content']
#         # 长度40
#         reward_len = -abs(40 - len(c)) / 10
#         # 一个逗号
#         reward_comma = -abs(c.count('，') - 1) 
#         # 一个句号
#         reward_period = -abs(c.count('。') - 1) 
#         # 如果有英文字母则分数-1
#         reward_language = -count_english_letters(c)
#         reward = reward_comma + reward_language + reward_period + reward_len
#         rewards.append(reward)
#     return rewards

def count_non_english_letters(input_string):
    count = 0
    for ch in input_string:
        if ch >= chr(255):  # 不是 ASCII 字符
            count += 1
    return count

def count_of_completions_in_prompts(prompt, completion):
    count = 0
    p_words = set(prompt.lower().split())
    for c_word in completion.split():
        if c_word not in p_words:
            count += 1
    return count

def count_word_length(completion):
    words = completion.split()
    return len(words)

def zipngram(text: str, ngram_size: int):
    """Helper function to generate n-grams from text."""
    words = text.lower().split() # Lowercase and split into words
    return zip(*[words[i:] for i in range(ngram_size)]) # Create n-grams

def repetition_penalty_reward(completion):
    ngram_size = 3
    max_penalty = 1.0 # Maximum penalty for repetition

    ngrams = set() # Use a set to store unique n-grams
    total = 0
    for ng in zipngram(completion, ngram_size): # Generate n-grams
        ngrams.add(ng) # Add n-gram to the set (duplicates are ignored)
        total += 1 # Count total n-grams

    # Calculate scaling factor: more repetition -> higher scaling
    scaling = 1 - len(ngrams) / total
    reward = scaling * max_penalty # Apply penalty based on scaling
    return reward

def count_punctuation(completion):
    # 一个逗号的奖励为 0
    reward_comma = abs(completion.count(',') - 1) 
    # 一个句号的奖励为 0
    reward_period = abs(completion.count('.') - 1) 
    return reward_comma + reward_period


# 用英文回答，一个句号(.)一个逗号(,)，总长度30个单词且必须在样本中包含（以防止产生非摘要信息）。
def reward_func(prompts, completions, **reward_kwargs):
    rewards = []
    assert(len(completions) == len(prompts))
    for i in range(len(completions)):
        c = completions[i][0]['content']
        p = prompts[i][1]['content']
        # 长度 25
        reward_len = -abs(25 - count_word_length(c)) / 10 # 如果50个单词将会得到 -2.5
        # 标点符号
        reward_punctuation = -count_punctuation(c)
        # 如果有英文字母则分数-1
        reward_language = -count_non_english_letters(c)
        # c 是否是 prompts 的摘要
        reward_summary = -count_of_completions_in_prompts(p, c) / 10
        reward_repetition = -repetition_penalty_reward(c) * 2  # 重复加倍惩罚
        reward = reward_punctuation \
                 + reward_language + reward_len \
                 + reward_summary + reward_repetition
        rewards.append(reward)
    return rewards


SYSTEM_PROMPT = (
    # "把下面的文字简化成20字以内,并把结果放在<result></result>之间。"
    # "把下面的文字简化成20字左右，使用中文输出，并把结果放在两个方括号之间。"
    #"把下面的文字简化成两句话，一共40字左右，用逗号隔开，用句号结尾，并使用中文输出。"
    "Simplify the following text into two sentences, totaling about 25 words, separated by commas, ending with a period, and output in English."
)

def make_conversation(example):
    """Convert dataset examples into conversation format."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["prompt"]},
        ],
    }

def load_data():
    print("加载数据并模板化")
    dataset = load_dataset("trl-lib/tldr", split="validation")
    print(dataset)
    print(dataset[1])
    dataset = dataset.map(make_conversation, load_from_cache_file=False)
    datasets = dataset.remove_columns(["prompt", "completion"])
    datasets = datasets.rename_column("messages", "prompt")
    return datasets

def set_log(logger):
    logger.setLevel(logging.DEBUG)  # 设置日志级别
    # 创建一个 file handler，用于将日志写入文件
    file_handler = logging.FileHandler("../model/ch9/5/tldr.log")  # 指定日志文件名
    file_handler.setLevel(logging.DEBUG)  # 设置文件处理器的日志级别
    # 创建一个 formatter，定义日志格式
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)  # 将 formatter 添加到 file handler
    # 将 file handler 添加到 logger
    logger.addHandler(file_handler)
   

if __name__=="__main__":
    MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
    OUTPUT_DIR = "../model/ch9/5/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger = logging.getLogger(__name__)
    set_log(logger)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets = load_data()
    tokenizer = load_tokenizer(MODEL_NAME)
    model = load_model(MODEL_NAME, device)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,          # 训练模型参数保存目录
        overwrite_output_dir=True,      # 覆盖输出目录
        num_train_epochs=2,             # 训练轮数
        per_device_train_batch_size=16, # 训练时的批大小
        #per_device_eval_batch_size=8,  # 验证时的批大小
        gradient_accumulation_steps=1,  # 梯度累积
        learning_rate=5e-5,             # 初始学习率
        warmup_ratio=0.1,               # 线性预热步数（比例）
        weight_decay=0.01,              # 权重衰减系数
        logging_steps=5,                # 日志记录频率
        # eval_strategy="steps",        # 验证计量单位（no|steps|epoch）
        # eval_steps=1000,              # 验证频率
        save_strategy="steps",          # 保存计量单位（no|steps|epoch|best）
        save_steps=200,                 # 保存频率
        save_total_limit=5,             # 最大保存数量（覆盖式）
        dataloader_num_workers=1,       # 数据读取进程
        seed=42,                        # 随机种子
        bf16=True,                      # 混合精度训练BFP16
        push_to_hub=False,              # 模型上传
        gradient_checkpointing=True,    # 梯度检查
        report_to="none",               # 日志上传
    )

    # Create GRPOConfig from TrainingArguments
    grpo_config = GRPOConfig(
        num_generations=4,
        use_vllm=True,
        vllm_mode="colocate",
        **training_args.to_dict(), # Convert TrainingArguments to dictionary and unpack
        **{ 
        # REMOVED model_init_kwargs here 
        # We are passing the instantiated 'model' object, so GRPOTrainer doesn't need model_init_kwargs
        }
    )

    grpo_trainer = GRPOTrainer(
        model=model,                    # 指定模型
        reward_funcs=reward_func,  # 指定奖励函数列表
        args=grpo_config,               # GRPOConfig (TrainingArguments)
        train_dataset=datasets,
        #eval_dataset=datasets['test'],
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
