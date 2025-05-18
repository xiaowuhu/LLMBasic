import torch
from transformers import(
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
)

BOT = "<|begin_of_thought|>"
EOT = "<|end_of_thought|>"
BOA = "<|begin_of_answer|>"
EOA = "<|end_of_answer|>"


def load_tokenizer(MODEL_NAME, add_special_token=False):
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
    if add_special_token:
        # 增加特殊 token
        sp_tokens = [BOT, EOT, BOA, EOA]
        # sp_tokens = ['<|im_start|>', '<|im_end|>']
        tokenizer.add_special_tokens({'additional_special_tokens': sp_tokens})
        # Set pad token if not set
        # if tokenizer.pad_token is None:
        #     print("设置 tokenizer.eos_token => tokenizer.pad_token")
        #     tokenizer.pad_token = tokenizer.eos_token
        print(f"特殊 token: {tokenizer.special_tokens_map}")
        print(f"词表大小: {len(tokenizer)}")

    return tokenizer


def load_model(MODEL_NAME, device, dtype=torch.float32):
    print(f"加载模型 {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=dtype
    )
    print(f"模型最大输出长度: {model.config.max_position_embeddings}")
    print(f"模型参数: {model.num_parameters():,}")
    model.to(device)
    return model

if __name__=="__main__":
    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer: AutoTokenizer = load_tokenizer(MODEL_NAME, add_special_token=False)
    model = load_model(MODEL_NAME, device)
    print("嵌入层大小:", model.model.embed_tokens.weight.shape)
    input_sample = f"{BOT}this is thought{EOT}"
    print("不增加特殊 token 的分词结果：")
    print(tokenizer.tokenize(input_sample))
    tokenizer: AutoTokenizer = load_tokenizer(MODEL_NAME, add_special_token=True)
    print("增加特殊 token 的分词结果：")
    print(tokenizer.tokenize(input_sample))
    config = AutoConfig.from_pretrained(MODEL_NAME)
    print(config)
