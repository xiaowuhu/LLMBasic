import torch
from transformers import GPT2LMHeadModel, BertTokenizer, TextGenerationPipeline
from H7_7_4_Alpaca_Test import sample_sequence, load_data

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("load tokenizer...")
    model_path = "uer/gpt2-distil-chinese-cluecorpussmall"
#    model_path = "uer/gpt2-chinese-cluecorpussmall"
#    model_path = "uer/gpt2-medium-chinese-cluecorpussmall"
    # model_path = "uer/gpt2-large-chinese-cluecorpussmall"
    # model_path = "uer/gpt2-xlarge-chinese-cluecorpussmall"
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained(model_path)
    print("特殊字符:", tokenizer.special_tokens_map)
    print("词表大小:", len(tokenizer))
    model = GPT2LMHeadModel.from_pretrained(model_path)
    print(f"最大长度:{model.config.n_ctx}")
    model_size = sum(t.numel() for t in model.parameters())
    # 计算并显示模型参数
    print(f"参数量: {model_size/1000**2:.1f}M parameters")
    model.to(device)
    model.eval()
    text_generator = TextGenerationPipeline(model, tokenizer)  # 文本生成
    text = text_generator("这件事是这样的：", max_length=100, do_sample=True)
    print(text.replace(" ", ""))

    ds_valid = load_data("../data/alpaca-chinese", "train")
    for i in range(0,10):
        data = ds_valid[i]
        print(f"-------{i+1} ---")
        new_data = data['instruction'] + data["input"]
        print(new_data.replace("\\n", "\r\n"))
        input_ids = tokenizer.encode(new_data, add_special_tokens=False)
        response = sample_sequence(input_ids, model, tokenizer, 128, device)
        l = len(response) - len(input_ids)
        output = response[-l:]
        text = tokenizer.decode(output, skip_special_tokens=False)
        print(text.replace(" ", ""))
