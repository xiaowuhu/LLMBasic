import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import json
from tqdm import tqdm

def extract_answer(text, count):
    n_str = text[count:]
    clean_str = n_str.replace("<|endoftext|>", "").strip()
    return clean_str


def get_answer(text):
    try:
        # 找到 answer is 后面的文字
        idx = text.find("answer is")
        if idx > 0: # found
            return extract_answer(text, idx + 9)
        idx = text.find("answer:")
        if idx > 0: # found
            return extract_answer(text, idx + 7)
    except:
        return ""

def sample(model, question, tokenizer: GPT2Tokenizer, device, sample_len):
    for _ in range(sample_len):
        with torch.no_grad():
            toks = tokenizer([question], padding=False, return_tensors="pt").to(device)
            orig_len = toks["input_ids"].shape[1]
            out = model.generate(
                **toks, max_length=orig_len + 1, pad_token_id=model.config.eos_token_id
            )
            text = tokenizer.batch_decode(out)[0]
            question = text
            if out[0][-1].item() == tokenizer.eos_token_id:
                break
    return question


def test_all(ds_valid):
    with open("../data/cot_flan/test_result_tmp.json", "w", encoding='utf-8') as f:
        results = []
        total_correct = 0
        for i in tqdm(range(len(ds_valid))):
            result = {}
            data = ds_valid[i]
            new_data = data['instruction']  + "\\n"
            try:
                output = sample(model, new_data, tokenizer, device, 256)
            except:
                continue
            result["answer"] = get_answer(data["output"])
            result["output"] = get_answer(output)
            if result["answer"] == result["output"]:
                result["correct"] = 1
                total_correct += 1
            else:
                result["correct"] = 0
            results.append(result)
            if (i+1)%100==0: json.dump(results, f)
    with open("../data/cot_flan/test_result.json", "w", encoding='utf-8') as f:
        json.dump(results, f)
    print(total_correct, len(ds_valid))

def test_some():
    for i in range(len(ds_valid)):
        print(f"-------{i+1}-------")
        data = ds_valid[i]
        new_data = data['instruction']  + "\\n"
        print("【问题】")
        print(new_data)
        print("【答案】")
        print(data["output"])
        output = sample(model, new_data, tokenizer, device, 256)
        print("【输出】")
        print(output)

if __name__ == "__main__":
    model_path = "gpt2-medium"
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    ds_valid = load_dataset("causal-lm/cot_flan", split="validation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("gpt without sft ---")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    model.to(device)
    test_all(ds_valid)

    print("gpt with sft --- ")
    model = GPT2LMHeadModel.from_pretrained("../model/ch8/cot_flan/model_epoch3")
    model.eval()
    model.to(device)
    test_all(ds_valid)
