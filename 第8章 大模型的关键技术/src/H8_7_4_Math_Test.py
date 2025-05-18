import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from H8_7_3_Math_Train import load_data
from tqdm import tqdm
import json


def use_calculator(sample):
    if "<<" not in sample:
        return None

    parts = sample.split("<<")
    remaining = parts[-1]
    if ">>" in remaining:
        return None
    if "=" not in remaining:
        return None
    lhs = remaining.split("=")[0]
    lhs = lhs.replace(",", "")
    if any([x not in "0123456789*+-/.()" for x in lhs]):
        return None
    try:
        result = eval(lhs)
    except:
        result = 0
    finally:
        return result

def sample(model, question, tokenizer, device, sample_len):
    EQUALS_TOKENS = set([28, 796, 47505])

    for _ in range(sample_len):
        with torch.no_grad():
            toks = tokenizer([question], padding=False, return_tensors="pt").to(device)
            orig_len = toks["input_ids"].shape[1]
            out = model.generate(
                **toks, max_length=orig_len + 1, pad_token_id=model.config.eos_token_id
            )
            text = tokenizer.batch_decode(out)[0]
            if out[0, -1].item() in EQUALS_TOKENS:
                answer = use_calculator(text)
                if answer is not None:
                    print("Triggered calculator, answer", answer)
                    text = text + str(answer) + ">>"

            question = text
            if out[0][-1].item() == tokenizer.eos_token_id:
                break
    return question

def get_answer(text):
    try:
        # 找到 #### 空格后面的数字
        idx = text.index("####")
        if idx > 0: # found
            n_str = text[idx+4:]
            n = int(n_str.replace("<|endoftext|>", ""))
            return n
    except:
        return 0

def test_all(model, tokenizer, device):
    total_correct = 0
    with open("../data/GSM8K_math/test_result.json", "w", encoding='utf-8') as file:
        results = []
        ds = load_data("test")
        print(len(ds))
        i = 0
        for data in tqdm(ds):
            result = {}
            result["answer"] = get_answer(data['answer'])
            new_data = data['question'] + "\\n"
            output = sample(model, new_data, tokenizer, device, 1024)
            #print(output)
            result["output"] = get_answer(output)
            results.append(result)
            if result["answer"] == result["output"]:
                result["correct"] = 1
                total_correct += 1
            else:
                result["correct"] = 0
            i += 1
            if (i+1)%10 == 0:
                print(f"{total_correct} / {i}")
        json.dump(results, file)
    print(total_correct, len(ds))

def test_some(testing_list, model, tokenizer, device):
#    ds = load_data("test")
    ds = load_data("train")
    for idx in testing_list:    
        print(f"---- {idx} ----")
        data = ds[idx-1]
        new_data = data['question'] + "\\n"
        output = sample(model, new_data, tokenizer, device, 1024)
        print("【答案】", data["answer"])
        print("【输出】", output)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "gpt2-large"
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    #model = GPT2LMHeadModel.from_pretrained("../model/ch8/math/model_epoch5")
    # model_path = "Qwen/Qwen2.5-0.5B-Instruct"
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model.to(device)
    model.eval()

    #test_all(model, tokenizer, device)
    test_some([1,2,4,5,24,30], model, tokenizer, device)
