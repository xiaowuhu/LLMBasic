import torch
from torch.utils.data import DataLoader
from H9_6_1_Datasets import load_bespoke_data, load_gsm8k_data, load_gsm8k_data_for_test
from H9_6_2_Model import load_model, load_tokenizer
from transformers import(
    AutoTokenizer,
    AutoModelForCausalLM,
)
from H9_6_2_Model import BOT, EOT, BOA, EOA

# Testing Inference with the Trained Model
def test_model_inference(
        sample,
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        device, 
        num_output=4,
        max_len=256):
    
    data, answer = remove_assistant_message(sample)
    
    print("等待模型输出...")
    # # Apply chat template using our tokenizer
    input_text = tokenizer.apply_chat_template(
        data,
        tokenize=False,
        add_generation_prompt=True
    )
    print("用户输入:\n", input_text)
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    # Generate output using our *trained_model*
    for i in range(num_output):
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_len, # Maybe generate a bit longer now
            do_sample=True,
            temperature=1.0,
            #num_return_sequences=4,
            #eos_token_id=tokenizer.eos_token_id
        )
        # Decode the generated tokens back to text
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        print(f"-------模型输出{i}-------")
        print(response[len(input_text):])

def remove_assistant_message(sample):
    data = []
    answer = None
    for i in range(len(sample)):
        if sample[i]['role'] != "assistant":
            data.append(sample[i])
        else:
            answer = get_answer(sample[i]['content'])

    return data, answer


def get_answer(text):
    # 先找到 <|begin_of_answer|> 和 <|end_of_answer|> 的位置
    boa = text.rfind(BOA)    
    eoa = text.rfind(EOA)
    if boa != -1 and eoa != -1:
        answer = text[boa + len(BOA):eoa]
        return answer
    else:
        return None


def load_device(MODEL_DIR, TOKENIZER_DIR):
    # 确定训练设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    # 加载分词器
    tokenizer = load_tokenizer(TOKENIZER_DIR)
    model = load_model(MODEL_DIR, device)  # 缺省为 bfloat16
    return model, tokenizer, device


def original_model_inference():
    print("原始模型预测")
    # 原始模型
    MODEL_DIR = "Qwen/Qwen2.5-0.5B-Instruct"  
    TOKENIZER_DIR = "Qwen/Qwen2.5-0.5B-Instruct"
    inference(MODEL_DIR, TOKENIZER_DIR)

def sft_model_inference():
    # 微调后的模型
    print("微调模型预测")
    MODEL_DIR = "../model/ch9/6/SFT0.5B/model/" 
    TOKENIZER_DIR = "../model/ch9/6/SFT0.5B/tokenizer/"
    inference(MODEL_DIR, TOKENIZER_DIR)

def inference(MODEL_DIR, TOKENIZER_DIR):
    model, tokenizer, device = load_device(MODEL_DIR, TOKENIZER_DIR)
    # 加载数据
    datasets = load_gsm8k_data() # load_bespoke_data()
    for i in range(1):
        sample = datasets["train"][i]['messages']
        #sample = datasets["test"][0]['messages']
        test_model_inference(sample, model, tokenizer, device)


def metric_inference(MODEL_DIR, TOKENIZER_DIR, batch_size=64, num_samples=1):
    # 评估模型
    model, tokenizer, device = load_device(MODEL_DIR, TOKENIZER_DIR)
    # 加载数据
    datasets, ground_truth = load_gsm8k_data_for_test(tokenizer) # load_bespoke_data()
    count_of_correct = 0
    test_dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=False)
    for batch_id, batch_data in enumerate(test_dataloader):
        inputs = tokenizer(batch_data, padding=True, padding_side='left', return_tensors="pt").to(device)
        # Generate output using our *trained_model*
        outputs = model.generate(
            **inputs,
            max_new_tokens=256, # Maybe generate a bit longer now
            do_sample=True,
            num_return_sequences=num_samples,
        )
        for i in range(len(batch_data)):
            for j in range(num_samples):
                # Decode the generated tokens back to text
                response = tokenizer.decode(outputs[i * num_samples + j], skip_special_tokens=False)
                answer = get_answer(response)
                id = batch_id * batch_size + i
                if answer == ground_truth[id]:
                    print(f"第{id}个样本正确")
                    count_of_correct += 1
                    print(count_of_correct)
                    break  # pass@5 we need only one
                else:
                    print(f"第{id}个样本错误,正确答案是{ground_truth[id]}, 模型输出是{answer}")

    print(f"模型在测试集上的准确数为{count_of_correct}")
    print(f"模型在测试集上的准确率为{count_of_correct/len(datasets)}")



if __name__=="__main__":
    MODEL_DIR = "../model/ch9/6/SFT0.5B/model/"
    TOKENIZER_DIR = "../model/ch9/6/SFT0.5B/tokenizer/"
    metric_inference(MODEL_DIR, TOKENIZER_DIR, batch_size=8, num_samples=5)
    #original_model_inference()
    #sft_model_inference()
