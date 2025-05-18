import torch
from H9_7_3_RFT_Train import load_tokenizer, load_model
from H9_7_1_Datasets import load_gsm8k_data
from H9_7_1_Datasets import hash2answer
from H9_6_4_SFT_Test import metric_inference

def test_model_inference(tokenizer, model, input, device):
    input_text = tokenizer.apply_chat_template(
        input["prompt"],
        tokenize=False,
        add_generation_prompt=True
    )    
    print("用户输入:\n", input_text)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    # Generate output using our *trained_model*
    outputs = model.generate(
        **inputs,
        max_new_tokens=256, # Maybe generate a bit longer now
        do_sample=True,
        temperature=1,
        #num_return_sequences=4, # 可以产生多个输出
    )
    # Decode the generated tokens back to text
    for i, output in enumerate(outputs):
        response = tokenizer.decode(output, skip_special_tokens=False)
        print(response[len(input_text):])


def test_rft_model(device, datasets, n):
    MODEL_NAME = "../model/ch9/7/RFT/checkpoint-1800/"
    #MODEL_NAME = "../model/ch9/6/SFT0.5B/model/"
    TOKENIZER_DIR = "../model/ch9/6/SFT0.5B/tokenizer/"

    #MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    #TOKENIZER_DIR = "Qwen/Qwen2.5-Math-1.5B-Instruct"

    tokenizer = load_tokenizer(TOKENIZER_DIR)
    model = load_model(MODEL_NAME, device)
    for i in range(n):
        print(f"-----{i}-----")
        data = datasets['train'][i]
        p = data['prompt'][1]['content']
        v = hash2answer[hash(p)]
        print(f"Correct answer is:{v}----")
        test_model_inference(tokenizer, model, data, device)
 
if __name__=="__main__":
    MODEL_DIR = "../model/ch9/7/RFT/"
    TOKENIZER_DIR = "../model/ch9/6/SFT0.5B/tokenizer/"
    metric_inference(MODEL_DIR, TOKENIZER_DIR, batch_size=64)    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # datasets = load_gsm8k_data()
    # # print("原始模型的TLDR")
    # # test_original_model(device, datasets, 3)
    # print("使用RL训练后的模型")
    # test_rft_model(device, datasets, 4)
