import torch
from H9_5_TLDR_Train import load_data, load_tokenizer, load_model
 
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
        max_new_tokens=1024, # Maybe generate a bit longer now
        do_sample=True,
        temperature=1,
        #num_return_sequences=4, # 可以产生多个输出
    )
    # Decode the generated tokens back to text
    for i, output in enumerate(outputs):
        response = tokenizer.decode(output, skip_special_tokens=False)
        print(response[len(input_text):])


def test_original_model(device, datasets, n):
    MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
    tokenizer = load_tokenizer(MODEL_NAME)
    model = load_model(MODEL_NAME, device)
    for i in range(n):
        print(f"-----{i}-----")
        data = datasets[i]
        test_model_inference(tokenizer, model, data, device)

def test_tldr_model(device, datasets, n):
    MODEL_NAME = "../model/ch9/5/"
    tokenizer = load_tokenizer(MODEL_NAME)
    model = load_model(MODEL_NAME, device)
    for i in range(n):
        print(f"-----{i}-----")
        data = datasets[i]
        test_model_inference(tokenizer, model, data, device)
    

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = load_data()
    print("原始模型的TLDR")
    test_original_model(device, datasets, 3)
    print("使用RL训练后的模型的TLDR")
    test_tldr_model(device, datasets, 3)
