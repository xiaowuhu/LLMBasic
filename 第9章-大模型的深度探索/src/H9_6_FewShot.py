from transformers import(
    AutoTokenizer,
    AutoModelForCausalLM,
)
import torch

from H9_6_ChatTemplate import default_chat_template

# Generate Long COT Response
def generate_response(tokenizer, model, prompt_text):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that provides step-by-step solutions."},
        {"role": "user", "content": prompt_text}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("---- tokenize ----")
    print(text)
    print("-----------")
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True) # Keep it deterministic for example
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return response.split("<|im_start|>assistant")[-1].strip() # Extract assistant's response

def test_base_model_with_few_shot(tokenizer, model):
    # Example problems with solutions (using | special_token | as delimiter)
    few_shot_prompt = """
    Problem: What's the square root of 9 plus 5?
    Solution: <|special_token|> First, find the square root of 9, which is 3. Then, add 5 to 3.  3 + 5 equals 8. <|special_token|> Summary: The answer is 8.

    Problem: Train travels at 60 mph for 2 hours, how far?
    Solution: <|special_token|> Use the formula: Distance = Speed times Time. Speed is 60 mph, Time is 2 hours. Distance = 60 * 2 = 120 miles. <|special_token|> Summary: Train travels 120 miles.
    
    """

    # Generate response for the target problem using few-shot examples
    target_problem_prompt = few_shot_prompt + "Problem: What is 2 + 3 * 4?"
    model_response_few_shot = generate_response(tokenizer, model, target_problem_prompt)

    print("------ Few-Shot CoT -----")
    print(target_problem_prompt)
    print("\n<<<Model Response>>>")
    print(model_response_few_shot)

def direct_prompt(tokenizer, model):
    # Direct prompting example
    direct_prompt_text = """
    Problem: Solve this, show reasoning step-by-step, and verify:
    What is 2 + 3 * 4?
    """

    model_response_direct = generate_response(tokenizer, model, direct_prompt_text)

    print("--------Direct Prompt--------")
    print(direct_prompt_text)
    print("\n<<<Model Response>>>")
    print(model_response_direct)    

if __name__=="__main__":
    # Loading Model and Tokenizer
    #MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    MODEL_NAME = "gpt2-medium"
    #MODEL_NAME = "data/SFT-training"  # gpt2-medium SFT
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side="right")
    print(tokenizer.special_tokens_map)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = model.to(device)
    if tokenizer.chat_template is None:
        tokenizer.chat_template = default_chat_template(tokenizer)['default']
    
    test_base_model_with_few_shot(tokenizer, model)
    direct_prompt(tokenizer, model)
