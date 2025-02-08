import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

sentences = [
    "很好的地理位置，一蹋糊涂的服务，萧条的酒店",
    "选择滨海花园的原因就是方便，有电动扶梯直接到达海边，周围餐馆、食廊、商场、超市、摊位一应俱全。酒店装修一般，但还算整洁。泳池在大堂的屋顶，女儿倒是喜欢。",
    "没有比这更差的酒店了。房间灯光暗淡，空调无法调节，前台服务僵化。建议大家不要住这家酒店，有被骗的感觉"
]

prompts = ["感觉很[MASK]。", "总体上来说很[MASK]。"]

def get_prompt(P, X):
    Z = P + X
    return Z

def MLM_result(tokenizer, P, X):
    Z = get_prompt(P, X)
    inputs = tokenizer(Z, return_tensors="pt")
    token_logits = model(**inputs).logits
    # 找到 [MASK] 的位置
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    # 获得 [MASK] 的位置的输出分数
    mask_token_logits = token_logits[0, mask_token_index, :]
    top_1_tokens = torch.topk(mask_token_logits, 1, dim=1).indices[0].tolist()
    for token in top_1_tokens:
        print(f"'>>> {Z.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")

if __name__=="__main__":
    pre_trained_model = "bert-base-chinese"
    model = AutoModelForMaskedLM.from_pretrained(pre_trained_model)
    tokenizer = AutoTokenizer.from_pretrained(pre_trained_model)
    for sentence in sentences:
        for prompt in prompts:
            MLM_result(tokenizer, prompt, sentence)
        print("----------")
