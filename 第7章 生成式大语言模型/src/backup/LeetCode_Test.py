import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, BertTokenizer

from LeetCode_Data import read_jsonl_file, format_input

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def sample_sequence(input_ids, model, tokenizer, max_length, device):
    repitition_penalty = 1.0
    temperature = 1
    top_k = 10
    top_p = 0.8
    n_ctx = model.config.n_ctx
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        for _ in range(max_length):
            inputs = {
                'input_ids': input_ids[0][-(n_ctx - 1):].unsqueeze(0),
            }
            # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            outputs = model(**inputs) 
            next_token_logits = outputs[0][0, -1, :]  # 取最后一个 token
            existing = input_ids[0].tolist()
            for id in set(existing): # 存在惩罚
                next_token_logits[id] /= repitition_penalty
            next_token_logits = next_token_logits / temperature  # 温度
            # top-k + top-p
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)
            if next_token.item() == tokenizer.sep_token_id:
                break 
        return input_ids.tolist()[0]

if __name__=="__main__":
    print("read data file...")
    file_path = "../data/leetcode/valid.jsonl"
    dataset = read_jsonl_file(file_path)
    print(f"数据集长度:{len(dataset)}")
    # str = format_input(data[10])
    # print(str)

    print("load model and tokenizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    vocab_path = "../model/ch7/code/new_vocab"
    model_path = "../model/ch7/code/model_epoch1/"
    #model_path = "uer/gpt2-distil-chinese-cluecorpussmall"
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained(vocab_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    max_length = model.config.n_ctx
    model = model.to(device)    
    model.eval()

    num_workers = 0
    batch_size = 1
    torch.manual_seed(123)

    entry = dataset[2]
    input_text = format_input(entry)
    input_ids = tokenizer.encode("[CLS]" + input_text, add_special_tokens=False)
    response = sample_sequence(input_ids, model, tokenizer, max_length, device)
    l = len(response) - len(input_ids)
    response = response[-l:]
    text = tokenizer.decode(response, skip_special_tokens=False)
    a = text.replace("\ n", "\r\n")
    print(a.replace("\\n", "\r\n"))
