import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, BertTokenizer, TextGenerationPipeline


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
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

def sample_sequence(chat_history, model, max_length, device):
    repitition_penalty = 1.0
    temperature = 1
    top_k = 10
    top_p = 0.8
    n_ctx = model.config.n_ctx
    input_ids = torch.tensor(chat_history, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        for _ in range(max_length):
            inputs = {
                'input_ids': input_ids[0][-(n_ctx - 1):].unsqueeze(0),
            }
            # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            outputs = model(**inputs)   # 1x8x9033
            next_token_logits = outputs[0][0, -1, :]  # 取最后一个 token
            existing = input_ids[0].tolist()
            for id in set(existing): # 存在惩罚
                next_token_logits[id] /= repitition_penalty
            next_token_logits = next_token_logits / temperature  # 温度
            # top-k + top-p
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)
            if next_token.item() == sep:
                break 
        return input_ids.tolist()[0]

# show only the latest response
def show_diologue(message, tokenizer):
    text = tokenizer.decode(message, skip_special_tokens=False)
    print("bot:", text)


def get_last_3_round_chat(list_chat, bos):
    chat_history = []
    for i in range(3, 0, -1):
        if i <= len(list_chat):
            chat_history += list_chat[-i]
    return [bos] + chat_history


def main():
    # 用双重列表保存对话
    list_chat = []
   
    while(True):
        message = input("input:")
        tokens = tokenizer.encode(message, add_special_tokens=False)
        user = tokens + [sep]
        list_chat.append(user)
        chat_history = get_last_3_round_chat(list_chat, cls)
        response = sample_sequence(chat_history, model, 32, device)
        l = len(response) - len(chat_history)
        response = response[-l:]
        show_diologue(response, tokenizer)
        list_chat.append(response)
        

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained("uer/gpt2-distil-chinese-cluecorpussmall")
    model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained("uer/gpt2-distil-chinese-cluecorpussmall")
    print(f"参数数量:{model.num_parameters() / 1_000_000} M")
    print("特殊token:", tokenizer.cls_token, tokenizer.sep_token)
    model.to(device)
    model.eval()
    sep = tokenizer.sep_token_id
    cls = tokenizer.cls_token_id
    text_generator = TextGenerationPipeline(model, tokenizer)
    text = text_generator("这件事是这样的：", max_length=100, do_sample=True)
    print(text)
    main()
