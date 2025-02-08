import torch
import torch.nn.functional as F
from tqdm import trange
from transformers import GPT2LMHeadModel, AutoTokenizer
from tokenizers import Tokenizer


def is_letter(word):
    for item in list(word):
        if item not in 'qwertyuiopasdfghjklzxcvbnm':
            return False
    return True

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


def sample_sequence(model, context, max_length, n_ctx, tokenizer: Tokenizer, temperature=1.0, top_k=30, top_p=0.0, repitition_penalty=1.0,
                    device='cpu'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0)
    generated = context
    with torch.no_grad():
        for _ in trange(max_length):
            inputs = {'input_ids': generated[0][-(n_ctx - 1):].unsqueeze(0)}
            # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            outputs = model(**inputs)   # 1x8x9033
            next_token_logits = outputs[0][0, -1, :]  # 取最后一个 token
            existing = generated[0].tolist()
            for id in set(existing): # 存在惩罚
                next_token_logits[id] /= repitition_penalty
            next_token_logits = next_token_logits / temperature  # 温度
            # top-k + top-p
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
            if next_token.item() == tokenizer.token_to_id("[EOS]"):
                break 
        return generated.tolist()[0]

def sample_sequence2(model, context1, context2, max_length, n_ctx, tokenizer: Tokenizer, temperature=1.0, top_k=30, top_p=0.0, repitition_penalty=1.0,
                    device='cpu'):
    context = torch.tensor(context1, dtype=torch.long, device=device)
    context = context.unsqueeze(0)
    generated = context
    with torch.no_grad():
        for _ in trange(max_length):
            inputs = {'input_ids': generated[0][-(n_ctx - 1):].unsqueeze(0)}
            # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            outputs = model(**inputs)   # 1x8x9033
            next_token_logits = outputs[0][0, -1, :]  # 取最后一个 token
            existing = generated[0].tolist()
            for id in set(existing): # 存在惩罚
                next_token_logits[id] /= repitition_penalty
            next_token_logits = next_token_logits / temperature  # 温度
            # top-k + top-p
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
            if next_token.item() == tokenizer.token_to_id("[EOS]"):
                break 
            if next_token.item() == tokenizer.token_to_id("[SEP]"):
                context = torch.tensor(context2, dtype=torch.long, device=device)
                context = context.unsqueeze(0)
                generated = torch.cat((generated, context), dim=1)
        return generated.tolist()[0]

def fast_sample_sequence(model, context, length, temperature=1.0, top_k=30, top_p=0.0, device='cpu'):
    inputs = torch.LongTensor(context).view(1, -1).to(device)
    if len(context) > 1:
        _, past = model(inputs[:, :-1], None)[:2]
        prev = inputs[:, -1].view(1, -1)
    else:
        past = None
        prev = inputs
    generate = [] + context
    with torch.no_grad():
        for i in trange(length):
            output = model(prev, past=past)
            output, past = output[:2]
            output = output[-1].squeeze(0) / temperature
            filtered_logits = top_k_top_p_filtering(output, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(torch.softmax(filtered_logits, dim=-1), num_samples=1)
            generate.append(next_token.item())
            prev = next_token.view(1, 1)
    return generate


# 通过命令行参数--fast_pattern，指定模式
def generate(n_ctx, model, context, length, tokenizer, temperature=1, top_k=0, top_p=0.0, repitition_penalty=1.0, device='cpu'):
    return sample_sequence(model, context, length, n_ctx, tokenizer=tokenizer, temperature=temperature, top_k=top_k, top_p=top_p,
                               repitition_penalty=repitition_penalty, device=device)

def main():
    model_path = "../model/ch7/pycpilot/model_epoch3/"
    max_length = 128
    nsamples = 2
    temperature = 1.2
    top_k = 5
    top_p = 0.75
    repetition_penalty = 1.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(device)
    model.eval()

    raw_text = "class MLP(nn.Module):"
    #raw_text = "def merge_files(folder_name):"
    tokenizer = AutoTokenizer.from_pretrained("../model/ch7/pycpilot/tokenizer")
    input_ids = tokenizer(raw_text, return_tensors="pt").to(device)
    output_ids = model.generate(**input_ids, max_length=128)
    text = tokenizer.batch_decode(output_ids)
    print(text[0])

    # n_ctx = model.config.n_ctx
    # if max_length == -1:
    #     max_length = model.config.n_ctx
    
    

#     raw_text = '''def is_even(value):
#     """Returns True if value is an even number."""
#     return value % 2 == 0

# # setup unit tests for is_even
# import unittest
# '''

    # tokens = tokenizer.encode(raw_text)
    # context_tokens = tokens.ids

    # generated_ids = model.generate(context_tokens)

    # for k in range(nsamples):
    #     out = generate(
    #         n_ctx=n_ctx,
    #         model=model,
    #         context=context_tokens,
    #         length=max_length,
    #         tokenizer=tokenizer,
    #         temperature=temperature, top_k=top_k, top_p=top_p, repitition_penalty=repetition_penalty, device=device
    #     )
    #     text = tokenizer.decode(out, skip_special_tokens=False)
    #     print(text)

    #     # text = str.replace(text, " ", "") # 去掉空格
    #     # text = str.replace(text, "[BOS]", " ")
    #     # text = str.replace(text, "[SEP]", " ")
    #     # text = str.replace(text, "[EOS]", "")
    #     # info = "=" * 40 + " SAMPLE " + str(k) + " " + "=" * 40
    #     # print(info)
    #     # text = ''.join(text).replace('##', '').strip()
    #     # print(text)
    # print("=" * 80)


if __name__ == '__main__':
    main()
