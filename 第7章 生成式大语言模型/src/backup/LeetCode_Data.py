'''
读取 jsonl 文件，把 instruction/input/output 变成以下形式:

### instruction
xxx

### input
xxx

### output
xxx
'''
import torch
from torch.utils.data import Dataset
import json
from tqdm import tqdm


def read_jsonl_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        datas = []
        for line in lines:
            data = json.loads(line)
            datas.append(data)
    return datas


def format_input(entry):
    instruction_text = (
        #f"Below is an instruction that describes a task. "
        #f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in tqdm(data):
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )
    
    def __getitem__(self, index):
        return self.encoded_texts[index]
    
    def __len__(self):
        return len(self.data)


def custom_collate_fn(
    batch,
    pad_token_id = 0,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    # Find the longest sequence in the batch
    batch_max_length = max(len(item) for item in batch)

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # Pad sequences to max_length
        padded = (new_item + [pad_token_id] * (batch_max_length - len(new_item)))
        # inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        # targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets
        inputs = torch.tensor(padded)
        targets = torch.tensor(padded)

        # New: Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices] = ignore_index

        # New: Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs and targets to tensors and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

