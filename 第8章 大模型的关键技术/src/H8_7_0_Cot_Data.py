from datasets import load_dataset
from tqdm import tqdm
import json

def dump_data(split, file_path):
    ds = load_dataset("causal-lm/cot_flan", split=split)
    ds_list = []
    for data in tqdm(ds):
        new_data = {}
        new_data["instruction"] = data["instruction"]
        new_data["input"] = data["input"]
        new_data["output"] = data["output"]
        ds_list.append(new_data)
    
    with open(file_path, "w", encoding='utf-8') as file:
        json.dump(ds_list, file)

if __name__ == "__main__":
    dump_data("train", "../data/cot/train.json")
    dump_data("validation", "../data/cot/valid.json")
    