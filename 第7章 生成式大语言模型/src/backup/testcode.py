# This cell will take a very long time to execute, so you should skip it and go to
# the next one!
from datasets import load_dataset, Dataset
import datasets
from collections import defaultdict
from tqdm import tqdm

def any_keyword_in_string(string, keywords):
    for keyword in keywords:
        if keyword in string:
            return True
    return False

def filter_streaming_dataset(dataset, filters):
    filtered_dict = defaultdict(list)
    total = 0
    for sample in tqdm(iter(dataset)):
        total += 1
        if any_keyword_in_string(sample["content"], filters):
            for k, v in sample.items():
                filtered_dict[k].append(v)
    print(f"{len(filtered_dict['content'])/total:.2%} of data after filtering.")
    return Dataset.from_dict(filtered_dict)

split = "train"  # "valid"
filters = ["torch"]

# data = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
# filtered_data = filter_streaming_dataset(data, filters)
# filtered_data.save_to_disk(f"../data/ch7/codeparrot_torch.json")

ds_train = datasets.load_from_disk("../data/ch7/codeparrot_torch.json")
print(len(ds_train))
for sample in ds_train:
    print(sample["content"].replace("\n", "\r\n"))
    break
