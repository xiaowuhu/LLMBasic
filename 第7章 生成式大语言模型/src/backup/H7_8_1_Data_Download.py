
from datasets import load_dataset, DatasetDict


def load_raw_data():
    print("load train dataset...")
    ds_train = load_dataset("huggingface-course/codeparrot-ds-train", split="train")
    print("load valid dataset...")
    ds_valid = load_dataset("huggingface-course/codeparrot-ds-valid", split="validation")
    raw_datasets = DatasetDict(
        {
            "train": ds_train,
            "valid": ds_valid
        }
    )

    print(raw_datasets)
    for key in raw_datasets["train"][10]:
        print(f"{key.upper()}: {raw_datasets['train'][0][key][0:2000]}")

    return raw_datasets

if __name__=="__main__":
    # 下载数据，如果已经下载好了，会从缓存里取数据
    # \users\{your_name}\/.cache/huggingface/datasets
    load_raw_data()