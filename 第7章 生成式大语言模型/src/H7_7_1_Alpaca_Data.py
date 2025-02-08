import os
import json

# read all json files one by one, extract the "zh_instruction, zh_input, zh_output"

def read_json_files(folder_path):
    dataset = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding='utf-8') as file:
                datas = json.load(file)
                for data in datas:
                    new_data = {}
                    new_data["instruction"] = data['zh_instruction'] 
                    new_data["input"] = data['zh_input'] 
                    new_data["output"] = data['zh_output'] 
                    dataset.append(new_data)

    return dataset

def write_json_file(dataset, valid):
    train_set = dataset[0:-valid]
    print(f"训练集:{len(train_set)}")
    with open("../data/alpaca-chinese/train.json", "w", encoding='utf-8') as f:
        json.dump(train_set, f, ensure_ascii=False)
    valid_set = dataset[-valid:]
    print(f"训练集:{len(valid_set)}")
    with open("../data/alpaca-chinese/valid.json", "w", encoding='utf-8') as f:
        json.dump(valid_set, f, ensure_ascii=False)

if __name__=="__main__":
    ds = read_json_files("../data/alpaca-chinese-dataset-main/data_v3")
    valid = 500
    write_json_file(ds, valid)

