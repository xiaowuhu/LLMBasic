from transformers import AutoTokenizer
from H7_8_1_Data_Download import load_raw_data


def test_tokenizer(tokenizer):
    outputs = tokenizer(
        raw_datasets["train"][0:3]["content"], # 取三个样本的 content 字段
        truncation=True, # 允许截断
        max_length=context_length,  # 截断长度
        return_overflowing_tokens=True,  # 返回全部分块
        return_length=True, # 返回分块长度
    )

    #print(raw_datasets["train"][0:3]["content"])
    print(f"Input IDs length: {len(outputs['input_ids'])}")
    print(f"Input chunk lengths: {(outputs['length'])}")
    print(f"Chunk mapping: {outputs['overflow_to_sample_mapping']}")


def tokenize_function(element):
    outputs = my_tokenizer(
        element["content"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        #if length == context_length:  # 过滤掉较短的序列
        input_batch.append(input_ids)
            
    return {"input_ids": input_batch}


if __name__=="__main__":
    raw_datasets = load_raw_data()
    print("load tokenizer...")
    context_length = 256  # 设置序列最大长度
    my_tokenizer = AutoTokenizer.from_pretrained("../model/ch7/pycpilot/tokenizer/")
    test_tokenizer(my_tokenizer)

    # 转换的结果将会保存到磁盘
    print("raw datasets mapping...")
    tokenized_datasets = raw_datasets.map(
        tokenize_function, 
        batched=True, 
        remove_columns=raw_datasets["train"].column_names
    )
    tokenized_datasets.set_format("torch") # 设置为 torch tensor 格式
    print(tokenized_datasets)
    tokenized_datasets.save_to_disk("../data/ch7/pycpilot/tokenized_datasets")
