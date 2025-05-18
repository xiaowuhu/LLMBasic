from datasets import load_dataset, Dataset
import re

from H9_6_2_Model import BOT, EOT, BOA, EOA

SYSTEM_PROMPT = (
    "The assistant first thinks about the reasoning process for the user's question, "
    f"and put the reasoning process within {BOT} and {EOT} tags."
    f"Then put the answer within {BOA} and {EOA} tags."
)

def load_bespoke_data(num_train=None):
    
    def make_conversation(example):
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example['conversations'][0]["value"]},
                {"role": "assistant", "content": example['conversations'][1]["value"]}, # 监督学习所必须的，但强化学习不需要
            ],
        }    

    print("加载数据 bespokelabs/Bespoke-Stratos-17k ...")
    dataset: Dataset = load_dataset("bespokelabs/Bespoke-Stratos-17k") 
    print("原始数据样本------")
    print(dataset['train'][0])
    print("过滤数据......") 
    # 只保留数学运算部分的样本，忽略后面的生成python代码的训练样本
    indices_to_keep = []
    for i, example in enumerate(dataset['train']):
        if example['conversations'][0]['value'].startswith("Return your final"):
            indices_to_keep.append(i)
    dataset = dataset['train'].select(indices_to_keep)
    print("数据模板化......")
    dataset = dataset.map(make_conversation, load_from_cache_file=True)
    dataset = dataset.remove_columns(["system", "conversations"])
    print(dataset)
    print(dataset.column_names)
    print("模板化后的样本------")
    print(dataset[0])
    print("分割数据...")
    num_sample = len(dataset)
    num_test = 100
    if num_train is None:
        num_train = num_sample - num_test
    datasets = dataset.train_test_split(num_test, num_train, shuffle=False)
    return datasets

def load_gsm8k_data():

    def make_conversation(example):
        user_questoin = example['question']
        assistant_answer = example['answer']
        split_token = "\n####"
        if split_token in assistant_answer:
            before, after = assistant_answer.split(split_token, 1)
            # 给原始数据中添加特殊标记
            assistant_answer = BOT + before + EOT + BOA + after.strip() + EOA
        else:
            # If no split token, wrap the whole string as thought
            assistant_answer = BOA + assistant_answer + EOA
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_questoin},
                {"role": "assistant", "content": assistant_answer}, # 监督学习所必须的，但强化学习不需要
            ],
        }    

    datasets = load_dataset("openai/gsm8k", "socratic", download_mode='reuse_dataset_if_exists') # main, socratic
    print(datasets)
    print(datasets['train'][0])
    print("数据模板化......")
    datasets = datasets.map(make_conversation)
    datasets = datasets.remove_columns(["question", "answer"])
    print("-----train [0]-----")
    print(datasets['train'][0])
    print("-----test [0]-----")
    print(datasets['test'][0])
    return datasets


def load_gsm8k_data_for_test(tokenizer):

    def make_conversation(example):
        user_questoin = example['question']
        assistant_answer = example['answer']
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_questoin},
            ],
        }    


    datasets = load_dataset("openai/gsm8k", "socratic", download_mode='reuse_dataset_if_exists') # main, socratic
    # 提取答案
    answers = []
    for data in datasets['test']:
        assistant_answer = data['answer']
        split_token = "\n####"
        assert(split_token in assistant_answer)
        _, answer = assistant_answer.split(split_token, 1)
        answers.append(answer.strip())


    print(datasets)
    datasets = datasets.map(make_conversation)
    datasets = datasets.remove_columns(["question", "answer"])
    new_datasets = []
    for data in datasets['test']:
        input_text = tokenizer.apply_chat_template(data['messages'], tokenize=False)
        new_datasets.append(input_text)
    return new_datasets, answers



def check_length_of_thought(datasets):
    all = 0
    for data in datasets['train']:
        for role in data['messages']:
            if role['role'] == "assistant":
                text = role['content']
                c = count_words_between_tags(text)
                all += c
    return all/len(datasets['train'])

def count_words_between_tags(text):
    # 使用正则表达式匹配 <A> 和 <B> 之间的内容
    match = re.search(r'<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>', text, re.DOTALL | re.MULTILINE)
    if match:
        # 提取匹配到的内容
        content = match.group(1)
        # 使用正则表达式匹配英文单词
        words = re.findall(r'\b[a-zA-Z]+\b', content)
        # 返回单词数量
        return len(words)
    else:
        return 0


if __name__=="__main__":
    #datasets = load_bespoke_data()
    datasets = load_gsm8k_data()
    average_len = check_length_of_thought(datasets)
    print("平均思考长度：", average_len)
