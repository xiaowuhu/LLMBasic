import re
from H9_7_1_Datasets import hash2answer
from H9_6_1_Datasets import BOT, EOT, BOA, EOA

start_answer = len(BOA)
end_answer = len(EOA)

# 必须有 <\|begin_of_thought\|>.*?<\|end_of_thought\|>
# 必须有 <\|begin_of_solution\|>.*?<\|end_of_solution\|>
def format_reward(completion, **kwargs):
    reward = 0
    p1 = r"<\|begin_of_thought\|>.*?<\|end_of_thought\|>"
    m1 = re.match(p1, completion, re.DOTALL | re.MULTILINE)
    if m1:
        reward += 0.5

    p2 = r"<\|begin_of_answer\|>.*?<\|end_of_answer\|>"
    m2 = re.search(p2, completion, re.DOTALL | re.MULTILINE)
    if m2:
        reward += 0.5

    return reward


# 推理部分的长度限定在 128~144 左右
# def length_reward(completion, **kwargs):
#     count_of_words = len(completion.split())
#     if count_of_words < 128:
#         return -abs(128 - count_of_words)/10
#     elif count_of_words > 144:
#         return -abs(144 - count_of_words)/10
#     else:
#         return 1


# def check_end_reward(completion, **kwargs):
#     pos_eoa = completion.find(EOA)
#     if pos_eoa > 0:  # should only <|im_end|> after the EOA, otherwise give penalty
#         c = completion.replace("<|endoftext|>", "")
#         len_nouse = len(c) - pos_eoa - end_answer - 10
#         return -len_nouse
#     return 0

# 有逐步的分析步骤标记，如 step, First, Second 等至少有3步     
# def reasoning_steps_reward(completion, **kwargs):
#     # 表示推理步骤的关键字
#     pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
#     # 搜索一共发现了几个关键字
#     count = len(re.findall(pattern, completion, re.MULTILINE))
#     # 鼓励至少有3步分析过程
#     return min(1.0, count / 3)


# 惩罚重复词汇 3-gram
# def repetition_penalty(completion):

#     def zipngram(text: str, ngram_size: int):
#         """Helper function to generate n-grams from text."""
#         words = text.lower().split() # Lowercase and split into words
#         return zip(*[words[i:] for i in range(ngram_size)]) # Create n-grams

#     ngram_size = 3
#     max_penalty = -10.0 # Maximum penalty for repetition

#     ngrams = set() # Use a set to store unique n-grams
#     total = 1
#     for ng in zipngram(completion, ngram_size): # Generate n-grams
#         ngrams.add(ng) # Add n-gram to the set (duplicates are ignored)
#         total += 1 # Count total n-grams

#     # Calculate scaling factor: more repetition -> higher scaling
#     scaling = 1 - len(ngrams) / total
#     reward = scaling * max_penalty # Apply penalty based on scaling
#     return reward


# 准确性奖励
def accuracy_reward(prompt, completion, **kwargs):
    p = r"<\|begin_of_answer\|>.*?<\|end_of_answer\|>"
    m = re.search(p, completion, re.DOTALL | re.MULTILINE)
    if m:
        answer = m.group(0)[start_answer:-end_answer]
        ground_truth = hash2answer[hash(prompt)]
        if answer == ground_truth:
            return 1.0   
    return 0.0


def reward_functions(prompts, completions, **reward_kwargs):
    rewards = []
    for i in range(len(completions)):
        c = completions[i][0]['content']
        p = prompts[i][1]['content']
        format_r = format_reward(c)
        # length_r = length_reward(c)
        # reasoning_r = reasoning_steps_reward(c)
        #repet_p = repetition_penalty(c)
        accuracy_r = accuracy_reward(p, c)
        #end_p = check_end_reward(c)
        # reward = format_r + length_r + reasoning_r + repet_p + accuracy_r
        reward = accuracy_r + format_r
        rewards.append(reward)
    return rewards


if __name__=="__main__":
    # func_list = get_reward_functions()
    completions = [
        [
          {
            "content":"<think>First, this is a question about math. Second, I'm find.</think><answer>the final answer is 3</answer>"
          },
        ],
        [
          {
            "content":"<|begin_of_thought|>First, this is a question about math. I'm find.<|end_of_thought|><|begin_of_solution|>the final answer is 3<|end_of_solution|>"
          },
        ],
        [
          {
            "content":"<|begin_of_thought|>First, this is a question about math. I'm find.<|end_of_thought|><|begin_of_solution|>the final answer is 3"
          },
        ]
    ]

    # for func in func_list:
    #     r = func(completions)
    #     print(r)

    r = reward_functions(completions)
    print(r)
