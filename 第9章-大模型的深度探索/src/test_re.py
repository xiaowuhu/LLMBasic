import re

pattern = r"^<\|begin_of_thought\|>.*?<\|end_of_thought\|>\s*<\|begin_of_solution\|>.*?<\|end_of_solution\|>$"
completion = "<|begin_of_thought|>First, this is a question about math. I'm find.<|end_of_thought|><|begin_of_solution|>the final answer is 3<|end_of_solution|>"
m1 = re.search(pattern, completion)
if m1: print(m1.group(0))

c2 = "<think>First, this is a question about math. Second, I'm find.</think><answer>the final answer is 3</answer>"
m2 = re.search(pattern, c2)
print(m2)

c3 = "<|begin_of_thought|>First, this is a question about math. I'm find.<|end_of_thought|><|begin_of_solution|>the final answer is 3"
p31 = r"<\|begin_of_thought\|>.*?<\|end_of_thought\|>"
p32 = r"<\|begin_of_solution\|>.*?<\|end_of_solution\|>"
m31 = re.search(p31, c3)
m32 = re.search(p32, c3)
if m31: 
    print(m31.group(0)) 
else: 
    print(m31)
print(m32)

# import re

# # 定义正则表达式
# pattern = r'^123'

# # 测试字符串
# test_string1 = "123abc"
# test_string2 = "abc123"

# # 使用 re.match() 检查是否匹配
# match1 = re.match(pattern, test_string1)
# match2 = re.search(pattern, test_string2)

# # 判断是否匹配
# print(f"'{test_string1}' 匹配结果: {'匹配成功' if match1 else '匹配失败'}")
# print(f"'{test_string2}' 匹配结果: {'匹配成功' if match2 else '匹配失败'}")


ss = "<|im_start|>system\
The assistant first thinks about the reasoning process for the user's question, and put the reasoning process within <|begin_of_thought|> and <|end_of_thought|> tags.Then put the answer within <|begin_of_answer|> and <|end_of_answer|> tags.<|im_end|>\
<|im_start|>user\
Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?<|im_end|>\
<|im_start|>assistant\
If you have 120 pages in the book, you can read 12 pages on day 1, so 12 pages + 120=28) The second page will be 30+ 20 (200 pages) and the same number for each subsequent one.\
In 100 days, I want to read 560 pages. This time I will read 100 – 28-300 = 560\
A few days ago, you were able to read 36, a few more than 24 pages! 240 pages, 360 - 3 = 324).\
30 more pages from today! 324 / 100 = 1.24 for the rest of the day\
, that’s 3,240\
. So, 120/2 = 60. It’s an interesting fact, but not all the pages are useful for all people, so it’s important to choose those that suit them best).\
This way we prevent children from falling behind ours due to learning problems caused by physical activities outside school halls).\
Additionally, studying data analytics helps"
a = ss.split()
print(len(a))
