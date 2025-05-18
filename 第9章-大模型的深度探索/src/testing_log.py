import logging

# 创建一个 logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # 设置日志级别

# 创建一个 file handler，用于将日志写入文件
file_handler = logging.FileHandler('example.log')  # 指定日志文件名
file_handler.setLevel(logging.DEBUG)  # 设置文件处理器的日志级别

# 创建一个 formatter，定义日志格式
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(message)s')
file_handler.setFormatter(formatter)  # 将 formatter 添加到 file handler

# 将 file handler 添加到 logger
logger.addHandler(file_handler)

# 测试日志
logger.debug('This is a debug message')
logger.info('This is an info message')
logger.warning('This is a warning message')
logger.error('This is an error message')
logger.critical('This is a critical message')

logger.info("{'loss': 0.0024, 'grad_norm': 2.175569534301758e-06, 'learning_rate': 2.1510868730317228e-05, 'rewards/reward_len': -236.0, 'reward': -236.0, 'reward_std': 0.0, 'completion_length': 256.0, 'kl': 0.06026326939463615, 'epoch': 1.23}")
logger.info("{'loss': 0.0024, 'grad_norm': 4.351139068603516e-06, 'learning_rate': 2.1491666026576543e-05, 'rewards/reward_len': -236.0, 'reward': -236.0, 'reward_std': 0.0, 'completion_length': 256.0, 'kl': 0.059102198109030724, 'epoch': 1.23}")
logger.info("{'loss': 0.0023, 'grad_norm': 1.9222497940063477e-06, 'learning_rate': 2.1472463322835858e-05, 'rewards/reward_len': -236.0, 'reward': -236.0, 'reward_std': 0.0, 'completion_length': 256.0, 'kl': 0.05818593129515648, 'epoch': 1.23}")
logger.info("{'loss': 0.0024, 'grad_norm': 1.691281795501709e-06, 'learning_rate': 2.145326061909517e-05, 'rewards/reward_len': -236.0, 'reward': -236.0, 'reward_std': 0.0, 'completion_length': 256.0, 'kl': 0.05939498096704483, 'epoch': 1.23}")
logger.info("{'loss': 0.0023, 'grad_norm': 1.773238182067871e-06, 'learning_rate': 2.1434057915354484e-05, 'rewards/reward_len': -236.0, 'reward': -236.0, 'reward_std': 0.0, 'completion_length': 256.0, 'kl': 0.05834215469658375, 'epoch': 1.23}")

import torch
from torch.nn.utils.rnn import pad_sequence

# 示例序列
sequences = [torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6])]

# 右侧填充
padded_sequences_right = pad_sequence(sequences, batch_first=True, padding_value=0)
print("右侧填充：")
print(padded_sequences_right)

# 左侧填充
padded_sequences_left = pad_sequence([torch.flip(seq, [0]) for seq in sequences], batch_first=True, padding_value=0)
padded_sequences_left = torch.flip(padded_sequences_left, [1])
print("左侧填充：")
print(padded_sequences_left)
