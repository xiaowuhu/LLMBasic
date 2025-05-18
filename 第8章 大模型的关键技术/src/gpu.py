import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Default device: {device}")

# 打印所有可用的GPU
if torch.cuda.is_available():
    print("Available GPUs:")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        