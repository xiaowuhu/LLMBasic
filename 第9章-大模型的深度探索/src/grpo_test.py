import torch

log_probs = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
log_probs_diff = log_probs - log_probs.detach()
print(f"原始值: {log_probs}")
print(f"减去无梯度值: {log_probs_diff}")
advantages = torch.tensor([1.0, 2.0, 3.0])
print(f"优势值: {advantages}")
policy_loss = -log_probs_diff * advantages
print(f"乘以优势值: {policy_loss}")
loss = policy_loss.mean().backward()
print(f"检查梯度:{log_probs.grad}")
