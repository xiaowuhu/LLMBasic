import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

# 定义 GRPO Trainer
class GRPOTrainer:
    def __init__(self, model, ref_model, tokenizer, dataset, config):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.config = config
        self.beta = 0.04
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            for batch in self.dataloader:
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)

                # 计算模型输出
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                # 计算参考模型输出
                with torch.no_grad():
                    ref_outputs = self.ref_model(input_ids, attention_mask=attention_mask)
                    ref_logits = ref_outputs.logits

                loss = self.comnpute_loss(logits, ref_logits)
                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(self.dataloader):.4f}")

    def compute_loss(self, logits, ref_logits, completion_mask):
        log_probs = torch.log_softmax(logits, dim=-1)
        ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
        policy_loss = torch.exp(log_probs - ref_log_probs) * self.advantages
        kl_loss = torch.exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1
        loss = -(policy_loss - self.beta * kl_loss)
        loss = ((loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        return loss

# 配置
config = {
    'learning_rate': 1e-5,
    'batch_size': 4,
    'beta': 0.04
}

# 加载预训练模型和 tokenizer
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
ref_model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 准备数据集
data = [{'text': 'This is a sample text.'}, {'text': 'Another sample text.'}]
dataset = MyDataset(data, tokenizer)

# 初始化 GRPO Trainer
trainer = GRPOTrainer(model, ref_model, tokenizer, dataset, config)

# 训练模型
trainer.train(num_epochs=3)


