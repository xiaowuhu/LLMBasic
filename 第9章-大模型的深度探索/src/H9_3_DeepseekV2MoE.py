import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
from dataclasses import dataclass

# 这是DeepSeekV2MoE 的部分代码实现，不完整，只供说明

config = {
  # 部分参数省略
  "hidden_act": "silu",
  "hidden_size": 5120,
  "initializer_range": 0.02,
  "intermediate_size": 12288,
  "model_type": "deepseek_v2",
  "moe_intermediate_size": 1536,
  "moe_layer_freq": 1,
  "n_group": 8,
  "n_routed_experts": 160,
  "n_shared_experts": 2,
  "norm_topk_prob": False,
  "num_experts_per_tok": 6,
  "num_hidden_layers": 60,
  "num_key_value_heads": 128,
  "topk_group": 3,
  "topk_method": "group_limited_greedy",
}

class DeepseekV2MLP(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act] # silu 激活函数

    def forward(self, x):
        mlp_out = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return mlp_out

class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.scoring_func = config.scoring_func
        self.topk_method = config.topk_method
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        
        # 静态化推理配置（假设配置固定）
        self.inference_norm = self.norm_topk_prob and (self.top_k > 1)
        self.use_group_limited = (self.topk_method == "group_limited_greedy")

        # 门控权重
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    @torch.inference_mode()  # 禁用梯度与训练逻辑
    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, h)
        
        # 门控分数计算（保持原始数据类型）
        logits = F.linear(hidden_states, self.weight)  # [n_tokens, n_experts]
        scores = logits.softmax(dim=-1)  # 自动推断 dtype

        # Top-K 选择（静态分支）
        if self.use_group_limited:
            # 分组限制逻辑优化
            group_scores = scores.view(bsz * seq_len, self.n_group, -1).max(dim=-1).values
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
            score_mask = group_mask.unsqueeze(-1).expand(-1, -1, self.n_routed_experts // self.n_group).reshape(bsz * seq_len, -1)
            scores = scores.masked_fill(~score_mask.bool(), 0.0)
        
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # 权重归一化（静态分支）
        if self.inference_norm:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        else:
            topk_weight = topk_weight * self.routed_scaling_factor

        return topk_idx, topk_weight, None  # aux_loss 始终为 None

@dataclass
class DeepseekV2Config:
    # 1, Position Config
    max_position_embeddings: int = 163840
    vocab_size: int = 102400

    # 2, MLA Config
    # down_linear config
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512

    # head_dim、heads and hidden_size config
    v_head_dim: int = 128
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    hidden_size: int = 5120
    num_attention_heads: int = 128
    num_key_value_heads: int = 128
    
    attention_bias: bool = False

    attention_dropout: float = 0.1
    # rope config
    rope_theta: float = 10000

    # 3, MOE Config
    n_group: int = 8
    n_routed_experts: int = 160
    num_experts_per_tok: int = 6
    topk_group: int = 3
    routed_scaling_factor: float = 1.0
    scoring_func: str="softmax"
    topk_method: str="greedy"
    norm_topk_prob: bool = True

# 初始化配置
config = DeepseekV2Config()

# 模拟输入，CPU 电脑可直接跑，去除了 cuda 设备限制代码
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_states = torch.randn(32, 64, 5120, device=device)

# 创建模块
moe_gate = MoEGate(config)  # 半精度推理

# gate 网络推理
topk_idx, topk_weight, _ = moe_gate(hidden_states)

print("topk_idx shape ", topk_idx.shape) # 32 * 64 = 2048 个 tokens
print("topk_weight shape", topk_weight.shape)

"""
# 输出如下，表示每个 token 会激活 6 个专家参与计算
topk_idx shape  torch.Size([2048, 6]) 
topk_weight shape torch.Size([2048, 6])
"""

# 为了单元测试，模拟不使用分布式（ep_size默认为1）
class DeepseekV2MoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok

        self.experts = nn.ModuleList(
            [
                DeepseekV2MLP(
                    config, intermediate_size=config.moe_intermediate_size
                )
                for i in range(config.n_routed_experts)
            ]
        )
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DummyMLP(config=config, intermediate_size=intermediate_size)

    # 此处为简化实现，仅做推理示例，不涉及分布式通信
    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        # x: [batch * seq_len, hidden_size]
        # 对每个 token 依然采用与训练类似的方式进行专家计算
        outputs = []
        flat_topk_ids = topk_ids.view(-1)
        for i, expert in enumerate(self.experts):
            mask = (flat_topk_ids == i)
            if mask.sum() == 0:
                continue
            outputs.append(expert(x[mask]))
        # 简单拼接，不做复杂排序和 all-to-all 操作
        outs = torch.cat(outputs, dim=0)
        new_x = torch.empty_like(outs)
        # 这里直接返回加权求和的结果（实际实现更复杂）
        final_out = (outs.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
        return final_out
