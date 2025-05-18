# DeepSeek-V3 模型参数
{
 "aux_loss_alpha": 0.001,
  "ep_size": 1,
  "hidden_act": "silu",#激活函数
  "hidden_size": 7168,# hidden size
  "initializer_range": 0.02,
  "intermediate_size": 18432,#这个是mlp中间层大小
  "kv_lora_rank": 512, # mla的kv低秩维度
  "max_position_embeddings": 163840,#最大长度
  "moe_intermediate_size": 2048,#专家的中间维度
  "n_group": 8,
  "n_routed_experts": 256, # 非共享专家数
  "n_shared_experts": 1,# 共享专家书
  "num_attention_heads": 128,#注意力头数
  "num_experts_per_tok": 8, # 每个token激活抓夹数
  "num_hidden_layers": 61,#transformer层数
  "num_key_value_heads": 128, # k，v头数
  "pretraining_tp": 1,# 预训练tp数
  "q_lora_rank": 1536,#query的lora维度
  "qk_nope_head_dim": 128,#qk的无rope维度
  "qk_rope_head_dim": 64,#qk的rope维度
  "quantization_config": {
    "activation_scheme": "dynamic",#activation动态量化
    "fmt": "e4m3",# 指数4位尾数三位
    "quant_method": "fp8",# 量化数据结构
    "weight_block_size": [
      128,# 量化分组数，即一个128*128的矩阵共享一组scaling和bias
      128
    ]
  },
  "rms_norm_eps": 1e-06,#rms 的参数
  "rope_scaling": {# yarn的上下文长度拓展参数
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 40,
    "scale": 1.0,
    "mscale_all_dim": 1.0,
    "original_max_position_embeddings": 4096,
    "type": "yarn"
  },
  "rope_theta": 10000, # theta的大小
  "routed_scaling_factor": 2.5, #router的放大系数
  "scoring_fun": "sigmoid",
  "seq_aux": true,
  "tie_word_embeddings": false, #lm_head和embeding不共享
  "topk_group": 4,#激活专家group
  "v_head_dim": 128,#v的维度
  "vocab_size": 129280 #词表数
}
