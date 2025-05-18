# 示例代码，不可运行，仅供参考
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习的缩放参数
    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x: torch.Tensor):
        return self.weight * self._norm(x.float()).type_as(x)

class share_embedding(nn.Module, config):
    def __init__(self, config: DeepseekV3Config):
        super().__init__()
        self.embed_tokens = nn.Embedding(
                config.vocab_size, config.hidden_size
            )
    def forward(self,input_ids):
        return self.embed_tokens(input_ids)
        
class share_output_head(nn.Module,config):
    def __init__(self,config:Deepseekv3Config):
        super().__init__()
        self.lm_head = nn.Linear(config.hidden_size,config.vocab_size)
    def forward(self,transformer_hidden):
        return self.lm_head(transformer_hidden)
        
class transformer_block(nn.Module,config):
    def __init__(self, config: DeepseekV3Config):
        super().__init__()
        self.mla = MLA(config)
        self.moe = MOE(config)
        self.pre_RMSNorm = RMSNorm(config)
        self.post_RMSNorm = RMSNorm(config)
    def forward(self,input_hidden):
        out_logits = self.pre_RMSNorm(input_hidden)
        out_logits = self.mla(out_logits)
        out_logits = self.post_RMSNorm(out_logits)        
        moe_input = input_hidden+out_logits
        moe_output = self.moe(moe_input)
        block_output = moe_output+moe_input
        return block_output
        
class MTP(nn.Module,config):
    def __init__(self,config:Deepseekv3Config):
        super().__init__()
        self.RMSNorm_right = RMSNorm(config)
        self.RMSNorm_left = RMSNorm(config)
        self.transformer = transformer_block(config)
        self.proj = nn.Linear(2*config.hidden_size,config.hidden_size)
    def forward(self,last_block_out,input_ids_truncated,share_embeding,share_lm_head):
        last_norm_out = self.RMSNorm_right(last_block_out)
        embeding_trunc = share_embeding(input_ids_truncated)
        concat_input = torch.cat((last_norm_out, embeding_trunc), dim=-1)
        proj_out = self.proj(concat_input)
        trans_out_logits = self.transformer(proj_out)
        return trans_out_logits
        
class MTP_and_deepseek(nn.Module,config):
    def __init__(self,config:Deepseekv3Config):
        super().__init__()
        self.share_embedding = share_embedding(config)
        self.share_lm_head = share_output_head(config)
        self.loss = nn.CrossEntropyLoss(ignore = config.pad_token_id)
        self.Model_trans_blocks = nn.ModuleList(
            [
                transformer_block(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.MTP_trans_blocks = nn.ModuleList(
            [
                transformer_block(config, layer_idx)
                for layer_idx in range(config.num_MTP_layers)
            ]
        )
        self.config = config
        self.alpha_list = config.alpha_list#对于每一个MTPloss的加权列表
   
   def forward(self,input_ids):
       ## input_ids :
       ###[bos,tok1,tok2......last_tok]
       ##labels_origin:
       ##[tok1,tok2,....,last_tok,eos/pad]
       ##labels_MTP1
       ##[tok2,.....last_tok,eos,pad]
       ##labels_MTP2
       ##[tok3,.....last_tok,eos,pad,pad]
       embeding_logits = self.share_embedding(input_ids)
       deepseek_hidden = embeding_logits
       for index,trans_block in enumerate(self.Model_trans_blocks):
           deepseek_hidden = trans_block(deepseek_hidden)
       deepseek_logits = self.share_lm_head(deepseek_hidden)
       labels = torch.cat([input_ids[:, 1:], torch.full((input_ids.size(0), 1), config.pad_token_id)], dim=1)
       Main_loss = self.loss(deepseek_logits,labels)
       
       
       last_mtp_out = deepseek_hidden
       for ind, MTP in enumerate(self.MTP_trans_blocks):
           input_ids_trunc = torch.cat([input_ids[:, ind + 1 :],  # 截取从 ind+1 开始的部分
              torch.full((input_ids.size(0), ind + 1), self.config.pad_token_id),  # 在后面添加 pad_token_id],
              dim=1,)    
           mtp_out = MTP(last_mtp_out,input_ids_trunc,self.share_embedding)
           mtp_logits = self.share_lm_head(mtp_out)
           last_mtp_out = mtp_out
           labels_trunc = torch.cat([input_ids_trunc[:, 1:], torch.full((input_ids.size(0), 1), config.pad_token_id)], dim=1)
           mtp_loss = self.loss(mtp_logits,labels_trunc)
           alpha = self.alpha_list[ind]
           Main_loss +=  alpha*mtp_loss
       return Main_loss
