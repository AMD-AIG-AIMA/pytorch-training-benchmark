import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te

from pydantic.dataclasses import dataclass
from flash_attn import flash_attn_func

@dataclass
class LLaMAConfig:
    num_layers: int    # L
    num_heads: int     # H
    num_kv_heads: int  # J
    embedding_dim: int      # E
    max_seq_len: int # T
    vocab_size: int  # V
    eps: float
    hidden_dim: int # K

    def estimate_flops_per_token(self, model,bsz):
        head_dim = self.embedding_dim // self.num_heads

        """Calculate training TFLOP"""
        ffn1_flops = (
            2
            * bsz
            * self.max_seq_len
            * self.hidden_dim
            * self.embedding_dim
            * 2 # len(config.mlp_activations)
        )
        ffn2_flops = 2 * bsz * self.max_seq_len * self.hidden_dim * self.embedding_dim
        total_ffn_flops = ffn1_flops + ffn2_flops

        qkv_flops = (
            2
            * bsz
            * self.max_seq_len
            * self.embedding_dim
            * (self.num_heads + 2 * self.num_kv_heads)
            * head_dim
        )
        attention_flops = 4 * bsz * self.max_seq_len**2 * self.num_heads * head_dim
        projection_flops = (
            2 * bsz * self.max_seq_len * self.embedding_dim * self.num_heads * head_dim
        )
        embedding_flops = 2 * bsz * self.max_seq_len * self.embedding_dim * self.vocab_size

        # multiply by 3 for both feed forward and back propagation flops
        learnable_weight_tflops = (
            ((total_ffn_flops + qkv_flops + projection_flops) * self.num_layers + embedding_flops) * 3
        )

        # megatron tflops calculation does not account for causality in attention
        attention_tflops = attention_flops * self.num_layers * 3

        total_tflops = learnable_weight_tflops + attention_tflops
        self.flops_per_token =  total_tflops / (bsz * self.max_seq_len)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    #freqs_cis = freqs_cis[:, None, :]
    freqs_cis = freqs_cis[None, None, :, :]

    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    def __init__(self,embedding_dim,num_heads,num_kv_heads):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.head_dim = embedding_dim // num_heads
        self.kv_dim = embedding_dim * num_kv_heads // num_heads 
        self.in_proj = nn.Linear(embedding_dim, embedding_dim+2*self.kv_dim, bias=False)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.use_sdpa = torch.cuda.is_available() and 'MI3' not in torch.cuda.get_device_name() 
    
    def forward(self,input,position_encoding):
        qkv = self.in_proj(input)
        q,k,v = qkv.split([self.embedding_dim, self.kv_dim, self.kv_dim], -1)
        q = q.unflatten(-1, [-1, self.head_dim]).transpose(1, 2)
        k = k.unflatten(-1, [-1, self.head_dim]).transpose(1, 2)
        v = v.unflatten(-1, [-1, self.head_dim]).transpose(1, 2)
        q, k = apply_rotary_emb(q, k, position_encoding)

        
        if self.use_sdpa:
            k = k.repeat_interleave(self.embedding_dim//self.kv_dim,1)
            v = v.repeat_interleave(self.embedding_dim//self.kv_dim,1)
            o = F.scaled_dot_product_attention(q,k,v,dropout_p=0, is_causal=True)
        else:
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            o = flash_attn_func(q,k,v,dropout_p=0, causal=True)

        o = self.out_proj(o.reshape(input.shape))
        return o

class MLP(nn.Module):
    def __init__(self,embedding_dim,hidden_dim):
        super().__init__()
        self.up_proj = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.gate_proj = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, embedding_dim, bias=False)
    
    def forward(self,input):
        hid = F.silu(self.gate_proj(input)) * self.up_proj(input)
        o = self.down_proj(hid)
        return o

class RMSNorm(nn.Module):
    def __init__(self, embedding_dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(embedding_dim))
        self.eps = eps

    def forward(self, input):
        # use high precision, see https://github.com/foundation-model-stack/foundation-model-stack/blob/d55a9f2ade65ef4157cdfd928300874e2348e5d0/fms/modules/layernorm.py#L64
        input_float = input.float() 
        output = (input_float * torch.rsqrt(input_float.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(input) * self.weight
        return output

class LLaMABlock(nn.Module):
    def __init__(self,embedding_dim,hidden_dim,num_heads,num_kv_heads,eps):
        super().__init__()
        self.attn_norm = RMSNorm(embedding_dim,eps)
        self.attn = Attention(embedding_dim,num_heads,num_kv_heads)
        self.mlp_norm = RMSNorm(embedding_dim,eps)
        self.mlp = MLP(embedding_dim,hidden_dim)

    def forward(self,input,position_encoding):
        hid = input + self.attn(self.attn_norm(input), position_encoding)
        output = hid + self.mlp(self.mlp_norm(hid))
        return output
    
def precompute_freq_cis(dim, max_seq_len):
    rope_base=500000.0
    assert dim % 2 == 0
    freqs = 1 / (rope_base ** (torch.arange(0, dim, 2).float() / dim))  # F = dim // 2
    t = torch.arange(max_seq_len, device=freqs.device) 
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)

class LLaMA(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,num_layers,num_heads,num_kv_heads,max_seq_len,eps):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        layers = []
        for i in range(num_layers):
            layers.append(LLaMABlock(embedding_dim,hidden_dim,num_heads,num_kv_heads,eps))
        self.layers = nn.ModuleList(layers)
        self.norm = RMSNorm(embedding_dim, eps)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        freqs = precompute_freq_cis(embedding_dim//num_heads,max_seq_len)
        self.register_buffer('position_encoding', freqs)

    def forward(self, idxs, is_first_microbatch):
        x = self.embedding(idxs)
        for layer in self.layers:
            x = layer(x, self.position_encoding)
        logits = self.lm_head(self.norm(x))
        return logits
    
class Fp8LLaMA(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,num_layers,num_heads,num_kv_heads,max_seq_len,eps):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        layers = []
        for i in range(num_layers):
            layers.append(Fp8LLaMABlock(embedding_dim,hidden_dim,num_heads,num_kv_heads,eps))
        self.layers = nn.ModuleList(layers)
        self.norm_lm_head = te.LayerNormLinear(embedding_dim, vocab_size, bias=False,normalization='RMSNorm', eps=eps)

        position_encoding = te.attention.RotaryPositionEmbedding(embedding_dim//num_heads)(max_seq_len=max_seq_len)
        self.register_buffer('position_encoding', position_encoding.to(torch.bfloat16))

    def forward(self, idxs, is_first_microbatch):
        x = self.embedding(idxs)
        for layer in self.layers:
            x = layer(x, rotary_pos_emb=self.position_encoding, is_first_microbatch=is_first_microbatch)
        logits = self.norm_lm_head(x)
        return logits

class Fp8LLaMABlock(te.TransformerLayer):
    def __init__(self, embedding_dim, hidden_dim, num_heads, num_kv_heads,eps):
        super().__init__(
            hidden_size=embedding_dim,
            num_attention_heads=num_heads,
            num_gqa_groups=num_heads//num_kv_heads,
            fuse_qkv_params=True,
            attn_input_format='bshd',
            attention_dropout=0.0,
            normalization='RMSNorm',
            layernorm_epsilon=eps,
            ffn_hidden_size=hidden_dim,
            bias=False,
            activation='swiglu',
            hidden_dropout=0.0
        )