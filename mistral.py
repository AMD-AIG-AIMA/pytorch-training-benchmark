from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te
from pydantic.dataclasses import dataclass

from torch.nn.attention.flex_attention import create_block_mask
from torch.nn.attention.flex_attention import flex_attention
from flash_attn.modules.mha import FlashSelfAttention  


@dataclass
class MistralConfig:
    num_layers: int    # L
    num_heads: int     # H
    num_kv_heads: int  # J
    embedding_dim: int      # E
    hidden_dim: int       # K
    vocab_size: int  # V
    max_seq_len: int # T
    window_size: int # W
    eps: float
    arch_name: str = 'mistral'

    def calculate_mistral_tflops_training_per_device(self, bsz, total_ffn_flops, qkv_flops, projection_flops, embedding_flops):
        """
        Calculate training TFLOP for mistral as in mistral we use sliding window attention
        """
        head_dim = self.embedding_dim // self.num_heads
        attention_flops = (
            # local attention
            4
            * bsz
            * self.max_seq_len
            * min(self.window_size, self.max_seq_len)
            * self.num_heads
            * head_dim
        )
        attention_tflops = attention_flops * self.num_layers * 3

        learnable_weight_tflops = (
            ((total_ffn_flops + qkv_flops + projection_flops) * self.num_layers + embedding_flops) * 3
        )

        return attention_tflops, learnable_weight_tflops

    def estimate_flops_per_token(self, model, bsz, rank=0):
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
        projection_flops = (
            2 * bsz * self.max_seq_len * self.embedding_dim * self.num_heads * head_dim
        )
        embedding_flops = 2 * bsz * self.max_seq_len * self.embedding_dim * self.vocab_size

        attention_tflops, learnable_weight_tflops = self.calculate_mistral_tflops_training_per_device(
            bsz, total_ffn_flops, qkv_flops, projection_flops, embedding_flops
        )
        
        total_tflops = learnable_weight_tflops + attention_tflops

        self.flops_per_token =  total_tflops / (bsz * self.max_seq_len)


class MistralAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_kv_heads, window_size, max_seq_len, **kwargs):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.head_dim = embedding_dim // num_heads
        self.kv_dim = num_kv_heads * embedding_dim // num_heads

        # self.q_proj = nn.Linear(embedding_dim, num_heads * self.head_dim, bias=False)
        # self.k_proj = nn.Linear(embedding_dim, num_kv_heads * self.head_dim, bias=False)
        # self.v_proj = nn.Linear(embedding_dim, num_kv_heads * self.head_dim, bias=False)
        self.in_proj = nn.Linear(embedding_dim, embedding_dim+2*self.kv_dim, bias=False)

        self.o_proj = nn.Linear(num_heads * self.head_dim, embedding_dim, bias=False)

        self.use_flex = torch.cuda.is_available() and 'MI3' not in torch.cuda.get_device_name() 
        if self.use_flex:
            self.sdpa = torch.compile(flex_attention, dynamic=False)
            self.blk_mask = blk_mask = create_block_mask_cached(window_size, max_seq_len)
        else:
            self.self_attn = FlashSelfAttention(attention_dropout=0.0, window_size=(window_size-1,0))

    def forward(self, input, position_encoding):
        # q, k, v = self.q_proj(input), self.k_proj(input), self.v_proj(input)
        qkv = self.in_proj(input)
        q,k,v = qkv.split([self.embedding_dim, self.kv_dim, self.kv_dim], -1)
        q = q.unflatten(-1, [-1, self.head_dim]).transpose(1, 2)
        k = k.unflatten(-1, [-1, self.head_dim]).transpose(1, 2)
        v = v.unflatten(-1, [-1, self.head_dim]).transpose(1, 2)

        q, k = apply_rotary_emb(q, k, freqs_cis=position_encoding)
        
        #q = apply_rotary_embd(q, position_encoding)
        #k = apply_rotary_embd(k, position_encoding)

        k = k.repeat_interleave(self.embedding_dim//self.kv_dim, 1)
        v = v.repeat_interleave(self.embedding_dim//self.kv_dim, 1)

        if self.use_flex:
            o_BHTD = self.sdpa(q, k, v, block_mask=self.blk_mask)
            y_BTE = self.o_proj(o_BHTD.transpose(1, 2).flatten(-2))
        else:
            qkv = torch.stack([q, k, v], dim=2)
            qkv_t = qkv.transpose(-2, 1)
            o = self.self_attn(qkv_t, causal=True)
            out = self.o_proj(o.reshape(input.shape))

        return out

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

def apply_rotary_embd(x_BXTD, freq_cis_TFC):
     x_BXTFC = x_BXTD.unflatten(-1, [-1, 2])  # C: Complex number dimension
     freq_cis_BXTFC = freq_cis_TFC.expand_as(x_BXTFC)

     out_BXTDC = torch.stack([
         x_BXTFC[..., 0] * freq_cis_BXTFC[..., 0] - x_BXTFC[..., 1] * freq_cis_BXTFC[..., 1],
         x_BXTFC[..., 1] * freq_cis_BXTFC[..., 0] + x_BXTFC[..., 0] * freq_cis_BXTFC[..., 1],
     ], dim=-1)
     out_BXTD = out_BXTDC.flatten(-2)

     return out_BXTD.type_as(x_BXTD)


@lru_cache
def create_block_mask_cached(window_size, max_seq_len):
    def swa_mask_mod(b, h, q_idx, kv_idx):
        causal_mask = (q_idx >= kv_idx)
        window_mask = (q_idx - kv_idx <= window_size)  # (window_size + 1) kv per q unmasked
        return causal_mask & window_mask
    blk_mask = create_block_mask(swa_mask_mod, 1, 1, max_seq_len, max_seq_len, device='cuda')
    return blk_mask


class SwiGLU(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.up_proj = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.gate_proj = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, embedding_dim, bias=False)

    def forward(self,input):
        o = self.down_proj(F.silu(self.gate_proj(input)) * self.up_proj(input))
        return o


class RMSNorm(nn.Module):
    def __init__(self, embedding_dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embedding_dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class MistralBlock(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_heads, num_kv_heads, eps, max_seq_len, window_size):
        super().__init__()
        self.attn_norm = RMSNorm(embedding_dim, eps)
        self.attn = MistralAttention(embedding_dim, num_heads, num_kv_heads, window_size, max_seq_len)
        self.ffn_norm = RMSNorm(embedding_dim, eps)
        self.ffn = SwiGLU(embedding_dim, hidden_dim)

    def forward(self, x, freq_cis):
        h = x + self.attn(self.attn_norm(x), freq_cis)
        out = h + self.ffn(self.ffn_norm(h))
        return out


class Mistral(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_heads, num_kv_heads, eps, max_seq_len, window_size, **kwargs):
        super().__init__()
        self.tok_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer_layers = nn.ModuleList(
            MistralBlock(embedding_dim, hidden_dim, num_heads, num_kv_heads, eps, max_seq_len, window_size) for _ in range(num_layers)
        )
        self.norm = RMSNorm(embedding_dim, eps)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)
        freqs = precompute_freqs_cis(embedding_dim//num_heads, max_seq_len)
        self.register_buffer('position_encoding', freqs.to(self.lm_head.weight.dtype))
        if not torch.cuda.is_available() or 'MI3' in torch.cuda.get_device_name():
            self.register_buffer('mask', create_sliding_window_attention_mask(window_size, max_seq_len))
        else:
            self.mask = None

    def forward(self, idx, **kwargs):
        x = self.tok_embedding(idx)
        for layer in self.transformer_layers:
            x = layer(x, self.position_encoding)
        logits = self.lm_head(self.norm(x))
        return logits

def precompute_freqs_cis(dim, end, theta: float = 1e4) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def create_sliding_window_attention_mask(window_size, max_seq_len):
    q_idx = torch.arange(max_seq_len).unsqueeze(1)   # [T, 1]
    kv_idx = torch.arange(max_seq_len).unsqueeze(0)  # [1, T]
    causal_mask = (q_idx >= kv_idx)
    window_mask = (q_idx - kv_idx < window_size)
    return causal_mask & window_mask


class Fp8Mistral(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_heads, num_kv_heads, window_size, max_seq_len, eps, **kwargs):
        super().__init__()
        self.tok_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer_layers = nn.ModuleList(
            Fp8MistralBlock(embedding_dim, hidden_dim, num_heads, num_kv_heads, window_size, eps) for _ in range(num_layers)
        )
        self.lm_head_te = te.LayerNormLinear(
            embedding_dim, vocab_size, bias=False,
            normalization='RMSNorm', eps=eps
        )
        position_encoding = te.attention.RotaryPositionEmbedding(embedding_dim//num_heads)(max_seq_len=max_seq_len)
        self.register_buffer('position_encoding', position_encoding.to(torch.bfloat16))

    def forward(self, idx, is_first_microbatch):
        x = self.tok_embedding(idx)
        for layer in self.transformer_layers:
            x = layer(x, rotary_pos_emb=self.position_encoding, is_first_microbatch=is_first_microbatch)
        logits = self.lm_head_te(x)
        return logits


class Fp8MistralBlock(te.TransformerLayer):
    def __init__(self, embedding_dim, hidden_dim, num_heads, num_kv_heads, window_size, eps):
        super().__init__(
            hidden_size=embedding_dim,
            num_attention_heads=num_heads,
            num_gqa_groups=num_heads//num_kv_heads,
            fuse_qkv_params=True,
            attn_input_format='bshd',
            self_attn_mask_type='causal',  # attn mask is ignored if not set
            window_size=(window_size, 0),
            attention_dropout=0.0,
            normalization='RMSNorm',
            layernorm_epsilon=eps,
            ffn_hidden_size=hidden_dim,
            bias=False,
            activation='swiglu',
            hidden_dropout=0.0
        )

