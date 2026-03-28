import math
import torch
from typing import Optional
import torch.nn as nn
import warnings
from einops import rearrange

from .flash_attn_triton import flash_attn_func
from .norm import LPLaterNorm
def _reset_is_causal(
        num_query_tokens: int, num_key_tokens: int, original_is_causal: bool
):
    if original_is_causal and num_key_tokens != num_query_tokens:
        if num_query_tokens != 1:
            raise NotImplementedError("query and key with different number of tokens")
        else:
            return False
    return original_is_causal

def scaled_multihead_dot_product_attention(
        query,
        key,
        value,
        n_heads,
        past_key_value=None, # 支持KV-Cache
        softmax_scale=None,
        attn_bias=None,
        key_padding_mask=None,
        is_causal=False,
        dropout_p=0.0,
        training=False,
        needs_weights=False,
        multiquery=False, # 在multi-query Attention中，Q仍然多头但是共享KV 
):
    q = rearrange(query,"b s (h d) -> b h s d",h = n_heads)
    kv_n_heads = 1 if multiquery else n_heads
    k = rearrange(key,"b s (h d) -> b h d s",h=kv_n_heads)
    v = rearrange(value,"b s (h d) -> b h s d",h=kv_n_heads)
    if past_key_value is not None:
        if len(past_key_value) != 0:
            k = torch.cat([past_key_value[0],k],dim=3)
            v = torch.cat([past_key_value[1],v],dim=2)
        past_key_value = (k,v)
    (b,_,s_q,d) = q.shape
    s_k = k.size(-1)
    if softmax_scale is None:
        softmax_scale = 1 / math.sqrt(d)
    attn_weight = q.matmul(k) * softmax_scale
    if attn_bias is not None:
        _s_q = max(0,attn_bias.size(2) - s_q)
        _s_k = max(0,attn_bias.size(3) - s_k)
        attn_bias = attn_bias[:,:,_s_q:,_s_k:]
        if (                                        #检查是否能够broadcas
            attn_bias.size(-1) != 1
            and attn_bias.size(-1) != s_k
            or (attn_bias.size(-2) != 1 and attn_bias.size(-2) != s_q)
        ):
            raise RuntimeError("attn_bias can't broadcast")
        attn_weight = attn_weight + attn_bias
    min_val = torch.finfo(q.dtype).min #极小值
    
    if key_padding_mask is not None:
        warnings.warn(
                "Propogating key_padding_mask to the attention module "
                + "and applying it within the attention module can cause "
                + "unneccessary computation/memory usage. Consider integrating "
                + "into attn_bias once and passing that to each attention "
                + "module instead."
            )
        attn_weight = attn_weight.masked_fill(
            ~key_padding_mask.view((b,1,1,s_k)),min_val
        )
    
    if is_causal and (not q.size(2) == 1):
        s = max(s_q,s_k)
        causal_mask = attn_weight.new_ones(s,s,dtype=torch.float16)
        causal_mask = causal_mask.tril() #生成下三角
        causal_mask = causal_mask.to(torch.bool)
        causal_mask = ~causal_mask
        causal_mask = causal_mask[-s_q:,-s_k:]
        attn_weight = attn_weight.masked_fill(causal_mask.view(1,1,s_q,s_k),min_val)

    attn_weight = torch.softmax(attn_weight,dim=-1)
    if dropout_p:
        attn_weight = torch.nn.functional.dropout(
            attn_weight,p=dropout_p,training=training,inplace=True
        )
    
    out = attn_weight.to(v.dtype).matmul(v)
    out = rearrange(out,"b h s d -> b s (h d)")

    if needs_weights:
        return (out,attn_weight,past_key_value)
    return (out,None,past_key_value)

def check_valid_inputs(*tensors,valid_dtypes=[torch.float16,torch.bfloat16]):
    for tensor in tensors:
        if tensor.dtype not in valid_dtypes:
            raise TypeError(
                f"tensor.dtype={tensor.dtype!r} must be in valid_dtypes={valid_dtypes!r}."
            )
        if not tensor.is_cuda:
            raise TypeError(
                f"Inputs must be cuda tensors (tensor.is_cuda={tensor.is_cuda!r})."
            )

    
def triton_flash_attn_fn(
    query,
    key,
    value,
    n_heads,
    past_key_value=None,
    softmax_scale=None,
    attn_bias=None,
    key_padding_mask=None,
    is_causal=False,
    dropout_p=0.0,
    training=False,
    needs_weights=False,
    multiquery=False,
):
    check_valid_inputs(query, key, value)
    if past_key_value is not None:
        if len(past_key_value) != 0:
            key = torch.cat([past_key_value[0], key], dim=1)
            value = torch.cat([past_key_value[1], value], dim=1)
        past_key_value = (key, value)
    if attn_bias is not None:
        _s_q = max(0, attn_bias.size(2) - query.size(1))
        _s_k = max(0, attn_bias.size(3) - key.size(1))
        attn_bias = attn_bias[:, :, _s_q:, _s_k:]
    if dropout_p:
        raise NotImplementedError(f"Dropout not implemented for attn_impl: triton.")
    if needs_weights:
        raise NotImplementedError(f"attn_impl: triton cannot return attn weights.")
    if key_padding_mask is not None:
        warnings.warn(
            "Propagating key_padding_mask to the attention module "
            + "and applying it within the attention module can cause "
            + "unnecessary computation/memory usage. Consider integrating "
            + "into attn_bias once and passing that to each attention "
            + "module instead."
        )
        (b_size, s_k) = key_padding_mask.shape[:2]
        if attn_bias is None:
            attn_bias = query.new_zeros(b_size, 1, 1, s_k)
        attn_bias = attn_bias.masked_fill(
            ~key_padding_mask.view((b_size, 1, 1, s_k)), torch.finfo(query.dtype).min
        )
    query = rearrange(query, "b s (h d) -> b s h d", h=n_heads)
    key = rearrange(key, "b s (h d) -> b s h d", h=1 if multiquery else n_heads)
    value = rearrange(value, "b s (h d) -> b s h d", h=1 if multiquery else n_heads)
    if multiquery:
        key = key.expand(*key.shape[:2], n_heads, key.size(-1))
        value = value.expand(*value.shape[:2], n_heads, value.size(-1))
    reset_is_causal = _reset_is_causal(query.size(1), key.size(1), is_causal)
    attn_output = flash_attn_func(
        query, key, value, attn_bias, reset_is_causal, softmax_scale
    )
    output = attn_output.view(*attn_output.shape[:2], -1)
    return (output, None, past_key_value)

class MultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attn_impl:str = "triton",
        clip_qkv:Optional[float] = None,
        qk_ln: bool = False,
        softmax_scale: Optional[float] = None,
        attn_pdrop: float = 0.0,
        low_precision_layernorm: bool = False,
        verbose: int = 0,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.attn_impl = attn_impl
        self.clip_qkv = clip_qkv
        self.qk_ln = qk_ln
        self.d_model = d_model
        self.n_heads = n_heads
        self.softmax_scale = softmax_scale
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.d_model / self.n_heads)
        self.attn_dropout_p = attn_pdrop
        self.Wqkv = nn.Linear(self.d_model,3*self.d_model,device=device)    #x -> [Q | K | V]
        fuse_splits = (d_model,2*d_model)
        self.Wqkv._fused = (0,fuse_splits)
        if self.qk_ln:
            layernorm_class = LPLayerNorm if low_precision_layernorm else nn.LayerNorm
            self.q_ln = layernorm_class(self.d_model,device=device)
            self.k_ln = layernorm_class(self.d_model,device=device)
        if self.attn_impl == "Triton":
            self.attn_fn = triton_flash_attn_fn
        elif self.attn_impl == "torch":
            self.attn_fn = scaled_multihead_dot_product_attention
        else:
            raise ValueError(f"attn_impl={attn_impl!r} is an invalid setting.")
        self.out_proj = nn.Linear(self.d_model,self.d_model,device=device)
        self.out_proj._is_residual = True
    def forward(
        self,
        x,
        past_key_value=None,
        attn_bias=None,
        attention_mask=None,
        is_causal=True,
        needs_weights=False,
    ):
        qkv = self.Wqkv(x)
        if self.clip_qkv:
            qkv.clamp_(min=-self.clip_qkv,max=self.clip_qkv)
        (query,key,value) = qkv.chunk(3,dim=2)
        key_padding_mask = attention_mask
        if self.qk_ln:
            dtype = query.dtype
            query = self.q_ln(query).to(dtype)
            key = self.k_ln(key).to(dtype)
        (context,attn_weights,past_key_value) = self.attn_fn(
            query,
            key,
            value,
            self.n_heads,
            past_key_value=past_key_value,
            softmax_scale=self.softmax_scale,
            attn_bias=attn_bias,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            dropout_p=self.attn_dropout_p,
            training=self.training,
            needs_weights=needs_weights,
        )
        return (self.out_proj(context),attn_weights,past_key_value)
