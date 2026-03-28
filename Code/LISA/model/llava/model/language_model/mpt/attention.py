import math
import torch
import warnings
from einops import rearrange
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

def flash_attn_fn(
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
    try:
        from flash_attn import bert_padding,flash_attn_interface
    except:
        raise RuntimeError("Please Install the flash-attn!")
    check_valid_inputs(query,key,value)
    if past_key_value is not None:
        if len(past_key_value) != 0:
            key = torch.cat([past_key_value[0],key],dim=1)
            value = torch.cat([past_key_value[1],value],dim=1)
        past_key_value = (key,value)
    
    if attn_bias is not None:
        raise NotImplementedError(f"attn_bias not implemented for flash attn.")
    
    (batch_size,seq_len) = query.shape[:2]
    
    if key_padding_mask is None:
        key_padding_mask = torch.ones_like(key[:,:,0],dtype=torch.bool)
    
    query_padding_mask = key_padding_mask[:,-query.size(1):]
    (query_unpad,indices_q,cu_seqlens_q,max_seqlen_q) = bert_padding.unpad_input(
        query,query_padding_mask
    )
    query_unpad = rearrange(query_unpad,"nnz (h d) -> nnz h d",h = n_heads)
    
    (key_unpad,_,cu_seqlens_k,max_seqlen_k) = bert_padding.unpad_input(
        key,key_padding_mask
    )
    key_unpad = rearrange(key_unpad,"nnz (h d) -> nnz h d",h = 1 if multiquery else n_heads)

    (value_unpad, _, _, _) = bert_padding.unpad_input(value, key_padding_mask)
    value_unpad = rearrange(
        value_unpad, "nnz (h d) -> nnz h d", h=1 if multiquery else n_heads
    )

    if multiquery:
        key_unpad = key_unpad.expand(key_unpad.size(0), n_heads, key_unpad.size(-1))
        value_unpad = value_unpad.expand(
            value_unpad.size(0), n_heads, value_unpad.size(-1)
        )
    dropout_p = dropout_p if training else 0.0
    reset_is_causal = _reset_is_causal(query.size(1),key.size(1),is_causal)
    output_unpad = flash_attn_interface.flash_attn_unpadded_func(
        query_unpad,
        key_unpad,
        value_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale = softmax_scale,
        causal = reset_is_causal,
        return_attn_probs = needs_weights,
    )
    output = bert_padding.pad_input(
        rearrange(output_unpad,"nnz h d -> nnz (h d)"),indices_q,batch_size,seq_len
    )
    return (output,None,past_key_value)
    
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
    try:
        from .flash_attn_triton import flash_attn_func
    except ImportError:
        try:
            from flash_attn.flash_attn_triton import flash_attn_func
        except ImportError as exc:
            raise RuntimeError(
                "Requirements for `attn_impl: triton` not installed. "
                "Install a CUDA-compatible Triton stack, typically via "
                "`pip install triton` plus your chosen flash-attn package, "
                "or switch to the torch/flash attention implementation."
            ) from exc

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
