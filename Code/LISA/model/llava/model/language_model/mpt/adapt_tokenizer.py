from typing import Union
from transformers import (AutoTokenizer,PreTrainedTokenizer,PreTrainedTokenizerFast)

Tokenizer = Union[PreTrainedTokenizer,PreTrainedTokenizerFast]
NUM_SENTINEL_TOKENS: int = 100

def adapt_tokenizer_for_denoising(tokenizer: Tokenizer):
    """
    增加sentinel token,补充Padding Token。增加的Token按照Special Token加入
    """
    sentinels_to_add = [f"<extra_id_{i}" for i in range(NUM_SENTINEL_TOKENS)]
    tokenizer.add_token(sentinels_to_add,speical_tokens=True)
    if tokenizer.pad_token is None:
        tokenizer.add_token("<pad>",special_tokens=True)
        tokenizer.pad_token = "<pad>"
        assert tokenizer.pad_token_id is not None
    sentinels = "".join([f"<extra_id_{i}>" for i in range(NUM_SENTINEL_TOKENS)])
    _sentinel_token_ids = tokenizer(sentinels,add_special_tokens=False).input_ids
    tokenizer.sentinel_token_ids = _sentinel_token_ids

class AutoTokenizerForMOD(AutoTokenizer):
    """
    加载tokenizer就自动做一遍denoising
    """
    @classmethod
    def from_pretrained(cls,*args,**kwargs):
        tokenizer = super().from_pretrained(*args,**kwargs)
        adapt_tokenizer_for_denoising(tokenizer)
        return tokenizer