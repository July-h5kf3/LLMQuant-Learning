import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel,PreTrainedTokenizer,PreTrainedTokenizerFast

from .configuration_mpt import MPTConfig
from .norm import NORM_CLASS_REGISTRY
from .attention import attn_bias_shape,build_attn_bias
from .blocks import MPTBlock
from .custom_embedding import SharedEmbedding
from .param_init_fns import MODEL_INIT_REGISTRY,generic_param_init_fn_

class MPTPreTrainedModel(PreTrainedModel):
    config_class = MPTConfig
    base_model_prefix = "model"
    _no_split_modules = ["MPTBlock"]

class MPTModel(MPTPreTrainedModel):
    def __init__(self,config: MPTConfig):
        config._validate_config()
        super().__init__(config)
        self.attn_impl = config.attn_config["attn_impl"]
        self.prefix_lm = config.attn_config["prefix_lm"]
        self.attn_uses_sequence_id = config.attn_config["attn_uses_sequence_id"]
        self.alibi = config.attn_config["alibi"]
        self.alibi_bias_max = config.attn_config["alibi_bias_max"]
        
        norm_class = NORM_CLASS_REGISTRY[config.norm_type.lower()]
        self.embedding_fraction = config.embedding_fraction
        self.wte = SharedEmbedding(
            config.vocab_size,config.d_model,device=config.init_device
        )
        self.wpe = torch.nn.Embedding(config.max_seq_len,config.d_model,device=config.init_device)
        self.emb_drop = nn.Dropout(config.emb_pdrop)
        self.blocks = nn.ModuleList(
            [
                MPTBlock(device=config.init_device,**config.to_dict())
                for _ in range(config.n_layers)
            ]
        )
        self.norm_f = norm_class(config.d_model,device = config.init_device)
        print(
                f'You are using config.init_device={config.init_device!r}, but you can also use config.init_device="meta" with Composer + FSDP for fast initialization.'
            )
        self.apply(self.param_init_fn)
        self.is_causal = not self.prefix_lm
        self._attn_bias_initialized = False
        self.attn_bias = None
        self.attn_bias_shape = attn_bias_shape(
            self.attn_impl,
            config.n_heads,
            config.max_seq_len,
            self.alibi,
            prefix_lm=self.prefix_lm,
            causal=self.is_causal,
            use_sequence_id=self.attn_uses_sequence_id,
        )
        if config.verbose and config.verbose > 2:
            print(self)
        if "verbose" not in self.config.init_config:
            self.config.init_config["verbose"] = self.config.verbose
        if self.config.init_config["verbose"] > 1:
            init_fn_name = self.config.init_config["name"]
            warnings.warn(f"Using {init_fn_name} initialization.")
        self.gradient_checkpointing = False
    def get_input_embeddings(self):
        return self.wte
    def set_input_embeddings(self, value):
        self.wte = value