import torch
import torch.nn as nn

class LlavaConfig(LlamaConfig):
    model_type = "llava"

class LlavaLlamaModel(LlavaMetaModel,LlamaModel):
    config_class = LlavaConfig

    def __init__(self,config:LlamaModel):
        super(LlavaLlamaModel,self).__init__(config)