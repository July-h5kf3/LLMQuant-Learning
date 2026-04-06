#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import gc
import json
import os
from typing import List, Optional, Tuple, Union

import bitsandbytes as bnb
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoConfig, AutoModelForCausalLM
try:
    from transformers.generation import GenerationMixin
except ImportError:
    from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
try:
    from transformers.initialization import no_init_weights
except ImportError:
    from transformers.modeling_utils import no_init_weights
from transformers.quantizers.quantizers_utils import should_convert_module

from model.compat_transformers_431 import (
    LlamaConfig,
    LlamaForCausalLM,
    LlamaModel,
)
from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

from torch.nn import CrossEntropyLoss
import copy

class LlavaConfig(LlamaConfig):
    model_type = "lisa_llava"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM, GenerationMixin):
    config_class = LlavaConfig

    _quantization_skip_modules = [
        "lm_head",
        "model.visual_model",
        "model.text_hidden_fcs",
        "model.mm_projector",
    ]

    @staticmethod
    def _requests_quantized_loading(kwargs):
        quantization_config = kwargs.get("quantization_config")
        return bool(
            kwargs.get("load_in_4bit")
            or kwargs.get("load_in_8bit")
            or quantization_config is not None
        )

    @classmethod
    def _coerce_config(cls, config):
        if config is None or isinstance(config, cls.config_class):
            return config

        config_dict = config.to_dict()
        if config_dict.get("model_type") == "llava":
            config_dict["model_type"] = cls.config_class.model_type
        return cls.config_class.from_dict(config_dict)

    @classmethod
    def _load_compat_config(cls, model_path):
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            return cls.config_class.from_pretrained(model_path)

        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        if config_dict.get("model_type") == "llava":
            config_dict["model_type"] = cls.config_class.model_type
        return cls.config_class.from_dict(config_dict)

    @staticmethod
    def _resolve_torch_dtype(config, torch_dtype):
        if torch_dtype in (None, "auto"):
            config_dtype = getattr(config, "torch_dtype", None)
            if isinstance(config_dtype, str) and hasattr(torch, config_dtype):
                return getattr(torch, config_dtype)
            return None
        if isinstance(torch_dtype, str) and hasattr(torch, torch_dtype):
            return getattr(torch, torch_dtype)
        return torch_dtype

    @staticmethod
    def _resolve_target_device(device_map):
        if device_map is None:
            return None
        target = device_map.get("") if isinstance(device_map, dict) else device_map
        if target in (None, "auto"):
            return None
        if isinstance(target, int):
            return f"cuda:{target}"
        return str(target)

    @staticmethod
    def _iter_checkpoint_files(model_path):
        index_path = os.path.join(model_path, "pytorch_model.bin.index.json")
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            seen = set()
            for filename in index["weight_map"].values():
                if filename in seen:
                    continue
                seen.add(filename)
                yield os.path.join(model_path, filename)
            return

        single_bin = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(single_bin):
            yield single_bin
            return

        raise FileNotFoundError(f"Could not find PyTorch checkpoint under {model_path}")

    @classmethod
    def _replace_with_bnb_layers(
        cls, module, quantization_config, prefix=""
    ):
        for child_name, child in list(module.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name

            if isinstance(child, nn.Linear) and should_convert_module(
                full_name, cls._quantization_skip_modules
            ):
                bias = child.bias is not None
                if quantization_config.load_in_8bit:
                    new_module = bnb.nn.Linear8bitLt(
                        child.in_features,
                        child.out_features,
                        bias=bias,
                        has_fp16_weights=quantization_config.llm_int8_has_fp16_weight,
                        threshold=quantization_config.llm_int8_threshold,
                    )
                    new_module.weight = bnb.nn.Int8Params(
                        child.weight.detach().cpu(),
                        requires_grad=False,
                        has_fp16_weights=quantization_config.llm_int8_has_fp16_weight,
                    )
                else:
                    new_module = bnb.nn.Linear4bit(
                        child.in_features,
                        child.out_features,
                        bias=bias,
                        compute_dtype=quantization_config.bnb_4bit_compute_dtype,
                        compress_statistics=quantization_config.bnb_4bit_use_double_quant,
                        quant_type=quantization_config.bnb_4bit_quant_type,
                        quant_storage=quantization_config.bnb_4bit_quant_storage,
                    )
                    new_module.weight = bnb.nn.Params4bit(
                        child.weight.detach().cpu(),
                        requires_grad=False,
                        compress_statistics=quantization_config.bnb_4bit_use_double_quant,
                        quant_type=quantization_config.bnb_4bit_quant_type,
                        quant_storage=quantization_config.bnb_4bit_quant_storage,
                        module=new_module,
                    )

                if bias:
                    new_module.bias = nn.Parameter(
                        child.bias.detach().cpu(), requires_grad=False
                    )
                new_module.source_cls = type(child)
                new_module.requires_grad_(False)
                setattr(module, child_name, new_module)
                continue

            cls._replace_with_bnb_layers(child, quantization_config, full_name)

    @classmethod
    def _apply_legacy_quantization(cls, model, quantization_config):
        cls._replace_with_bnb_layers(model, quantization_config)

        if quantization_config.load_in_4bit:
            model.is_loaded_in_4bit = True
            model.is_4bit_serializable = True
        if quantization_config.load_in_8bit:
            model.is_loaded_in_8bit = True
            model.is_8bit_serializable = True
        return model

    @classmethod
    def _legacy_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = cls._coerce_config(kwargs.pop("config", None))
        torch_dtype = kwargs.pop("torch_dtype", None)
        device_map = kwargs.pop("device_map", None)
        kwargs.pop("low_cpu_mem_usage", None)
        load_in_4bit = kwargs.pop("load_in_4bit", False)
        load_in_8bit = kwargs.pop("load_in_8bit", False)
        quantization_config = kwargs.pop("quantization_config", None)

        if config is None:
            config = cls._load_compat_config(pretrained_model_name_or_path)

        resolved_dtype = cls._resolve_torch_dtype(config, torch_dtype)
        init_dtype = (
            resolved_dtype
            if resolved_dtype in (torch.float16, torch.bfloat16, torch.float32)
            else None
        )

        with no_init_weights():
            if init_dtype is None:
                model = cls(config, *model_args, **kwargs)
            else:
                original_dtype = torch.get_default_dtype()
                torch.set_default_dtype(init_dtype)
                try:
                    model = cls(config, *model_args, **kwargs)
                finally:
                    torch.set_default_dtype(original_dtype)

        quantized_loading = quantization_config is not None or load_in_4bit or load_in_8bit

        if quantized_loading:
            if quantization_config is None:
                raise ValueError("quantization_config is required for legacy quantized loading.")

        for checkpoint_file in cls._iter_checkpoint_files(pretrained_model_name_or_path):
            shard_state = torch.load(checkpoint_file, map_location="cpu")
            model.load_state_dict(shard_state, strict=False)
            del shard_state
            gc.collect()

        if quantized_loading:
            model = cls._apply_legacy_quantization(model, quantization_config)

        if (
            resolved_dtype is not None
            and resolved_dtype != init_dtype
            and not quantized_loading
        ):
            model = model.to(dtype=resolved_dtype)
        target_device = cls._resolve_target_device(device_map)
        if target_device is not None:
            model = model.to(device=target_device)
        return model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        version_parts = transformers.__version__.split(".")[:2]
        transformers_version = tuple(int(part) for part in version_parts)
        if transformers_version > (4, 31):
            return cls._legacy_from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs
            )
        return super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    # def process_embeds(self, input_ids, inputs_embeds, correct_emb):
    #     # 1. 检查inputs_embeds和raw_input_ids第二个维度的差值是否为575
    #     assert inputs_embeds.shape[1] - input_ids.shape[1] == 575
        
    #     # 2. 检查raw_input_ids中的32000的数量是否与correct_emb的形状匹配
    #     count_32000 = (input_ids == 32000).sum(dim=1).tolist()
    #     assert all(count % 2 == 0 for count in count_32000)
        
    #     # for i, count in enumerate(count_32000):
    #     #     assert count == len(correct_emb[i]) * 2
        
    #     # 3. 找到32000的位置，将前1/2的位置替换为correct_emb中的对应值
    #     for batch_idx in range(input_ids.shape[0]):
    #         indices = torch.nonzero(input_ids[batch_idx] == 32000).flatten().tolist()
    #         num_32000 = len(indices)
            
    #         half_num_32000 = num_32000 // 2
    #         for idx, correct in zip(indices[:half_num_32000], correct_emb[batch_idx]):
    #             inputs_embeds[batch_idx, idx + 575, :] = correct
        
    #     return inputs_embeds
    def process_embeds(self, input_ids, inputs_embeds, correct_emb):
        # 1. 检查inputs_embeds和raw_input_ids第二个维度的差值是否为575
        assert inputs_embeds.shape[1] - input_ids.shape[1] == 575
        
        # 2. 检查raw_input_ids中的32000的数量是否与correct_emb的形状匹配
        count_32000 = (input_ids == 32000).sum(dim=1).tolist()
        assert all(count % 2 == 0 for count in count_32000)
        
        # for i, count in enumerate(count_32000):
        #     assert count == len(correct_emb[i]) * 2
        
        # 3. 找到32000的位置，将前1/2的位置替换为correct_emb中的对应值
        half_correct_num = 0
        for batch_idx in range(input_ids.shape[0]):
            indices = torch.nonzero(input_ids[batch_idx] == 32000).flatten().tolist()
            num_32000 = len(indices)
            
            half_num_32000 = num_32000 // 2
            for idx, correct in zip(indices[:half_num_32000], correct_emb[half_correct_num : half_correct_num + half_num_32000]):
                inputs_embeds[batch_idx, idx + 575, :] = correct
            half_correct_num += half_num_32000
        assert half_correct_num*2 == sum(count_32000)
        return inputs_embeds

    def process_embeds_inference(self, input_ids, inputs_embeds, correct_emb):
        # 1. 检查inputs_embeds和raw_input_ids第二个维度的差值是否为575
        assert inputs_embeds.shape[1] - input_ids.shape[1] == 575
                
        # 3. 找到32000的位置，将前1/2的位置替换为correct_emb中的对应值
        for batch_idx in range(input_ids.shape[0]):
            indices = torch.nonzero(input_ids[batch_idx] == 32000).flatten().tolist()
            indices = indices[:len(correct_emb)]
            
            for idx, correct in zip(indices, correct_emb):
                inputs_embeds[batch_idx, idx + 575, :] = correct
        return inputs_embeds
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        correct_emb: Optional[List[torch.FloatTensor]] = None,
        return_dict: Optional[bool] = None,
        if_inference: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        if correct_emb is not None:
            raw_input_ids  = copy.deepcopy(input_ids)

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if correct_emb is not None and if_inference is None:
            inputs_embeds = self.process_embeds(raw_input_ids, inputs_embeds, correct_emb)
        elif correct_emb is not None and if_inference is not None:
            inputs_embeds = self.process_embeds_inference(raw_input_ids, inputs_embeds, correct_emb)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        if self.training:
            output_hidden_states = outputs.hidden_states
        else:
            output_hidden_states = hidden_states

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=output_hidden_states, #outputs.hidden_states,
            attentions=outputs.attentions,
        )


        # return super().forward(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     past_key_values=past_key_values,
        #     inputs_embeds=inputs_embeds,
        #     labels=labels,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict
        # )

    # def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
    #     images = kwargs.pop("images", None)
    #     _inputs = super().prepare_inputs_for_generation(
    #         input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
    #     )
    #     if images is not None:
    #         _inputs['images'] = images
    #     return _inputs

    # def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, images=None, **kwargs):
    #     # images = kwargs.pop("images", None)
    #     _inputs = super().prepare_inputs_for_generation(
    #         input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
    #     )
    #     if images is not None:
    #         _inputs['images'] = images
    #     return _inputs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        images=None,
        correct_emb=None,
        if_inference=None,
        **kwargs
    ):
        past_key_values=None
        if past_key_values:
            input_ids = input_ids[:, -1:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": images,
                "correct_emb":correct_emb,
                "if_inference":if_inference,
            }
        )
        return model_inputs

try:
    AutoConfig.register("lisa_llava", LlavaConfig)
except ValueError:
    pass
try:
    AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
except ValueError:
    pass
