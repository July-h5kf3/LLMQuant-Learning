import json
import os
import shutil
from pathlib import Path

import torch
import yaml
from transformers import AwqConfig, BitsAndBytesConfig

from utils.tokenizer_compat import add_lisa_seg_token, load_lisa_tokenizer

WEIGHT_FILES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)

QUANTIZED_LINEAR_NAMES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def validate_quantization_config(quant_method, quant_kwargs=None):
    if quant_kwargs is None:
        quant_kwargs = {}

    supported_methods = {
        "none",
        "bnb_8bit",
        "bnb_4bit",
        "awq",
        "gptq",
        "hqq",
        "quanto",
        "smoothquant",
    }
    if quant_method not in supported_methods:
        raise ValueError(
            f"Unsupported quant_method: {quant_method}. "
            f"Supported values: {sorted(supported_methods)}"
        )

    if not isinstance(quant_kwargs, dict):
        raise ValueError("quant_kwargs must be a mapping.")


def load_quant_config(config_path, base_dir=None):
    if not config_path:
        return {}

    resolved_path = config_path
    if not os.path.isabs(resolved_path) and base_dir is not None:
        resolved_path = os.path.join(base_dir, resolved_path)

    with open(resolved_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise ValueError("Quantization config YAML must be a mapping.")
    return config


def resolve_path(path_value, *, base_dir):
    path = Path(path_value)
    if not path.is_absolute():
        path = Path(base_dir) / path
    return path.resolve()


def load_method_quant_config(config_path, *, base_dir, path_keys=(), bool_defaults=None):
    config_path = resolve_path(config_path, base_dir=base_dir)
    config = load_quant_config(str(config_path))
    config_dir = config_path.parent
    config["_config_path"] = str(config_path)

    for key in path_keys:
        if key in config and config[key]:
            config[key] = str(resolve_path(config[key], base_dir=config_dir))

    for key, default in (bool_defaults or {}).items():
        config[key] = bool(config.get(key, default))

    return config


def patch_transformers_compat():
    import transformers
    import transformers.modeling_utils as modeling_utils

    if not hasattr(modeling_utils, "no_init_weights"):
        try:
            from transformers.initialization import no_init_weights
        except ImportError:
            from transformers.modeling_utils import no_init_weights

        modeling_utils.no_init_weights = no_init_weights

    if not hasattr(transformers, "AutoModelForVision2Seq") and hasattr(
        transformers, "AutoModelForImageTextToText"
    ):
        transformers.AutoModelForVision2Seq = transformers.AutoModelForImageTextToText

    try:
        from awq.models.base import TRANSFORMERS_AUTO_MAPPING_DICT
    except ImportError:
        return

    if TRANSFORMERS_AUTO_MAPPING_DICT.get("llava") == "AutoModelForVision2Seq":
        TRANSFORMERS_AUTO_MAPPING_DICT["llava"] = "AutoModelForCausalLM"


def build_lisa_tokenizer(model_path, model_max_length):
    tokenizer = load_lisa_tokenizer(
        model_path,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    add_lisa_seg_token(tokenizer)
    return tokenizer


def checkpoint_exists(output_dir, *required_files):
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return False

    return all((output_dir / name).exists() for name in required_files) and any(
        (output_dir / name).exists() for name in WEIGHT_FILES
    )


def reset_dir(path, force=False):
    path = Path(path)
    if force and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def remove_dir(path):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def link_or_copy_file(src, dst):
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)


def build_quant_artifact_dirs(output_dir):
    output_dir = Path(output_dir)
    return (
        output_dir.parent / f"{output_dir.name}_lm_backbone",
        output_dir.parent / f"{output_dir.name}.tmp",
    )


def finalize_quantized_output(temp_output_dir, output_dir):
    temp_output_dir = Path(temp_output_dir)
    output_dir = Path(output_dir)
    remove_dir(output_dir)
    temp_output_dir.replace(output_dir)
    return output_dir


def cleanup_export_dir(export_dir, keep_export=False):
    if not keep_export:
        remove_dir(export_dir)


def is_lisa_lm_weight(weight_name):
    return weight_name == "lm_head.weight" or weight_name.startswith(
        ("model.embed_tokens.", "model.layers.", "model.norm.")
    )


def estimate_lm_weight_bytes(model_path, lm_weight_map):
    shard_to_names = {}
    for weight_name, shard_name in lm_weight_map.items():
        shard_to_names.setdefault(shard_name, []).append(weight_name)

    total_size = 0
    for shard_name, weight_names in shard_to_names.items():
        shard_state = torch.load(Path(model_path) / shard_name, map_location="cpu")
        for weight_name in weight_names:
            tensor = shard_state[weight_name]
            total_size += tensor.numel() * tensor.element_size()
        del shard_state
    return total_size


def can_materialize_filtered_shards(
    export_dir,
    required_bytes,
    safety_margin_bytes=2 * 1024**3,
):
    free_bytes = shutil.disk_usage(export_dir).free
    return free_bytes >= required_bytes + safety_margin_bytes


def export_lisa_lm_backbone(model_path, export_dir, force=False):
    model_path = Path(model_path)
    export_dir = reset_dir(export_dir, force=force)

    if checkpoint_exists(export_dir, "config.json") and not force:
        return export_dir

    index_path = model_path / "pytorch_model.bin.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing checkpoint index: {index_path}")

    with open(index_path, "r", encoding="utf-8") as f:
        source_index = json.load(f)

    weight_map = source_index["weight_map"]
    lm_weight_map = {
        name: shard_name for name, shard_name in weight_map.items() if is_lisa_lm_weight(name)
    }
    if not lm_weight_map:
        raise ValueError("No LLM backbone weights were found in the LISA checkpoint.")

    shard_filenames = list(dict.fromkeys(lm_weight_map.values()))
    lm_total_size = estimate_lm_weight_bytes(model_path, lm_weight_map)
    storage_mode = "filtered_shards"

    if can_materialize_filtered_shards(export_dir, lm_total_size):
        for shard_name in shard_filenames:
            shard_path = model_path / shard_name
            if not shard_path.exists():
                raise FileNotFoundError(f"Missing checkpoint shard: {shard_path}")

            shard_state = torch.load(shard_path, map_location="cpu")
            filtered_state = {
                weight_name: shard_state[weight_name]
                for weight_name, weight_shard in lm_weight_map.items()
                if weight_shard == shard_name
            }
            torch.save(
                filtered_state,
                export_dir / shard_name,
                _use_new_zipfile_serialization=False,
            )
            del shard_state
    else:
        storage_mode = "hardlink_or_symlink_full_shards"
        for shard_name in shard_filenames:
            shard_path = model_path / shard_name
            if not shard_path.exists():
                raise FileNotFoundError(f"Missing checkpoint shard: {shard_path}")
            link_or_copy_file(shard_path, export_dir / shard_name)

    with open(model_path / "config.json", "r", encoding="utf-8") as f:
        source_config = json.load(f)
    source_config.update(
        {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "torch_dtype": "float16",
        }
    )
    write_json(export_dir / "config.json", source_config)

    generation_config_path = model_path / "generation_config.json"
    if generation_config_path.exists():
        shutil.copy2(generation_config_path, export_dir / "generation_config.json")

    exported_index = {
        "metadata": {"total_size": lm_total_size},
        "weight_map": lm_weight_map,
    }
    write_json(export_dir / "pytorch_model.bin.index.json", exported_index)

    export_meta = {
        "source_model_path": str(model_path),
        "exported_model_path": str(export_dir),
        "export_type": "lisa_lm_backbone",
        "num_weights": len(lm_weight_map),
        "num_shards": len(shard_filenames),
        "storage_mode": storage_mode,
        "estimated_lm_bytes": lm_total_size,
    }
    write_json(export_dir / "lisa_lm_export_meta.json", export_meta)

    return export_dir


def collect_calibration_records(loader):
    records = []

    for batch in loader:
        batch_size = len(batch["conversation_list"])
        for idx in range(batch_size):
            input_ids = batch["input_ids"][idx]
            attention_mask = batch["attention_masks"][idx]
            prompt = batch["conversation_list"][idx]

            seq_len = int(attention_mask.sum().item())
            records.append(
                {
                    "prompt": prompt,
                    "input_ids": input_ids[:seq_len].tolist(),
                    "attention_mask": attention_mask[:seq_len].tolist(),
                }
            )

    return records


def build_lisa_model(model_path, config, tokenizer):
    from model.LISA import LISAForCausalLM

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = LISAForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        train_mask_decoder=False,
        out_dim=256,
        ce_loss_weight=1.0,
        dice_loss_weight=0.5,
        bce_loss_weight=2.0,
        seg_token_idx=tokenizer.convert_tokens_to_ids("[SEG]"),
        vision_pretrained=config.get("vision_pretrained"),
        vision_tower=config["vision_tower"],
        use_mm_start_end=config["use_mm_start_end"],
    )
    vision_tower = model.get_vision_tower()
    if vision_tower is not None and getattr(vision_tower, "is_loaded", 1) is False:
        vision_tower.load_model()
    model.eval()
    return model


def expand_images_clip(images_clip, offset):
    if offset is None:
        return images_clip

    total_prompts = int(offset[-1].item())
    if images_clip.shape[0] == total_prompts:
        return images_clip

    expanded = []
    for image_idx in range(len(offset) - 1):
        start = int(offset[image_idx].item())
        end = int(offset[image_idx + 1].item())
        repeat_count = end - start
        if repeat_count <= 0:
            continue
        expanded.append(
            images_clip[image_idx : image_idx + 1]
            .expand(repeat_count, -1, -1, -1)
            .contiguous()
        )

    if not expanded:
        raise ValueError("Failed to expand multimodal calibration images.")

    return torch.cat(expanded, dim=0)


def pad_hidden_states(hidden_states, target_len):
    if hidden_states.shape[1] == target_len:
        return hidden_states

    pad_len = target_len - hidden_states.shape[1]
    return torch.cat(
        [
            hidden_states,
            hidden_states.new_zeros(hidden_states.shape[0], pad_len, hidden_states.shape[2]),
        ],
        dim=1,
    )


def pad_position_ids(position_ids, target_len):
    if position_ids is None or position_ids.shape[1] == target_len:
        return position_ids

    pad_len = target_len - position_ids.shape[1]
    return torch.cat(
        [
            position_ids,
            position_ids.new_zeros(position_ids.shape[0], pad_len),
        ],
        dim=1,
    )


def pad_attention_mask(attention_mask, target_len, *, error_context="multimodal calibration"):
    if attention_mask is None:
        return None

    if attention_mask.ndim == 2 and attention_mask.shape[1] == target_len:
        return attention_mask
    if attention_mask.ndim == 3 and attention_mask.shape[2] == target_len:
        return attention_mask
    if (
        attention_mask.ndim == 4
        and attention_mask.shape[2] == target_len
        and attention_mask.shape[3] == target_len
    ):
        return attention_mask

    fill_value = False if attention_mask.dtype == torch.bool else torch.finfo(attention_mask.dtype).min

    if attention_mask.ndim == 2:
        padded = attention_mask.new_full((attention_mask.shape[0], target_len), fill_value)
        padded[:, : attention_mask.shape[1]] = attention_mask
        return padded

    if attention_mask.ndim == 3:
        padded = attention_mask.new_full(
            (attention_mask.shape[0], attention_mask.shape[1], target_len),
            fill_value,
        )
        padded[:, :, : attention_mask.shape[2]] = attention_mask
        return padded

    if attention_mask.ndim == 4:
        padded = attention_mask.new_full(
            (
                attention_mask.shape[0],
                attention_mask.shape[1],
                target_len,
                target_len,
            ),
            fill_value,
        )
        padded[:, :, : attention_mask.shape[2], : attention_mask.shape[3]] = attention_mask
        return padded

    raise ValueError(
        f"Unsupported attention_mask rank for {error_context}: {attention_mask.ndim}"
    )


def capture_multimodal_layer_inputs(model, batch, device):
    import torch.nn as nn

    from model.llava1p5.model.language_model.llava_llama import LlavaLlamaForCausalLM

    first_layer = model.model.layers[0]
    captured = {}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            hidden_states = args[0] if args else kwargs["hidden_states"]
            captured["hidden_states"] = hidden_states.detach().cpu()

            attention_mask = kwargs.get("attention_mask")
            captured["attention_mask"] = (
                attention_mask.detach().cpu() if attention_mask is not None else None
            )

            position_ids = kwargs.get("position_ids")
            captured["position_ids"] = (
                position_ids.detach().cpu() if position_ids is not None else None
            )

            raise ValueError("Captured multimodal decoder inputs.")

    model.model.layers[0] = Catcher(first_layer)

    try:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_masks"].to(device)
        vision_tower = model.get_vision_tower()
        vision_dtype = getattr(vision_tower, "dtype", next(model.parameters()).dtype)
        images_clip = expand_images_clip(batch["images_clip"], batch.get("offset")).to(
            device=device,
            dtype=vision_dtype,
        )

        with torch.inference_mode():
            LlavaLlamaForCausalLM.forward(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images_clip,
                use_cache=False,
            )
    except ValueError as exc:
        if str(exc) != "Captured multimodal decoder inputs.":
            raise
    finally:
        model.model.layers[0] = first_layer

    if "hidden_states" not in captured:
        raise RuntimeError("Failed to capture multimodal calibration activations.")

    return captured


def collect_multimodal_calibration_inputs(model_path, config, tokenizer, loader):
    model = build_lisa_model(model_path, config, tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    vision_tower = model.get_vision_tower()
    if vision_tower is not None:
        projector_dtype = next(model.get_model().mm_projector.parameters()).dtype
        vision_tower.to(device=device, dtype=projector_dtype)

    hidden_state_batches = []
    attention_mask_batches = []
    position_id_batches = []
    max_seq_len = 0

    try:
        for batch in loader:
            captured = capture_multimodal_layer_inputs(model, batch, device)
            hidden_states = captured["hidden_states"]
            attention_mask = captured["attention_mask"]
            position_ids = captured["position_ids"]

            if (
                position_ids is not None
                and position_ids.shape[0] == 1
                and hidden_states.shape[0] > 1
            ):
                position_ids = position_ids.expand(hidden_states.shape[0], -1).contiguous()

            hidden_state_batches.append(hidden_states)
            attention_mask_batches.append(attention_mask)
            position_id_batches.append(position_ids)
            max_seq_len = max(max_seq_len, hidden_states.shape[1])

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not hidden_state_batches:
        raise ValueError("Multimodal calibration activations are empty.")

    inputs_embeds = torch.cat(
        [pad_hidden_states(hidden_states, max_seq_len) for hidden_states in hidden_state_batches],
        dim=0,
    )

    if all(mask is None for mask in attention_mask_batches):
        attention_mask = None
    else:
        attention_mask = torch.cat(
            [
                pad_attention_mask(mask, max_seq_len, error_context="multimodal calibration")
                for mask in attention_mask_batches
            ],
            dim=0,
        )

    if all(position_ids is None for position_ids in position_id_batches):
        position_ids = None
    else:
        position_ids = torch.cat(
            [pad_position_ids(position_ids, max_seq_len) for position_ids in position_id_batches],
            dim=0,
        )

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "max_seq_len": max_seq_len,
        "num_samples": inputs_embeds.shape[0],
    }


def pad_token_tensor(tensor, target_len, *, pad_value=0):
    if tensor is None or tensor.shape[1] == target_len:
        return tensor

    padded = tensor.new_full((tensor.shape[0], target_len), pad_value)
    padded[:, : tensor.shape[1]] = tensor
    return padded


def build_default_position_ids(batch_size, seq_len, *, device, dtype=torch.long):
    return (
        torch.arange(seq_len, dtype=dtype, device=device)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .contiguous()
    )


def build_position_ids_from_attention_mask(attention_mask, *, dtype=torch.long):
    if attention_mask is None:
        return None

    attention_mask = attention_mask.to(torch.bool)
    position_ids = attention_mask.to(dtype).cumsum(dim=-1) - 1
    position_ids.masked_fill_(~attention_mask, 0)
    return position_ids


def build_causal_decoder_attention_mask(attention_mask, *, dtype):
    if attention_mask is None:
        return None

    attention_mask = attention_mask.to(torch.bool)
    bsz, src_len = attention_mask.shape
    tgt_len = src_len
    device = attention_mask.device
    fill_value = torch.finfo(dtype).min

    causal_mask = torch.full((tgt_len, tgt_len), fill_value, dtype=dtype, device=device)
    mask_cond = torch.arange(tgt_len, device=device)
    causal_mask.masked_fill_(mask_cond < (mask_cond + 1).view(tgt_len, 1), 0)
    causal_mask = causal_mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)

    expanded_mask = attention_mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    expanded_mask = inverted_mask.masked_fill(inverted_mask.to(torch.bool), fill_value)
    return expanded_mask + causal_mask


def build_padded_layer_attention_mask(attention_mask, *, dtype):
    if attention_mask is None:
        return None

    attention_mask = attention_mask.to(torch.bool)
    bsz, seq_len = attention_mask.shape
    device = attention_mask.device
    fill_value = torch.finfo(dtype).min

    layer_attention_mask = torch.full(
        (bsz, 1, seq_len, seq_len),
        fill_value,
        dtype=dtype,
        device=device,
    )

    for sample_idx in range(bsz):
        valid_len = int(attention_mask[sample_idx].sum().item())
        if valid_len <= 0:
            continue

        sample_mask = torch.full(
            (valid_len, valid_len),
            fill_value,
            dtype=dtype,
            device=device,
        )
        mask_cond = torch.arange(valid_len, device=device)
        sample_mask.masked_fill_(mask_cond < (mask_cond + 1).view(valid_len, 1), 0)
        layer_attention_mask[sample_idx, 0, :valid_len, :valid_len] = sample_mask

    return layer_attention_mask


def build_vision_mask_from_prepared_inputs(
    input_ids,
    input_attention_mask,
    prepared_labels,
    prepared_attention_mask,
):
    from model.llava1p5.constants import IMAGE_TOKEN_INDEX

    if prepared_labels is None or prepared_attention_mask is None:
        return None

    vision_masks = []
    for batch_idx, pre_input_ids in enumerate(input_ids):
        post_attn_mask = prepared_attention_mask[batch_idx].bool()
        current_vision_mask = torch.zeros_like(post_attn_mask, dtype=torch.bool)

        num_images = int((pre_input_ids == IMAGE_TOKEN_INDEX).sum().item())
        if num_images <= 0:
            vision_masks.append(current_vision_mask)
            continue

        pre_len = int(input_attention_mask[batch_idx].bool().sum().item())
        post_len = int(post_attn_mask.sum().item())
        image_emb_len = int((post_len - pre_len + num_images) / num_images)

        image_emb_start = torch.where(pre_input_ids == IMAGE_TOKEN_INDEX)[0]
        image_emb_start = image_emb_start.clone()
        for image_idx in range(len(image_emb_start)):
            image_emb_start[image_idx] = image_emb_start[image_idx] + (image_emb_len - 1) * image_idx

        image_emb_end = image_emb_start + image_emb_len
        for image_start, image_end in zip(image_emb_start.tolist(), image_emb_end.tolist()):
            current_vision_mask[image_start:image_end] = True

        vision_masks.append(current_vision_mask)

    return torch.stack(vision_masks, dim=0)


@torch.no_grad()
def prepare_multimodal_prompt_inputs(model, batch, device):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_masks"].to(device)
    labels = batch["labels"].to(device)

    vision_tower = model.get_vision_tower()
    vision_dtype = getattr(vision_tower, "dtype", next(model.parameters()).dtype)
    images_clip = expand_images_clip(batch["images_clip"], batch.get("offset")).to(
        device=device,
        dtype=vision_dtype,
    )

    (
        _,
        position_ids,
        prepared_attention_mask,
        _,
        inputs_embeds,
        prepared_labels,
    ) = model.prepare_inputs_labels_for_multimodal(
        input_ids,
        None,
        attention_mask,
        None,
        labels,
        images_clip,
    )

    vision_mask = build_vision_mask_from_prepared_inputs(
        input_ids,
        attention_mask,
        prepared_labels,
        prepared_attention_mask,
    )
    caption_mask = (
        prepared_labels.ne(-100) if prepared_labels is not None else None
    )

    return {
        "inputs_embeds": inputs_embeds.detach().cpu(),
        "attention_mask": (
            prepared_attention_mask.detach().cpu()
            if prepared_attention_mask is not None
            else None
        ),
        "position_ids": position_ids.detach().cpu() if position_ids is not None else None,
        "labels": prepared_labels.detach().cpu() if prepared_labels is not None else None,
        "vision_mask": vision_mask.detach().cpu() if vision_mask is not None else None,
        "caption_mask": caption_mask.detach().cpu() if caption_mask is not None else None,
    }


def collect_mbq_multimodal_calibration_inputs(model_path, config, tokenizer, loader):
    model = build_lisa_model(model_path, config, tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    vision_tower = model.get_vision_tower()
    if vision_tower is not None:
        projector_dtype = next(model.get_model().mm_projector.parameters()).dtype
        vision_tower.to(device=device, dtype=projector_dtype)

    prompt_input_batches = []
    prompt_attention_batches = []
    prompt_position_batches = []
    prompt_label_batches = []
    prompt_vision_mask_batches = []
    prompt_caption_mask_batches = []
    max_seq_len = 0

    try:
        for batch in loader:
            prepared = prepare_multimodal_prompt_inputs(model, batch, device)
            inputs_embeds = prepared["inputs_embeds"]
            attention_mask = prepared["attention_mask"]
            position_ids = prepared["position_ids"]
            labels = prepared["labels"]
            vision_mask = prepared["vision_mask"]
            caption_mask = prepared["caption_mask"]

            prompt_input_batches.append(inputs_embeds)
            prompt_attention_batches.append(attention_mask)
            prompt_position_batches.append(position_ids)
            prompt_label_batches.append(labels)
            prompt_vision_mask_batches.append(vision_mask)
            prompt_caption_mask_batches.append(caption_mask)
            max_seq_len = max(max_seq_len, inputs_embeds.shape[1])

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not prompt_input_batches:
        raise ValueError("MBQ-style multimodal calibration activations are empty.")

    inputs_embeds = torch.cat(
        [pad_hidden_states(hidden_states, max_seq_len) for hidden_states in prompt_input_batches],
        dim=0,
    )

    attention_mask = torch.cat(
        [
            pad_token_tensor(mask, max_seq_len, pad_value=False)
            for mask in prompt_attention_batches
        ],
        dim=0,
    )

    if all(position_ids is None for position_ids in prompt_position_batches):
        position_ids = build_position_ids_from_attention_mask(attention_mask)
    else:
        position_ids = torch.cat(
            [
                pad_position_ids(position_ids, max_seq_len)
                for position_ids in prompt_position_batches
            ],
            dim=0,
        )

    labels = torch.cat(
        [
            pad_token_tensor(label, max_seq_len, pad_value=-100)
            for label in prompt_label_batches
        ],
        dim=0,
    )

    if all(mask is None for mask in prompt_vision_mask_batches):
        vision_mask = None
    else:
        vision_mask = torch.cat(
            [
                pad_token_tensor(mask, max_seq_len, pad_value=False)
                for mask in prompt_vision_mask_batches
            ],
            dim=0,
        )

    if all(mask is None for mask in prompt_caption_mask_batches):
        caption_mask = None
    else:
        caption_mask = torch.cat(
            [
                pad_token_tensor(mask, max_seq_len, pad_value=False)
                for mask in prompt_caption_mask_batches
            ],
            dim=0,
        )

    prompt_inputs = {"inputs_embeds": inputs_embeds}
    prompt_kwargs = {
        "attention_mask": attention_mask,
        "labels": labels,
    }
    if position_ids is not None:
        prompt_kwargs["position_ids"] = position_ids
    if vision_mask is not None:
        prompt_kwargs["vision_mask"] = vision_mask
    if caption_mask is not None:
        prompt_kwargs["caption_mask"] = caption_mask

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "labels": labels,
        "vision_mask": vision_mask,
        "caption_mask": caption_mask,
        "prompt_inputs": prompt_inputs,
        "prompt_kwargs": prompt_kwargs,
        "max_seq_len": max_seq_len,
        "num_samples": inputs_embeds.shape[0],
        "calibration_mode": "mbq",
    }


def build_awq_layer_calibration_inputs(multimodal_inputs):
    inputs_embeds = multimodal_inputs["inputs_embeds"]
    attention_mask = multimodal_inputs.get("attention_mask")
    position_ids = multimodal_inputs.get("position_ids")

    if attention_mask is not None and attention_mask.ndim == 2:
        layer_attention_mask = build_padded_layer_attention_mask(
            attention_mask,
            dtype=inputs_embeds.dtype,
        )
    else:
        layer_attention_mask = attention_mask

    if position_ids is None:
        position_ids = build_position_ids_from_attention_mask(attention_mask)

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": layer_attention_mask,
        "position_ids": position_ids,
        "max_seq_len": multimodal_inputs["max_seq_len"],
        "num_samples": multimodal_inputs["num_samples"],
    }


def build_calibration_data(model_path, config):
    from quantization.calibration_dataset import build_calibration_loader

    tokenizer = build_lisa_tokenizer(model_path, config["model_max_length"])
    loader = build_calibration_loader(
        base_image_dir=config["dataset_dir"],
        tokenizer=tokenizer,
        vision_tower=config["vision_tower"],
        reason_seg_data=config["calibration_dataset"],
        image_size=config["image_size"],
        max_samples=config["max_calib_samples"],
        questions_per_image=config["questions_per_image"],
        seed=config["seed"],
        conv_type=config["conv_type"],
        use_mm_start_end=config["use_mm_start_end"],
        batch_size=config["calibration_batch_size"],
        num_workers=config["calibration_num_workers"],
        shuffle=False,
    )
    calibration_records = collect_calibration_records(loader)
    calibration_mode = config.get("calibration_mode", "mbq")
    if calibration_mode == "legacy":
        multimodal_inputs = collect_multimodal_calibration_inputs(
            str(model_path), config, tokenizer, loader
        )
        multimodal_inputs["calibration_mode"] = "legacy"
    elif calibration_mode == "mbq":
        multimodal_inputs = collect_mbq_multimodal_calibration_inputs(
            str(model_path), config, tokenizer, loader
        )
    else:
        raise ValueError(
            f"Unsupported calibration_mode: {calibration_mode}. "
            "Supported values are 'mbq' and 'legacy'."
        )
    return tokenizer, calibration_records, multimodal_inputs


def infer_inputs_embeds_seq_len(inputs_embeds):
    non_padding = inputs_embeds.abs().sum(dim=-1) > 0
    if torch.any(non_padding):
        return int(non_padding.nonzero(as_tuple=False)[-1].item()) + 1
    return int(inputs_embeds.shape[0])


def slice_attention_mask_sample(attention_mask, sample_idx, seq_len):
    if attention_mask is None:
        return None

    sample_mask = attention_mask[sample_idx]
    if sample_mask.ndim == 1:
        return sample_mask[:seq_len].contiguous()
    if sample_mask.ndim == 2:
        if sample_mask.shape[-2] == sample_mask.shape[-1]:
            return sample_mask[:seq_len, :seq_len].contiguous()
        return sample_mask[..., :seq_len].contiguous()
    if sample_mask.ndim == 3:
        if sample_mask.shape[-2] == sample_mask.shape[-1]:
            return sample_mask[..., :seq_len, :seq_len].contiguous()
        return sample_mask[..., :seq_len].contiguous()

    raise ValueError(
        f"Unsupported attention_mask rank for multimodal GPTQ calibration: {sample_mask.ndim}"
    )


def split_multimodal_calibration_inputs(multimodal_inputs):
    inputs_embeds = multimodal_inputs["inputs_embeds"]
    attention_mask = multimodal_inputs["attention_mask"]
    position_ids = multimodal_inputs["position_ids"]

    calibration_records = []
    for sample_idx in range(inputs_embeds.shape[0]):
        sample_inputs_embeds = inputs_embeds[sample_idx]
        seq_len = infer_inputs_embeds_seq_len(sample_inputs_embeds)

        record = {
            "inputs_embeds": sample_inputs_embeds[:seq_len].contiguous(),
        }
        if attention_mask is not None:
            record["attention_mask"] = slice_attention_mask_sample(
                attention_mask, sample_idx, seq_len
            )
        if position_ids is not None:
            record["position_ids"] = position_ids[sample_idx, :seq_len].contiguous()
        calibration_records.append(record)

    if not calibration_records:
        raise ValueError("Multimodal GPTQ calibration activations are empty.")

    return calibration_records


def pad_sample_attention_mask(attention_mask, target_len):
    if attention_mask is None:
        return None

    attention_mask = attention_mask.unsqueeze(0)
    if attention_mask.ndim == 2 and attention_mask.shape[1] == target_len:
        return attention_mask.squeeze(0)
    if attention_mask.ndim == 3 and attention_mask.shape[2] == target_len:
        return attention_mask.squeeze(0)
    if (
        attention_mask.ndim == 4
        and attention_mask.shape[2] == target_len
        and attention_mask.shape[3] == target_len
    ):
        return attention_mask.squeeze(0)

    if attention_mask.dtype == torch.bool:
        fill_value = False
    elif attention_mask.dtype.is_floating_point:
        fill_value = torch.finfo(attention_mask.dtype).min
    else:
        fill_value = 0

    if attention_mask.ndim == 2:
        padded = attention_mask.new_full((attention_mask.shape[0], target_len), fill_value)
        padded[:, : attention_mask.shape[1]] = attention_mask
        return padded.squeeze(0)

    if attention_mask.ndim == 3:
        padded = attention_mask.new_full(
            (attention_mask.shape[0], attention_mask.shape[1], target_len),
            fill_value,
        )
        padded[:, :, : attention_mask.shape[2]] = attention_mask
        return padded.squeeze(0)

    if attention_mask.ndim == 4:
        padded = attention_mask.new_full(
            (
                attention_mask.shape[0],
                attention_mask.shape[1],
                target_len,
                target_len,
            ),
            fill_value,
        )
        padded[:, :, : attention_mask.shape[2], : attention_mask.shape[3]] = attention_mask
        return padded.squeeze(0)

    raise ValueError(
        f"Unsupported attention_mask rank for multimodal GPTQ calibration: {attention_mask.ndim - 1}"
    )


def prepare_multimodal_gptq_dataset(
    self,
    calibration_dataset,
    calibration_dataset_concat_size=None,
    batch_size=1,
    calibration_data_min_length=10,
):
    from quantization.calibration_dataset import MultimodalCalibrationExample

    del self
    del calibration_dataset_concat_size

    prepared_examples = []
    for record in calibration_dataset:
        seq_len = record["inputs_embeds"].shape[0]
        if seq_len <= calibration_data_min_length:
            continue
        prepared_examples.append(record)

    if not prepared_examples:
        raise ValueError("All multimodal GPTQ calibration samples are too short.")

    batched_examples = []
    for start in range(0, len(prepared_examples), batch_size):
        chunk = prepared_examples[start : start + batch_size]
        target_len = max(record["inputs_embeds"].shape[0] for record in chunk)

        batch_inputs_embeds = torch.stack(
            [
                pad_hidden_states(record["inputs_embeds"].unsqueeze(0), target_len).squeeze(0)
                for record in chunk
            ],
            dim=0,
        )

        if all(record.get("attention_mask") is None for record in chunk):
            batch_attention_mask = None
        else:
            batch_attention_mask = torch.stack(
                [pad_sample_attention_mask(record.get("attention_mask"), target_len) for record in chunk],
                dim=0,
            )

        if all(record.get("position_ids") is None for record in chunk):
            batch_position_ids = None
        else:
            batch_position_ids = torch.stack(
                [
                    pad_position_ids(record.get("position_ids").unsqueeze(0), target_len).squeeze(0)
                    for record in chunk
                ],
                dim=0,
            )

        fake_input_ids = torch.zeros((len(chunk), target_len), dtype=torch.long)
        batched_examples.append(
            MultimodalCalibrationExample(
                inputs_embeds=batch_inputs_embeds,
                fake_input_ids=fake_input_ids,
                attention_mask=batch_attention_mask,
                position_ids=batch_position_ids,
            )
        )

    return batched_examples


def _get_causal_lm_components(model):
    if not hasattr(model, "model") or not hasattr(model, "lm_head"):
        raise TypeError(
            "Quantized backbone must expose `model` and `lm_head` like a standard causal LM."
        )
    return model.model, model.lm_head


def _inject_quantized_layer_modules(lisa_layer, quant_layer):
    for module_name in QUANTIZED_LINEAR_NAMES:
        lisa_parent = lisa_layer
        quant_parent = quant_layer
        *parents, leaf = module_name.split(".")
        for part in parents:
            lisa_parent = getattr(lisa_parent, part)
            quant_parent = getattr(quant_parent, part)
        setattr(lisa_parent, leaf, getattr(quant_parent, leaf))

    lisa_layer.input_layernorm.load_state_dict(quant_layer.input_layernorm.state_dict())
    lisa_layer.post_attention_layernorm.load_state_dict(
        quant_layer.post_attention_layernorm.state_dict()
    )


def _validate_backbone_compatibility(lisa_model, quantized_model):
    lisa_backbone = lisa_model.get_model()
    quant_backbone, quant_lm_head = _get_causal_lm_components(quantized_model)

    if len(lisa_backbone.layers) != len(quant_backbone.layers):
        raise ValueError(
            "Layer count mismatch between LISA backbone and quantized backbone: "
            f"{len(lisa_backbone.layers)} != {len(quant_backbone.layers)}"
        )

    if lisa_backbone.embed_tokens.weight.shape != quant_backbone.embed_tokens.weight.shape:
        raise ValueError(
            "Embedding shape mismatch between LISA backbone and quantized backbone: "
            f"{tuple(lisa_backbone.embed_tokens.weight.shape)} != "
            f"{tuple(quant_backbone.embed_tokens.weight.shape)}"
        )

    if lisa_model.lm_head.weight.shape != quant_lm_head.weight.shape:
        raise ValueError(
            "LM head shape mismatch between LISA model and quantized backbone: "
            f"{tuple(lisa_model.lm_head.weight.shape)} != "
            f"{tuple(quant_lm_head.weight.shape)}"
        )


def load_quantized_backbone_into_lisa(
    lisa_model,
    quantized_model,
    *,
    quantization_method=None,
    quantization_config=None,
):
    _validate_backbone_compatibility(lisa_model, quantized_model)

    lisa_backbone = lisa_model.get_model()
    quant_backbone, quant_lm_head = _get_causal_lm_components(quantized_model)

    lisa_backbone.embed_tokens.load_state_dict(quant_backbone.embed_tokens.state_dict())
    for lisa_layer, quant_layer in zip(lisa_backbone.layers, quant_backbone.layers):
        _inject_quantized_layer_modules(lisa_layer, quant_layer)
    lisa_backbone.norm.load_state_dict(quant_backbone.norm.state_dict())
    lisa_model.lm_head.load_state_dict(quant_lm_head.state_dict())

    if quantization_method is not None:
        lisa_model.quantization_method = quantization_method

    if quantization_config is None:
        quantization_config = getattr(
            getattr(quantized_model, "config", None), "quantization_config", None
        )
    if quantization_config is not None:
        lisa_model.config.quantization_config = quantization_config

    return lisa_model


def build_bnb_8bit_kwargs():
    return {
        "load_in_8bit": True,
        "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
    }


def build_bnb_4bit_kwargs(torch_dtype, quant_kwargs=None):
    if quant_kwargs is None:
        quant_kwargs = {}

    quant_type = quant_kwargs.get("quant_type", "nf4")
    use_double_quant = quant_kwargs.get("use_double_quant", True)

    return {
        "load_in_4bit": True,
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=use_double_quant,
            bnb_4bit_quant_type=quant_type,
        ),
    }


def build_awq_kwargs(quant_kwargs=None):
    if quant_kwargs is None:
        quant_kwargs = {}

    bits = quant_kwargs.get("bits", 4)
    group_size = quant_kwargs.get("group_size", 128)
    zero_point = quant_kwargs.get("zero_point", True)
    backend = quant_kwargs.get("backend", "auto")
    modules_to_not_convert = quant_kwargs.get("modules_to_not_convert")

    awq_config = AwqConfig(
        bits=bits,
        group_size=group_size,
        zero_point=zero_point,
        backend=backend,
        modules_to_not_convert=modules_to_not_convert,
    )

    extra_kwargs = {
        k: v
        for k, v in quant_kwargs.items()
        if k
        not in {
            "bits",
            "group_size",
            "zero_point",
            "backend",
            "modules_to_not_convert",
        }
    }

    return {
        "quantization_config": awq_config,
        **extra_kwargs,
    }


def build_quantization_kwargs(quant_method, torch_dtype, quant_kwargs=None):
    validate_quantization_config(quant_method, quant_kwargs)

    if quant_method == "none":
        return {}
    if quant_method == "bnb_8bit":
        return build_bnb_8bit_kwargs()
    if quant_method == "bnb_4bit":
        return build_bnb_4bit_kwargs(torch_dtype, quant_kwargs)
    if quant_method == "awq":
        return build_awq_kwargs(quant_kwargs)
    if quant_method in {"gptq", "hqq", "quanto", "smoothquant"}:
        raise ValueError(
            f"{quant_method} uses the exported-backbone quantization path and should "
            "not call build_quantization_kwargs()."
        )

    raise ValueError(f"Unsupported quant_method: {quant_method}")


def is_quantized_model(model):
    return bool(
        getattr(model, "is_loaded_in_4bit", False)
        or getattr(model, "is_loaded_in_8bit", False)
        or getattr(model, "quantization_method", None)
        in {"awq", "gptq", "hqq", "quanto", "smoothquant"}
        or getattr(getattr(model, "config", None), "quantization_config", None) is not None
    )
