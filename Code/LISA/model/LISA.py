from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .llava1p5.model.language_model.llava_llama import (
    LlavaLlamaForCausalLM,
    LlavaLlamaModel,
)
from .segment_anything import build_sam_vit_h


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,
    eps=1e-6,
):
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


class LisaMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaMetaModel, self).__init__(config)

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_lisa_modules(self.config)

    def initialize_lisa_modules(self, config):
        print("if you have trained models, please be careful when you see this!")
        #SAM
        self.visual_model = build_sam_vit_h(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True
        
        #Proj Layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class LisaModel(LisaMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class LISAForCausalLM(LlavaLlamaForCausalLM):

    def __init__(
        self,
        config,
        **kwargs,
    ):
        self.ce_loss_weight = kwargs.pop(
            "ce_loss_weight", getattr(config, "ce_loss_weight", None)
        )
        self.dice_loss_weight = kwargs.pop(
            "dice_loss_weight", getattr(config, "dice_loss_weight", None)
        )
        self.bce_loss_weight = kwargs.pop(
            "bce_loss_weight", getattr(config, "bce_loss_weight", None)
        )

        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )
        else:
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get("vision_tower", config.vision_tower)

        self.seg_token_idx = kwargs.pop("seg_token_idx")

        super().__init__(config)

        self.model = LisaModel(config, **kwargs)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                image_embeddings = self.model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, dim=0)
        return image_embeddings

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def batch_dice_loss(self, inputs: torch.Tensor, targets: torch.Tensor):
        inputs = inputs.sigmoid()
        numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
        denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
        return 1 - (numerator + 1) / (denominator + 1)

    def batch_sigmoid_ce_loss(self, inputs: torch.Tensor, targets: torch.Tensor):
        hw = inputs.shape[1]
        pos = F.binary_cross_entropy_with_logits(
            inputs, torch.ones_like(inputs), reduction="none"
        )
        neg = F.binary_cross_entropy_with_logits(
            inputs, torch.zeros_like(inputs), reduction="none"
        )
        loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
            "nc,mc->nm", neg, (1 - targets)
        )
        return loss / hw

    def _align_seg_token_mask(
        self, seg_token_mask: torch.Tensor, target_len: int
    ) -> torch.Tensor:
        current_len = seg_token_mask.shape[1]
        if current_len == target_len:
            return seg_token_mask
        if current_len > target_len:
            return seg_token_mask[:, :target_len]
        pad = torch.zeros(
            (seg_token_mask.shape[0], target_len - current_len),
            dtype=seg_token_mask.dtype,
            device=seg_token_mask.device,
        )
        return torch.cat([seg_token_mask, pad], dim=1)

    def _hungarian_match_indices(
        self, pred_masks_flat: torch.Tensor, gt_masks_flat: torch.Tensor
    ):
        cost = self.batch_dice_loss(pred_masks_flat, gt_masks_flat) + self.batch_sigmoid_ce_loss(
            pred_masks_flat, gt_masks_flat
        )
        pred_indices, gt_indices = linear_sum_assignment(cost.detach().cpu().numpy())
        sorted_pred_order = np.argsort(pred_indices)
        adjusted_gt_indices = gt_indices[sorted_pred_order]
        return adjusted_gt_indices

    def reorder_gt_masks_by_change_list(self, pred_masks, gt_masks, change_list):
        reordered_gt_masks = list(gt_masks)
        if change_list is None:
            return reordered_gt_masks

        max_len = min(len(reordered_gt_masks), len(pred_masks), len(change_list))
        for batch_idx in range(max_len):
            groups = change_list[batch_idx]
            if not isinstance(groups, (list, tuple)):
                continue

            batch_pred = pred_masks[batch_idx]
            batch_gt = reordered_gt_masks[batch_idx]
            batch_reordered = batch_gt.clone()

            for group in groups:
                if not isinstance(group, (list, tuple)) or len(group) <= 1:
                    continue

                group_idx = list(group)
                group_pred = batch_pred[group_idx, :, :].reshape(len(group_idx), -1)
                group_gt = batch_gt[group_idx, :, :].reshape(len(group_idx), -1)

                group_gt_indices = self._hungarian_match_indices(group_pred, group_gt)
                for local_i, gt_local_i in enumerate(group_gt_indices):
                    batch_reordered[group_idx[local_i]] = batch_gt[group_idx[int(gt_local_i)]]

            reordered_gt_masks[batch_idx] = batch_reordered

        return reordered_gt_masks

    def model_forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor = None,
        offset: torch.LongTensor = None,
        masks_list: List[torch.FloatTensor] = None,
        label_list: List[torch.Tensor] = None,
        resize_list: List[tuple] = None,
        change_list: List = None,
        inference: bool = False,
        **kwargs,
    ):
        image_embeddings = self.get_visual_embs(images)
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1

        device = input_ids.device
        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx
        # 因为seg_token_mask是根据input_ids[:,1:]生成的，所以它的shape是(batch_size, seq_len-1)，需要补一个False，保证和input_ids的shape一致
        seg_token_mask = torch.cat(
            [
                seg_token_mask,
                torch.zeros((seg_token_mask.shape[0], 1), dtype=torch.bool, device=device),
            ],
            dim=1,
        )
        # 模型前面会插入图像token，所以seg_token_mask需要在前面补255个False，保证和input_ids的shape一致
        image_token_len = (images_clip.shape[2] // 14) * (images_clip.shape[3] // 14)
        seg_token_mask = torch.cat(
            [
                torch.zeros((seg_token_mask.shape[0], image_token_len - 1), dtype=torch.bool, device=device),#bug fix from LISA++
                #torch.zeros((seg_token_mask.shape[0], 255), dtype=torch.bool, device=device),
                seg_token_mask,
            ],
            dim=1,
        )

        if inference:
            n_batch = 1
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()
            # 一张图片可能对应多个文本输入，所以需要在前面补齐图片特征，保证和input_ids的shape一致

            output_hidden_states = []
            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                output_i = super().forward(
                    images=images_clip_extend[: end_i - start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    output_hidden_states=True,
                )
                output_hidden_states.append(output_i.hidden_states)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            output_hidden_states_list = []
            output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
            output_hidden_states_list.append(output_hidden_states_level)
            output_hidden_states = output_hidden_states_list
            output = None
        else:
            images_clip_list = []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = (
                    images_clip[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                images_clip_list.append(images_clip_i)
            images_clip = torch.cat(images_clip_list, dim=0)

            output = super().forward(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states

        hidden_states = []
        assert len(self.model.text_hidden_fcs) == 1
        # 取最后一层的hidden state，并映射到SAM的文本embedding空间
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))

        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        seg_token_mask = self._align_seg_token_mask(
            seg_token_mask, last_hidden_state.shape[1]
        )
        pred_embeddings = last_hidden_state[seg_token_mask]  # 取出文本中所有seg_token_idx对应的hidden state，作为SAM的文本prompt embedding
        seg_token_counts = seg_token_mask.int().sum(-1)

        # seg_token_counts表示每个样本中seg_token_idx的数量，
        # seg_token_offset表示每个样本中seg_token_idx的起始位置，
        # pred_embeddings_表示每个样本中seg_token_idx对应的hidden state列表
        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=device), seg_token_offset], dim=0
        )
        seg_token_offset = seg_token_offset[offset]

        pred_embeddings_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_

        multimask_output = False
        pred_masks = []
        for i in range(len(pred_embeddings)):
            sparse_embeddings, dense_embeddings = self.model.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=pred_embeddings[i].unsqueeze(1),
            )
            sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
            low_res_masks, _ = self.model.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            pred_mask = self.model.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )
            pred_masks.append(pred_mask[:, 0])

        model_output = output
        gt_masks = masks_list
        gt_masks = self.reorder_gt_masks_by_change_list(pred_masks, gt_masks, change_list)

        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
            }

        ce_loss = model_output.loss * self.ce_loss_weight
        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0

        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]

            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), f"gt_mask.shape: {gt_mask.shape}, pred_mask.shape: {pred_mask.shape}"
            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        loss = ce_loss + mask_loss

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
        }

    def evaluate(
        self,
        images_clip,
        images,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=32,
        tokenizer=None,
    ):
        with torch.no_grad():
            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output_hidden_states = outputs.hidden_states[-1]
            output_ids = outputs.sequences

            device = output_ids.device
            seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
            image_token_len = (images_clip.shape[2] // 14) * (images_clip.shape[3] // 14)
            seg_token_mask = torch.cat(
                [
                    torch.zeros((seg_token_mask.shape[0], image_token_len - 1), dtype=torch.bool, device=device),
                    seg_token_mask,
                ],
                dim=1,
            )

            hidden_states = []
            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            pred_embeddings = last_hidden_state[seg_token_mask]

            seg_token_counts = seg_token_mask.int().sum(-1)
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1, dtype=torch.long, device=device), seg_token_offset], dim=0
            )

            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_

            image_embeddings = self.get_visual_embs(images)

            multimask_output = False
            pred_masks = []
            for i in range(len(pred_embeddings)):
                sparse_embeddings, dense_embeddings = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )

                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                low_res_masks, _ = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=original_size_list[i],
                )
                pred_masks.append(pred_mask[:, 0])

        return output_ids, pred_masks
