import torch
import torch.nn

from abc import ABC,abstractmethod

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN, IGNORE_INDEX,
                         IMAGE_TOKEN_INDEX)
from .multimodel_encoder.builder import build_vision_tower

class LlavaMetaModel:
    def __init__(self,config):
        super(LlavaMetaModel,self).__init__(config)
        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size) #视觉投影层
    def get_vision_tower(self):
        vision_tower = getattr(self,"vision_tower",None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0] 
        return vision_tower
    def initialize_vision_modules(self,model_args,fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer      #拿哪一层的hidden_states取特征
        mm_vision_select_feature = model_args.mm_vision_select_feature  #取哪种token特征 [patch,cls_patch] 
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter    #多模态投影层

        self.config.mm_vision_tower = vision_tower

        vision_tower = build_vision_tower(model_args)

        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower = [vision_tower]
        else:
            self.vision_tower =vision_tower
        
        self.config.use_mm_proj = True
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if not hasattr(self,"mm_projector"):
            self.mm_projector = nn.Linear(
                self.config.mm_hidden_size,map_location="cpu"
            )
            def get_w(weights, keyword):
                return {
                    k.split(keyword + ".")[1]: v
                    for k, v in weights.items()
                    if keyword in k
                }
            self.mm_projector.load_state_dict(
                get_w(mm_projector_weights, "mm_projector")
            )


class LlavaMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass
    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features
    def prepare_inputs_labels_for_multimodal(
        self,input_ids,attention_mask,past_key_values,labels,images
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if (
                past_key_values is not None
                and vision_tower is not None
                and images is not None
                and input_ids.shape[1] == 1
            ):
                attention_mask = torch.ones(
                    (attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
            return input_ids,attention_mask,past_key_values,None,labels

        #将图片转化为图片特征
        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images(images)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0

        for batch_idx, cur_input_idx in enumerate(images):
            if(cur_input_idx == IMAGE_TOKEN_INDEX).sum() == 0:
                #说明没图片
                cur_input_embeds = self.get_model().embed_tokens(cur_input_idx)
                cur_input_embeds = (
                    cur_input_embeds
                    + (
                        0.0 * self.get_model().mm_projector(vision_tower.dummy_feature)
                    ).sum()
                )
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            
            image_token_indices = torch.where(cur_input_idx == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_idx.shape
            while image_token_indices.numel > 0:
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                #将图片拼接入序列
                if getattr(self.config,"tune_mm_mlp_adapter",False) and getattr(
                    self.config,"mm_use_im_start_end",False
                ):
                #text + <im_start> + <image> + <im_end> + text -> text + <im_start> + im_features + <im_end> + text
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_idx[:image_token_start - 1]).detach()
                        #训练过程中不更新文本部分的embedding
                    )#text
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_idx[image_token_start - 1 : image_token_start])
                    )
                    #<im_start>
                    cur_new_input_embeds.append(cur_image_features)
                    #im_features
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_idx[image_token_start + 1 : image_token_start + 2])
                    )
                    #<im_end>

                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=labels.device,
                                dtype=labels.dtype,
                            )
                        )
                        cur_new_labels.append(
                            cur_labels[image_token_start + 1: image_token_start + 2]#Bug Fix
                            # cur_labels[image_token_start:image_token_start + 1]
                        )
                        cur_labels = cur_labels[image_token_start + 2 :]
                elif getattr(self.config,"mm_use_im_start_end",False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_idx[:image_token_start]))#text
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_idx[image_token_start + 1 : image_token_start + 2])
                    )
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=labels.device,
                                dtype=labels.dtype,
                            )
                        )
                        cur_new_labels.append(
                            cur_labels[image_token_start + 1: image_token_start + 2]
                        )
                        cur_labels = cur_labels[image_token_start + 2 :]
                else:
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_idx[:image_token_start])
                    )
                    cur_new_input_embeds.append(cur_image_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=labels.device,
                                dtype=labels.dtype,
                            )
                        )
                        cur_labels = cur_labels[image_token_start + 1:]
                cur_image_idx += 1
                if getattr(self.config,"tune_mm_mlp_adapter",False) and getattr(
                    self.config,"mm_use_im_start_end",False
                ):
                    cur_input_idx = cur_input_idx[image_token_start + 2 : ]
                elif getattr(self.config,"mm_use_im_start_end",False):
                    cur_input_idx = cur_input_idx[image_token_start + 2 : ]
                else:
                    cur_input_idx = cur_input_idx[image_token_start + 1 : ]
                image_token_indices = torch.where(cur_input_idx == IMAGE_TOKEN_INDEX)[0]

            if cur_input_idx.numel() > 0:#处理尾端
                if getattr(self.config,"tune_mm_mlp_adapter",False) and getattr(
                    self.config,"mm_use_im_start_end",False
                ):
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_idx).detach()
                    )#text
                elif getattr(self.config,"mm_use_im_start_end",False):
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_idx)
                    )
                else:
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_idx)
                    )
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [
                x.to(device=self.device) for x in cur_new_input_embeds
            ]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds,dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels,dim=0)
                new_labels.append(cur_new_labels)
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat(
                    cur_new_embed,
                    torch.zeros(
                        (max_len - cur_new_embed.shape[0],cur_new_embed.shape[1]),
                        device=cur_new_embed.device,
                        dtype=cur_new_embed.dtype,
                    ),
                    dim=0,
                )
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align,dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat(
                        (
                            cur_new_label,
                            torch.full(
                                (max_len - cur_new_label.shape[0],),
                                IGNORE_INDEX,
                                dtype=cur_new_label.dtype,
                                device=cur_new_label.device,
                            ),
                        ),
                        dim=0,
                    )
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)
            
            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(
                    attention_mask, _new_labels, new_labels
                ):
                    new_attn_mask_pad_left = torch.full(
                        (cur_new_labels.shape[0] - labels.shape[1],),
                        True,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    new_attn_mask_pad_right = torch.full(
                        (cur_new_labels_align.shape[0] - cur_new_labels.shape[0],),
                        False,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    cur_new_attention_mask = torch.cat(
                        (
                            new_attn_mask_pad_left,
                            cur_attention_mask,
                            new_attn_mask_pad_right,
                        ),
                        dim=0,
                    )
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds,dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels,dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (
                        attention_mask.shape[0],
                        new_input_embeds.shape[1] - input_ids.shape[1],
                    ),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat(
                    (new_attn_mask_pad_left, attention_mask), dim=1
                )
                assert attention_mask.shape == new_input_embeds.shape[:2]
        return None,attention_mask,past_key_values,new_input_embeds,new_labels

