import glob
import os
import random
from functools import partial

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPImageProcessor

from model.llava1p5 import conversation as conversation_lib
from model.llava1p5.constants import DEFAULT_IMAGE_TOKEN
from model.llava1p5.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.data_processing import get_mask_from_json
from utils.utils import (
    ANSWER_LIST,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    LONG_QUESTION_LIST,
    SHORT_QUESTION_LIST,
)


class CalibrationDataset(Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        *,
        reason_seg_data="ReasonSeg|train",
        image_size=1024,
        max_samples=128,
        questions_per_image=1,
        seed=3407,
        conv_type="llava_v1",
    ):
        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.questions_per_image = questions_per_image
        self.seed = seed
        self.conv_type = conv_type

        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.samples = self._build_sample_index(reason_seg_data, max_samples)

    def _build_sample_index(self, reason_seg_data, max_samples):
        # Build a fixed image/json list once so calibration is reproducible.
        dataset_name, split_spec = reason_seg_data.split("|", 1)
        splits = split_spec.split("_")

        image_paths = []
        for split in splits:
            image_paths.extend(
                glob.glob(
                    os.path.join(
                        self.base_image_dir,
                        "reason_seg",
                        dataset_name,
                        split,
                        "*.jpg",
                    )
                )
            )

        image_paths = sorted(image_paths)
        samples = [
            (image_path, image_path.replace(".jpg", ".json")) for image_path in image_paths
        ]

        if max_samples is not None and max_samples > 0 and len(samples) > max_samples:
            rng = random.Random(self.seed)
            rng.shuffle(samples)
            samples = sorted(samples[:max_samples], key=lambda item: item[0])

        return samples

    def __len__(self):
        return len(self.samples)

    def preprocess(self, x):
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        image_path, json_path = self.samples[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        mask, sents, is_sentence = get_mask_from_json(json_path, image)
        del mask

        if isinstance(sents, str):
            sents = [sents]
        else:
            sents = [sent for sent in sents if isinstance(sent, str) and sent.strip()]

        if not sents:
            raise ValueError(f"No valid text annotations found in {json_path}")

        # Keep only a small number of prompts per image to reduce calibration cost.
        if len(sents) > self.questions_per_image:
            rng = random.Random(f"{self.seed}:{image_path}")
            indices = list(range(len(sents)))
            rng.shuffle(indices)
            indices = sorted(indices[: self.questions_per_image])
            sampled_sents = [sents[i] for i in indices]
        else:
            sampled_sents = sents

        questions = []
        conversations = []
        for text in sampled_sents:
            if is_sentence:
                question = LONG_QUESTION_LIST[0].format(sent=text)
            else:
                question = SHORT_QUESTION_LIST[0].format(class_name=text.lower())

            conv = conversation_lib.conv_templates[self.conv_type].copy()
            conv.messages = []
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], ANSWER_LIST[0])

            questions.append(question)
            conversations.append(conv.get_prompt())

        image_sam = self.transform.apply_image(image)
        resize = image_sam.shape[:2]
        image_sam = self.preprocess(
            torch.from_numpy(image_sam).permute(2, 0, 1).contiguous()
        )

        return {
            "image_path": image_path,
            "image": image_sam,
            "image_clip": image_clip,
            "resize": resize,
            "questions": questions,
            "sampled_sents": sampled_sents,
            "conversations": conversations,
        }


def calibration_collate_fn(batch, tokenizer, use_mm_start_end=True):
    # Flatten multiple prompts from one image into a single token batch.
    image_paths = []
    images = []
    images_clip = []
    resize_list = []
    questions_list = []
    sampled_sents_list = []
    conversation_list = []
    offset_list = [0]
    count = 0

    for item in batch:
        image_paths.append(item["image_path"])
        images.append(item["image"])
        images_clip.append(item["image_clip"])
        resize_list.append(item["resize"])
        questions_list.append(item["questions"])
        sampled_sents_list.append(item["sampled_sents"])
        conversation_list.extend(item["conversations"])
        count += len(item["conversations"])
        offset_list.append(count)

    if use_mm_start_end:
        replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        conversation_list = [
            prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
            for prompt in conversation_list
        ]

    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    return {
        "image_paths": image_paths,
        "images": torch.stack(images, dim=0),
        "images_clip": torch.stack(images_clip, dim=0),
        "input_ids": input_ids,
        "attention_masks": attention_masks,
        "resize_list": resize_list,
        "questions_list": questions_list,
        "sampled_sents_list": sampled_sents_list,
        "conversation_list": conversation_list,
        "offset": torch.LongTensor(offset_list),
    }


def build_calibration_loader(
    *,
    base_image_dir,
    tokenizer,
    vision_tower,
    reason_seg_data="ReasonSeg|train",
    image_size=1024,
    max_samples=128,
    questions_per_image=1,
    seed=3407,
    conv_type="llava_v1",
    use_mm_start_end=True,
    batch_size=1,
    num_workers=0,
    shuffle=False,
):
    dataset = CalibrationDataset(
        base_image_dir=base_image_dir,
        tokenizer=tokenizer,
        vision_tower=vision_tower,
        reason_seg_data=reason_seg_data,
        image_size=image_size,
        max_samples=max_samples,
        questions_per_image=questions_per_image,
        seed=seed,
        conv_type=conv_type,
    )

    collate = partial(
        calibration_collate_fn,
        tokenizer=tokenizer,
        use_mm_start_end=use_mm_start_end,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
    )
