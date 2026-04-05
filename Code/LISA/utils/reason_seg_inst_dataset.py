import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor

from model.llava1p5 import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .utils import DEFAULT_IMAGE_TOKEN, DEFAULT_INSTANT_SEG


class ReasonSegInstDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(self, base_image_dir, tokenizer, vision_tower, split, image_size=1024):
        self.dataset_root = Path(base_image_dir) / "ReasonSeg-inst"
        self.items_path = self.dataset_root / f"{split}.json"
        self.image_root = self.dataset_root / split
        if not self.items_path.is_file():
            raise FileNotFoundError(f"ReasonSeg-Inst json not found: {self.items_path}")
        if not self.image_root.is_dir():
            raise FileNotFoundError(
                f"ReasonSeg-Inst image directory not found: {self.image_root}"
            )

        with self.items_path.open("r", encoding="utf-8") as f:
            self.items = json.load(f)
        if not isinstance(self.items, list):
            raise ValueError(
                f"Expected a list in ReasonSeg-Inst json, got {type(self.items)}"
            )

        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

    def __len__(self):
        return len(self.items)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def get_multi_mask_from_item(self, item, image):
        height, width = image.shape[:2]
        masks = []
        for instance_id in item["ID"]:
            mask = np.zeros((height, width), dtype=np.uint8)
            polygons = item["points"][instance_id]["points"]
            for polygon in polygons:
                pts = np.array(polygon, dtype=np.float32)
                if pts.ndim != 2 or pts.shape[0] < 3:
                    continue
                pts = np.round(pts).astype(np.int32)
                cv2.polylines(mask, [pts], True, 1, 1)
                cv2.fillPoly(mask, [pts], 1)
            masks.append(mask.astype(np.float32))
        return masks

    def __getitem__(self, idx):
        item = self.items[idx]
        image_name = item["img_path"].strip()
        image_path = self.image_root / image_name
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        sampled_masks = self.get_multi_mask_from_item(item, image)
        if not sampled_masks:
            raise ValueError(f"No valid polygons found for sample index {idx}: {image_name}")

        question = (
            DEFAULT_IMAGE_TOKEN
            + "\n"
            + item["English Question"].strip()
            + " Please output segmentation mask."
            + DEFAULT_INSTANT_SEG
        )

        seg_count = len(item["ID"])
        seg_tokens = ["[SEG]" for _ in range(seg_count)]
        if len(seg_tokens) == 1:
            answer = f"{seg_tokens[0]}."
        elif len(seg_tokens) == 2:
            answer = f"{seg_tokens[0]} and {seg_tokens[1]}."
        else:
            answer = f"{', '.join(seg_tokens[:-1])}, and {seg_tokens[-1]}."

        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], answer)
        conversations = [conv.get_prompt()]

        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        masks = torch.from_numpy(np.stack(sampled_masks, axis=0))
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        inference = True

        return (
            [str(image_path), [list(range(seg_count))]],
            image,
            image_clip,
            conversations,
            masks,
            labels,
            resize,
            None,
            None,
            inference,
        )
