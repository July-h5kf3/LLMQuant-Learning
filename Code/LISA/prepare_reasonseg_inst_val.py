import argparse
import base64
import json
import os
import random
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path

import cv2
import requests
from pycocotools import mask as mask_utils


SYSTEM_PROMPT = """You are asked to generate the instruction tuning data for language-guided reasoning instance segmentation. Requirements are:
(1) Create a series of specific questions (Q1, Q2, Q3, etc.)(but no more than 5 questions) focusing on identifying and isolating different elements within the image, based on the polygon information. Each question should not refer to previous questions, and facilitate the generation of segmented masks for objects when processed by an imaging system. Ensure the questions are clear, precise, logical, and interesting, and avoid directly mentioning coordinates, label names, and polygons. The questions should try to consider the use and nature of the object, not just its appearance. The output format must be 'Q[number]: [question]'. If the question is about humans, do not ask questions without extra modifiers, but ask questions simply like 'Please find out all the individuals in the image.'
(2) Answer all your questions (A1, A2, A3, etc.) indicating which polygons in <anno> correspond to each question. For items with multiple instances in the same category, list ALL instances for that category in the answer! Do not output full information; the format MUST follow: 'A[number]: instance id is [id1], label name is [name]; instance id is [id2], label name is [name]; instance id is [id3], label name is [name]; ...'
(3) Never invent instance ids. Use only the ids provided in <anno>.
(4) Return only Q/A lines. No extra commentary."""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a ReasonSeg-inst style dataset from COCO val2017 with GLM-5."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/root/autodl-tmp/LLMQuant-Learning/Code/LISA/dataset"),
    )
    parser.add_argument(
        "--ann-file",
        type=Path,
        default=None,
        help="Defaults to dataset-root/COCO2017/annotations/instances_val2017.json",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Defaults to dataset-root/COCO2017/val2017",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Defaults to dataset-root/ReasonSeg-inst-val",
    )
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument(
        "--grouped-json-path",
        type=Path,
        default=None,
        help="Optional aggregated per-image json list, similar to /root/autodl-tmp/data.json.",
    )
    parser.add_argument(
        "--data-json-path",
        type=Path,
        default=None,
        help="Optional flat data.json-style output path. Defaults to output-root/data.json.",
    )
    parser.add_argument("--target-pairs", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=20260404)
    parser.add_argument("--min-width", type=int, default=512)
    parser.add_argument("--min-height", type=int, default=512)
    parser.add_argument("--min-area", type=float, default=400.0)
    parser.add_argument("--max-questions", type=int, default=5)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "copy", "symlink"),
        default="hardlink",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default="https://www.autodl.art/api/v1",
    )
    parser.add_argument("--model", type=str, default="qwen3.6-plus")
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="AUTODL_API_KEY",
        help="Environment variable that stores the API key.",
    )
    parser.add_argument(
        "--explode-multi-instance",
        action="store_true",
        help="Split one QA pair with multiple ids into multiple flat entries.",
    )
    parser.add_argument(
        "--limit-images",
        type=int,
        default=None,
        help="Optional smaller cap for debugging.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional hard cap on how many images to process before stopping.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip images that already have per-image json outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only filter and sample images. Do not call the API.",
    )
    return parser.parse_args()


def load_coco_data(ann_file: Path):
    with ann_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    images = {img["id"]: img for img in data["images"]}
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}
    anns_by_image = defaultdict(list)
    for ann in data["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)
    return images, categories, anns_by_image


def decode_segmentation_to_polygons(segmentation, height, width):
    if isinstance(segmentation, list):
        polygons = []
        for poly in segmentation:
            if len(poly) < 6:
                continue
            pts = []
            for i in range(0, len(poly), 2):
                pts.append([round(float(poly[i]), 2), round(float(poly[i + 1]), 2)])
            if len(pts) >= 3:
                polygons.append(pts)
        return polygons

    rle = segmentation
    if isinstance(rle, dict) and isinstance(rle.get("counts"), str):
        rle = dict(rle)
        rle["counts"] = rle["counts"].encode()
    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = mask.max(axis=2)
    mask = mask.astype("uint8")
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        pts = contour.squeeze(1).tolist()
        if len(pts) < 3:
            continue
        polygons.append([[round(float(x), 2), round(float(y), 2)] for x, y in pts])
    return polygons


def build_valid_annotations(image, anns, categories, min_area):
    valid = []
    for ann in anns:
        if ann.get("iscrowd", 0) == 1:
            continue
        if float(ann.get("area", 0.0)) < min_area:
            continue
        try:
            polygons = decode_segmentation_to_polygons(
                ann["segmentation"], image["height"], image["width"]
            )
        except Exception:
            continue
        if not polygons:
            continue
        category_name = categories.get(ann["category_id"])
        if not category_name:
            continue
        valid.append(
            {
                "ann_id": str(ann["id"]),
                "category_id": ann["category_id"],
                "label_name": category_name,
                "area": float(ann["area"]),
                "bbox": [round(float(v), 2) for v in ann.get("bbox", [])],
                "points": polygons,
            }
        )
    return valid


def choose_images(images, categories, anns_by_image, seed, min_width, min_height, min_area):
    eligible = []
    for image_id, image in images.items():
        if image["width"] < min_width or image["height"] < min_height:
            continue
        valid = build_valid_annotations(image, anns_by_image.get(image_id, []), categories, min_area)
        if not valid:
            continue
        eligible.append((image, valid))
    rng = random.Random(seed)
    rng.shuffle(eligible)
    return eligible


def image_to_data_url(image_path: Path):
    suffix = image_path.suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def format_anno_for_prompt(image, valid_annotations):
    lines = [
        f"image file: {image['file_name']}",
        f"image size: {image['width']}x{image['height']}",
        "<anno>",
    ]
    for ann in valid_annotations:
        lines.append(
            json.dumps(
                {
                    "instance_id": ann["ann_id"],
                    "label_name": ann["label_name"],
                    "area": round(ann["area"], 2),
                    "bbox_xywh": ann["bbox"],
                    "polygons": ann["points"],
                },
                ensure_ascii=False,
            )
        )
    lines.append("</anno>")
    lines.append(
        "Generate up to {} question-answer pairs for this image.".format(
            min(5, len(valid_annotations))
        )
    )
    return "\n".join(lines)


def call_glm_chat(api_base, api_key, model, image_path, prompt_text, max_retries, sleep_seconds):
    url = api_base.rstrip("/") + "/chat/completions"
    data_url = image_to_data_url(image_path)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error = None
    for _ in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=180)
            if not resp.ok:
                error_text = resp.text[:4000]
                raise RuntimeError(
                    f"HTTP {resp.status_code} from {url} with model={model}: {error_text}"
                )
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as exc:
            last_error = exc
            time.sleep(sleep_seconds)
    raise RuntimeError(f"GLM request failed after retries: {last_error}")


def parse_qa_pairs(text):
    questions = {}
    answers = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        q_match = re.match(r"Q(\d+)\s*:\s*(.+)", line, flags=re.IGNORECASE)
        if q_match:
            questions[int(q_match.group(1))] = q_match.group(2).strip()
            continue
        a_match = re.match(r"A(\d+)\s*:\s*(.+)", line, flags=re.IGNORECASE)
        if a_match:
            answers[int(a_match.group(1))] = a_match.group(2).strip()
    qa_pairs = []
    for idx in sorted(set(questions) & set(answers)):
        qa_pairs.append((questions[idx], answers[idx]))
    return qa_pairs


def parse_answer_instances(answer_text):
    pattern = re.compile(
        r"instance id is\s+(\d+)\s*,\s*label name is\s*([^;]+)",
        flags=re.IGNORECASE,
    )
    parsed = []
    for match in pattern.finditer(answer_text):
        parsed.append(
            {
                "ann_id": match.group(1).strip(),
                "label_name": match.group(2).strip(),
            }
        )
    return parsed


def build_entries_for_pairs(image, qa_pairs, ann_lookup, explode_multi_instance):
    entries = []
    for question, answer in qa_pairs:
        parsed_instances = parse_answer_instances(answer)
        valid_instances = [item for item in parsed_instances if item["ann_id"] in ann_lookup]
        if not valid_instances:
            continue

        if explode_multi_instance:
            for item in valid_instances:
                ann = ann_lookup[item["ann_id"]]
                entries.append(
                    {
                        "English Question": question,
                        "English Answer": f"instance id is {ann['ann_id']}, label name is {ann['label_name']}",
                        "ID": [ann["ann_id"]],
                        "points": {
                            ann["ann_id"]: {
                                "label name": ann["label_name"],
                                "points": ann["points"],
                            }
                        },
                        "img_path": image["file_name"],
                    }
                )
            continue

        points = {}
        ids = []
        answer_parts = []
        for item in valid_instances:
            ann = ann_lookup[item["ann_id"]]
            ids.append(ann["ann_id"])
            points[ann["ann_id"]] = {
                "label name": ann["label_name"],
                "points": ann["points"],
            }
            answer_parts.append(
                f"instance id is {ann['ann_id']}, label name is {ann['label_name']}"
            )
        entries.append(
            {
                "English Question": question,
                "English Answer": "; ".join(answer_parts),
                "ID": ids,
                "points": points,
                "img_path": image["file_name"],
            }
        )
    return entries


def materialize_image(src_path: Path, dst_path: Path, link_mode: str):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        return
    if link_mode == "hardlink":
        os.link(src_path, dst_path)
    elif link_mode == "symlink":
        os.symlink(src_path, dst_path)
    else:
        shutil.copy2(src_path, dst_path)


def main():
    args = parse_args()
    ann_file = args.ann_file or args.dataset_root / "COCO2017" / "annotations" / "instances_val2017.json"
    image_root = args.image_root or args.dataset_root / "COCO2017" / "val2017"
    output_root = args.output_root or args.dataset_root / "ReasonSeg-inst-val"
    split_dir = output_root / args.split
    split_json_path = output_root / f"{args.split}.json"
    manifest_path = output_root / f"{args.split}_manifest.json"
    grouped_json_path = args.grouped_json_path or output_root / f"{args.split}_data_grouped.json"
    data_json_path = args.data_json_path or output_root / "data.json"

    images, categories, anns_by_image = load_coco_data(ann_file)
    sampled = choose_images(
        images,
        categories,
        anns_by_image,
        seed=args.seed,
        min_width=args.min_width,
        min_height=args.min_height,
        min_area=args.min_area,
    )

    if args.limit_images is not None:
        sampled = sampled[: args.limit_images]

    output_root.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get(args.api_key_env)
    if not args.dry_run and not api_key:
        raise RuntimeError(
            f"Environment variable {args.api_key_env} is not set. "
            "Export it before running the script."
        )

    flat_entries = []
    grouped_entries = []
    per_image_records = []
    summary = {
        "ann_file": str(ann_file),
        "image_root": str(image_root),
        "output_root": str(output_root),
        "split": args.split,
        "target_pairs": args.target_pairs,
        "eligible_images": len(sampled),
        "min_width": args.min_width,
        "min_height": args.min_height,
        "min_area": args.min_area,
        "link_mode": args.link_mode,
        "model": args.model,
        "explode_multi_instance": args.explode_multi_instance,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    generated_pairs = 0
    for index, (image, valid_annotations) in enumerate(sampled, start=1):
        if args.max_images is not None and len(per_image_records) >= args.max_images:
            print(f"Reached max-images={args.max_images}, stopping.")
            break
        if generated_pairs >= args.target_pairs:
            print(f"Reached target-pairs={args.target_pairs}, stopping.")
            break

        image_path = image_root / image["file_name"]
        output_image_path = split_dir / image["file_name"]
        output_image_json = split_dir / f"{Path(image['file_name']).stem}.json"

        if args.resume and output_image_json.exists():
            print(f"[{index}/{len(sampled)}] skip existing {image['file_name']}")
            continue

        materialize_image(image_path, output_image_path, args.link_mode)
        ann_lookup = {ann["ann_id"]: ann for ann in valid_annotations}

        if args.dry_run:
            per_image_entries = []
            response_text = ""
        else:
            prompt_text = format_anno_for_prompt(image, valid_annotations)
            response_text = call_glm_chat(
                args.api_base,
                api_key,
                args.model,
                image_path,
                prompt_text,
                args.max_retries,
                args.sleep_seconds,
            )
            qa_pairs = parse_qa_pairs(response_text)
            per_image_entries = build_entries_for_pairs(
                image, qa_pairs, ann_lookup, args.explode_multi_instance
            )

        image_record = {
            "image": image["file_name"],
            "image_path": image["file_name"],
            "source_split": "val2017",
            "num_annotations": len(per_image_entries),
            "annotations": per_image_entries,
            "raw_response": response_text,
            "candidate_instances": [
                {
                    "instance_id": ann["ann_id"],
                    "label_name": ann["label_name"],
                    "area": ann["area"],
                }
                for ann in valid_annotations
            ],
        }
        with output_image_json.open("w", encoding="utf-8") as f:
            json.dump(image_record, f, ensure_ascii=False)

        flat_entries.extend(per_image_entries)
        grouped_entries.append(image_record)
        generated_pairs += len(per_image_entries)
        per_image_records.append(
            {
                "image": image["file_name"],
                "num_annotations": len(per_image_entries),
                "candidate_instances": len(valid_annotations),
            }
        )
        print(
            f"[{index}/{len(sampled)}] {image['file_name']} "
            f"candidates={len(valid_annotations)} generated={len(per_image_entries)} "
            f"total_pairs={generated_pairs}"
        )

        if generated_pairs >= args.target_pairs:
            print(f"Reached target-pairs={args.target_pairs}, stopping.")
            break

    with split_json_path.open("w", encoding="utf-8") as f:
        json.dump(flat_entries, f, ensure_ascii=False)

    with data_json_path.open("w", encoding="utf-8") as f:
        json.dump(flat_entries, f, ensure_ascii=False)

    with grouped_json_path.open("w", encoding="utf-8") as f:
        json.dump(grouped_entries, f, ensure_ascii=False)

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "images": per_image_records,
                "total_annotations": len(flat_entries),
                "data_json_path": str(data_json_path),
                "grouped_json_path": str(grouped_json_path),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        json.dumps(
            {
                "split_json": str(split_json_path),
                "data_json": str(data_json_path),
                "grouped_json": str(grouped_json_path),
                "manifest": str(manifest_path),
                "images_dir": str(split_dir),
                "generated_annotations": len(flat_entries),
                "grouped_images": len(grouped_entries),
                "processed_images": len(per_image_records),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
