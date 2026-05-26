import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from prune_quant_baseline.compressors.conv1d_compressor import RelevanceCompressor


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a relevance compressor from oracle labels.")
    parser.add_argument("--labels-path", required=True, help="Path to JSONL/PT labels generated on remote.")
    parser.add_argument("--output-checkpoint", required=True, help="Where to save the compressor checkpoint.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--target-retention-ratio", type=float, default=0.5)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


class CompressorLabelDataset(Dataset):
    """Variable-length proxy-attention / oracle-relevance pairs."""

    def __init__(self, records: list[dict]) -> None:
        self.records = records

    @classmethod
    def from_path(cls, path: Path) -> "CompressorLabelDataset":
        if path.suffix in {".pt", ".pth"}:
            data = torch.load(path, map_location="cpu")
            records = data["records"] if isinstance(data, dict) and "records" in data else data
        else:
            records = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    records.append(json.loads(line))
        if not records:
            raise ValueError(f"No compressor labels found in {path}.")
        return cls(list(records))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.records[idx]
        proxy = torch.as_tensor(record["proxy_attention"], dtype=torch.float32)
        oracle = torch.as_tensor(record["gae_scores"], dtype=torch.float32)
        if proxy.dim() != 1 or oracle.dim() != 1:
            raise ValueError("proxy_attention and gae_scores must be one-dimensional.")
        if proxy.numel() != oracle.numel():
            raise ValueError("proxy_attention and gae_scores must have the same length.")
        return {"proxy": proxy, "oracle": oracle}


def _target_distribution(oracle: torch.Tensor, mask: torch.Tensor, retention_ratio: float) -> torch.Tensor:
    if not (0 < retention_ratio <= 1):
        raise ValueError("target_retention_ratio must be in the range (0, 1].")
    target = torch.zeros_like(oracle)
    lengths = mask.sum(dim=-1).to(dtype=torch.long)
    for row_idx, length in enumerate(lengths.tolist()):
        if length <= 0:
            raise ValueError("Encountered empty oracle label.")
        scores = oracle[row_idx, :length].clamp_min(0)
        k = max(1, min(length, math.ceil(length * retention_ratio)))
        topk = torch.topk(scores, k=k).indices
        kept = scores.index_select(dim=0, index=topk)
        denom = kept.sum()
        if denom <= 0:
            target[row_idx, :length] = 1.0 / length
        else:
            target[row_idx, topk] = kept / denom
    return target.masked_fill(~mask, 0.0)


def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_len = max(item["proxy"].numel() for item in batch)
    proxy = torch.zeros(len(batch), max_len, dtype=torch.float32)
    oracle = torch.zeros(len(batch), max_len, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for row_idx, item in enumerate(batch):
        length = item["proxy"].numel()
        proxy[row_idx, :length] = item["proxy"]
        oracle[row_idx, :length] = item["oracle"]
        mask[row_idx, :length] = True
    return {"proxy": proxy, "oracle": oracle, "mask": mask}


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    labels_path = Path(args.labels_path)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels path does not exist: {labels_path}")
    dataset = CompressorLabelDataset.from_path(labels_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=_collate)
    device = torch.device(args.device)
    model = RelevanceCompressor(channels=args.channels, num_blocks=args.num_blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        total_loss = 0.0
        total_items = 0
        model.train()
        for batch in loader:
            proxy = batch["proxy"].to(device)
            oracle = batch["oracle"].to(device)
            mask = batch["mask"].to(device)
            target = _target_distribution(oracle, mask, args.target_retention_ratio)
            pred = model(proxy, mask=mask)
            loss = F.kl_div((pred.clamp_min(1e-12)).log(), target, reduction="batchmean")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * proxy.shape[0]
            total_items += proxy.shape[0]
        print(f"epoch={epoch + 1} loss={total_loss / max(total_items, 1):.6f}")

    output_path = Path(args.output_checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "channels": args.channels,
            "num_blocks": args.num_blocks,
            "target_retention_ratio": args.target_retention_ratio,
        },
        output_path,
    )


if __name__ == "__main__":
    main()
