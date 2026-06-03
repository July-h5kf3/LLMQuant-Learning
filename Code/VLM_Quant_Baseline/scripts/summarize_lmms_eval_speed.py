#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
import re
from pathlib import Path


SPEED_RE = re.compile(r"\[(?P<engine>VLLM|TRTLLM)_SPEED\]\s+(?P<body>.*)")


def parse_kv_line(body):
    values = {}
    for item in body.split():
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        try:
            if "." in value:
                values[key] = float(value)
            else:
                values[key] = int(value)
        except ValueError:
            values[key] = value
    return values


def latest_result(output_dir):
    paths = glob.glob(os.path.join(output_dir, "**", "*_results.json"), recursive=True)
    if not paths:
        return None, {}
    path = max(paths, key=os.path.getmtime)
    with open(path, "r", encoding="utf-8") as f:
        return path, json.load(f)


def parse_elapsed(time_path):
    values = {}
    if not os.path.exists(time_path):
        return values
    with open(time_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            try:
                values[key] = float(value)
            except ValueError:
                values[key] = value
    return values


def parse_speed(log_path):
    latest = {}
    if not os.path.exists(log_path):
        return latest
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            match = SPEED_RE.search(line)
            if match:
                latest = parse_kv_line(match.group("body"))
                latest["engine"] = match.group("engine").lower()
    return latest


def flatten_metrics(results):
    rows = []
    for task, metrics in results.get("results", {}).items():
        for key, value in metrics.items():
            if key == "alias" or key.endswith("_stderr") or not isinstance(value, (int, float)):
                continue
            rows.append((task, key, value))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Summarize lmms-eval wall-clock and generation speed logs.")
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--output_csv", default=None)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    rows = []
    for variant_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        logs_dir = variant_dir / "logs"
        log_paths = sorted(logs_dir.glob("*.log"))
        log_path = str(log_paths[-1]) if log_paths else ""
        time_values = parse_elapsed(f"{log_path}.time") if log_path else {}
        speed_values = parse_speed(log_path) if log_path else {}
        result_path, result = latest_result(str(variant_dir))

        elapsed = time_values.get("elapsed_sec", "")
        requests = speed_values.get("requests", "")
        end_to_end_samples_per_sec = ""
        if elapsed and requests:
            end_to_end_samples_per_sec = float(requests) / float(elapsed)

        base = {
            "variant": variant_dir.name,
            "status": time_values.get("status", ""),
            "elapsed_sec": elapsed,
            "engine": speed_values.get("engine", ""),
            "requests": requests,
            "generated_tokens": speed_values.get("generated_tokens", ""),
            "generate_sec": speed_values.get("generate_sec", ""),
            "generate_samples_per_sec": speed_values.get("samples_per_sec", ""),
            "output_tokens_per_sec": speed_values.get("output_tokens_per_sec", ""),
            "avg_output_tokens": speed_values.get("avg_output_tokens", ""),
            "end_to_end_samples_per_sec": end_to_end_samples_per_sec,
            "result_path": result_path or "",
            "log_path": log_path,
        }

        metrics = flatten_metrics(result)
        if not metrics:
            rows.append({**base, "task": "", "metric": "", "value": ""})
            continue
        for task, metric, value in metrics:
            rows.append({**base, "task": task, "metric": metric, "value": value})

    fieldnames = [
        "variant",
        "status",
        "elapsed_sec",
        "engine",
        "requests",
        "generated_tokens",
        "generate_sec",
        "generate_samples_per_sec",
        "output_tokens_per_sec",
        "avg_output_tokens",
        "end_to_end_samples_per_sec",
        "task",
        "metric",
        "value",
        "result_path",
        "log_path",
    ]
    output_csv = Path(args.output_csv) if args.output_csv else run_root / "summary.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(output_csv)


if __name__ == "__main__":
    main()
