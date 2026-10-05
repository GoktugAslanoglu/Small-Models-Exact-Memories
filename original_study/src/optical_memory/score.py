from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable

from .common import project_path, read_jsonl


THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
BOX_RE = re.compile(
    r"^\s*<\|begin_of_box\|>(.*?)<\|end_of_box\|>\s*$",
    re.DOTALL | re.IGNORECASE,
)
TOKEN_RE = re.compile(r"[A-Za-z0-9_.:/\\+-]+")


def clean_prediction(prediction: str) -> str:
    value = THINK_RE.sub("", prediction).strip()
    box_match = BOX_RE.match(value)
    if box_match:
        value = box_match.group(1).strip()
    value = value.replace("```json", "").replace("```text", "").replace("```", "").strip()
    try:
        parsed = json.loads(value)
        if isinstance(parsed, dict) and "answer" in parsed:
            value = str(parsed["answer"])
        elif isinstance(parsed, str):
            value = parsed
    except (json.JSONDecodeError, TypeError):
        pass
    value = re.sub(r"(?i)^answer\s*[:=]\s*", "", value).strip()
    if "\n" in value:
        value = next((line.strip() for line in value.splitlines() if line.strip()), "")
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"', "`"}:
        value = value[1:-1]
    return value.strip()


def normalize_semantic(value: str) -> str:
    tokens = TOKEN_RE.findall(value.lower())
    return " ".join(tokens)


def levenshtein(reference: str, hypothesis: str) -> int:
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, ref_char in enumerate(reference, start=1):
        current = [i]
        for j, hyp_char in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (ref_char != hyp_char),
                )
            )
        previous = current
    return previous[-1]


def token_f1(reference: str, hypothesis: str) -> float:
    reference_tokens = normalize_semantic(reference).split()
    hypothesis_tokens = normalize_semantic(hypothesis).split()
    if not reference_tokens and not hypothesis_tokens:
        return 1.0
    if not reference_tokens or not hypothesis_tokens:
        return 0.0
    remaining = list(hypothesis_tokens)
    overlap = 0
    for token in reference_tokens:
        if token in remaining:
            overlap += 1
            remaining.remove(token)
    precision = overlap / len(hypothesis_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def score_row(row: dict[str, Any]) -> dict[str, Any]:
    prediction = clean_prediction(str(row["prediction"]))
    reference = str(row["answer"]).strip()
    if row.get("scoring") == "semantic":
        exact_match = float(normalize_semantic(prediction) == normalize_semantic(reference))
    else:
        exact_match = float(prediction == reference)
    return {
        **row,
        "clean_prediction": prediction,
        "exact_match": exact_match,
        "cer": levenshtein(reference, prediction) / max(1, len(reference)),
        "token_f1": token_f1(reference, prediction),
    }


def bootstrap_ci(values: list[float], seed: int, samples: int = 2000) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    estimates = sorted(mean(rng.choices(values, k=len(values))) for _ in range(samples))
    return estimates[int(0.025 * samples)], estimates[min(samples - 1, int(0.975 * samples))]


def summarize(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["model"]),
            str(row["condition"]),
            str(row["memory_type"]),
            int(row["target_compression"]),
        )
        grouped[key].append(row)

    summary: list[dict[str, Any]] = []
    for group_index, (key, group) in enumerate(sorted(grouped.items())):
        em_values = [float(row["exact_match"]) for row in group]
        low, high = bootstrap_ci(em_values, seed + group_index)
        realized = [
            float(row["realized_compression"])
            for row in group
            if row.get("realized_compression") is not None
        ]
        vision = [
            float(row.get("vision_tokens") or row.get("grid_vision_tokens"))
            for row in group
            if row.get("vision_tokens") is not None or row.get("grid_vision_tokens") is not None
        ]
        latency = [float(row["latency_seconds"]) for row in group if row.get("latency_seconds")]
        summary.append(
            {
                "model": key[0],
                "condition": key[1],
                "memory_type": key[2],
                "target_compression": key[3],
                "n": len(group),
                "exact_match": mean(em_values),
                "exact_match_ci_low": low,
                "exact_match_ci_high": high,
                "cer": mean(float(row["cer"]) for row in group),
                "token_f1": mean(float(row["token_f1"]) for row in group),
                "mean_realized_compression": mean(realized) if realized else None,
                "mean_vision_tokens": mean(vision) if vision else None,
                "mean_latency_seconds": mean(latency) if latency else None,
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score optical-memory model outputs.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--output-dir", default="results/scored")
    parser.add_argument("--seed", type=int, default=20260826)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_path = project_path(args.results)
    output_dir = project_path(args.output_dir)
    scored = [score_row(row) for row in read_jsonl(results_path)]
    summary = summarize(scored, args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_example_path = output_dir / "per_example.csv"
    summary_path = output_dir / "summary.csv"
    write_csv(per_example_path, scored)
    write_csv(summary_path, summary)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"Scored {len(scored)} rows")
    print(f"Per-example metrics: {per_example_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
