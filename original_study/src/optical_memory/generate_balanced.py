from __future__ import annotations

import argparse
from typing import Any

from .common import load_json, project_path, write_jsonl
from .generate import MEMORY_TYPES, generate_episode


def generate_balanced_dataset(config: dict[str, Any]) -> list[dict[str, object]]:
    """Generate the pilot with globally balanced target-position buckets.

    Ten samples cannot be split equally into three buckets within one class. Rotating
    the bucket offset across six classes yields 20 early, 20 middle, and 20 late
    examples globally while retaining a 4/3/3 permutation within each class.
    """

    seed = int(config["seed"])
    samples_per_type = int(config["samples_per_type"])
    target_chars = int(config["target_chars"])
    rows: list[dict[str, object]] = []
    for type_index, memory_type in enumerate(MEMORY_TYPES):
        for public_index in range(samples_per_type):
            internal_index = public_index + type_index
            row = generate_episode(memory_type, internal_index, seed, target_chars)
            row["sample_index"] = public_index
            row["episode_id"] = f"{memory_type}-{public_index:03d}"
            rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a position-balanced pilot dataset.")
    parser.add_argument("--config", default="configs/pilot_pages.json")
    parser.add_argument("--output", help="Override output JSONL path from the config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(project_path(args.config))
    rows = generate_balanced_dataset(config)
    output = project_path(args.output or config["output"])
    write_jsonl(output, rows)
    bucket_counts = {
        bucket: sum(row["position_bucket"] == bucket for row in rows)
        for bucket in ("early", "middle", "late")
    }
    print(f"Wrote {len(rows)} episodes to {output}")
    print(f"Position buckets: {bucket_counts}")


if __name__ == "__main__":
    main()

