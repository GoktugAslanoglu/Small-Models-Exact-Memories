from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any

from .common import project_path, read_jsonl, write_jsonl


BUCKET_ORDER = ("early", "middle", "late")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a stratified pilot render manifest.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-type", type=int, default=1)
    parser.add_argument(
        "--conditions",
        default="uniform_c4,uniform_c8,uniform_c16,exactstrip_c8,exactstrip_c16",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.per_type < 1:
        raise SystemExit("--per-type must be positive")
    dataset_path = project_path(args.dataset)
    manifest_path = project_path(args.manifest)
    output_path = project_path(args.output)
    episodes = list(read_jsonl(dataset_path))
    rows = list(read_jsonl(manifest_path))
    conditions = {value.strip() for value in args.conditions.split(",") if value.strip()}

    grouped: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for episode in sorted(episodes, key=lambda row: str(row["episode_id"])):
        grouped[str(episode["memory_type"])][str(episode["position_bucket"])].append(
            str(episode["episode_id"])
        )

    selected: set[str] = set()
    for memory_type, by_bucket in sorted(grouped.items()):
        cursors = {bucket: 0 for bucket in BUCKET_ORDER}
        while len([episode_id for episode_id in selected if episode_id.startswith(memory_type)]) < args.per_type:
            made_progress = False
            for bucket in BUCKET_ORDER:
                values = by_bucket.get(bucket, [])
                cursor = cursors[bucket]
                if cursor < len(values):
                    selected.add(values[cursor])
                    cursors[bucket] += 1
                    made_progress = True
                    if len([episode_id for episode_id in selected if episode_id.startswith(memory_type)]) >= args.per_type:
                        break
            if not made_progress:
                raise SystemExit(
                    f"Requested {args.per_type} episodes for {memory_type}, but only "
                    f"{sum(len(values) for values in by_bucket.values())} exist."
                )

    subset = [
        row
        for row in rows
        if str(row["episode_id"]) in selected and str(row["condition"]) in conditions
    ]
    present_conditions = {str(row["condition"]) for row in subset}
    missing = conditions - present_conditions
    if missing:
        raise SystemExit(f"Conditions missing from source manifest: {sorted(missing)}")
    write_jsonl(output_path, subset)
    print(f"Selected episodes: {len(selected)}")
    print(f"Manifest rows: {len(subset)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()

