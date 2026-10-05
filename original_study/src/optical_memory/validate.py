from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from PIL import Image

from .common import project_path, read_jsonl
from .exactspan import detect_exact_spans, format_protection_strip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated pilot data and render invariants.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = project_path(args.dataset)
    manifest_path = project_path(args.manifest)
    episodes = list(read_jsonl(dataset_path))
    rows = list(read_jsonl(manifest_path))
    errors: list[str] = []

    ids = [str(row["episode_id"]) for row in episodes]
    if len(ids) != len(set(ids)):
        errors.append("dataset contains duplicate episode_id values")
    episode_map = {str(row["episode_id"]): row for row in episodes}

    bucket_counts = Counter(str(row["position_bucket"]) for row in episodes)
    if max(bucket_counts.values()) - min(bucket_counts.values()) > 1:
        errors.append(f"position buckets are unbalanced: {dict(bucket_counts)}")

    for episode in episodes:
        answer = str(episode["answer"])
        history = str(episode["history"])
        question = str(episode["question"])
        if answer not in history:
            errors.append(f"{episode['episode_id']}: answer is absent from history")
        if answer in question:
            errors.append(f"{episode['episode_id']}: question leaks the answer")
        if episode["scoring"] == "exact":
            strip = format_protection_strip(detect_exact_spans(history))
            if answer not in strip:
                errors.append(f"{episode['episode_id']}: syntax detector misses exact target")

    condition_counts = Counter(str(row["condition"]) for row in rows)
    for condition, count in condition_counts.items():
        if count != len(episodes):
            errors.append(f"{condition}: expected {len(episodes)} rows, found {count}")

    row_map = {(str(row["episode_id"]), str(row["condition"])): row for row in rows}
    for row in rows:
        if str(row["episode_id"]) not in episode_map:
            errors.append(f"manifest references unknown episode {row['episode_id']}")
        for relative in row["image_paths"]:
            path = (manifest_path.parent / str(relative)).resolve()
            if not path.exists():
                errors.append(f"missing image: {path}")
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception as exc:  # pragma: no cover - diagnostic path
                errors.append(f"invalid image {path}: {exc}")

    exact_targets = sorted(
        {
            int(row["target_compression"])
            for row in rows
            if row["render_kind"] == "exactstrip"
        }
    )
    for episode_id in ids:
        for compression in exact_targets:
            uniform = row_map.get((episode_id, f"uniform_c{compression}"))
            strip = row_map.get((episode_id, f"exactstrip_c{compression}"))
            if uniform is None or strip is None:
                errors.append(f"{episode_id} c{compression}: missing matched render pair")
                continue
            invariant_fields = (
                "page_count",
                "image_width",
                "image_height",
                "total_pixel_budget",
            )
            for field in invariant_fields:
                if uniform.get(field) != strip.get(field):
                    errors.append(
                        f"{episode_id} c{compression}: {field} differs "
                        f"({uniform.get(field)} != {strip.get(field)})"
                    )

    if errors:
        print("VALIDATION FAILED")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("VALIDATION PASSED")
    print(f"episodes={len(episodes)}")
    print(f"manifest_rows={len(rows)}")
    print(f"conditions={dict(sorted(condition_counts.items()))}")
    print(f"position_buckets={dict(sorted(bucket_counts.items()))}")
    print(f"validated_images={sum(len(row['image_paths']) for row in rows)}")
    print(f"matched_exactstrip_budgets={exact_targets}")


if __name__ == "__main__":
    main()

