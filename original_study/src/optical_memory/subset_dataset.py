from __future__ import annotations

import argparse

from .common import project_path, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter a dataset to episodes referenced by a manifest.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = project_path(args.dataset)
    manifest_path = project_path(args.manifest)
    selected_ids = {str(row["episode_id"]) for row in read_jsonl(manifest_path)}
    rows = [row for row in read_jsonl(dataset_path) if str(row["episode_id"]) in selected_ids]
    found_ids = {str(row["episode_id"]) for row in rows}
    missing = selected_ids - found_ids
    if missing:
        raise SystemExit(f"Manifest references missing dataset episodes: {sorted(missing)}")
    rows.sort(key=lambda row: (str(row["memory_type"]), str(row["episode_id"])))
    output = project_path(args.output)
    write_jsonl(output, rows)
    print(f"Wrote {len(rows)} selected episodes to {output}")


if __name__ == "__main__":
    main()

