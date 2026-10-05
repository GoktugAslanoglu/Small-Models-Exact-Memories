"""Run the 18-episode early/middle/late Qwen3.5-2B calibration pilot.

Default Colab invocation::

    python -u /content/slm_agents/scripts/run_balanced18_pilot_colab.py --zip-output

The script prepares deterministic readable sources, derives the 10/8/6 scale
sweep, runs 54 optical and 18 raw-text calls with resume support, validates the
72-result matrix, scores it, and optionally creates a compact result ZIP.
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

from optical_memory.common import read_jsonl, write_jsonl
from optical_memory.generate import MEMORY_TYPES, POSITION_BUCKETS


PROJECT_ROOT = Path("/content/slm_agents")
SOURCE_DATASET = Path("data/generated/pilot.jsonl")
BALANCED_DATASET = Path("data/generated/balanced18.jsonl")
SOURCE_RENDER_DIR = Path("renders/generated/balanced18_controls")
SWEEP_MANIFEST = Path("renders/generated/balanced18_sweep/render_manifest.jsonl")
OPTICAL_RESULTS = Path("results/qwen35_2b_balanced18_optical.jsonl")
RAW_RESULTS = Path("results/qwen35_2b_balanced18_raw.jsonl")
COMBINED_RESULTS = Path("results/qwen35_2b_balanced18_combined.jsonl")
OPTICAL_SCORED = Path("results/scored/qwen35_2b_balanced18_optical")
COMBINED_SCORED = Path("results/scored/qwen35_2b_balanced18_combined")
RESULT_ZIP = Path("results/qwen35_2b_balanced18_bundle.zip")
TARGET_FONTS = "10,8,6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Qwen3.5-2B balanced 18-episode 10/8/6 pilot."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Reuse existing balanced dataset, readable sources, and sweep images.",
    )
    parser.add_argument(
        "--zip-output",
        nargs="?",
        type=Path,
        const=RESULT_ZIP,
        default=None,
        help="Create the compact result ZIP at the default or supplied path.",
    )
    return parser.parse_args()


def rooted(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def run_live(command: Sequence[str], *, cwd: Path, label: str) -> None:
    printable = " ".join(shlex.quote(str(item)) for item in command)
    print(f"\n=== {label} ===\n+ {printable}", flush=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        [str(item) for item in command],
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return_code = process.wait()
    if return_code:
        raise SystemExit(
            f"{label} failed with return code {return_code}. "
            "The original error is printed above; rerun after fixing it."
        )


def locate_font() -> Path:
    try:
        from matplotlib import font_manager

        candidate = Path(
            font_manager.findfont("DejaVu Sans Mono", fallback_to_default=False)
        )
    except Exception as exc:
        raise SystemExit(
            "Matplotlib's DejaVu Sans Mono font is required. Re-run the notebook "
            "dependency cell before this script."
        ) from exc
    if not candidate.is_file():
        raise SystemExit(f"Resolved font does not exist: {candidate}")
    print(f"Deterministic font: {candidate}")
    return candidate


def validate_balanced_dataset(path: Path) -> list[dict[str, Any]]:
    rows = list(read_jsonl(path))
    if len(rows) != 18:
        raise SystemExit(f"Expected 18 balanced episodes; found {len(rows)}")
    ids = [str(row["episode_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("Balanced dataset contains duplicate episode IDs")
    type_counts = Counter(str(row["memory_type"]) for row in rows)
    bucket_counts = Counter(str(row["position_bucket"]) for row in rows)
    pairs = Counter(
        (str(row["memory_type"]), str(row["position_bucket"])) for row in rows
    )
    expected_pairs = {
        (memory_type, bucket)
        for memory_type in MEMORY_TYPES
        for bucket in POSITION_BUCKETS
    }
    if type_counts != Counter({memory_type: 3 for memory_type in MEMORY_TYPES}):
        raise SystemExit(f"Unbalanced memory types: {dict(type_counts)}")
    if bucket_counts != Counter({bucket: 6 for bucket in POSITION_BUCKETS}):
        raise SystemExit(f"Unbalanced positions: {dict(bucket_counts)}")
    if set(pairs) != expected_pairs or any(count != 1 for count in pairs.values()):
        raise SystemExit("Dataset is not a one-per-type/position Cartesian grid")
    print("Balanced dataset integrity: PASSED (18 episodes; 6 types x 3 positions)")
    return rows


def validate_result_matrix(
    dataset_rows: Sequence[dict[str, Any]],
    optical_path: Path,
    raw_path: Path,
    model: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    optical = list(read_jsonl(optical_path))
    raw = list(read_jsonl(raw_path))
    episode_ids = {str(row["episode_id"]) for row in dataset_rows}
    conditions = sorted({str(row["condition"]) for row in optical})
    if len(conditions) != 3:
        raise SystemExit(f"Expected three optical conditions; found {conditions}")
    if len(optical) != 54 or len(raw) != 18:
        raise SystemExit(
            f"Expected 54 optical and 18 raw rows; found {len(optical)} and {len(raw)}"
        )
    expected_optical = {
        (episode_id, condition, model)
        for episode_id in episode_ids
        for condition in conditions
    }
    expected_raw = {(episode_id, "raw_text", model) for episode_id in episode_ids}
    optical_keys = {
        (str(row["episode_id"]), str(row["condition"]), str(row["model"]))
        for row in optical
    }
    raw_keys = {
        (str(row["episode_id"]), str(row["condition"]), str(row["model"]))
        for row in raw
    }
    if optical_keys != expected_optical or raw_keys != expected_raw:
        raise SystemExit("Result rows do not match the expected 18-episode matrix")
    if len(optical_keys) != len(optical) or len(raw_keys) != len(raw):
        raise SystemExit("Result files contain duplicate resume keys")
    combined = optical + raw
    print("Result matrix integrity: PASSED (54 optical + 18 raw = 72)")
    return combined, conditions


def print_summary(per_example: Path, conditions: Sequence[str]) -> None:
    with per_example.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    order = list(conditions) + ["raw_text"]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition"])].append(row)

    print("\n=== Aggregate results ===")
    for condition in order:
        selected = grouped.get(condition, [])
        correct = sum(float(row["exact_match"]) for row in selected)
        vision = [
            float(row["vision_tokens"] or row["grid_vision_tokens"])
            for row in selected
            if row.get("vision_tokens") or row.get("grid_vision_tokens")
        ]
        compression = [
            float(row["realized_compression"])
            for row in selected
            if row.get("realized_compression")
        ]
        details: list[str] = []
        if vision:
            details.append(f"mean vision={mean(vision):.1f}")
        if compression:
            details.append(f"mean compression={mean(compression):.3f}x")
        suffix = f" ({', '.join(details)})" if details else ""
        print(f"{condition}: {correct:g}/{len(selected)} correct{suffix}")

    print("\n=== Accuracy by position ===")
    for condition in order:
        values = []
        for bucket in POSITION_BUCKETS:
            selected = [
                row
                for row in grouped.get(condition, [])
                if row["position_bucket"] == bucket
            ]
            correct = sum(float(row["exact_match"]) for row in selected)
            values.append(f"{bucket}={correct:g}/{len(selected)}")
        print(f"{condition}: " + ", ".join(values))

    print("\n=== Accuracy by memory type ===")
    for condition in order:
        values = []
        for memory_type in MEMORY_TYPES:
            selected = [
                row
                for row in grouped.get(condition, [])
                if row["memory_type"] == memory_type
            ]
            correct = sum(float(row["exact_match"]) for row in selected)
            values.append(f"{memory_type}={correct:g}/{len(selected)}")
        print(f"{condition}: " + ", ".join(values))


def archive_inputs(
    dataset: Path,
    source_manifest: Path,
    sweep_manifest: Path,
    optical: Path,
    raw: Path,
    combined: Path,
    optical_scored: Path,
    combined_scored: Path,
) -> Iterable[Path]:
    for path in (dataset, source_manifest, sweep_manifest, optical, raw, combined):
        if path.is_file():
            yield path
    for directory in (optical_scored, combined_scored):
        if directory.is_dir():
            yield from sorted(path for path in directory.rglob("*") if path.is_file())


def create_archive(output: Path, root: Path, paths: Iterable[Path]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in dict.fromkeys(paths):
            try:
                name = path.relative_to(root)
            except ValueError:
                name = Path("external") / path.name
            bundle.write(path, name.as_posix())
    print(f"Result archive: {output}")


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Project root does not exist: {root}")
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive")

    source_dataset = rooted(root, SOURCE_DATASET)
    balanced_dataset = rooted(root, BALANCED_DATASET)
    source_render_dir = rooted(root, SOURCE_RENDER_DIR)
    source_manifest = source_render_dir / "render_manifest.jsonl"
    sweep_manifest = rooted(root, SWEEP_MANIFEST)
    optical_results = rooted(root, OPTICAL_RESULTS)
    raw_results = rooted(root, RAW_RESULTS)
    combined_results = rooted(root, COMBINED_RESULTS)
    optical_scored = rooted(root, OPTICAL_SCORED)
    combined_scored = rooted(root, COMBINED_SCORED)

    if not args.skip_prepare:
        font = locate_font()
        run_live(
            [
                sys.executable,
                "-u",
                "-m",
                "optical_memory.balanced_pilot",
                "--dataset",
                str(source_dataset),
                "--output-dataset",
                str(balanced_dataset),
                "--output-render-dir",
                str(source_render_dir),
                "--font-path",
                str(font),
            ],
            cwd=root,
            label="Prepare 18 balanced readable sources",
        )
    if not balanced_dataset.is_file() or not source_manifest.is_file():
        raise SystemExit("Balanced dataset or readable source manifest is missing")
    dataset_rows = validate_balanced_dataset(balanced_dataset)

    sweep_command = [
        sys.executable,
        "-u",
        str(root / "scripts" / "run_readability_sweep_colab.py"),
        "--project-root",
        str(root),
        "--dataset",
        str(balanced_dataset),
        "--source-manifest",
        str(source_manifest),
        "--sweep-manifest",
        str(sweep_manifest),
        "--target-fonts",
        TARGET_FONTS,
        "--model",
        args.model,
        "--results",
        str(optical_results),
        "--scored-dir",
        str(optical_scored),
        "--seed",
        str(args.seed),
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if args.skip_prepare:
        sweep_command.append("--skip-prepare")
    run_live(
        sweep_command,
        cwd=root,
        label="Run balanced 10/8/6 optical sweep",
    )

    raw_results.parent.mkdir(parents=True, exist_ok=True)
    run_live(
        [
            sys.executable,
            "-u",
            "-m",
            "optical_memory.colab_runner",
            "--mode",
            "raw",
            "--dataset",
            str(balanced_dataset),
            "--model",
            args.model,
            "--output",
            str(raw_results),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--seed",
            str(args.seed),
            "--resume",
        ],
        cwd=root,
        label="Run balanced raw-text upper bound",
    )

    combined, conditions = validate_result_matrix(
        dataset_rows,
        optical_results,
        raw_results,
        args.model,
    )
    write_jsonl(combined_results, combined)
    run_live(
        [
            sys.executable,
            "-u",
            "-m",
            "optical_memory.score",
            "--results",
            str(combined_results),
            "--output-dir",
            str(combined_scored),
            "--seed",
            str(args.seed),
        ],
        cwd=root,
        label="Score combined balanced pilot",
    )
    print_summary(combined_scored / "per_example.csv", conditions)

    if args.zip_output is not None:
        output = rooted(root, args.zip_output)
        create_archive(
            output,
            root,
            archive_inputs(
                balanced_dataset,
                source_manifest,
                sweep_manifest,
                optical_results,
                raw_results,
                combined_results,
                optical_scored,
                combined_scored,
            ),
        )
    print("\nBalanced 18-episode pilot complete.")


if __name__ == "__main__":
    main()
