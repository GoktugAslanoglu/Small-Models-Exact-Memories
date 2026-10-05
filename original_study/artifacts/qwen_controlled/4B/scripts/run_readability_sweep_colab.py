"""Run the 12/10/8/6 readability sweep in one resumable Colab process.

Default invocation::

    python /content/slm_agents/scripts/run_readability_sweep_colab.py

Pass ``--zip-output`` to also package the manifests, JSONL, and score tables.
Rendered PNGs are deliberately omitted from that compact result archive.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path("/content/slm_agents")
DATASET = Path("data/generated/readability_check.jsonl")
SOURCE_MANIFEST = Path("renders/generated/readability_controls/render_manifest.jsonl")
SWEEP_MANIFEST = Path("renders/generated/readability_sweep/render_manifest.jsonl")
RESULTS = Path("results/qwen35_2b_readability_sweep.jsonl")
SCORED_DIR = Path("results/scored/qwen35_2b_readability_sweep")
RESULT_ZIP = Path("results/qwen35_2b_readability_sweep_bundle.zip")
TARGET_FONTS = "12,10,8,6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare, validate, run, score, and optionally archive the "
            "Qwen3.5-2B 12/10/8/6 readability sweep."
        )
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--source-manifest", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--sweep-manifest", type=Path, default=SWEEP_MANIFEST)
    parser.add_argument("--source-condition", default="full_history_12px")
    parser.add_argument(
        "--target-fonts",
        default=TARGET_FONTS,
        help="Comma-separated source-scale sizes (default: 12,10,8,6).",
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--scored-dir", type=Path, default=SCORED_DIR)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Reuse an existing sweep manifest and its rendered images.",
    )
    parser.add_argument(
        "--zip-output",
        nargs="?",
        type=Path,
        const=RESULT_ZIP,
        default=None,
        help=(
            "Create a compact result ZIP; omit the value to use "
            "results/qwen35_2b_readability_sweep_bundle.zip."
        ),
    )
    return parser.parse_args()


def rooted(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"Missing {label}: {path}")


def run_live(command: Sequence[str], *, cwd: Path, label: str) -> None:
    """Stream stdout and stderr together, preserving the original failure."""

    printable = " ".join(shlex.quote(item) for item in command)
    print(f"\n=== {label} ===\n+ {printable}", flush=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        list(command),
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
            f"\n{label} failed with return code {return_code}.\n"
            f"Command: {printable}\n"
            "The original error is printed above. Inference is checkpointed after "
            "each row, so fix the error and rerun this script to resume."
        )


def parse_fonts(value: str) -> list[int]:
    try:
        fonts = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise SystemExit("--target-fonts must be comma-separated integers") from exc
    if not fonts or any(font <= 0 for font in fonts):
        raise SystemExit("--target-fonts requires positive integers")
    if len(fonts) != len(set(fonts)):
        raise SystemExit("--target-fonts cannot contain duplicates")
    return fonts


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at {path}:{number}: {exc}") from exc
            if not isinstance(row, dict):
                raise SystemExit(f"Expected a JSON object at {path}:{number}")
            rows.append(row)
    return rows


def inspect_manifest(
    manifest_path: Path,
    dataset_path: Path,
    requested_fonts: Sequence[int],
) -> tuple[list[str], dict[str, int], list[str]]:
    """Verify the Cartesian product and discover exact size-aware names."""

    dataset_rows = read_jsonl(dataset_path)
    manifest_rows = read_jsonl(manifest_path)
    episode_ids = [str(row["episode_id"]) for row in dataset_rows]
    if not episode_ids or len(episode_ids) != len(set(episode_ids)):
        raise SystemExit("Dataset must contain unique, nonempty episode IDs")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    keys: list[tuple[str, str]] = []
    for row in manifest_rows:
        condition = str(row["condition"])
        episode_id = str(row["episode_id"])
        grouped[condition].append(row)
        keys.append((episode_id, condition))
    if len(keys) != len(set(keys)):
        raise SystemExit("Sweep manifest has duplicate episode/condition rows")

    expected_ids = set(episode_ids)
    sides: dict[str, int] = {}
    for condition, rows in grouped.items():
        actual_ids = {str(row["episode_id"]) for row in rows}
        if len(rows) != len(episode_ids) or actual_ids != expected_ids:
            raise SystemExit(
                f"{condition} is not a complete {len(episode_ids)}-episode slice"
            )
        widths = {int(row["image_width"]) for row in rows}
        heights = {int(row["image_height"]) for row in rows}
        if len(widths) != 1 or heights != widths:
            raise SystemExit(f"{condition} does not have one consistent square size")
        sides[condition] = next(iter(widths))

    if len(grouped) != len(requested_fonts):
        raise SystemExit(
            f"Expected {len(requested_fonts)} conditions; found {len(grouped)}: "
            f"{sorted(grouped)}"
        )
    for font in requested_fonts:
        matches = [name for name in grouped if f"_{font}px_" in name]
        if len(matches) != 1:
            raise SystemExit(f"Expected one {font}px condition; found {matches}")

    conditions = sorted(grouped, key=lambda name: sides[name], reverse=True)
    expected_rows = len(episode_ids) * len(conditions)
    if len(manifest_rows) != expected_rows:
        raise SystemExit(
            f"Expected {expected_rows} unique manifest rows; found {len(manifest_rows)}"
        )

    print("\n=== Manifest integrity ===")
    print(f"episodes={len(episode_ids)} manifest_rows={len(manifest_rows)}")
    for condition in conditions:
        print(f"  {condition}: {len(grouped[condition])} rows at {sides[condition]}px")
    return conditions, sides, episode_ids


def check_resume(path: Path, conditions: Sequence[str], model: str) -> None:
    if not path.exists():
        return
    rows = read_jsonl(path)
    allowed = set(conditions)
    bad = [
        row
        for row in rows
        if str(row.get("condition")) not in allowed or str(row.get("model")) != model
    ]
    if bad:
        examples = [f"{row.get('condition')}/{row.get('model')}" for row in bad[:5]]
        raise SystemExit(
            f"Resume file is incompatible: {path}\nUnexpected rows: {examples}\n"
            "Move it or choose a new --results path; the script will not delete it."
        )
    keys = [
        (str(row.get("episode_id")), str(row.get("condition")), str(row.get("model")))
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise SystemExit(f"Resume file contains duplicate keys: {path}")


def validate_results(
    path: Path,
    *,
    conditions: Sequence[str],
    sides: dict[str, int],
    episode_ids: Sequence[str],
    model: str,
    max_new_tokens: int,
) -> None:
    rows = read_jsonl(path)
    keys = [
        (str(row.get("episode_id")), str(row.get("condition")), str(row.get("model")))
        for row in rows
    ]
    expected = {
        (episode_id, condition, model)
        for episode_id in episode_ids
        for condition in conditions
    }
    if len(keys) != len(set(keys)):
        raise SystemExit("Results contain duplicate episode/condition/model keys")
    if set(keys) != expected:
        missing = sorted(expected - set(keys))
        extra = sorted(set(keys) - expected)
        raise SystemExit(
            f"Incomplete/incompatible result matrix; missing={missing[:8]}, "
            f"extra={extra[:8]}"
        )

    counts = Counter(str(row["condition"]) for row in rows)
    for condition in conditions:
        if counts[condition] != len(episode_ids):
            raise SystemExit(
                f"{condition}: expected {len(episode_ids)} rows, got {counts[condition]}"
            )

    lookup = {
        (str(row["episode_id"]), str(row["condition"])): row for row in rows
    }
    ordered = sorted(conditions, key=lambda name: sides[name], reverse=True)
    for episode_id in episode_ids:
        episode_rows = [lookup[(episode_id, condition)] for condition in ordered]
        if len({int(row["text_tokens"]) for row in episode_rows}) != 1:
            raise SystemExit(f"{episode_id}: text_tokens vary across conditions")

        vision: list[float] = []
        compression: list[float] = []
        for row in episode_rows:
            token_value = row.get("vision_tokens") or row.get("grid_vision_tokens")
            ratio = row.get("realized_compression")
            if token_value is None or float(token_value) <= 0:
                raise SystemExit(f"{episode_id}: missing/nonpositive vision tokens")
            if ratio is None or float(ratio) <= 0:
                raise SystemExit(f"{episode_id}: missing/nonpositive compression")
            vision.append(float(token_value))
            compression.append(float(ratio))
        if any(b > a for a, b in zip(vision, vision[1:])):
            raise SystemExit(f"{episode_id}: vision tokens are not nonincreasing: {vision}")
        if any(b < a for a, b in zip(compression, compression[1:])):
            raise SystemExit(
                f"{episode_id}: realized compression is not nondecreasing: {compression}"
            )

    truncated = [
        row for row in rows if int(row.get("generated_tokens") or 0) >= max_new_tokens
    ]
    print("\n=== Result integrity ===")
    print(f"results={len(rows)} unique_keys={len(set(keys))}")
    print("Vision-token and realized-compression monotonicity: PASSED")
    if truncated:
        print(
            f"WARNING: {len(truncated)} prediction(s) reached max_new_tokens="
            f"{max_new_tokens}; inspect them for truncation."
        )


def print_predictions(path: Path, conditions: Sequence[str]) -> None:
    require_file(path, "per-example score table")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("condition") in conditions:
            grouped[str(row["condition"])].append(row)

    print("\n=== Sweep predictions ===")
    for condition in conditions:
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
        detail = []
        if vision:
            detail.append(f"mean vision tokens={mean(vision):.1f}")
        if compression:
            detail.append(f"mean realized compression={mean(compression):.3f}x")
        suffix = f" ({', '.join(detail)})" if detail else ""
        print(f"\n{condition}: {correct:g}/{len(selected)} correct{suffix}")
        for row in selected:
            print(
                f"  {row['memory_type']} | expected: {row['answer']!r} "
                f"| predicted: {row['clean_prediction']!r}"
            )


def archive_files(
    dataset: Path,
    source_manifest: Path,
    sweep_manifest: Path,
    results: Path,
    scored_dir: Path,
) -> Iterable[Path]:
    for path in (dataset, source_manifest, sweep_manifest, results):
        if path.is_file():
            yield path
    if scored_dir.is_dir():
        yield from sorted(path for path in scored_dir.rglob("*") if path.is_file())


def create_archive(
    output: Path,
    *,
    root: Path,
    dataset: Path,
    source_manifest: Path,
    sweep_manifest: Path,
    results: Path,
    scored_dir: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    files = list(
        dict.fromkeys(
            archive_files(dataset, source_manifest, sweep_manifest, results, scored_dir)
        )
    )
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            try:
                name = path.relative_to(root)
            except ValueError:
                name = Path("external") / path.name
            bundle.write(path, name.as_posix())
    print(f"\nResult archive: {output} ({len(files)} files; PNGs excluded)")


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Project root does not exist: {root}")
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive")

    dataset = rooted(root, args.dataset)
    source_manifest = rooted(root, args.source_manifest)
    sweep_manifest = rooted(root, args.sweep_manifest)
    results = rooted(root, args.results)
    scored_dir = rooted(root, args.scored_dir)
    fonts = parse_fonts(args.target_fonts)
    require_file(dataset, "readability-control dataset")

    if not args.skip_prepare:
        require_file(source_manifest, "12px readability-control manifest")
        sweep_manifest.parent.mkdir(parents=True, exist_ok=True)
        run_live(
            [
                sys.executable,
                "-u",
                "-m",
                "optical_memory.readability_sweep",
                "--manifest",
                str(source_manifest),
                "--output",
                str(sweep_manifest),
                "--source-condition",
                args.source_condition,
                "--target-fonts",
                ",".join(str(font) for font in fonts),
            ],
            cwd=root,
            label="Prepare 12/10/8/6 readability sweep",
        )
    require_file(sweep_manifest, "readability-sweep manifest")

    run_live(
        [
            sys.executable,
            "-u",
            "-m",
            "optical_memory.validate",
            "--dataset",
            str(dataset),
            "--manifest",
            str(sweep_manifest),
        ],
        cwd=root,
        label="Validate sweep data and images",
    )
    conditions, sides, episode_ids = inspect_manifest(sweep_manifest, dataset, fonts)
    check_resume(results, conditions, args.model)

    results.parent.mkdir(parents=True, exist_ok=True)
    run_live(
        [
            sys.executable,
            "-u",
            "-m",
            "optical_memory.colab_runner",
            "--mode",
            "optical",
            "--dataset",
            str(dataset),
            "--manifest",
            str(sweep_manifest),
            "--model",
            args.model,
            "--conditions",
            ",".join(conditions),
            "--output",
            str(results),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--seed",
            str(args.seed),
            "--resume",
        ],
        cwd=root,
        label=f"Run {args.model} sweep (resumable)",
    )
    validate_results(
        results,
        conditions=conditions,
        sides=sides,
        episode_ids=episode_ids,
        model=args.model,
        max_new_tokens=args.max_new_tokens,
    )

    run_live(
        [
            sys.executable,
            "-u",
            "-m",
            "optical_memory.score",
            "--results",
            str(results),
            "--output-dir",
            str(scored_dir),
            "--seed",
            str(args.seed),
        ],
        cwd=root,
        label="Score sweep results",
    )
    print_predictions(scored_dir / "per_example.csv", conditions)

    if args.zip_output is not None:
        create_archive(
            rooted(root, args.zip_output),
            root=root,
            dataset=dataset,
            source_manifest=source_manifest,
            sweep_manifest=sweep_manifest,
            results=results,
            scored_dir=scored_dir,
        )
    print("\nReadability sweep complete.")


if __name__ == "__main__":
    main()
