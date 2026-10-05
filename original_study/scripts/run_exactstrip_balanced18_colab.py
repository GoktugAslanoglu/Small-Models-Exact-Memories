"""Run the paired balanced-18 ExactStrip experiment on Google Colab.

Default invocation::

    python -u scripts/run_exactstrip_balanced18_colab.py --zip-output

The script renders ExactStrip at the exact page counts and dimensions of the
existing uniform 8px/6px conditions, performs 36 resumable Qwen3.5-2B calls,
verifies equal measured vision-token budgets, scores the paired comparison, and
creates a compact result ZIP. Existing uniform predictions are never rerun.
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
from optical_memory.exactstrip_sweep import (
    DEFAULT_REFERENCE_CONDITIONS,
    build_exactstrip_sweep,
    parse_conditions,
)
from optical_memory.generate import MEMORY_TYPES, POSITION_BUCKETS


PROJECT_ROOT = Path("/content/slm_agents")
DATASET = Path("data/generated/balanced18.jsonl")
UNIFORM_MANIFEST = Path("renders/generated/balanced18_sweep/render_manifest.jsonl")
EXACTSTRIP_MANIFEST = Path(
    "renders/generated/balanced18_exactstrip/render_manifest.jsonl"
)
MATCHED_UNIFORM_MANIFEST = Path(
    "renders/generated/balanced18_exactstrip/matched_uniform_manifest.jsonl"
)
UNIFORM_RESULTS = Path("results/qwen35_2b_balanced18_optical.jsonl")
EXACTSTRIP_RESULTS = Path("results/qwen35_2b_balanced18_exactstrip.jsonl")
PAIRED_RESULTS = Path("results/qwen35_2b_balanced18_uniform_vs_exactstrip.jsonl")
EXACTSTRIP_SCORED = Path("results/scored/qwen35_2b_balanced18_exactstrip")
PAIRED_SCORED = Path("results/scored/qwen35_2b_balanced18_uniform_vs_exactstrip")
RESULT_ZIP = Path("results/qwen35_2b_balanced18_exactstrip_bundle.zip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired ExactStrip versus uniform 8px/6px on balanced-18."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--uniform-manifest", type=Path, default=UNIFORM_MANIFEST)
    parser.add_argument("--exactstrip-manifest", type=Path, default=EXACTSTRIP_MANIFEST)
    parser.add_argument("--uniform-results", type=Path, default=UNIFORM_RESULTS)
    parser.add_argument("--results", type=Path, default=EXACTSTRIP_RESULTS)
    parser.add_argument("--paired-results", type=Path, default=PAIRED_RESULTS)
    parser.add_argument("--scored-dir", type=Path, default=EXACTSTRIP_SCORED)
    parser.add_argument("--paired-scored-dir", type=Path, default=PAIRED_SCORED)
    parser.add_argument(
        "--reference-conditions",
        type=parse_conditions,
        default=DEFAULT_REFERENCE_CONDITIONS,
        help="Comma-separated uniform conditions to match exactly.",
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Reuse an existing, already validated ExactStrip manifest and images.",
    )
    parser.add_argument(
        "--zip-output",
        nargs="?",
        type=Path,
        const=RESULT_ZIP,
        default=None,
        help="Create a compact result ZIP at the default or supplied path.",
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
    expected_pairs = {
        (memory_type, bucket)
        for memory_type in MEMORY_TYPES
        for bucket in POSITION_BUCKETS
    }
    actual_pairs = {
        (str(row["memory_type"]), str(row["position_bucket"])) for row in rows
    }
    if type_counts != Counter({memory_type: 3 for memory_type in MEMORY_TYPES}):
        raise SystemExit(f"Unbalanced memory types: {dict(type_counts)}")
    if bucket_counts != Counter({bucket: 6 for bucket in POSITION_BUCKETS}):
        raise SystemExit(f"Unbalanced positions: {dict(bucket_counts)}")
    if actual_pairs != expected_pairs:
        raise SystemExit("Dataset is not a one-per-type/position Cartesian grid")
    print("Balanced dataset integrity: PASSED (18 episodes; 6 types x 3 positions)")
    return rows


def validate_exactstrip_manifest(
    dataset_rows: Sequence[dict[str, Any]],
    exactstrip_manifest: Path,
    uniform_manifest: Path,
    reference_conditions: Sequence[str],
) -> tuple[list[dict[str, Any]], list[tuple[str, str]], list[dict[str, Any]]]:
    rows = list(read_jsonl(exactstrip_manifest))
    episode_ids = {str(row["episode_id"]) for row in dataset_rows}
    uniform_rows = [
        row
        for row in read_jsonl(uniform_manifest)
        if str(row.get("condition")) in reference_conditions
    ]
    uniform_lookup = {
        (str(row["episode_id"]), str(row["condition"])): row for row in uniform_rows
    }
    if len(uniform_lookup) != 18 * len(reference_conditions):
        raise SystemExit("Uniform reference manifest is incomplete or contains duplicates")

    pairs_by_uniform: dict[str, str] = {}
    exact_keys: set[tuple[str, str]] = set()
    for row in rows:
        episode_id = str(row["episode_id"])
        condition = str(row["condition"])
        uniform_condition = str(row.get("matched_uniform_condition", ""))
        if episode_id not in episode_ids:
            raise SystemExit(f"ExactStrip manifest has unknown episode: {episode_id}")
        if uniform_condition not in reference_conditions:
            raise SystemExit(
                f"{episode_id}/{condition} has invalid matched condition {uniform_condition!r}"
            )
        reference = uniform_lookup[(episode_id, uniform_condition)]
        paired_fields = ("page_count", "image_width", "image_height", "total_pixel_budget")
        for field in paired_fields:
            if int(row[field]) != int(reference[field]):
                raise SystemExit(
                    f"Budget mismatch for {episode_id}: ExactStrip {field}={row[field]} "
                    f"but {uniform_condition} has {reference[field]}"
                )
        if str(row.get("detector_input")) != "history_only":
            raise SystemExit(f"{episode_id}/{condition}: detector provenance is invalid")
        for value in row["image_paths"]:
            if not (exactstrip_manifest.parent / str(value)).is_file():
                raise SystemExit(f"Missing ExactStrip image: {value}")
        pairs_by_uniform.setdefault(uniform_condition, condition)
        if pairs_by_uniform[uniform_condition] != condition:
            raise SystemExit(f"Uniform condition {uniform_condition} maps to multiple outputs")
        key = (episode_id, condition)
        if key in exact_keys:
            raise SystemExit(f"Duplicate ExactStrip manifest key: {key}")
        exact_keys.add(key)

    expected = {
        (episode_id, condition)
        for episode_id in episode_ids
        for condition in pairs_by_uniform.values()
    }
    if len(pairs_by_uniform) != len(reference_conditions) or exact_keys != expected:
        raise SystemExit("ExactStrip manifest is not a complete 18 x 2 paired matrix")
    pairs = [
        (condition, pairs_by_uniform[condition]) for condition in reference_conditions
    ]
    print("ExactStrip manifest integrity: PASSED (36 rows; equal page/pixel budgets)")
    for uniform_condition, exact_condition in pairs:
        sample = next(row for row in rows if row["condition"] == exact_condition)
        print(
            f"  {uniform_condition} <-> {exact_condition}: "
            f"{sample['page_count']} pages at "
            f"{sample['image_width']}x{sample['image_height']}"
        )
    return rows, pairs, uniform_rows


def _measured_tokens(row: dict[str, Any]) -> int:
    value = row.get("vision_tokens") or row.get("grid_vision_tokens")
    if value is None:
        raise SystemExit(
            f"{row.get('episode_id')}/{row.get('condition')} has no vision-token count"
        )
    return int(value)


def validate_results(
    dataset_rows: Sequence[dict[str, Any]],
    exactstrip_manifest_rows: Sequence[dict[str, Any]],
    uniform_results_path: Path,
    exactstrip_results_path: Path,
    model: str,
    pairs: Sequence[tuple[str, str]],
) -> list[dict[str, Any]]:
    episode_ids = {str(row["episode_id"]) for row in dataset_rows}
    exact_results = list(read_jsonl(exactstrip_results_path))
    exact_conditions = {exact for _, exact in pairs}
    expected_exact = {
        (episode_id, condition, model)
        for episode_id in episode_ids
        for condition in exact_conditions
    }
    actual_exact = [
        (str(row["episode_id"]), str(row["condition"]), str(row["model"]))
        for row in exact_results
    ]
    if len(actual_exact) != len(set(actual_exact)) or set(actual_exact) != expected_exact:
        raise SystemExit("ExactStrip results are incomplete, duplicated, or from another model")

    uniform_conditions = {uniform for uniform, _ in pairs}
    uniform_results = [
        row
        for row in read_jsonl(uniform_results_path)
        if str(row.get("condition")) in uniform_conditions
    ]
    expected_uniform = {
        (episode_id, condition, model)
        for episode_id in episode_ids
        for condition in uniform_conditions
    }
    actual_uniform = [
        (str(row["episode_id"]), str(row["condition"]), str(row["model"]))
        for row in uniform_results
    ]
    if len(actual_uniform) != len(set(actual_uniform)) or set(actual_uniform) != expected_uniform:
        raise SystemExit("Existing uniform results do not contain the required paired matrix")

    exact_lookup = {
        (str(row["episode_id"]), str(row["condition"])): row for row in exact_results
    }
    uniform_lookup = {
        (str(row["episode_id"]), str(row["condition"])): row for row in uniform_results
    }
    manifest_lookup = {
        (str(row["episode_id"]), str(row["condition"])): row
        for row in exactstrip_manifest_rows
    }
    for uniform_condition, exact_condition in pairs:
        for episode_id in episode_ids:
            uniform = uniform_lookup[(episode_id, uniform_condition)]
            exact = exact_lookup[(episode_id, exact_condition)]
            manifest = manifest_lookup[(episode_id, exact_condition)]
            if _measured_tokens(uniform) != _measured_tokens(exact):
                raise SystemExit(
                    f"Measured token mismatch for {episode_id}: "
                    f"{uniform_condition}={_measured_tokens(uniform)}, "
                    f"{exact_condition}={_measured_tokens(exact)}"
                )
            for field in ("image_width", "image_height"):
                if int(exact[field]) != int(uniform[field]) or int(exact[field]) != int(
                    manifest[field]
                ):
                    raise SystemExit(f"Result dimension mismatch for {episode_id}/{field}")

    print("Measured-budget integrity: PASSED (paired vision-token counts are identical)")
    return uniform_results + exact_results


def print_summary(per_example: Path, pairs: Sequence[tuple[str, str]]) -> None:
    with per_example.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition"])].append(row)

    order = [condition for pair in pairs for condition in pair]
    print("\n=== Paired aggregate results ===")
    for condition in order:
        selected = grouped[condition]
        correct = sum(float(row["exact_match"]) for row in selected)
        vision = [
            float(row["vision_tokens"] or row["grid_vision_tokens"])
            for row in selected
        ]
        compression = [float(row["realized_compression"]) for row in selected]
        print(
            f"{condition}: {correct:g}/{len(selected)} correct "
            f"(mean vision={mean(vision):.1f}, "
            f"mean compression={mean(compression):.3f}x)"
        )

    print("\n=== Paired changes ===")
    for uniform_condition, exact_condition in pairs:
        uniform = {row["episode_id"]: row for row in grouped[uniform_condition]}
        exact = {row["episode_id"]: row for row in grouped[exact_condition]}
        gains = losses = ties = 0
        for episode_id in sorted(uniform):
            before = float(uniform[episode_id]["exact_match"])
            after = float(exact[episode_id]["exact_match"])
            if after > before:
                gains += 1
            elif after < before:
                losses += 1
            else:
                ties += 1
        print(
            f"{uniform_condition} -> {exact_condition}: "
            f"gains={gains}, losses={losses}, ties={ties}"
        )

    print("\n=== Accuracy by memory type ===")
    for condition in order:
        values = []
        for memory_type in MEMORY_TYPES:
            selected = [
                row for row in grouped[condition] if row["memory_type"] == memory_type
            ]
            correct = sum(float(row["exact_match"]) for row in selected)
            values.append(f"{memory_type}={correct:g}/{len(selected)}")
        print(f"{condition}: " + ", ".join(values))

    print("\n=== Accuracy by position ===")
    for condition in order:
        values = []
        for bucket in POSITION_BUCKETS:
            selected = [
                row for row in grouped[condition] if row["position_bucket"] == bucket
            ]
            correct = sum(float(row["exact_match"]) for row in selected)
            values.append(f"{bucket}={correct:g}/{len(selected)}")
        print(f"{condition}: " + ", ".join(values))


def archive_inputs(
    dataset: Path,
    exactstrip_manifest: Path,
    matched_uniform_manifest: Path,
    exactstrip_results: Path,
    paired_results: Path,
    exactstrip_scored: Path,
    paired_scored: Path,
) -> Iterable[Path]:
    for path in (
        dataset,
        exactstrip_manifest,
        matched_uniform_manifest,
        exactstrip_results,
        paired_results,
    ):
        if path.is_file():
            yield path
    for directory in (exactstrip_scored, paired_scored):
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

    dataset = rooted(root, args.dataset)
    uniform_manifest = rooted(root, args.uniform_manifest)
    exactstrip_manifest = rooted(root, args.exactstrip_manifest)
    matched_uniform_manifest = rooted(root, MATCHED_UNIFORM_MANIFEST)
    uniform_results = rooted(root, args.uniform_results)
    exactstrip_results = rooted(root, args.results)
    paired_results = rooted(root, args.paired_results)
    exactstrip_scored = rooted(root, args.scored_dir)
    paired_scored = rooted(root, args.paired_scored_dir)

    for required in (dataset, uniform_manifest, uniform_results):
        if not required.is_file():
            raise SystemExit(
                f"Required prior-stage file is missing: {required}\n"
                "Continue in the balanced-18 Colab runtime or restore its result bundle."
            )
    dataset_rows = validate_balanced_dataset(dataset)

    if not args.skip_prepare:
        font = locate_font()
        print("\n=== Render paired ExactStrip 8px/6px images ===", flush=True)
        rendered = build_exactstrip_sweep(
            dataset,
            uniform_manifest,
            exactstrip_manifest,
            reference_conditions=args.reference_conditions,
            font_path=font,
        )
        print(f"Rendered {len(rendered)} ExactStrip rows to {exactstrip_manifest}")
    if not exactstrip_manifest.is_file():
        raise SystemExit(f"ExactStrip manifest is missing: {exactstrip_manifest}")

    manifest_rows, pairs, matched_uniform_rows = validate_exactstrip_manifest(
        dataset_rows,
        exactstrip_manifest,
        uniform_manifest,
        args.reference_conditions,
    )
    write_jsonl(matched_uniform_manifest, matched_uniform_rows)
    exact_conditions = [exact for _, exact in pairs]

    exactstrip_results.parent.mkdir(parents=True, exist_ok=True)
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
            str(exactstrip_manifest),
            "--conditions",
            ",".join(exact_conditions),
            "--model",
            args.model,
            "--output",
            str(exactstrip_results),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--seed",
            str(args.seed),
            "--resume",
        ],
        cwd=root,
        label="Run 36 ExactStrip model calls",
    )

    paired = validate_results(
        dataset_rows,
        manifest_rows,
        uniform_results,
        exactstrip_results,
        args.model,
        pairs,
    )
    write_jsonl(paired_results, paired)

    run_live(
        [
            sys.executable,
            "-u",
            "-m",
            "optical_memory.score",
            "--results",
            str(exactstrip_results),
            "--output-dir",
            str(exactstrip_scored),
            "--seed",
            str(args.seed),
        ],
        cwd=root,
        label="Score ExactStrip results",
    )
    run_live(
        [
            sys.executable,
            "-u",
            "-m",
            "optical_memory.score",
            "--results",
            str(paired_results),
            "--output-dir",
            str(paired_scored),
            "--seed",
            str(args.seed),
        ],
        cwd=root,
        label="Score paired uniform-versus-ExactStrip comparison",
    )
    print_summary(paired_scored / "per_example.csv", pairs)

    if args.zip_output is not None:
        output = rooted(root, args.zip_output)
        create_archive(
            output,
            root,
            archive_inputs(
                dataset,
                exactstrip_manifest,
                matched_uniform_manifest,
                exactstrip_results,
                paired_results,
                exactstrip_scored,
                paired_scored,
            ),
        )
    print("\nBalanced-18 ExactStrip experiment complete.")


if __name__ == "__main__":
    main()
