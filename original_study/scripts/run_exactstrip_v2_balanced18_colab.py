"""Run the 18-call matched-8px ExactStrip-v2 development ablation.

Default Colab invocation::

    python -u scripts/run_exactstrip_v2_balanced18_colab.py --zip-output

The existing uniform 8px results are reused. V2 predictions are written after
every call and resumed by episode-condition-model key.
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
from optical_memory.exactstrip_v2_sweep import (
    DEFAULT_REFERENCE_CONDITION,
    build_exactstrip_v2_sweep,
)
from optical_memory.generate import MEMORY_TYPES, POSITION_BUCKETS


PROJECT_ROOT = Path("/content/slm_agents")
DATASET = Path("data/generated/balanced18.jsonl")
UNIFORM_MANIFEST = Path("renders/generated/balanced18_sweep/render_manifest.jsonl")
V2_MANIFEST = Path("renders/generated/balanced18_exactstrip_v2/render_manifest.jsonl")
MATCHED_UNIFORM_MANIFEST = Path(
    "renders/generated/balanced18_exactstrip_v2/matched_uniform_manifest.jsonl"
)
UNIFORM_RESULTS = Path("results/qwen35_2b_balanced18_optical.jsonl")
V2_RESULTS = Path("results/qwen35_2b_balanced18_exactstrip_v2.jsonl")
PAIRED_RESULTS = Path("results/qwen35_2b_balanced18_uniform8_vs_exactstrip_v2.jsonl")
V2_SCORED = Path("results/scored/qwen35_2b_balanced18_exactstrip_v2")
PAIRED_SCORED = Path(
    "results/scored/qwen35_2b_balanced18_uniform8_vs_exactstrip_v2"
)
RESULT_ZIP = Path("results/qwen35_2b_balanced18_exactstrip_v2_bundle.zip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ExactStrip-v2 versus the existing uniform 8px baseline."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--uniform-manifest", type=Path, default=UNIFORM_MANIFEST)
    parser.add_argument("--v2-manifest", type=Path, default=V2_MANIFEST)
    parser.add_argument("--uniform-results", type=Path, default=UNIFORM_RESULTS)
    parser.add_argument("--results", type=Path, default=V2_RESULTS)
    parser.add_argument("--paired-results", type=Path, default=PAIRED_RESULTS)
    parser.add_argument("--scored-dir", type=Path, default=V2_SCORED)
    parser.add_argument("--paired-scored-dir", type=Path, default=PAIRED_SCORED)
    parser.add_argument("--reference-condition", default=DEFAULT_REFERENCE_CONDITION)
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Reuse an existing, already validated v2 manifest and images.",
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


def validate_v2_manifest(
    dataset_rows: Sequence[dict[str, Any]],
    v2_manifest: Path,
    uniform_manifest: Path,
    reference_condition: str,
) -> tuple[list[dict[str, Any]], str, list[dict[str, Any]]]:
    rows = list(read_jsonl(v2_manifest))
    episode_ids = {str(row["episode_id"]) for row in dataset_rows}
    uniform_rows = [
        row
        for row in read_jsonl(uniform_manifest)
        if str(row.get("condition")) == reference_condition
    ]
    uniform_lookup = {str(row["episode_id"]): row for row in uniform_rows}
    if len(uniform_rows) != 18 or set(uniform_lookup) != episode_ids:
        raise SystemExit("Uniform 8px reference manifest is incomplete or duplicated")
    conditions = {str(row["condition"]) for row in rows}
    if len(rows) != 18 or len(conditions) != 1:
        raise SystemExit("Expected exactly 18 rows in one ExactStrip-v2 condition")
    condition = next(iter(conditions))

    keys: set[str] = set()
    for row in rows:
        episode_id = str(row["episode_id"])
        if episode_id in keys or episode_id not in episode_ids:
            raise SystemExit(f"Duplicate or unknown v2 episode: {episode_id}")
        keys.add(episode_id)
        reference = uniform_lookup[episode_id]
        for field in ("page_count", "image_width", "image_height", "total_pixel_budget"):
            if int(row[field]) != int(reference[field]):
                raise SystemExit(
                    f"Budget mismatch for {episode_id}: v2 {field}={row[field]}, "
                    f"uniform={reference[field]}"
                )
        if str(row.get("matched_uniform_condition")) != reference_condition:
            raise SystemExit(f"{episode_id}: incorrect matched uniform condition")
        if str(row.get("detector_input")) != "history_only":
            raise SystemExit(f"{episode_id}: invalid detector provenance")
        if str(row.get("span_detector")) != "syntax_only_v2":
            raise SystemExit(f"{episode_id}: invalid v2 detector version")
        if str(row.get("strip_format")) != "raw_span_no_type_prefix":
            raise SystemExit(f"{episode_id}: bracket-free strip format was not recorded")
        if str(row.get("development_status")) != "posthoc_v2_ablation":
            raise SystemExit(f"{episode_id}: development status is missing")
        if int(row["protected_span_count"]) > int(row["detected_span_count"]):
            raise SystemExit(f"{episode_id}: protected-span counts are inconsistent")
        for value in row["image_paths"]:
            if not (v2_manifest.parent / str(value)).is_file():
                raise SystemExit(f"Missing v2 image: {value}")
    if keys != episode_ids:
        raise SystemExit("ExactStrip-v2 manifest is missing episodes")

    sample = rows[0]
    print("ExactStrip-v2 manifest integrity: PASSED (18 equal-budget rows)")
    print(
        f"  {reference_condition} <-> {condition}: {sample['page_count']} pages at "
        f"{sample['image_width']}x{sample['image_height']}"
    )
    print(
        "  Effective fonts: "
        f"dense={sample['actual_effective_dense_font_size_px']}px, "
        f"strip={sample['actual_effective_strip_font_size_px']}px"
    )
    return rows, condition, uniform_rows


def _measured_tokens(row: dict[str, Any]) -> int:
    value = row.get("vision_tokens") or row.get("grid_vision_tokens")
    if value is None:
        raise SystemExit(
            f"{row.get('episode_id')}/{row.get('condition')} has no vision-token count"
        )
    return int(value)


def validate_results(
    dataset_rows: Sequence[dict[str, Any]],
    uniform_results_path: Path,
    v2_results_path: Path,
    model: str,
    reference_condition: str,
    v2_condition: str,
) -> list[dict[str, Any]]:
    episode_ids = {str(row["episode_id"]) for row in dataset_rows}
    uniform = [
        row
        for row in read_jsonl(uniform_results_path)
        if str(row.get("condition")) == reference_condition
    ]
    v2 = list(read_jsonl(v2_results_path))
    expected_uniform = {
        (episode_id, reference_condition, model) for episode_id in episode_ids
    }
    expected_v2 = {(episode_id, v2_condition, model) for episode_id in episode_ids}
    actual_uniform = [
        (str(row["episode_id"]), str(row["condition"]), str(row["model"]))
        for row in uniform
    ]
    actual_v2 = [
        (str(row["episode_id"]), str(row["condition"]), str(row["model"]))
        for row in v2
    ]
    if len(actual_uniform) != len(set(actual_uniform)) or set(actual_uniform) != expected_uniform:
        raise SystemExit("Existing uniform 8px results are incomplete or from another model")
    if len(actual_v2) != len(set(actual_v2)) or set(actual_v2) != expected_v2:
        raise SystemExit("V2 results are incomplete, duplicated, or from another model")

    uniform_lookup = {str(row["episode_id"]): row for row in uniform}
    v2_lookup = {str(row["episode_id"]): row for row in v2}
    for episode_id in episode_ids:
        before = uniform_lookup[episode_id]
        after = v2_lookup[episode_id]
        if _measured_tokens(before) != _measured_tokens(after):
            raise SystemExit(
                f"Measured token mismatch for {episode_id}: "
                f"uniform={_measured_tokens(before)}, v2={_measured_tokens(after)}"
            )
        for field in ("image_width", "image_height"):
            if int(before[field]) != int(after[field]):
                raise SystemExit(f"Result dimension mismatch for {episode_id}/{field}")
    print("Measured-budget integrity: PASSED (all paired vision-token counts identical)")
    return uniform + v2


def print_summary(
    per_example: Path,
    reference_condition: str,
    v2_condition: str,
) -> bool:
    with per_example.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition"])].append(row)

    print("\n=== ExactStrip-v2 development result ===")
    for condition in (reference_condition, v2_condition):
        selected = grouped[condition]
        correct = sum(float(row["exact_match"]) for row in selected)
        cer = mean(float(row["cer"]) for row in selected)
        vision = mean(
            float(row["vision_tokens"] or row["grid_vision_tokens"])
            for row in selected
        )
        compression = mean(float(row["realized_compression"]) for row in selected)
        print(
            f"{condition}: {correct:g}/{len(selected)} correct, CER={cer:.4f}, "
            f"mean vision={vision:.1f}, compression={compression:.3f}x"
        )

    uniform = {row["episode_id"]: row for row in grouped[reference_condition]}
    v2 = {row["episode_id"]: row for row in grouped[v2_condition]}
    gains: list[str] = []
    losses: list[str] = []
    for episode_id in sorted(uniform):
        before = float(uniform[episode_id]["exact_match"])
        after = float(v2[episode_id]["exact_match"])
        if after > before:
            gains.append(episode_id)
        elif after < before:
            losses.append(episode_id)
    print(f"Paired gains ({len(gains)}): {gains}")
    print(f"Paired losses ({len(losses)}): {losses}")

    print("\n=== Accuracy by memory type ===")
    for condition in (reference_condition, v2_condition):
        parts = []
        for memory_type in MEMORY_TYPES:
            selected = [
                row for row in grouped[condition] if row["memory_type"] == memory_type
            ]
            correct = sum(float(row["exact_match"]) for row in selected)
            parts.append(f"{memory_type}={correct:g}/{len(selected)}")
        print(f"{condition}: " + ", ".join(parts))

    print("\n=== Accuracy by position ===")
    for condition in (reference_condition, v2_condition):
        parts = []
        for bucket in POSITION_BUCKETS:
            selected = [
                row for row in grouped[condition] if row["position_bucket"] == bucket
            ]
            correct = sum(float(row["exact_match"]) for row in selected)
            parts.append(f"{bucket}={correct:g}/{len(selected)}")
        print(f"{condition}: " + ", ".join(parts))

    v2_correct = sum(float(row["exact_match"]) for row in grouped[v2_condition])
    path_correct = sum(
        float(row["exact_match"])
        for row in grouped[v2_condition]
        if row["memory_type"] == "symbol_path"
    )
    protected_types = {"semantic_decision", "structured_config", "code_tool_state"}
    protected_correct = sum(
        float(row["exact_match"])
        for row in grouped[v2_condition]
        if row["memory_type"] in protected_types
    )
    passed = v2_correct >= 15 and path_correct >= 2 and protected_correct == 9
    print("\n=== Development gate ===")
    print(
        f"overall={v2_correct:g}/18 (requires >=15), "
        f"paths={path_correct:g}/3 (requires >=2), "
        f"semantic+config+code={protected_correct:g}/9 (requires 9)"
    )
    print("GATE:", "PASS - confirm on held-out episodes" if passed else "FAIL - do not scale v2")
    return passed


def archive_inputs(
    dataset: Path,
    v2_manifest: Path,
    matched_uniform_manifest: Path,
    v2_results: Path,
    paired_results: Path,
    v2_scored: Path,
    paired_scored: Path,
) -> Iterable[Path]:
    for path in (
        dataset,
        v2_manifest,
        matched_uniform_manifest,
        v2_results,
        paired_results,
    ):
        if path.is_file():
            yield path
    for directory in (v2_scored, paired_scored):
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
    v2_manifest = rooted(root, args.v2_manifest)
    matched_uniform_manifest = rooted(root, MATCHED_UNIFORM_MANIFEST)
    uniform_results = rooted(root, args.uniform_results)
    v2_results = rooted(root, args.results)
    paired_results = rooted(root, args.paired_results)
    v2_scored = rooted(root, args.scored_dir)
    paired_scored = rooted(root, args.paired_scored_dir)

    for required in (dataset, uniform_manifest, uniform_results):
        if not required.is_file():
            raise SystemExit(
                f"Required prior-stage file is missing: {required}\n"
                "Continue in the balanced-18 Colab runtime or restore its files."
            )
    dataset_rows = validate_balanced_dataset(dataset)

    if not args.skip_prepare:
        font = locate_font()
        print("\n=== Render matched ExactStrip-v2 8px images ===", flush=True)
        rendered = build_exactstrip_v2_sweep(
            dataset,
            uniform_manifest,
            v2_manifest,
            reference_condition=args.reference_condition,
            font_path=font,
        )
        print(f"Rendered {len(rendered)} v2 rows to {v2_manifest}")
    if not v2_manifest.is_file():
        raise SystemExit(f"ExactStrip-v2 manifest is missing: {v2_manifest}")

    manifest_rows, v2_condition, matched_uniform_rows = validate_v2_manifest(
        dataset_rows,
        v2_manifest,
        uniform_manifest,
        args.reference_condition,
    )
    write_jsonl(matched_uniform_manifest, matched_uniform_rows)

    v2_results.parent.mkdir(parents=True, exist_ok=True)
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
            str(v2_manifest),
            "--conditions",
            v2_condition,
            "--model",
            args.model,
            "--output",
            str(v2_results),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--seed",
            str(args.seed),
            "--resume",
        ],
        cwd=root,
        label="Run 18 ExactStrip-v2 model calls",
    )

    paired = validate_results(
        dataset_rows,
        uniform_results,
        v2_results,
        args.model,
        args.reference_condition,
        v2_condition,
    )
    write_jsonl(paired_results, paired)

    run_live(
        [
            sys.executable,
            "-u",
            "-m",
            "optical_memory.score",
            "--results",
            str(v2_results),
            "--output-dir",
            str(v2_scored),
            "--seed",
            str(args.seed),
        ],
        cwd=root,
        label="Score ExactStrip-v2 results",
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
        label="Score paired uniform-8px versus ExactStrip-v2",
    )
    print_summary(paired_scored / "per_example.csv", args.reference_condition, v2_condition)

    if args.zip_output is not None:
        output = rooted(root, args.zip_output)
        create_archive(
            output,
            root,
            archive_inputs(
                dataset,
                v2_manifest,
                matched_uniform_manifest,
                v2_results,
                paired_results,
                v2_scored,
                paired_scored,
            ),
        )
    print("\nExactStrip-v2 8px development ablation complete.")


if __name__ == "__main__":
    main()
