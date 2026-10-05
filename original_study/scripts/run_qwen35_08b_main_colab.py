"""Run the frozen Qwen3.5-0.8B balanced-18 baseline experiment in Colab.

Default invocation::

    python -u scripts/run_qwen35_08b_main_colab.py \
        --project-root /content/slm_agents --zip-output

The experiment contains 18 raw-text calls and 72 uniform optical calls
(12/10/8/6px). Results are checkpointed after every inference and are safe to
resume only when the model, condition set, and required GPU match.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

from optical_memory.common import read_jsonl, write_jsonl
from optical_memory.generate import MEMORY_TYPES, POSITION_BUCKETS


PROJECT_ROOT = Path("/content/slm_agents")
MODEL = "Qwen/Qwen3.5-0.8B"
SEED = 20260826
TARGET_FONTS = (12, 10, 8, 6)
SOURCE_DATASET = Path("data/generated/pilot.jsonl")
BALANCED_DATASET = Path("data/generated/balanced18.jsonl")
SOURCE_RENDER_DIR = Path("renders/generated/balanced18_controls")
SWEEP_MANIFEST = Path("renders/generated/balanced18_sweep/render_manifest.jsonl")
OPTICAL_RESULTS = Path("results/qwen35_08b_balanced18_optical_main.jsonl")
RAW_RESULTS = Path("results/qwen35_08b_balanced18_raw_main.jsonl")
COMBINED_RESULTS = Path("results/qwen35_08b_balanced18_combined_main.jsonl")
OPTICAL_SCORED = Path("results/scored/qwen35_08b_balanced18_optical_main")
COMBINED_SCORED = Path("results/scored/qwen35_08b_balanced18_combined_main")
RUN_METADATA = Path("results/qwen35_08b_balanced18_main_metadata.json")
RESULT_ZIP = Path("results/qwen35_08b_balanced18_main_bundle.zip")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen Qwen3.5-0.8B balanced-18 raw + uniform 12/10/8/6px "
            "baseline experiment."
        )
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--required-gpu",
        default="A100",
        help="Case-insensitive substring required in the CUDA GPU name.",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Reuse an existing balanced dataset, manifests, and rendered images.",
    )
    parser.add_argument(
        "--zip-output",
        nargs="?",
        type=Path,
        const=RESULT_ZIP,
        default=None,
        help="Create a compact result ZIP at the default or supplied path.",
    )
    return parser.parse_args(argv)


def rooted(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"Missing {label}: {path}")


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
            f"\n{label} failed with return code {return_code}.\n"
            f"Command: {printable}\n"
            "The original error is printed above. Inference is checkpointed after "
            "every row, so fix the error and rerun this same command to resume."
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_font() -> Path:
    try:
        from matplotlib import font_manager

        candidate = Path(
            font_manager.findfont("DejaVu Sans Mono", fallback_to_default=False)
        )
    except Exception as exc:
        raise SystemExit(
            "Matplotlib's DejaVu Sans Mono font is required. Rerun the notebook "
            "dependency cell before this script."
        ) from exc
    if not candidate.is_file():
        raise SystemExit(f"Resolved font does not exist: {candidate}")
    print(f"Deterministic font: {candidate}")
    return candidate


def gpu_preflight(required_gpu: str) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required; rerun the notebook dependency cell.") from exc
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU runtime is required.")
    gpu_name = torch.cuda.get_device_name(0)
    if required_gpu and required_gpu.casefold() not in gpu_name.casefold():
        raise SystemExit(
            f"GPU guard failed: required a name containing {required_gpu!r}, "
            f"but Colab assigned {gpu_name!r}. Do not run this comparison on a "
            "different GPU; change the runtime and try again."
        )
    properties = torch.cuda.get_device_properties(0)
    info = {
        "gpu_name": gpu_name,
        "gpu_vram_gib": round(properties.total_memory / 2**30, 3),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    print(
        f"GPU guard: PASSED ({gpu_name}, {info['gpu_vram_gib']:.1f} GiB; "
        f"Torch {torch.__version__}, CUDA {torch.version.cuda})"
    )
    return info


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


def inspect_sweep_manifest(
    manifest: Path,
    dataset_rows: Sequence[dict[str, Any]],
) -> tuple[list[str], dict[str, int]]:
    rows = list(read_jsonl(manifest))
    episode_ids = {str(row["episode_id"]) for row in dataset_rows}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    keys: list[tuple[str, str]] = []
    for row in rows:
        condition = str(row["condition"])
        episode_id = str(row["episode_id"])
        grouped[condition].append(row)
        keys.append((episode_id, condition))
    if len(keys) != len(set(keys)):
        raise SystemExit("Sweep manifest contains duplicate episode/condition rows")
    if len(rows) != 18 * len(TARGET_FONTS):
        raise SystemExit(
            f"Expected {18 * len(TARGET_FONTS)} sweep rows; found {len(rows)}"
        )

    sides: dict[str, int] = {}
    missing_images: list[Path] = []
    for condition, selected in grouped.items():
        selected_ids = {str(row["episode_id"]) for row in selected}
        if len(selected) != 18 or selected_ids != episode_ids:
            raise SystemExit(f"{condition} is not a complete 18-episode slice")
        widths = {int(row["image_width"]) for row in selected}
        heights = {int(row["image_height"]) for row in selected}
        if len(widths) != 1 or heights != widths:
            raise SystemExit(f"{condition} does not have one consistent square size")
        sides[condition] = next(iter(widths))
        for row in selected:
            for image_path in row["image_paths"]:
                resolved = (manifest.parent / str(image_path)).resolve()
                if not resolved.is_file():
                    missing_images.append(resolved)
    if missing_images:
        raise SystemExit(
            f"Sweep is missing {len(missing_images)} rendered image(s); "
            f"first={missing_images[0]}"
        )
    if len(grouped) != len(TARGET_FONTS):
        raise SystemExit(f"Expected four optical conditions; found {sorted(grouped)}")
    for font in TARGET_FONTS:
        matches = [condition for condition in grouped if f"_{font}px_" in condition]
        if len(matches) != 1:
            raise SystemExit(f"Expected one {font}px condition; found {matches}")

    conditions = sorted(grouped, key=lambda condition: sides[condition], reverse=True)
    print("Sweep integrity: PASSED (18 episodes x 4 uniform conditions = 72 rows)")
    for condition in conditions:
        page_counts = sorted({int(row["page_count"]) for row in grouped[condition]})
        print(
            f"  {condition}: 18 rows, side={sides[condition]}px, "
            f"page_counts={page_counts}"
        )
    return conditions, sides


def check_resume(
    path: Path,
    *,
    allowed_conditions: Sequence[str],
    model: str,
    required_gpu: str,
) -> None:
    if not path.exists():
        print(f"Fresh output: {path}")
        return
    rows = list(read_jsonl(path))
    allowed = set(allowed_conditions)
    problems: list[str] = []
    for row in rows:
        if str(row.get("condition")) not in allowed:
            problems.append(f"condition={row.get('condition')!r}")
        if str(row.get("model")) != model:
            problems.append(f"model={row.get('model')!r}")
        gpu_name = str(row.get("gpu_name", ""))
        if required_gpu and required_gpu.casefold() not in gpu_name.casefold():
            problems.append(f"gpu={gpu_name!r}")
    keys = [
        (str(row.get("episode_id")), str(row.get("condition")), str(row.get("model")))
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        problems.append("duplicate resume keys")
    if problems:
        raise SystemExit(
            f"Unsafe/incompatible resume file: {path}\n"
            f"Problems: {problems[:8]}\n"
            "Choose a fresh output path; this script will not delete or overwrite it."
        )
    print(f"Safe resume: {path} ({len(rows)} completed rows)")


def validate_result_matrix(
    dataset_rows: Sequence[dict[str, Any]],
    optical_path: Path,
    raw_path: Path,
    *,
    conditions: Sequence[str],
    sides: dict[str, int],
    model: str,
    required_gpu: str,
) -> list[dict[str, Any]]:
    optical = list(read_jsonl(optical_path))
    raw = list(read_jsonl(raw_path))
    episode_ids = {str(row["episode_id"]) for row in dataset_rows}
    if len(optical) != 72 or len(raw) != 18:
        raise SystemExit(
            f"Expected 72 optical and 18 raw rows; found {len(optical)} and {len(raw)}"
        )
    expected_optical = {
        (episode_id, condition, model)
        for episode_id in episode_ids
        for condition in conditions
    }
    expected_raw = {(episode_id, "raw_text", model) for episode_id in episode_ids}
    optical_keys = [
        (str(row["episode_id"]), str(row["condition"]), str(row["model"]))
        for row in optical
    ]
    raw_keys = [
        (str(row["episode_id"]), str(row["condition"]), str(row["model"]))
        for row in raw
    ]
    if set(optical_keys) != expected_optical or set(raw_keys) != expected_raw:
        raise SystemExit("Result rows do not match the expected 18-episode matrix")
    if len(optical_keys) != len(set(optical_keys)) or len(raw_keys) != len(set(raw_keys)):
        raise SystemExit("Result files contain duplicate resume keys")

    all_rows = optical + raw
    wrong_gpu = [
        row
        for row in all_rows
        if required_gpu
        and required_gpu.casefold() not in str(row.get("gpu_name", "")).casefold()
    ]
    if wrong_gpu:
        raise SystemExit(
            f"GPU provenance failed: {len(wrong_gpu)} rows do not report "
            f"a GPU containing {required_gpu!r}"
        )
    wrong_model = [row for row in all_rows if str(row.get("model")) != model]
    if wrong_model:
        raise SystemExit(f"Model provenance failed for {len(wrong_model)} rows")

    optical_lookup = {
        (str(row["episode_id"]), str(row["condition"])): row for row in optical
    }
    for episode_id in episode_ids:
        selected = [optical_lookup[(episode_id, condition)] for condition in conditions]
        if len({int(row["text_tokens"]) for row in selected}) != 1:
            raise SystemExit(f"{episode_id}: text-token count varies across optical conditions")
        vision = [
            int(row.get("vision_tokens") or row.get("grid_vision_tokens"))
            for row in selected
        ]
        compression = [float(row["realized_compression"]) for row in selected]
        if any(next_value > value for value, next_value in zip(vision, vision[1:])):
            raise SystemExit(f"{episode_id}: visual tokens are not nonincreasing: {vision}")
        if any(next_value < value for value, next_value in zip(compression, compression[1:])):
            raise SystemExit(
                f"{episode_id}: compression is not nondecreasing: {compression}"
            )
        for condition, row in zip(conditions, selected):
            expected_side = sides[condition]
            if (int(row["image_width"]), int(row["image_height"])) != (
                expected_side,
                expected_side,
            ):
                raise SystemExit(f"{episode_id}/{condition}: result dimension mismatch")

    print("Result matrix: PASSED (72 optical + 18 raw = 90 unique A100 rows)")
    print("Visual-token/compression monotonicity: PASSED")
    return all_rows


def print_summary(per_example: Path, conditions: Sequence[str]) -> None:
    with per_example.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    order = list(conditions) + ["raw_text"]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition"])].append(row)

    print("\n=== Aggregate 0.8B results ===")
    for condition in order:
        selected = grouped.get(condition, [])
        correct = sum(float(row["exact_match"]) for row in selected)
        cer = mean(float(row["cer"]) for row in selected)
        latency = mean(float(row["latency_seconds"]) for row in selected)
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
        details = [f"CER={cer:.4f}", f"descriptive latency={latency:.3f}s"]
        if vision:
            details.append(f"mean vision={mean(vision):.1f}")
        if compression:
            details.append(f"mean compression={mean(compression):.3f}x")
        print(
            f"{condition}: {correct:g}/{len(selected)} correct "
            f"({', '.join(details)})"
        )

    print("\n=== Accuracy by memory type ===")
    for condition in order:
        parts: list[str] = []
        for memory_type in MEMORY_TYPES:
            selected = [
                row
                for row in grouped.get(condition, [])
                if row["memory_type"] == memory_type
            ]
            correct = sum(float(row["exact_match"]) for row in selected)
            parts.append(f"{memory_type}={correct:g}/{len(selected)}")
        print(f"{condition}: " + ", ".join(parts))

    print("\n=== Accuracy by target position ===")
    for condition in order:
        parts = []
        for bucket in POSITION_BUCKETS:
            selected = [
                row
                for row in grouped.get(condition, [])
                if row["position_bucket"] == bucket
            ]
            correct = sum(float(row["exact_match"]) for row in selected)
            parts.append(f"{bucket}={correct:g}/{len(selected)}")
        print(f"{condition}: " + ", ".join(parts))


def write_run_metadata(
    output: Path,
    *,
    root: Path,
    model: str,
    seed: int,
    required_gpu: str,
    gpu_info: dict[str, Any],
    conditions: Sequence[str],
    dataset: Path,
    source_manifest: Path,
    sweep_manifest: Path,
    optical_results: Path,
    raw_results: Path,
    combined_results: Path,
) -> None:
    try:
        import transformers

        transformers_version: str | None = transformers.__version__
    except Exception:
        transformers_version = None
    try:
        import torchvision

        torchvision_version: str | None = torchvision.__version__
    except Exception:
        torchvision_version = None
    tracked = [
        dataset,
        source_manifest,
        sweep_manifest,
        optical_results,
        raw_results,
        combined_results,
    ]
    metadata = {
        "schema_version": 1,
        "experiment": "qwen35_08b_balanced18_uniform_main",
        "experiment_role": "frozen cross-model baseline comparison",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "seed": seed,
        "decode": {"do_sample": False, "max_new_tokens": 64, "use_cache": True},
        "required_gpu_substring": required_gpu,
        "runtime": {
            **gpu_info,
            "python_version": platform.python_version(),
            "transformers_version": transformers_version,
            "torchvision_version": torchvision_version,
        },
        "conditions": [*conditions, "raw_text"],
        "target_effective_font_sizes_px": list(TARGET_FONTS),
        "expected_calls": {"optical": 72, "raw": 18, "total": 90},
        "latency_note": (
            "latency_seconds is synchronized single-pass model inference timing; "
            "it is descriptive, not a dedicated repeated latency benchmark"
        ),
        "sha256": {
            path.relative_to(root).as_posix(): sha256(path) for path in tracked
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Run metadata: {output}")


def archive_inputs(
    *,
    root: Path,
    dataset: Path,
    source_manifest: Path,
    sweep_manifest: Path,
    optical_results: Path,
    raw_results: Path,
    combined_results: Path,
    optical_scored: Path,
    combined_scored: Path,
    metadata: Path,
) -> Iterable[Path]:
    for path in (
        dataset,
        source_manifest,
        sweep_manifest,
        optical_results,
        raw_results,
        combined_results,
        metadata,
        root / "scripts" / "run_qwen35_08b_main_colab.py",
        root / "scripts" / "run_readability_sweep_colab.py",
        root / "configs" / "pilot_pages.json",
        root / "pyproject.toml",
    ):
        if path.is_file():
            yield path
    for directory in (optical_scored, combined_scored):
        if directory.is_dir():
            yield from sorted(path for path in directory.rglob("*") if path.is_file())


def create_archive(output: Path, *, root: Path, paths: Iterable[Path]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    files = list(dict.fromkeys(paths))
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            try:
                archive_name = path.relative_to(root)
            except ValueError:
                archive_name = Path("external") / path.name
            bundle.write(path, archive_name.as_posix())
    print(f"Result archive: {output} ({len(files)} files; rendered PNGs excluded)")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = args.project_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Project root does not exist: {root}")
    if args.seed != SEED:
        raise SystemExit(f"This frozen experiment requires seed {SEED}; received {args.seed}")
    if args.model != MODEL:
        raise SystemExit(f"This runner is frozen to {MODEL}; received {args.model}")
    if args.max_new_tokens != 64:
        raise SystemExit("This frozen experiment requires --max-new-tokens 64")

    gpu_info = gpu_preflight(args.required_gpu)
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
    metadata = rooted(root, RUN_METADATA)

    require_file(source_dataset, "frozen 60-episode source dataset")
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
            label="Prepare frozen balanced-18 readable sources",
        )
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
                "full_history_12px",
                "--target-fonts",
                ",".join(str(font) for font in TARGET_FONTS),
            ],
            cwd=root,
            label="Render uniform 12/10/8/6px conditions",
        )

    require_file(balanced_dataset, "balanced-18 dataset")
    require_file(source_manifest, "12px source manifest")
    require_file(sweep_manifest, "12/10/8/6px sweep manifest")
    dataset_rows = validate_balanced_dataset(balanced_dataset)
    run_live(
        [
            sys.executable,
            "-u",
            "-m",
            "optical_memory.validate",
            "--dataset",
            str(balanced_dataset),
            "--manifest",
            str(sweep_manifest),
        ],
        cwd=root,
        label="Validate frozen dataset and all rendered images",
    )
    conditions, sides = inspect_sweep_manifest(sweep_manifest, dataset_rows)

    check_resume(
        optical_results,
        allowed_conditions=conditions,
        model=args.model,
        required_gpu=args.required_gpu,
    )
    check_resume(
        raw_results,
        allowed_conditions=["raw_text"],
        model=args.model,
        required_gpu=args.required_gpu,
    )
    optical_results.parent.mkdir(parents=True, exist_ok=True)
    run_live(
        [
            sys.executable,
            "-u",
            "-m",
            "optical_memory.colab_runner",
            "--mode",
            "optical",
            "--dataset",
            str(balanced_dataset),
            "--manifest",
            str(sweep_manifest),
            "--model",
            args.model,
            "--conditions",
            ",".join(conditions),
            "--output",
            str(optical_results),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--seed",
            str(args.seed),
            "--resume",
        ],
        cwd=root,
        label="Run Qwen3.5-0.8B uniform optical matrix (72 resumable calls)",
    )
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
        label="Run Qwen3.5-0.8B raw-text control (18 resumable calls)",
    )

    combined = validate_result_matrix(
        dataset_rows,
        optical_results,
        raw_results,
        conditions=conditions,
        sides=sides,
        model=args.model,
        required_gpu=args.required_gpu,
    )
    write_jsonl(combined_results, combined)
    run_live(
        [
            sys.executable,
            "-u",
            "-m",
            "optical_memory.score",
            "--results",
            str(optical_results),
            "--output-dir",
            str(optical_scored),
            "--seed",
            str(args.seed),
        ],
        cwd=root,
        label="Score 0.8B optical results",
    )
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
        label="Score complete 0.8B matrix",
    )
    print_summary(combined_scored / "per_example.csv", conditions)
    write_run_metadata(
        metadata,
        root=root,
        model=args.model,
        seed=args.seed,
        required_gpu=args.required_gpu,
        gpu_info=gpu_info,
        conditions=conditions,
        dataset=balanced_dataset,
        source_manifest=source_manifest,
        sweep_manifest=sweep_manifest,
        optical_results=optical_results,
        raw_results=raw_results,
        combined_results=combined_results,
    )

    if args.zip_output is not None:
        create_archive(
            rooted(root, args.zip_output),
            root=root,
            paths=archive_inputs(
                root=root,
                dataset=balanced_dataset,
                source_manifest=source_manifest,
                sweep_manifest=sweep_manifest,
                optical_results=optical_results,
                raw_results=raw_results,
                combined_results=combined_results,
                optical_scored=optical_scored,
                combined_scored=combined_scored,
                metadata=metadata,
            ),
        )
    print("\nQwen3.5-0.8B frozen main experiment complete.")
    print("Download the result bundle before stopping the Colab runtime.")


if __name__ == "__main__":
    main()
