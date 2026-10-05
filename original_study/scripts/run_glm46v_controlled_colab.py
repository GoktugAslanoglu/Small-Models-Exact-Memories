"""Run the frozen 90-call controlled replication with GLM-4.6V-Flash."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

from optical_memory.common import read_jsonl, write_jsonl
from optical_memory.glm46v import MODEL_ID


ROOT_DEFAULT = Path("/content/slm_agents")
SEED = 20260826
TARGET_FONTS = (12, 10, 8, 6)
BALANCED_DATASET = Path("data/generated/balanced18.jsonl")
SOURCE_DATASET = Path("data/generated/pilot.jsonl")
SOURCE_RENDER_DIR = Path("renders/generated/balanced18_controls")
SWEEP_MANIFEST = Path("renders/generated/balanced18_sweep/render_manifest.jsonl")
OPTICAL_RESULTS = Path("results/glm46v_flash_balanced18_optical.jsonl")
RAW_RESULTS = Path("results/glm46v_flash_balanced18_raw.jsonl")
COMBINED_RESULTS = Path("results/glm46v_flash_balanced18_combined.jsonl")
OPTICAL_SCORED = Path("results/scored/glm46v_flash_balanced18_optical")
COMBINED_SCORED = Path("results/scored/glm46v_flash_balanced18_combined")
RUN_METADATA = Path("results/glm46v_flash_balanced18_metadata.json")
RESULT_ZIP = Path("results/glm46v_flash_balanced18_bundle.zip")


def _helpers() -> Any:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from scripts import run_qwen35_9b_main_colab as helpers

    return helpers


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run GLM-4.6V-Flash on raw text and uniform 12/10/8/6px "
            "for the frozen balanced-18 matrix."
        )
    )
    parser.add_argument("--project-root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--required-gpu", default="A100")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument(
        "--zip-output",
        nargs="?",
        type=Path,
        const=RESULT_ZIP,
        default=None,
    )
    return parser.parse_args(argv)


def print_summary(per_example: Path, conditions: Sequence[str]) -> None:
    with per_example.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition"])].append(row)

    print("\n=== GLM-4.6V-Flash controlled replication ===")
    for condition in [*conditions, "raw_text"]:
        selected = grouped[condition]
        correct = sum(float(row["exact_match"]) for row in selected)
        details = [
            f"CER={mean(float(row['cer']) for row in selected):.4f}",
            f"descriptive latency={mean(float(row['latency_seconds']) for row in selected):.3f}s",
        ]
        vision = [
            float(row["vision_tokens"])
            for row in selected
            if row.get("vision_tokens")
        ]
        compression = [
            float(row["model_specific_compression"])
            for row in selected
            if row.get("model_specific_compression")
        ]
        if vision:
            details.append(f"GLM vision={mean(vision):.1f}")
        if compression:
            details.append(f"GLM C={mean(compression):.3f}x")
        print(
            f"{condition}: {correct:g}/{len(selected)} correct "
            f"({', '.join(details)})"
        )

    print("\n=== Exact match by memory type ===")
    memory_types = sorted({row["memory_type"] for row in rows})
    for condition in [*conditions, "raw_text"]:
        parts = []
        for memory_type in memory_types:
            selected = [
                row
                for row in grouped[condition]
                if row["memory_type"] == memory_type
            ]
            parts.append(
                f"{memory_type}={sum(float(row['exact_match']) for row in selected):g}/{len(selected)}"
            )
        print(f"{condition}: " + ", ".join(parts))


def metadata_payload(
    *,
    model: str,
    seed: int,
    gpu_info: dict[str, Any],
    conditions: Sequence[str],
    required_gpu: str,
) -> dict[str, Any]:
    import transformers

    return {
        "schema_version": 2,
        "experiment": "glm46v_flash_balanced18_cross_family",
        "experiment_role": "cross-family replication",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "seed": seed,
        "decode": {"do_sample": False, "max_new_tokens": 64, "use_cache": True},
        "conditions": [*conditions, "raw_text"],
        "target_effective_font_sizes_px": list(TARGET_FONTS),
        "expected_calls": {"optical": 72, "raw": 18, "total": 90},
        "required_gpu_substring": required_gpu,
        "runtime": {
            **gpu_info,
            "python_version": platform.python_version(),
            "transformers_version": transformers.__version__,
        },
        "token_accounting": {
            "text": "GLM tokenizer on the source memory text",
            "vision": "count of GLM image placeholders in realized input_ids",
            "crosscheck": "sum(image_grid_thw products / merge_size^2)",
            "C": "GLM memory-text tokens / GLM realized vision tokens",
            "qwen_token_counts_reused": False,
        },
        "latency_note": (
            "Synchronized single-pass generation timing is descriptive and is "
            "not a randomized repeated latency benchmark."
        ),
    }


def archive_paths(root: Path) -> Iterable[Path]:
    fixed = (
        root / BALANCED_DATASET,
        root / SOURCE_RENDER_DIR / "render_manifest.jsonl",
        root / SWEEP_MANIFEST,
        root / OPTICAL_RESULTS,
        root / RAW_RESULTS,
        root / COMBINED_RESULTS,
        root / RUN_METADATA,
        root / "scripts" / "run_glm46v_controlled_colab.py",
        root / "src" / "optical_memory" / "glm46v.py",
        root / "src" / "optical_memory" / "glm46v_controlled_runner.py",
    )
    for path in fixed:
        if path.is_file():
            yield path
    for directory in (root / OPTICAL_SCORED, root / COMBINED_SCORED):
        if directory.is_dir():
            yield from sorted(path for path in directory.rglob("*") if path.is_file())


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.model != MODEL_ID:
        raise SystemExit(f"This runner is frozen to {MODEL_ID}; received {args.model}")
    if args.seed != SEED:
        raise SystemExit(f"This frozen experiment requires seed {SEED}")
    if args.max_new_tokens != 64:
        raise SystemExit("This frozen experiment requires 64 output tokens")

    h = _helpers()
    root = args.project_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Project root does not exist: {root}")
    gpu_info = h.gpu_preflight(args.required_gpu)
    source_dataset = h.rooted(root, SOURCE_DATASET)
    balanced_dataset = h.rooted(root, BALANCED_DATASET)
    source_render_dir = h.rooted(root, SOURCE_RENDER_DIR)
    source_manifest = source_render_dir / "render_manifest.jsonl"
    sweep_manifest = h.rooted(root, SWEEP_MANIFEST)
    optical_results = h.rooted(root, OPTICAL_RESULTS)
    raw_results = h.rooted(root, RAW_RESULTS)
    combined_results = h.rooted(root, COMBINED_RESULTS)
    optical_scored = h.rooted(root, OPTICAL_SCORED)
    combined_scored = h.rooted(root, COMBINED_SCORED)
    metadata_path = h.rooted(root, RUN_METADATA)

    h.require_file(source_dataset, "frozen source dataset")
    if not args.skip_prepare:
        font = h.locate_font()
        h.run_live(
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
            label="Prepare frozen balanced-18 sources",
        )
        h.run_live(
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
                ",".join(str(value) for value in TARGET_FONTS),
            ],
            cwd=root,
            label="Render uniform 12/10/8/6px conditions",
        )

    dataset_rows = h.validate_balanced_dataset(balanced_dataset)
    h.run_live(
        [
            sys.executable,
            "-m",
            "optical_memory.validate",
            "--dataset",
            str(balanced_dataset),
            "--manifest",
            str(sweep_manifest),
        ],
        cwd=root,
        label="Validate controlled dataset and PNGs",
    )
    conditions, sides = h.inspect_sweep_manifest(sweep_manifest, dataset_rows)
    h.check_resume(
        optical_results,
        allowed_conditions=conditions,
        model=args.model,
        required_gpu=args.required_gpu,
    )
    h.check_resume(
        raw_results,
        allowed_conditions=["raw_text"],
        model=args.model,
        required_gpu=args.required_gpu,
    )

    optical_results.parent.mkdir(parents=True, exist_ok=True)
    common = [
        "--dataset",
        str(balanced_dataset),
        "--model",
        args.model,
        "--max-new-tokens",
        "64",
        "--seed",
        str(args.seed),
        "--resume",
    ]
    h.run_live(
        [
            sys.executable,
            "-u",
            "-m",
            "optical_memory.glm46v_controlled_runner",
            "--mode",
            "optical",
            "--manifest",
            str(sweep_manifest),
            "--conditions",
            ",".join(conditions),
            "--output",
            str(optical_results),
            *common,
        ],
        cwd=root,
        label="Run GLM optical matrix (72 resumable calls)",
    )
    h.run_live(
        [
            sys.executable,
            "-u",
            "-m",
            "optical_memory.glm46v_controlled_runner",
            "--mode",
            "raw",
            "--output",
            str(raw_results),
            *common,
        ],
        cwd=root,
        label="Run GLM raw-text control (18 resumable calls)",
    )

    combined = h.validate_result_matrix(
        dataset_rows,
        optical_results,
        raw_results,
        conditions=conditions,
        sides=sides,
        model=args.model,
        required_gpu=args.required_gpu,
    )
    optical = list(read_jsonl(optical_results))
    if any(
        row.get("vision_token_source")
        != "glm_input_ids_image_placeholders_crosschecked_with_grid"
        for row in optical
    ):
        raise SystemExit("GLM-specific visual-token provenance is missing")
    if any(row.get("model_specific_compression") is None for row in optical):
        raise SystemExit("GLM-specific compression C is missing")
    write_jsonl(combined_results, combined)

    for source, target, label in (
        (optical_results, optical_scored, "Score GLM optical matrix"),
        (combined_results, combined_scored, "Score complete GLM matrix"),
    ):
        h.run_live(
            [
                sys.executable,
                "-m",
                "optical_memory.score",
                "--results",
                str(source),
                "--output-dir",
                str(target),
                "--seed",
                str(args.seed),
            ],
            cwd=root,
            label=label,
        )
    print_summary(combined_scored / "per_example.csv", conditions)

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            metadata_payload(
                model=args.model,
                seed=args.seed,
                gpu_info=gpu_info,
                conditions=conditions,
                required_gpu=args.required_gpu,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if args.zip_output is not None:
        h.create_archive(
            h.rooted(root, args.zip_output),
            root=root,
            paths=archive_paths(root),
        )
    print("\nGLM controlled cross-family replication complete.")


if __name__ == "__main__":
    main()
