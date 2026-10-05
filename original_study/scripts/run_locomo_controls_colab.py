"""Run the acceptance-focused LoCoMo control matrix in one Colab runtime.

Primary stages (2B and 9B) run 430 reviewer-control calls each and reuse the
already completed original LoCoMo matrix during paper analysis. Optional 0.8B
and 4B stages first generate their missing 400-call original matrix and then run
the same 430 controls so those checkpoints are not compared incompletely.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from optical_memory.common import read_jsonl
from optical_memory.locomo_controls import (
    EXPECTED_CALLS_PER_MODEL,
    MODEL_IDS,
    SEED,
    model_slug,
    package_control_results,
    prepare_controls_protocol,
    render_controls_protocol,
    validate_control_results,
    validate_prepared_controls,
)
from optical_memory.locomo_external import (
    CONDITIONS as ORIGINAL_CONDITIONS,
    prepare_protocol,
    render_protocol,
)


STAGE_ORDER = ("08b", "2b", "4b", "9b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and run the frozen LoCoMo reviewer controls."
    )
    parser.add_argument(
        "stage", choices=("prepare", "2b", "9b", "08b", "4b", "finalize")
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--require-a100", action="store_true")
    return parser.parse_args()


def _gpu_report(require_a100: bool) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU runtime is required.")
    name = torch.cuda.get_device_name(0)
    if require_a100 and "A100" not in name.upper():
        raise SystemExit(f"This protocol requested an A100, but Colab assigned: {name}")
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": name,
    }


def _manifest_ready(path: Path, expected_rows: int) -> bool:
    if not path.is_file():
        return False
    rows = list(read_jsonl(path))
    if len(rows) != expected_rows:
        return False
    return all(
        (path.parent / str(relative)).is_file()
        for row in rows
        for relative in row["image_paths"]
    )


def prepare(root: Path, seed: int) -> None:
    from matplotlib import font_manager

    source_path = root / "data" / "source" / "locomo10.json"
    original_paths = prepare_protocol(root, source_path, seed=seed)
    control_paths = prepare_controls_protocol(root, source_path, seed=seed)
    print("Frozen original protocol:")
    for label, path in original_paths.items():
        print(f"  {label}: {path}")
    print("Versioned reviewer-control protocol:")
    for label, path in control_paths.items():
        print(f"  {label}: {path}")

    font_path = font_manager.findfont("DejaVu Sans Mono", fallback_to_default=True)
    print(f"Deterministic font: {font_path}")
    original_render_dir = root / "renders" / "generated" / "locomo_external100"
    if not (
        _manifest_ready(original_render_dir / "full_manifest.jsonl", 10)
        and _manifest_ready(original_render_dir / "oracle_manifest.jsonl", 100)
    ):
        rendered = render_protocol(root, font_path)
        print("Rendered original shared inputs:", rendered)
    else:
        print("Reusing complete original render manifests and PNGs.")

    controls_render_dir = root / "renders" / "generated" / "locomo_controls100"
    if not (
        _manifest_ready(controls_render_dir / "selected_8px_manifest.jsonl", 100)
        and _manifest_ready(
            controls_render_dir / "temporal_oracle24_fixed_manifest.jsonl", 30
        )
    ):
        rendered = render_controls_protocol(root, font_path)
        print("Rendered reviewer-control inputs:", rendered)
    else:
        print("Reusing complete reviewer-control manifests and PNGs.")
    validate_prepared_controls(root)
    print("Prepared-control integrity: PASSED")


def _write_environment(root: Path, stage: str, report: dict[str, Any]) -> None:
    output = root / "results" / "locomo_controls100" / f"environment_{stage}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _run(command: Sequence[str], root: Path, label: str) -> None:
    print(f"\n=== {label} ===", flush=True)
    print("Launching isolated process:", " ".join(command), flush=True)
    subprocess.run(list(command), cwd=root, check=True)


def run_controls(
    root: Path,
    stage: str,
    max_new_tokens: int,
    require_a100: bool,
) -> Path:
    report = _gpu_report(require_a100)
    _write_environment(root, stage, report)
    model = MODEL_IDS[stage]
    output = root / "results" / "locomo_controls100" / f"qwen35_{stage}.jsonl"
    command = [
        sys.executable,
        "-m",
        "optical_memory.locomo_controls_runner",
        "--root",
        str(root),
        "--model",
        model,
        "--output",
        str(output),
        "--max-new-tokens",
        str(max_new_tokens),
        "--seed",
        str(SEED),
        "--resume",
    ]
    _run(command, root, f"Run {model} reviewer controls ({EXPECTED_CALLS_PER_MODEL} calls)")
    validate_control_results(root, [model], require_a100=require_a100)
    return output


def run_original_matrix_for_optional_model(
    root: Path,
    stage: str,
    max_new_tokens: int,
    require_a100: bool,
) -> Path:
    model = MODEL_IDS[stage]
    output = root / "results" / "locomo_external100" / f"qwen35_{stage}.jsonl"
    command = [
        sys.executable,
        "-m",
        "optical_memory.locomo_runner",
        "--root",
        str(root),
        "--model",
        model,
        "--output",
        str(output),
        "--max-new-tokens",
        str(max_new_tokens),
        "--seed",
        str(SEED),
        "--resume",
    ]
    _run(command, root, f"Run {model} original LoCoMo reference matrix (400 calls)")
    questions = list(
        read_jsonl(root / "data" / "generated" / "locomo_external100" / "questions.jsonl")
    )
    rows = list(read_jsonl(output))
    expected = {
        (str(question["question_id"]), condition, model)
        for question in questions
        for condition in ORIGINAL_CONDITIONS
    }
    actual = {
        (str(row["question_id"]), str(row["condition"]), str(row["model"]))
        for row in rows
    }
    if len(rows) != 400 or actual != expected:
        raise ValueError(f"{model}: expected 400 unique original-matrix rows")
    if any(int(row["seed"]) != SEED for row in rows):
        raise ValueError(f"{model}: original-matrix seed mismatch")
    if require_a100 and any("A100" not in str(row["gpu_name"]).upper() for row in rows):
        raise ValueError(f"{model}: original-matrix rows must come from an A100")
    return output


def completed_control_models(root: Path) -> list[str]:
    models: list[str] = []
    for stage in STAGE_ORDER:
        path = root / "results" / "locomo_controls100" / f"qwen35_{stage}.jsonl"
        if path.is_file():
            try:
                validate_control_results(root, [MODEL_IDS[stage]])
            except (FileNotFoundError, ValueError):
                continue
            models.append(MODEL_IDS[stage])
    return models


def package_completed(root: Path) -> Path:
    models = completed_control_models(root)
    if not models:
        raise SystemExit("No complete control model stage is available to package.")
    optional_originals = [
        model
        for model in models
        if model in (MODEL_IDS["08b"], MODEL_IDS["4b"])
    ]
    return package_control_results(
        root, models, original_model_ids=optional_originals
    )


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    if args.seed != SEED:
        raise SystemExit(f"Protocol seed is frozen at {SEED}; received {args.seed}")
    if args.stage == "prepare":
        prepare(root, args.seed)
        return
    if args.stage == "finalize":
        bundle = package_completed(root)
        print(f"Final validated bundle: {bundle}")
        return

    if args.stage in ("08b", "4b"):
        run_original_matrix_for_optional_model(
            root, args.stage, args.max_new_tokens, args.require_a100
        )
    run_controls(root, args.stage, args.max_new_tokens, args.require_a100)
    bundle = package_completed(root)
    print(f"Validated checkpoint bundle: {bundle}")


if __name__ == "__main__":
    main()
