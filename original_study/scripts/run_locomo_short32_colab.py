"""Run the frozen 32-token full-optical LoCoMo sensitivity on Qwen 2B and 9B."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import subprocess
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

from optical_memory.common import read_jsonl, write_jsonl
from optical_memory.locomo_external import (
    SEED,
    normalize_answer,
    sha256_file,
    score_prediction,
    score_results,
)


MODELS = {
    "2b": "Qwen/Qwen3.5-2B",
    "9b": "Qwen/Qwen3.5-9B",
}
CONDITION = "full_optical_8px"
PROMPT_PROFILE = "short32"
MAX_NEW_TOKENS = 32
EXPECTED_ROWS = 100
RESULT_DIR = Path("results/locomo_short32")
BASELINE_DIR = Path("results/locomo_external100")
BUNDLE_PATH = RESULT_DIR / "qwen35_2b_9b_short32_sensitivity_bundle.zip"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run only the full-optical LoCoMo condition with the frozen "
            "short-answer prompt and 32-token generation cap."
        )
    )
    parser.add_argument(
        "stage",
        choices=("check", "2b", "9b", "finalize", "all"),
    )
    parser.add_argument("--project-root", type=Path, default=Path("/content/slm_agents"))
    parser.add_argument("--required-gpu", default="A100")
    return parser.parse_args(argv)


def _result_path(root: Path, stage: str) -> Path:
    return root / RESULT_DIR / f"qwen35_{stage}_full_optical_short32.jsonl"


def _baseline_path(root: Path, stage: str) -> Path:
    return root / BASELINE_DIR / f"qwen35_{stage}.jsonl"


def validate_inputs(root: Path) -> None:
    data_dir = root / "data/generated/locomo_external100"
    render_dir = root / "renders/generated/locomo_external100"
    questions = list(read_jsonl(data_dir / "questions.jsonl"))
    conversations = list(read_jsonl(data_dir / "conversations.jsonl"))
    manifest_path = render_dir / "full_manifest.jsonl"
    manifest = list(read_jsonl(manifest_path))
    if len(questions) != 100 or len(conversations) != 10 or len(manifest) != 10:
        raise ValueError(
            "Expected 100 questions, 10 conversations, and 10 full-history manifests"
        )
    question_ids = {str(row["question_id"]) for row in questions}
    if len(question_ids) != 100:
        raise ValueError("Question IDs are not unique")
    for row in manifest:
        for relative in row["image_paths"]:
            path = manifest_path.parent / str(relative)
            if not path.is_file():
                raise FileNotFoundError(path)
    for stage in MODELS:
        baseline = _baseline_path(root, stage)
        if not baseline.is_file():
            raise FileNotFoundError(
                f"Missing archived 96-token baseline needed for comparison: {baseline}"
            )
    print("Inputs: PASSED (100 questions, 10 histories, all rendered pages present)")


def gpu_preflight(required_gpu: str) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU runtime is required")
    gpu = torch.cuda.get_device_name(0)
    if required_gpu and required_gpu.casefold() not in gpu.casefold():
        raise SystemExit(f"Expected {required_gpu}, but Colab assigned {gpu}")
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": gpu,
    }


def validate_model_rows(
    root: Path,
    stage: str,
    *,
    required_gpu: str,
) -> list[dict[str, Any]]:
    model = MODELS[stage]
    rows = list(read_jsonl(_result_path(root, stage)))
    keys = {
        (str(row["question_id"]), str(row["condition"]), str(row["model"]))
        for row in rows
    }
    if len(rows) != EXPECTED_ROWS or len(keys) != EXPECTED_ROWS:
        raise ValueError(f"{stage}: expected 100 unique rows; found {len(rows)}")
    for row in rows:
        if (
            row["model"] != model
            or row["condition"] != CONDITION
            or int(row["seed"]) != SEED
            or int(row["max_new_tokens"]) != MAX_NEW_TOKENS
            or row.get("prompt_profile") != PROMPT_PROFILE
        ):
            raise ValueError(f"{stage}: protocol metadata mismatch")
        if required_gpu and required_gpu.casefold() not in str(row["gpu_name"]).casefold():
            raise ValueError(f"{stage}: GPU provenance mismatch")
    return rows


def run_model(root: Path, stage: str, required_gpu: str) -> None:
    environment = gpu_preflight(required_gpu)
    result_dir = root / RESULT_DIR
    result_dir.mkdir(parents=True, exist_ok=True)
    environment.update(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "seed": SEED,
            "prompt_profile": PROMPT_PROFILE,
            "max_new_tokens": MAX_NEW_TOKENS,
        }
    )
    (result_dir / f"environment_{stage}.json").write_text(
        json.dumps(environment, indent=2) + "\n",
        encoding="utf-8",
    )
    output = _result_path(root, stage)
    command = [
        sys.executable,
        "-u",
        "-m",
        "optical_memory.locomo_runner",
        "--root",
        str(root),
        "--model",
        MODELS[stage],
        "--output",
        str(output),
        "--conditions",
        CONDITION,
        "--prompt-profile",
        PROMPT_PROFILE,
        "--max-new-tokens",
        str(MAX_NEW_TOKENS),
        "--seed",
        str(SEED),
        "--resume",
    ]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)
    rows = validate_model_rows(root, stage, required_gpu=required_gpu)
    score_results(output, result_dir / f"analysis_{stage}")
    print(f"{stage}: PASSED ({len(rows)} resumable full-optical calls)")


def _scored_by_question(
    rows: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    scored: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["condition"] != CONDITION:
            continue
        f1_value, exact = score_prediction(
            str(row["prediction"]),
            str(row["answer"]),
            int(row["category"]),
        )
        scored[str(row["question_id"])] = {
            **row,
            "official_token_f1": f1_value,
            "normalized_exact_match": int(exact),
        }
    return scored


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = probability * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _cluster_inference(
    paired: list[dict[str, Any]],
    *,
    bootstrap_draws: int = 10000,
) -> dict[str, Any]:
    by_conversation: dict[str, list[float]] = defaultdict(list)
    for row in paired:
        by_conversation[str(row["conversation_id"])].append(float(row["delta_f1"]))
    cluster_effects = {
        key: mean(values) for key, values in sorted(by_conversation.items())
    }
    values = list(cluster_effects.values())
    observed = mean(values)
    rng = random.Random(SEED)
    boot = [
        mean(rng.choice(values) for _ in values)
        for _ in range(bootstrap_draws)
    ]
    extreme = 0
    for mask in range(1 << len(values)):
        signed = [
            value if mask & (1 << index) else -value
            for index, value in enumerate(values)
        ]
        if abs(mean(signed)) >= abs(observed) - 1e-12:
            extreme += 1
    return {
        "conversation_effects": cluster_effects,
        "mean_conversation_effect": observed,
        "cluster_bootstrap_95_ci": [
            _percentile(boot, 0.025),
            _percentile(boot, 0.975),
        ],
        "exact_two_sided_sign_flip_p": extreme / (1 << len(values)),
        "bootstrap_draws": bootstrap_draws,
    }


def _condition_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "mean_official_token_f1": mean(float(row["official_token_f1"]) for row in rows),
        "normalized_exact_match_rate": mean(
            int(row["normalized_exact_match"]) for row in rows
        ),
        "mean_generated_tokens": mean(int(row["generated_tokens"]) for row in rows),
        "generation_cap_rate": mean(
            int(row["generated_tokens"]) >= int(row["max_new_tokens"]) for row in rows
        ),
        "abstention_rate": mean(
            normalize_answer(str(row["prediction"]))
            == normalize_answer("No information available")
            for row in rows
        ),
    }


def finalize(root: Path, required_gpu: str) -> Path:
    analysis_dir = root / RESULT_DIR / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    combined: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    comparison: dict[str, Any] = {
        "protocol": {
            "condition": CONDITION,
            "prompt_profile": PROMPT_PROFILE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "seed": SEED,
            "claim_status": "sensitivity only; not part of frozen claims until executed",
        },
        "models": {},
    }
    for stage, model in MODELS.items():
        short_rows = validate_model_rows(root, stage, required_gpu=required_gpu)
        combined.extend(short_rows)
        baseline_all = list(read_jsonl(_baseline_path(root, stage)))
        short = _scored_by_question(short_rows)
        baseline = _scored_by_question(baseline_all)
        if set(short) != set(baseline) or len(short) != EXPECTED_ROWS:
            raise ValueError(f"{stage}: baseline/sensitivity question mismatch")
        paired: list[dict[str, Any]] = []
        for question_id in sorted(short):
            current = short[question_id]
            prior = baseline[question_id]
            paired.append(
                {
                    "question_id": question_id,
                    "conversation_id": current["conversation_id"],
                    "category_name": current["category_name"],
                    "baseline_f1": prior["official_token_f1"],
                    "short32_f1": current["official_token_f1"],
                    "delta_f1": current["official_token_f1"] - prior["official_token_f1"],
                }
            )
        baseline_rows = list(baseline.values())
        short_scored = list(short.values())
        baseline_summary = _condition_summary(baseline_rows)
        short_summary = _condition_summary(short_scored)
        inference = _cluster_inference(paired)
        model_summary = {
            "model": model,
            "baseline_96": baseline_summary,
            "short32": short_summary,
            "paired_question_mean_delta_f1": mean(
                float(row["delta_f1"]) for row in paired
            ),
            "clustered_inference": inference,
        }
        comparison["models"][stage] = model_summary
        for label, metrics in (
            ("baseline_96", baseline_summary),
            ("short32", short_summary),
        ):
            summary_rows.append(
                {
                    "model": model,
                    "protocol": label,
                    **{key: round(value, 6) for key, value in metrics.items()},
                }
            )
        with (analysis_dir / f"paired_{stage}.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
            writer.writeheader()
            writer.writerows(paired)

    write_jsonl(root / RESULT_DIR / "combined_short32.jsonl", combined)
    score_results(
        root / RESULT_DIR / "combined_short32.jsonl",
        analysis_dir / "scored_short32",
    )
    (analysis_dir / "sensitivity_summary.json").write_text(
        json.dumps(comparison, indent=2) + "\n",
        encoding="utf-8",
    )
    with (analysis_dir / "sensitivity_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    bundle = root / BUNDLE_PATH
    candidates = [
        _result_path(root, "2b"),
        _result_path(root, "9b"),
        root / RESULT_DIR / "combined_short32.jsonl",
        root / RESULT_DIR / "environment_2b.json",
        root / RESULT_DIR / "environment_9b.json",
        *sorted(path for path in analysis_dir.rglob("*") if path.is_file()),
    ]
    manifest: list[dict[str, Any]] = []
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in candidates:
            name = path.relative_to(root).as_posix()
            archive.write(path, name)
            manifest.append(
                {
                    "path": name,
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
        archive.writestr("bundle_manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(comparison, indent=2))
    print(f"Final bundle: {bundle}")
    return bundle


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = args.project_root.resolve()
    validate_inputs(root)
    if args.stage == "check":
        return
    if args.stage in ("2b", "all"):
        run_model(root, "2b", args.required_gpu)
    if args.stage in ("9b", "all"):
        run_model(root, "9b", args.required_gpu)
    if args.stage in ("finalize", "all"):
        finalize(root, args.required_gpu)


if __name__ == "__main__":
    main()
