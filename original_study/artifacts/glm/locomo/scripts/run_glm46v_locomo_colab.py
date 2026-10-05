"""Prepare and run the four-condition GLM LoCoMo cross-family replication."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

from optical_memory.common import read_jsonl, write_jsonl
from optical_memory.glm46v import MODEL_ID
from optical_memory.glm46v_locomo_runner import CONDITIONS, EXPECTED_CALLS
from optical_memory.locomo_controls import (
    PROTOCOL_VERSION as CONTROLS_PROTOCOL,
    prepare_controls_protocol,
)
from optical_memory.locomo_external import (
    SEED,
    _render_text,
    prepare_protocol,
    score_results,
)


ROOT_DEFAULT = Path("/content/slm_agents")
RESULT_PATH = Path("results/locomo_glm46v_flash_fourway.jsonl")
ANALYSIS_DIR = Path("results/locomo_glm46v_flash_fourway_analysis")
METADATA_PATH = Path("results/locomo_glm46v_flash_fourway_metadata.json")
RESULT_ZIP = Path("results/locomo_glm46v_flash_fourway_bundle.zip")


def _helpers() -> Any:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from scripts import run_qwen35_9b_main_colab as helpers

    return helpers


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen four-condition GLM LoCoMo-100 replication."
    )
    parser.add_argument(
        "stage",
        choices=("prepare", "run"),
        help="Prepare/download/render inputs, then run inference in a later call.",
    )
    parser.add_argument("--project-root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--required-gpu", default="A100")
    parser.add_argument("--zip-output", action="store_true")
    return parser.parse_args(argv)


def _manifest_ready(path: Path, expected_rows: int) -> bool:
    if not path.is_file():
        return False
    rows = list(read_jsonl(path))
    return len(rows) == expected_rows and all(
        (path.parent / str(relative)).is_file()
        for row in rows
        for relative in row["image_paths"]
    )


def _render_full_history(root: Path, font_path: str) -> Path:
    data_path = root / "data" / "generated" / "locomo_external100" / "conversations.jsonl"
    render_dir = root / "renders" / "generated" / "locomo_external100"
    manifest_path = render_dir / "full_manifest.jsonl"
    if _manifest_ready(manifest_path, 10):
        print("Reusing complete full-optical 8px manifest.")
        return manifest_path

    rows: list[dict[str, Any]] = []
    conversations = list(read_jsonl(data_path))
    for ordinal, conversation in enumerate(conversations, start=1):
        image_paths, metadata = _render_text(
            str(conversation["full_history"]),
            render_dir,
            Path("images") / "full_8px",
            str(conversation["conversation_id"]),
            font_path,
            8,
            24,
        )
        rows.append(
            {
                "schema_version": 1,
                "conversation_id": conversation["conversation_id"],
                "condition": "full_optical_8px",
                "render_kind": "uniform",
                "image_paths": image_paths,
                **metadata,
            }
        )
        print(
            f"Rendered full LoCoMo conversation {ordinal}/{len(conversations)}: "
            f"{len(image_paths)} pages"
        )
    write_jsonl(manifest_path, rows)
    return manifest_path


def _render_selected_evidence(root: Path, font_path: str) -> Path:
    data_path = root / "data" / "generated" / "locomo_controls100" / "questions.jsonl"
    render_dir = root / "renders" / "generated" / "locomo_controls100"
    manifest_path = render_dir / "selected_8px_manifest.jsonl"
    if _manifest_ready(manifest_path, 100):
        print("Reusing complete selected-evidence 8px manifest.")
        return manifest_path

    rows: list[dict[str, Any]] = []
    questions = list(read_jsonl(data_path))
    for ordinal, question in enumerate(questions, start=1):
        image_paths, metadata = _render_text(
            str(question["selected_history"]),
            render_dir,
            Path("images") / "selected_8px",
            str(question["question_id"]),
            font_path,
            8,
            8,
        )
        rows.append(
            {
                "schema_version": 1,
                "protocol_version": CONTROLS_PROTOCOL,
                "question_id": question["question_id"],
                "conversation_id": question["conversation_id"],
                "condition": "selected_evidence_8px",
                "render_kind": "selected_uniform_8px",
                "source_text_sha256": question["selected_history_sha256"],
                "image_paths": image_paths,
                **metadata,
            }
        )
        if ordinal % 20 == 0:
            print(f"Rendered selected evidence {ordinal}/{len(questions)}")
    write_jsonl(manifest_path, rows)
    return manifest_path


def validate_prepared(root: Path) -> None:
    original_dir = root / "data" / "generated" / "locomo_external100"
    controls_dir = root / "data" / "generated" / "locomo_controls100"
    questions = list(read_jsonl(controls_dir / "questions.jsonl"))
    conversations = list(read_jsonl(controls_dir / "conversations.jsonl"))
    full_path = root / "renders" / "generated" / "locomo_external100" / "full_manifest.jsonl"
    selected_path = (
        root
        / "renders"
        / "generated"
        / "locomo_controls100"
        / "selected_8px_manifest.jsonl"
    )
    full = list(read_jsonl(full_path))
    selected = list(read_jsonl(selected_path))
    if len(questions) != 100 or len(conversations) != 10:
        raise ValueError("Expected the frozen 100-question, 10-conversation subset")
    if Counter(str(row["category_name"]) for row in questions) != Counter(
        {"single_hop": 50, "temporal": 30, "multi_hop": 20}
    ):
        raise ValueError("Unexpected LoCoMo category balance")
    if len(full) != 10 or len(selected) != 100:
        raise ValueError("Incomplete GLM LoCoMo render manifests")
    by_question = {str(row["question_id"]): row for row in questions}
    if any(
        str(row["source_text_sha256"])
        != str(by_question[str(row["question_id"])]["selected_history_sha256"])
        for row in selected
    ):
        raise ValueError("Selected evidence hash mismatch")
    for path, rows in ((full_path, full), (selected_path, selected)):
        for row in rows:
            for relative in row["image_paths"]:
                if not (path.parent / str(relative)).is_file():
                    raise FileNotFoundError(path.parent / str(relative))
    if not (original_dir / "protocol.json").is_file():
        raise FileNotFoundError(original_dir / "protocol.json")
    print(
        "GLM LoCoMo inputs: PASSED "
        "(100 questions; raw/full optical 8px/selected optical 8px/BM25)"
    )


def prepare(root: Path, seed: int) -> None:
    from matplotlib import font_manager

    source_path = root / "data" / "source" / "locomo10.json"
    prepare_protocol(root, source_path, seed=seed)
    prepare_controls_protocol(root, source_path, seed=seed)
    font_path = font_manager.findfont(
        "DejaVu Sans Mono",
        fallback_to_default=True,
    )
    print("Deterministic font:", font_path)
    _render_full_history(root, font_path)
    _render_selected_evidence(root, font_path)
    validate_prepared(root)


def validate_results(
    root: Path,
    *,
    model: str,
    required_gpu: str,
) -> list[dict[str, Any]]:
    questions = list(
        read_jsonl(
            root / "data" / "generated" / "locomo_controls100" / "questions.jsonl"
        )
    )
    rows = list(read_jsonl(root / RESULT_PATH))
    expected = {
        (str(question["question_id"]), condition, model)
        for question in questions
        for condition in CONDITIONS
    }
    actual = {
        (str(row["question_id"]), str(row["condition"]), str(row["model"]))
        for row in rows
    }
    if len(rows) != EXPECTED_CALLS or actual != expected:
        raise ValueError(
            f"Expected {EXPECTED_CALLS} unique GLM LoCoMo rows; found {len(rows)}"
        )
    if any(int(row["seed"]) != SEED for row in rows):
        raise ValueError("LoCoMo seed mismatch")
    if required_gpu and any(
        required_gpu.casefold() not in str(row["gpu_name"]).casefold()
        for row in rows
    ):
        raise ValueError("GPU provenance mismatch")
    optical = [
        row
        for row in rows
        if row["condition"] in ("full_optical_8px", "selected_evidence_8px")
    ]
    if any(
        row.get("vision_token_source")
        != "glm_input_ids_image_placeholders_crosschecked_with_grid"
        for row in optical
    ):
        raise ValueError("GLM visual-token provenance is missing")
    bm25 = [
        row for row in rows if row["condition"] == "bm25_equal_vision_budget_text"
    ]
    if any(
        int(row["text_tokens"]) > int(row["target_memory_tokens"])
        for row in bm25
    ):
        raise ValueError("BM25 exceeded GLM's full-optical budget")
    print("GLM LoCoMo result matrix: PASSED (400 unique rows)")
    return rows


def print_summary(summary_path: Path, result_rows: list[dict[str, Any]]) -> None:
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        summary = [
            row for row in csv.DictReader(handle) if row["category_name"] == "ALL"
        ]
    order = {condition: index for index, condition in enumerate(CONDITIONS)}
    summary.sort(key=lambda row: order[row["condition"]])
    print("\n=== GLM LoCoMo-100 four-condition replication ===")
    for row in summary:
        condition = row["condition"]
        selected = [item for item in result_rows if item["condition"] == condition]
        vision = [
            float(item["vision_tokens"])
            for item in selected
            if item.get("vision_tokens")
        ]
        compression = [
            float(item["model_specific_compression"])
            for item in selected
            if item.get("model_specific_compression") is not None
        ]
        details = [
            f"official F1={float(row['mean_official_token_f1']):.3f}",
            f"exact={float(row['normalized_exact_match_rate']):.3f}",
        ]
        if vision:
            details.append(f"GLM vision={mean(vision):.1f}")
        if compression:
            details.append(f"GLM C={mean(compression):.3f}x")
        print(f"{condition}: " + ", ".join(details))

    overall = {
        row["condition"]: float(row["mean_official_token_f1"]) for row in summary
    }
    gap = overall["selected_evidence_8px"] - overall["full_optical_8px"]
    print(f"Selected-optical minus full-optical F1: {gap:+.3f}")


def archive_paths(root: Path) -> Iterable[Path]:
    candidates = [
        root / RESULT_PATH,
        root / METADATA_PATH,
        root / "data" / "generated" / "locomo_external100" / "protocol.json",
        root / "data" / "generated" / "locomo_controls100" / "protocol.json",
        root / "data" / "generated" / "locomo_controls100" / "questions.jsonl",
        root / "data" / "generated" / "locomo_controls100" / "retrieval_documents.jsonl",
        root / "renders" / "generated" / "locomo_external100" / "full_manifest.jsonl",
        root / "renders" / "generated" / "locomo_controls100" / "selected_8px_manifest.jsonl",
        root / "scripts" / "run_glm46v_locomo_colab.py",
        root / "src" / "optical_memory" / "glm46v.py",
        root / "src" / "optical_memory" / "glm46v_locomo_runner.py",
    ]
    for path in candidates:
        if path.is_file():
            yield path
    analysis_dir = root / ANALYSIS_DIR
    if analysis_dir.is_dir():
        yield from sorted(path for path in analysis_dir.rglob("*") if path.is_file())


def run(args: argparse.Namespace, root: Path) -> None:
    h = _helpers()
    validate_prepared(root)
    gpu_info = h.gpu_preflight(args.required_gpu)
    output = root / RESULT_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    h.run_live(
        [
            sys.executable,
            "-u",
            "-m",
            "optical_memory.glm46v_locomo_runner",
            "--root",
            str(root),
            "--model",
            args.model,
            "--output",
            str(output),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--seed",
            str(args.seed),
            "--resume",
        ],
        cwd=root,
        label="Run GLM LoCoMo four-condition matrix (400 resumable calls)",
    )
    rows = validate_results(
        root,
        model=args.model,
        required_gpu=args.required_gpu,
    )
    score_paths = score_results(output, root / ANALYSIS_DIR)
    print_summary(score_paths["summary_csv"], rows)

    import transformers

    metadata = {
        "schema_version": 2,
        "experiment": "glm46v_flash_locomo_cross_family_fourway",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "seed": args.seed,
        "conditions": list(CONDITIONS),
        "expected_calls": EXPECTED_CALLS,
        "decode": {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "use_cache": True,
        },
        "runtime": {
            **gpu_info,
            "python_version": platform.python_version(),
            "transformers_version": transformers.__version__,
        },
        "token_accounting": {
            "full_optical_budget": "measured by the GLM processor per conversation",
            "BM25_budget": "GLM-tokenized packed text <= GLM full-optical vision tokens",
            "C": "rendered memory's GLM text tokens / GLM realized vision tokens",
            "qwen_token_counts_reused": False,
        },
    }
    metadata_path = root / METADATA_PATH
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.zip_output:
        h.create_archive(
            root / RESULT_ZIP,
            root=root,
            paths=archive_paths(root),
        )
    print("\nGLM LoCoMo cross-family replication complete.")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.model != MODEL_ID:
        raise SystemExit(f"This runner is frozen to {MODEL_ID}; received {args.model}")
    if args.seed != SEED:
        raise SystemExit(f"Protocol seed is frozen at {SEED}; received {args.seed}")
    if args.max_new_tokens != 96:
        raise SystemExit("This LoCoMo protocol requires 96 output tokens")
    root = args.project_root.resolve()
    if args.stage == "prepare":
        prepare(root, args.seed)
    else:
        run(args, root)


if __name__ == "__main__":
    main()
