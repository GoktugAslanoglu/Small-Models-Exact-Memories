from __future__ import annotations

import argparse
import gc
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .colab_runner import _load_runtime, _model_device, _sync
from .common import project_path, read_jsonl, write_jsonl
from .locomo_controls import (
    CONTROL_CONDITIONS,
    EXPECTED_CALLS_PER_MODEL,
    PROTOCOL_VERSION,
    SEED,
    build_control_work,
    pack_bm25_documents,
)
from .locomo_runner import INSTRUCTION, _apply_chat_template, _prepare_optical, _prepare_text
from .modeling import (
    configure_image_budget,
    count_text_tokens,
    count_vision_tokens,
    load_images,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one Qwen3.5 model on the frozen LoCoMo reviewer-control matrix."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, help="Debug-only cap on calls.")
    return parser.parse_args()


def _prepare_question_only(processor: Any, question: str) -> Any:
    content = (
        "No conversation memory is supplied for this baseline.\n\n"
        + INSTRUCTION.format(question=question)
    )
    return _apply_chat_template(processor, content)


def _result_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["question_id"]), str(row["condition"]), str(row["model"])


def _measure_full_optical_budgets(
    processor: Any,
    conversations: dict[str, dict[str, Any]],
    full_manifest: dict[str, dict[str, Any]],
    full_manifest_path: Path,
) -> dict[str, int]:
    budgets: dict[str, int] = {}
    for conversation_id in sorted(conversations):
        row = full_manifest[conversation_id]
        paths = [(full_manifest_path.parent / value).resolve() for value in row["image_paths"]]
        images = load_images(paths)
        configure_image_budget(
            processor, max(image.width * image.height for image in images)
        )
        inputs = _prepare_optical(
            processor,
            images,
            "Budget measurement only; do not answer.",
        )
        vision_tokens, grid_tokens = count_vision_tokens(processor, inputs)
        budget = vision_tokens or grid_tokens
        if budget is None or budget <= 0:
            raise RuntimeError(f"Could not measure the visual-token budget for {conversation_id}")
        budgets[conversation_id] = int(budget)
        for image in images:
            image.close()
        del images, inputs
    return budgets


def main() -> None:
    args = parse_args()
    if args.seed != SEED:
        raise SystemExit(f"Protocol seed is frozen at {SEED}; received {args.seed}")
    root = project_path(args.root)
    controls_data_dir = root / "data" / "generated" / "locomo_controls100"
    controls_render_dir = root / "renders" / "generated" / "locomo_controls100"
    original_data_dir = root / "data" / "generated" / "locomo_external100"
    original_render_dir = root / "renders" / "generated" / "locomo_external100"

    questions = list(read_jsonl(controls_data_dir / "questions.jsonl"))
    conversations = {
        str(row["conversation_id"]): row
        for row in read_jsonl(controls_data_dir / "conversations.jsonl")
    }
    retrieval_documents: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(controls_data_dir / "retrieval_documents.jsonl"):
        retrieval_documents[str(row["conversation_id"])].append(row)
    for rows in retrieval_documents.values():
        rows.sort(key=lambda row: int(row["document_index"]))

    selected_manifest_path = controls_render_dir / "selected_8px_manifest.jsonl"
    temporal_manifest_path = controls_render_dir / "temporal_oracle24_fixed_manifest.jsonl"
    full_manifest_path = original_render_dir / "full_manifest.jsonl"
    selected_manifest = {
        str(row["question_id"]): row for row in read_jsonl(selected_manifest_path)
    }
    temporal_manifest = {
        str(row["question_id"]): row for row in read_jsonl(temporal_manifest_path)
    }
    full_manifest = {
        str(row["conversation_id"]): row for row in read_jsonl(full_manifest_path)
    }
    original_conversations = {
        str(row["conversation_id"]): row
        for row in read_jsonl(original_data_dir / "conversations.jsonl")
    }
    if (
        len(questions) != 100
        or len(conversations) != 10
        or len(selected_manifest) != 100
        or len(temporal_manifest) != 30
        or len(full_manifest) != 10
        or len(original_conversations) != 10
    ):
        raise SystemExit("Run the preparation cell before control inference.")

    work = build_control_work(questions)
    if len(work) != EXPECTED_CALLS_PER_MODEL:
        raise RuntimeError(f"Expected {EXPECTED_CALLS_PER_MODEL} control calls")
    if args.limit is not None:
        work = work[: args.limit]

    output_path = project_path(args.output, root)
    existing: list[dict[str, Any]] = []
    completed: set[tuple[str, str, str]] = set()
    if args.resume and output_path.is_file():
        existing = list(read_jsonl(output_path))
        completed = {_result_key(row) for row in existing}

    random.seed(args.seed)
    torch, processor, model, dtype_name = _load_runtime(args.model)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    gpu_name = torch.cuda.get_device_name(0)
    full_optical_budgets = _measure_full_optical_budgets(
        processor,
        original_conversations,
        full_manifest,
        full_manifest_path,
    )
    results = list(existing)
    print(
        f"Loaded {args.model} on {gpu_name} ({dtype_name}); "
        f"{len(completed)}/{len(work)} calls already complete"
    )
    print("Measured full-optical memory-token budgets:", full_optical_budgets)

    for ordinal, (question, condition) in enumerate(work, start=1):
        key = (str(question["question_id"]), condition, args.model)
        if key in completed:
            continue
        question_id = str(question["question_id"])
        conversation_id = str(question["conversation_id"])
        question_text = str(question["question"])
        selected_history = str(question["selected_history"])
        source_full_history = str(conversations[conversation_id]["full_history"])
        images: list[Any] = []
        manifest_row: dict[str, Any] | None = None
        retrieval_metadata: dict[str, Any] = {}
        target_memory_tokens: int | None = None

        if condition == "oracle_evidence_24px_temporal_fixed":
            memory_text = selected_history
            manifest_row = temporal_manifest[question_id]
            paths = [
                (temporal_manifest_path.parent / value).resolve()
                for value in manifest_row["image_paths"]
            ]
            images = load_images(paths)
            configure_image_budget(
                processor, max(image.width * image.height for image in images)
            )
            inputs = _prepare_optical(processor, images, question_text)
            render_kind = "temporal_oracle_fixed_24px"
        elif condition == "selected_evidence_8px":
            memory_text = selected_history
            manifest_row = selected_manifest[question_id]
            paths = [
                (selected_manifest_path.parent / value).resolve()
                for value in manifest_row["image_paths"]
            ]
            images = load_images(paths)
            configure_image_budget(
                processor, max(image.width * image.height for image in images)
            )
            inputs = _prepare_optical(processor, images, question_text)
            render_kind = "selected_uniform_8px"
        elif condition == "selected_evidence_text":
            memory_text = selected_history
            inputs = _prepare_text(processor, memory_text, question_text)
            render_kind = "selected_text_identical_source"
        elif condition == "question_only":
            memory_text = ""
            inputs = _prepare_question_only(processor, question_text)
            render_kind = "no_memory"
        elif condition == "bm25_equal_vision_budget_text":
            target_memory_tokens = full_optical_budgets[conversation_id]
            memory_text, retrieval_metadata = pack_bm25_documents(
                question_text,
                retrieval_documents[conversation_id],
                lambda text: count_text_tokens(processor, text),
                target_memory_tokens,
            )
            inputs = _prepare_text(processor, memory_text, question_text)
            render_kind = "bm25_turn_retrieval_text"
        else:
            raise ValueError(condition)

        memory_text_tokens = count_text_tokens(processor, memory_text)
        source_full_text_tokens = count_text_tokens(processor, source_full_history)
        if images:
            vision_tokens, grid_vision_tokens = count_vision_tokens(processor, inputs)
        else:
            vision_tokens, grid_vision_tokens = None, None
        realized_vision_tokens = vision_tokens or grid_vision_tokens
        inputs = inputs.to(_model_device(model))
        _sync(torch)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
        _sync(torch)
        latency = time.perf_counter() - started
        prompt_length = int(inputs["input_ids"].shape[-1])
        prediction = processor.decode(
            generated[0][prompt_length:], skip_special_tokens=True
        ).strip()
        result = {
            "schema_version": 2,
            "protocol_version": PROTOCOL_VERSION,
            "benchmark": "LoCoMo",
            "question_id": question_id,
            "conversation_id": conversation_id,
            "source_qa_index": question["source_qa_index"],
            "category": question["category"],
            "category_name": question["category_name"],
            "condition": condition,
            "render_kind": render_kind,
            "model": args.model,
            "question": question_text,
            "answer": question["answer"],
            "evidence_ids": question["evidence_ids"],
            "selected_history_sha256": question["selected_history_sha256"],
            "prediction": prediction,
            "text_tokens": memory_text_tokens,
            "source_full_text_tokens": source_full_text_tokens,
            "target_memory_tokens": target_memory_tokens,
            "vision_tokens": vision_tokens,
            "grid_vision_tokens": grid_vision_tokens,
            "realized_compression": (
                round(source_full_text_tokens / realized_vision_tokens, 6)
                if realized_vision_tokens
                else None
            ),
            "page_count": manifest_row.get("page_count") if manifest_row else None,
            "font_size": manifest_row.get("font_size") if manifest_row else None,
            "image_width": manifest_row.get("image_width") if manifest_row else None,
            "image_height": manifest_row.get("image_height") if manifest_row else None,
            "latency_seconds": round(latency, 6),
            "generated_tokens": int(generated.shape[-1] - prompt_length),
            "torch_dtype": dtype_name,
            "gpu_name": gpu_name,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            **retrieval_metadata,
        }
        results.append(result)
        completed.add(key)
        write_jsonl(output_path, results)
        print(
            f"[{ordinal:03d}/{len(work)}] {condition:<38} "
            f"{question_id} -> {prediction[:72]!r} ({latency:.2f}s)"
        )
        del inputs, generated
        for image in images:
            image.close()
        del images
        gc.collect()

    print(f"Complete: {len(results)} rows in {output_path}")


if __name__ == "__main__":
    main()
