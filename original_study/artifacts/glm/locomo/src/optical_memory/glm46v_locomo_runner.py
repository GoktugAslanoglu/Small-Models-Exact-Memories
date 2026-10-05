from __future__ import annotations

import argparse
import gc
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import project_path, read_jsonl, write_jsonl
from .glm46v import (
    MODEL_ID,
    load_runtime,
    measure_budget,
    model_device,
    sanitize_inputs,
    sync_cuda,
    text_budget,
)
from .locomo_controls import pack_bm25_documents
from .locomo_external import SEED
from .locomo_runner import _prepare_optical, _prepare_text
from .modeling import count_text_tokens, load_images


PROTOCOL_VERSION = "locomo-cross-family-four-way-v1"
CONDITIONS = (
    "full_raw_text",
    "full_optical_8px",
    "selected_evidence_8px",
    "bm25_equal_vision_budget_text",
)
EXPECTED_CALLS = 400


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run GLM-4.6V-Flash on the frozen 100-question LoCoMo "
            "raw/full-optical/selected-optical/BM25 replication."
        )
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, help="Debug-only cap on calls.")
    return parser.parse_args()


def _result_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["question_id"]), str(row["condition"]), str(row["model"])


def _load_manifest(path: Path, key: str) -> dict[str, dict[str, Any]]:
    return {str(row[key]): row for row in read_jsonl(path)}


def _open_manifest_images(path: Path, row: dict[str, Any]) -> list[Any]:
    paths = [(path.parent / str(value)).resolve() for value in row["image_paths"]]
    return load_images(paths)


def _measure_full_optical_budgets(
    processor: Any,
    conversations: dict[str, dict[str, Any]],
    manifest: dict[str, dict[str, Any]],
    manifest_path: Path,
) -> dict[str, int]:
    budgets: dict[str, int] = {}
    for conversation_id in sorted(conversations):
        row = manifest[conversation_id]
        images = _open_manifest_images(manifest_path, row)
        inputs = _prepare_optical(
            processor,
            images,
            "Budget measurement only; do not answer.",
        )
        inputs = sanitize_inputs(inputs)
        memory_tokens = count_text_tokens(
            processor,
            str(conversations[conversation_id]["full_history"]),
        )
        budget = measure_budget(
            processor,
            inputs,
            memory_text_tokens=memory_tokens,
        )
        vision_tokens = budget["vision_tokens"]
        if not isinstance(vision_tokens, int) or vision_tokens <= 0:
            raise RuntimeError(
                f"Could not measure GLM's full-optical budget for {conversation_id}"
            )
        budgets[conversation_id] = vision_tokens
        for image in images:
            image.close()
        del images, inputs
    return budgets


def main() -> None:
    args = parse_args()
    if args.model != MODEL_ID:
        raise SystemExit(f"This runner is frozen to {MODEL_ID}; received {args.model}")
    if args.seed != SEED:
        raise SystemExit(f"Protocol seed is frozen at {SEED}; received {args.seed}")

    root = project_path(args.root)
    original_data_dir = root / "data" / "generated" / "locomo_external100"
    controls_data_dir = root / "data" / "generated" / "locomo_controls100"
    original_render_dir = root / "renders" / "generated" / "locomo_external100"
    controls_render_dir = root / "renders" / "generated" / "locomo_controls100"

    questions = list(read_jsonl(controls_data_dir / "questions.jsonl"))
    conversations = {
        str(row["conversation_id"]): row
        for row in read_jsonl(controls_data_dir / "conversations.jsonl")
    }
    original_conversations = {
        str(row["conversation_id"]): row
        for row in read_jsonl(original_data_dir / "conversations.jsonl")
    }
    retrieval_documents: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(controls_data_dir / "retrieval_documents.jsonl"):
        retrieval_documents[str(row["conversation_id"])].append(row)
    for rows in retrieval_documents.values():
        rows.sort(key=lambda row: int(row["document_index"]))

    full_manifest_path = original_render_dir / "full_manifest.jsonl"
    selected_manifest_path = controls_render_dir / "selected_8px_manifest.jsonl"
    full_manifest = _load_manifest(full_manifest_path, "conversation_id")
    selected_manifest = _load_manifest(selected_manifest_path, "question_id")
    if (
        len(questions) != 100
        or len(conversations) != 10
        or len(original_conversations) != 10
        or len(full_manifest) != 10
        or len(selected_manifest) != 100
    ):
        raise SystemExit("Run the GLM LoCoMo preparation stage before inference.")

    work = [
        (question, condition)
        for question in questions
        for condition in CONDITIONS
    ]
    if args.limit is not None:
        work = work[: args.limit]

    output_path = project_path(args.output, root)
    existing: list[dict[str, Any]] = []
    completed: set[tuple[str, str, str]] = set()
    if args.resume and output_path.is_file():
        existing = list(read_jsonl(output_path))
        if any(str(row.get("model")) != args.model for row in existing):
            raise SystemExit(f"Resume output contains a different model: {output_path}")
        completed = {_result_key(row) for row in existing}

    random.seed(args.seed)
    torch, processor, model, dtype_name = load_runtime(args.model)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    gpu_name = torch.cuda.get_device_name(0)
    model_commit = getattr(model.config, "_commit_hash", None)
    full_optical_budgets = _measure_full_optical_budgets(
        processor,
        original_conversations,
        full_manifest,
        full_manifest_path,
    )
    print(
        f"Loaded {args.model} on {gpu_name} ({dtype_name}); "
        f"{len(completed)}/{len(work)} calls already complete"
    )
    print("GLM full-optical vision-token budgets:", full_optical_budgets)

    results = list(existing)
    for ordinal, (question, condition) in enumerate(work, start=1):
        key = (str(question["question_id"]), condition, args.model)
        if key in completed:
            continue
        question_id = str(question["question_id"])
        conversation_id = str(question["conversation_id"])
        question_text = str(question["question"])
        source_full_history = str(conversations[conversation_id]["full_history"])
        source_full_tokens = count_text_tokens(processor, source_full_history)
        selected_history = str(question["selected_history"])
        manifest_row: dict[str, Any] | None = None
        images: list[Any] = []
        retrieval_metadata: dict[str, Any] = {}
        target_memory_tokens: int | None = None

        if condition == "full_raw_text":
            memory_text = source_full_history
            memory_text_tokens = source_full_tokens
            inputs = _prepare_text(processor, memory_text, question_text)
            inputs = sanitize_inputs(inputs)
            budget = text_budget(
                processor,
                inputs,
                memory_text_tokens=memory_text_tokens,
            )
            render_kind = "raw_text"
        elif condition == "full_optical_8px":
            memory_text = source_full_history
            memory_text_tokens = source_full_tokens
            manifest_row = full_manifest[conversation_id]
            images = _open_manifest_images(full_manifest_path, manifest_row)
            inputs = _prepare_optical(processor, images, question_text)
            inputs = sanitize_inputs(inputs)
            budget = measure_budget(
                processor,
                inputs,
                memory_text_tokens=memory_text_tokens,
            )
            render_kind = "uniform_8px"
        elif condition == "selected_evidence_8px":
            memory_text = selected_history
            memory_text_tokens = count_text_tokens(processor, memory_text)
            manifest_row = selected_manifest[question_id]
            images = _open_manifest_images(selected_manifest_path, manifest_row)
            inputs = _prepare_optical(processor, images, question_text)
            inputs = sanitize_inputs(inputs)
            budget = measure_budget(
                processor,
                inputs,
                memory_text_tokens=memory_text_tokens,
            )
            render_kind = "selected_uniform_8px"
        elif condition == "bm25_equal_vision_budget_text":
            target_memory_tokens = full_optical_budgets[conversation_id]
            memory_text, retrieval_metadata = pack_bm25_documents(
                question_text,
                retrieval_documents[conversation_id],
                lambda text: count_text_tokens(processor, text),
                target_memory_tokens,
            )
            memory_text_tokens = count_text_tokens(processor, memory_text)
            inputs = _prepare_text(processor, memory_text, question_text)
            inputs = sanitize_inputs(inputs)
            budget = text_budget(
                processor,
                inputs,
                memory_text_tokens=memory_text_tokens,
            )
            render_kind = "bm25_turn_retrieval_text"
        else:
            raise ValueError(condition)

        vision_tokens = budget["vision_tokens"]
        source_to_vision = (
            round(source_full_tokens / vision_tokens, 6)
            if isinstance(vision_tokens, int) and vision_tokens > 0
            else None
        )
        inputs = inputs.to(model_device(model))
        sync_cuda(torch)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
        sync_cuda(torch)
        latency = time.perf_counter() - started
        prompt_length = int(inputs["input_ids"].shape[-1])
        prediction = processor.decode(
            generated[0][prompt_length:],
            skip_special_tokens=True,
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
            "model_commit": model_commit,
            "question": question_text,
            "answer": question["answer"],
            "evidence_ids": question["evidence_ids"],
            "selected_history_sha256": question["selected_history_sha256"],
            "prediction": prediction,
            "text_tokens": memory_text_tokens,
            "memory_text_tokens": memory_text_tokens,
            "source_full_text_tokens": source_full_tokens,
            "target_memory_tokens": target_memory_tokens,
            **budget,
            "source_to_vision_compression": source_to_vision,
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
            f"[{ordinal:03d}/{len(work)}] {condition:<34} {question_id} "
            f"-> {prediction[:64]!r} ({latency:.2f}s; "
            f"vision={vision_tokens}; C={budget['model_specific_compression']})"
        )

        del inputs, generated
        for image in images:
            image.close()
        del images
        gc.collect()

    print(f"Complete: {len(results)} rows in {output_path}")


if __name__ == "__main__":
    main()
