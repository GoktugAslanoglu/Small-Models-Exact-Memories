from __future__ import annotations

import argparse
import gc
import random
import time
from pathlib import Path
from typing import Any

from .colab_runner import _load_runtime, _model_device, _sync
from .common import project_path, read_jsonl, write_jsonl
from .locomo_external import CONDITIONS, SEED
from .modeling import (
    configure_image_budget,
    count_text_tokens,
    count_vision_tokens,
    load_images,
)


INSTRUCTION = (
    "Answer the question using only the supplied conversation memory. "
    "Give a concise answer with no explanation. If the memory does not support "
    "an answer, reply exactly: No information available.\n\nQuestion: {question}"
)

SHORT_ANSWER_INSTRUCTION = (
    "Answer the question using only the supplied conversation memory. "
    "Return only the shortest answer needed. No explanation. If the memory "
    "does not contain an answer, reply exactly: No information available."
    "\n\nQuestion: {question}"
)

PROMPT_PROFILES = {
    "main96": INSTRUCTION,
    "short32": SHORT_ANSWER_INSTRUCTION,
}


def parse_conditions(value: str) -> tuple[str, ...]:
    conditions = tuple(part.strip() for part in value.split(",") if part.strip())
    invalid = sorted(set(conditions) - set(CONDITIONS))
    if not conditions:
        raise argparse.ArgumentTypeError("At least one condition is required")
    if invalid:
        raise argparse.ArgumentTypeError(
            "Unknown condition(s): " + ", ".join(invalid)
        )
    if len(set(conditions)) != len(conditions):
        raise argparse.ArgumentTypeError("Conditions must be unique")
    return conditions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Qwen3.5 model on the frozen LoCoMo external protocol.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--conditions",
        type=parse_conditions,
        default=tuple(CONDITIONS),
        help="Comma-separated subset of the frozen conditions.",
    )
    parser.add_argument(
        "--prompt-profile",
        choices=tuple(PROMPT_PROFILES),
        default="main96",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, help="Debug-only cap on question-condition calls.")
    return parser.parse_args()


def _apply_chat_template(processor: Any, content: Any) -> Any:
    messages = [{"role": "user", "content": content}]
    kwargs = {
        "add_generation_prompt": True,
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    try:
        return processor.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return processor.apply_chat_template(messages, **kwargs)


def _prepare_text(
    processor: Any,
    history: str,
    question: str,
    instruction: str = INSTRUCTION,
) -> Any:
    content = (
        "--- CONVERSATION MEMORY START ---\n"
        f"{history}\n"
        "--- CONVERSATION MEMORY END ---\n\n"
        + instruction.format(question=question)
    )
    return _apply_chat_template(processor, content)


def _prepare_optical(
    processor: Any,
    images: list[Any],
    question: str,
    instruction: str = INSTRUCTION,
) -> Any:
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": instruction.format(question=question)})
    return _apply_chat_template(processor, content)


def _result_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["question_id"]), str(row["condition"]), str(row["model"])


def main() -> None:
    args = parse_args()
    root = project_path(args.root)
    data_dir = root / "data" / "generated" / "locomo_external100"
    render_dir = root / "renders" / "generated" / "locomo_external100"
    questions = list(read_jsonl(data_dir / "questions.jsonl"))
    conversations = {
        str(row["conversation_id"]): row
        for row in read_jsonl(data_dir / "conversations.jsonl")
    }
    conditions = tuple(args.conditions)
    instruction = PROMPT_PROFILES[args.prompt_profile]
    full_manifest_path = render_dir / "full_manifest.jsonl"
    oracle_manifest_path = render_dir / "oracle_manifest.jsonl"
    full_manifest = (
        {
            str(row["conversation_id"]): row
            for row in read_jsonl(full_manifest_path)
        }
        if "full_optical_8px" in conditions
        else {}
    )
    oracle_manifest = (
        {str(row["question_id"]): row for row in read_jsonl(oracle_manifest_path)}
        if "oracle_evidence_24px" in conditions
        else {}
    )
    if len(questions) != 100 or len(conversations) != 10:
        raise SystemExit("Run the LoCoMo preparation and render cells before inference.")

    work: list[tuple[dict[str, Any], str]] = [
        (question, condition) for question in questions for condition in conditions
    ]
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
    results = list(existing)
    print(
        f"Loaded {args.model} on {gpu_name} ({dtype_name}); "
        f"{len(completed)}/{len(work)} calls already complete"
    )

    for ordinal, (question, condition) in enumerate(work, start=1):
        key = (str(question["question_id"]), condition, args.model)
        if key in completed:
            continue
        conversation = conversations[str(question["conversation_id"])]
        images: list[Any] = []
        manifest_row: dict[str, Any] | None = None
        if condition == "full_raw_text":
            memory_text = str(conversation["full_history"])
            inputs = _prepare_text(
                processor,
                memory_text,
                str(question["question"]),
                instruction,
            )
            render_kind = "raw_text"
        elif condition == "session_summary_text":
            memory_text = str(conversation["session_summary"])
            inputs = _prepare_text(
                processor,
                memory_text,
                str(question["question"]),
                instruction,
            )
            render_kind = "provided_session_summary"
        elif condition == "full_optical_8px":
            memory_text = str(conversation["full_history"])
            manifest_row = full_manifest[str(question["conversation_id"])]
            image_paths = [
                (full_manifest_path.parent / relative).resolve()
                for relative in manifest_row["image_paths"]
            ]
            images = load_images(image_paths)
            configure_image_budget(
                processor, max(image.width * image.height for image in images)
            )
            inputs = _prepare_optical(
                processor,
                images,
                str(question["question"]),
                instruction,
            )
            render_kind = "uniform_8px"
        elif condition == "oracle_evidence_24px":
            memory_text = str(question["oracle_history"])
            manifest_row = oracle_manifest[str(question["question_id"])]
            image_paths = [
                (oracle_manifest_path.parent / relative).resolve()
                for relative in manifest_row["image_paths"]
            ]
            images = load_images(image_paths)
            configure_image_budget(
                processor, max(image.width * image.height for image in images)
            )
            inputs = _prepare_optical(
                processor,
                images,
                str(question["question"]),
                instruction,
            )
            render_kind = "oracle_24px"
        else:
            raise ValueError(condition)

        text_tokens = count_text_tokens(processor, memory_text)
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
            "schema_version": 1,
            "benchmark": "LoCoMo",
            "question_id": question["question_id"],
            "conversation_id": question["conversation_id"],
            "source_qa_index": question["source_qa_index"],
            "category": question["category"],
            "category_name": question["category_name"],
            "condition": condition,
            "render_kind": render_kind,
            "model": args.model,
            "question": question["question"],
            "answer": question["answer"],
            "evidence_ids": question["evidence_ids"],
            "prediction": prediction,
            "text_tokens": text_tokens,
            "vision_tokens": vision_tokens,
            "grid_vision_tokens": grid_vision_tokens,
            "realized_compression": (
                round(text_tokens / realized_vision_tokens, 6)
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
            "prompt_profile": args.prompt_profile,
        }
        results.append(result)
        completed.add(key)
        write_jsonl(output_path, results)
        print(
            f"[{ordinal:03d}/{len(work)}] {condition:<27} "
            f"{question['question_id']} -> {prediction[:72]!r} ({latency:.2f}s)"
        )
        del inputs, generated
        for image in images:
            image.close()
        del images
        gc.collect()

    print(f"Complete: {len(results)} rows in {output_path}")


if __name__ == "__main__":
    main()
