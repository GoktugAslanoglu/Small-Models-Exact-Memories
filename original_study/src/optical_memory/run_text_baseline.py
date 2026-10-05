from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any

from .common import project_path, read_jsonl, write_jsonl
from .modeling import count_text_tokens


PROMPT = (
    "The following is earlier agent history. Answer the memory question using only "
    "that history. Return only the requested value, without explanation.\n\n"
    "--- HISTORY START ---\n{history}\n--- HISTORY END ---\n\nQuestion: {question}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the raw-text upper-bound baseline.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _prepare(processor: Any, history: str, question: str) -> Any:
    messages = [
        {
            "role": "user",
            "content": PROMPT.format(history=history, question=question),
        }
    ]
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


def _device(model: Any) -> Any:
    for parameter in model.parameters():
        if str(parameter.device) != "meta":
            return parameter.device
    return model.device


def main() -> None:
    args = parse_args()
    try:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor
    except ImportError as exc:
        raise SystemExit("Install the Qwen3.5 HF dependencies described in RUNBOOK.md") from exc

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset_path = project_path(args.dataset)
    output_path = project_path(args.output)
    episodes = list(read_jsonl(dataset_path))
    if args.limit is not None:
        episodes = episodes[: args.limit]

    existing: list[dict[str, Any]] = []
    completed: set[tuple[str, str]] = set()
    if args.resume and output_path.exists():
        existing = list(read_jsonl(output_path))
        completed = {(str(row["episode_id"]), str(row["model"])) for row in existing}

    processor = AutoProcessor.from_pretrained(args.model)
    model_kwargs: dict[str, Any] = {"low_cpu_mem_usage": True, "torch_dtype": "auto"}
    if args.device == "auto":
        model_kwargs["device_map"] = "auto"
    model = AutoModelForMultimodalLM.from_pretrained(args.model, **model_kwargs)
    if args.device != "auto":
        model.to(args.device)
    model.eval()

    results = list(existing)
    for index, episode in enumerate(episodes, start=1):
        key = (str(episode["episode_id"]), args.model)
        if key in completed:
            continue
        inputs = _prepare(processor, str(episode["history"]), str(episode["question"]))
        inputs = inputs.to(_device(model))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.perf_counter() - started
        prompt_length = inputs["input_ids"].shape[-1]
        prediction = processor.decode(generated[0][prompt_length:], skip_special_tokens=True)
        result = {
            "schema_version": 1,
            "episode_id": episode["episode_id"],
            "memory_type": episode["memory_type"],
            "position_bucket": episode["position_bucket"],
            "condition": "raw_text",
            "render_kind": "raw_text",
            "target_compression": 0,
            "model": args.model,
            "question": episode["question"],
            "answer": episode["answer"],
            "scoring": episode["scoring"],
            "prediction": prediction,
            "text_tokens": count_text_tokens(processor, str(episode["history"])),
            "vision_tokens": None,
            "grid_vision_tokens": None,
            "realized_compression": None,
            "image_width": None,
            "image_height": None,
            "latency_seconds": round(latency, 6),
            "generated_tokens": int(generated.shape[-1] - prompt_length),
        }
        results.append(result)
        write_jsonl(output_path, results)
        print(f"[{index}/{len(episodes)}] {episode['episode_id']} -> {prediction[:80]!r}")

    print(f"Wrote {len(results)} raw-text results to {output_path}")


if __name__ == "__main__":
    main()

