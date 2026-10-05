from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any

from .common import project_path, read_jsonl, write_jsonl
from .modeling import (
    configure_image_budget,
    count_text_tokens,
    count_vision_tokens,
    load_images,
    prepare_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen3.5 on rendered optical-memory queries.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--conditions", help="Comma-separated condition names to retain.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _sync_cuda(torch: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _input_device(model: Any) -> Any:
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
        raise SystemExit(
            "torch and the latest transformers are required. Install the HF extras "
            "documented in README.md."
        ) from exc

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset_path = project_path(args.dataset)
    manifest_path = project_path(args.manifest)
    output_path = project_path(args.output)
    episodes = {row["episode_id"]: row for row in read_jsonl(dataset_path)}
    selected_conditions = set(args.conditions.split(",")) if args.conditions else None

    existing: list[dict[str, Any]] = []
    completed: set[tuple[str, str, str]] = set()
    if args.resume and output_path.exists():
        existing = list(read_jsonl(output_path))
        completed = {
            (row["episode_id"], row["condition"], row["model"]) for row in existing
        }

    processor = AutoProcessor.from_pretrained(args.model)
    model_kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
    if args.device == "auto":
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = "auto"
    else:
        model_kwargs["torch_dtype"] = "auto"
    model = AutoModelForMultimodalLM.from_pretrained(args.model, **model_kwargs)
    if args.device != "auto":
        model.to(args.device)
    model.eval()

    rows_to_run = [
        row
        for row in read_jsonl(manifest_path)
        if selected_conditions is None or row["condition"] in selected_conditions
    ]
    if args.limit is not None:
        rows_to_run = rows_to_run[: args.limit]

    results = list(existing)
    for ordinal, row in enumerate(rows_to_run, start=1):
        key = (row["episode_id"], row["condition"], args.model)
        if key in completed:
            continue
        episode = episodes[row["episode_id"]]
        image_paths = [(manifest_path.parent / path).resolve() for path in row["image_paths"]]
        images = load_images(image_paths)
        configure_image_budget(processor, max(image.width * image.height for image in images))
        inputs = prepare_inputs(processor, images, episode["question"])
        text_tokens = count_text_tokens(processor, episode["history"])
        vision_tokens, grid_vision_tokens = count_vision_tokens(processor, inputs)
        realized_tokens = vision_tokens or grid_vision_tokens
        device = _input_device(model)
        inputs = inputs.to(device)

        _sync_cuda(torch)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
        _sync_cuda(torch)
        latency = time.perf_counter() - started
        prompt_length = inputs["input_ids"].shape[-1]
        prediction = processor.decode(generated[0][prompt_length:], skip_special_tokens=True)

        result = {
            "schema_version": 1,
            "episode_id": row["episode_id"],
            "memory_type": episode["memory_type"],
            "position_bucket": episode["position_bucket"],
            "condition": row["condition"],
            "render_kind": row["render_kind"],
            "target_compression": row["target_compression"],
            "model": args.model,
            "question": episode["question"],
            "answer": episode["answer"],
            "scoring": episode["scoring"],
            "prediction": prediction,
            "text_tokens": text_tokens,
            "vision_tokens": vision_tokens,
            "grid_vision_tokens": grid_vision_tokens,
            "realized_compression": (
                round(text_tokens / realized_tokens, 6) if realized_tokens else None
            ),
            "image_width": row["image_width"],
            "image_height": row["image_height"],
            "latency_seconds": round(latency, 6),
            "generated_tokens": int(generated.shape[-1] - prompt_length),
        }
        results.append(result)
        write_jsonl(output_path, results)
        print(
            f"[{ordinal}/{len(rows_to_run)}] {row['episode_id']} {row['condition']} "
            f"-> {prediction[:80]!r} ({latency:.2f}s)"
        )

    print(f"Wrote {len(results)} results to {output_path}")


if __name__ == "__main__":
    main()

