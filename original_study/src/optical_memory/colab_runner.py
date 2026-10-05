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
from .run_text_baseline import _prepare as prepare_text_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Colab-optimized Qwen3.5 experiment runner.")
    parser.add_argument("--mode", choices=("optical", "raw"), required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", help="Required for optical mode.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--conditions", help="Comma-separated optical conditions.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _model_device(model: Any) -> Any:
    for parameter in model.parameters():
        if str(parameter.device) != "meta":
            return parameter.device
    return model.device


def _sync(torch: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _cuda_dtype(torch: Any) -> Any:
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def _load_runtime(model_id: str) -> tuple[Any, Any, Any, str]:
    try:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor
    except ImportError as exc:
        raise SystemExit("Install the Colab dependencies from the supplied notebook.") from exc
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU runtime is required for the Colab experiment.")
    dtype = _cuda_dtype(torch)
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return torch, processor, model, str(dtype).replace("torch.", "")


def main() -> None:
    args = parse_args()
    if args.mode == "optical" and not args.manifest:
        raise SystemExit("--manifest is required in optical mode")

    import torch

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dataset_path = project_path(args.dataset)
    output_path = project_path(args.output)
    episodes = {str(row["episode_id"]): row for row in read_jsonl(dataset_path)}

    if args.mode == "optical":
        manifest_path = project_path(args.manifest)
        selected_conditions = set(args.conditions.split(",")) if args.conditions else None
        work_rows = [
            row
            for row in read_jsonl(manifest_path)
            if selected_conditions is None or row["condition"] in selected_conditions
        ]
    else:
        manifest_path = None
        work_rows = [
            {
                "episode_id": episode_id,
                "condition": "raw_text",
                "render_kind": "raw_text",
                "target_compression": 0,
                "image_width": None,
                "image_height": None,
            }
            for episode_id in episodes
        ]
    if args.limit is not None:
        work_rows = work_rows[: args.limit]

    existing: list[dict[str, Any]] = []
    completed: set[tuple[str, str, str]] = set()
    if args.resume and output_path.exists():
        existing = list(read_jsonl(output_path))
        completed = {
            (str(row["episode_id"]), str(row["condition"]), str(row["model"]))
            for row in existing
        }

    torch, processor, model, dtype_name = _load_runtime(args.model)
    gpu_name = torch.cuda.get_device_name(0)
    results = list(existing)
    for ordinal, row in enumerate(work_rows, start=1):
        key = (str(row["episode_id"]), str(row["condition"]), args.model)
        if key in completed:
            continue
        episode = episodes[str(row["episode_id"])]
        text_tokens = count_text_tokens(processor, str(episode["history"]))
        if args.mode == "optical":
            assert manifest_path is not None
            paths = [(manifest_path.parent / path).resolve() for path in row["image_paths"]]
            images = load_images(paths)
            configure_image_budget(processor, max(image.width * image.height for image in images))
            inputs = prepare_inputs(processor, images, str(episode["question"]))
            vision_tokens, grid_tokens = count_vision_tokens(processor, inputs)
        else:
            inputs = prepare_text_inputs(
                processor,
                str(episode["history"]),
                str(episode["question"]),
            )
            vision_tokens, grid_tokens = None, None
        realized_tokens = vision_tokens or grid_tokens
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
        prompt_length = inputs["input_ids"].shape[-1]
        prediction = processor.decode(generated[0][prompt_length:], skip_special_tokens=True)
        results.append(
            {
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
                "grid_vision_tokens": grid_tokens,
                "realized_compression": (
                    round(text_tokens / realized_tokens, 6) if realized_tokens else None
                ),
                "image_width": row.get("image_width"),
                "image_height": row.get("image_height"),
                "latency_seconds": round(latency, 6),
                "generated_tokens": int(generated.shape[-1] - prompt_length),
                "torch_dtype": dtype_name,
                "gpu_name": gpu_name,
            }
        )
        write_jsonl(output_path, results)
        print(
            f"[{ordinal}/{len(work_rows)}] {row['episode_id']} {row['condition']} "
            f"-> {prediction[:80]!r} ({latency:.2f}s)"
        )

    print(f"Wrote {len(results)} results to {output_path}")


if __name__ == "__main__":
    main()

