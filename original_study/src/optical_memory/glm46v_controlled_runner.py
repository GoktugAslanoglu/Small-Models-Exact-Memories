from __future__ import annotations

import argparse
import gc
import random
import time
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
from .modeling import count_text_tokens, load_images, prepare_inputs
from .run_text_baseline import _prepare as prepare_text_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GLM-4.6V-Flash on the frozen controlled memory matrix."
    )
    parser.add_argument("--mode", choices=("optical", "raw"), required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", help="Required for optical mode.")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--output", required=True)
    parser.add_argument("--conditions", help="Comma-separated optical conditions.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model != MODEL_ID:
        raise SystemExit(f"This runner is frozen to {MODEL_ID}; received {args.model}")
    if args.mode == "optical" and not args.manifest:
        raise SystemExit("--manifest is required in optical mode")

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
    if args.resume and output_path.is_file():
        existing = list(read_jsonl(output_path))
        if any(str(row.get("model")) != args.model for row in existing):
            raise SystemExit(f"Resume output contains a different model: {output_path}")
        completed = {
            (str(row["episode_id"]), str(row["condition"]), str(row["model"]))
            for row in existing
        }

    random.seed(args.seed)
    torch, processor, model, dtype_name = load_runtime(args.model)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    gpu_name = torch.cuda.get_device_name(0)
    model_commit = getattr(model.config, "_commit_hash", None)
    results = list(existing)
    print(
        f"Loaded {args.model} on {gpu_name} ({dtype_name}); "
        f"{len(completed)}/{len(work_rows)} calls already complete"
    )
    print(
        "Processor-specific accounting:",
        type(processor).__name__,
        "/",
        type(processor.image_processor).__name__,
    )

    for ordinal, row in enumerate(work_rows, start=1):
        key = (str(row["episode_id"]), str(row["condition"]), args.model)
        if key in completed:
            continue
        episode = episodes[str(row["episode_id"])]
        memory_text = str(episode["history"])
        memory_text_tokens = count_text_tokens(processor, memory_text)
        images: list[Any] = []
        if args.mode == "optical":
            assert manifest_path is not None
            paths = [
                (manifest_path.parent / str(path)).resolve()
                for path in row["image_paths"]
            ]
            images = load_images(paths)
            inputs = prepare_inputs(processor, images, str(episode["question"]))
            inputs = sanitize_inputs(inputs)
            budget = measure_budget(
                processor,
                inputs,
                memory_text_tokens=memory_text_tokens,
            )
        else:
            inputs = prepare_text_inputs(
                processor,
                memory_text,
                str(episode["question"]),
            )
            inputs = sanitize_inputs(inputs)
            budget = text_budget(
                processor,
                inputs,
                memory_text_tokens=memory_text_tokens,
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
            "protocol": "controlled-balanced18-cross-family-v1",
            "episode_id": row["episode_id"],
            "memory_type": episode["memory_type"],
            "position_bucket": episode["position_bucket"],
            "condition": row["condition"],
            "render_kind": row["render_kind"],
            "target_compression": row["target_compression"],
            "model": args.model,
            "model_commit": model_commit,
            "question": episode["question"],
            "answer": episode["answer"],
            "scoring": episode["scoring"],
            "prediction": prediction,
            "text_tokens": memory_text_tokens,
            "memory_text_tokens": memory_text_tokens,
            **budget,
            "image_width": row.get("image_width"),
            "image_height": row.get("image_height"),
            "page_count": row.get("page_count"),
            "latency_seconds": round(latency, 6),
            "generated_tokens": int(generated.shape[-1] - prompt_length),
            "torch_dtype": dtype_name,
            "gpu_name": gpu_name,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
        }
        results.append(result)
        completed.add(key)
        write_jsonl(output_path, results)
        print(
            f"[{ordinal:03d}/{len(work_rows)}] {row['condition']:<34} "
            f"{row['episode_id']} -> {prediction[:72]!r} ({latency:.2f}s; "
            f"vision={budget['vision_tokens']}; "
            f"C={budget['model_specific_compression']})"
        )

        del inputs, generated
        for image in images:
            image.close()
        del images
        gc.collect()

    print(f"Complete: {len(results)} rows in {output_path}")


if __name__ == "__main__":
    main()
