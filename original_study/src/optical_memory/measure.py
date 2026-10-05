from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Measure realized Qwen text and vision token counts.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--output", help="Defaults to measured_manifest_<model>.jsonl beside manifest.")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from transformers import AutoProcessor
    except ImportError as exc:
        raise SystemExit(
            "transformers is required. Install the HF extras documented in README.md."
        ) from exc

    dataset_path = project_path(args.dataset)
    manifest_path = project_path(args.manifest)
    episodes = {row["episode_id"]: row for row in read_jsonl(dataset_path)}
    processor = AutoProcessor.from_pretrained(args.model)
    output_rows: list[dict[str, Any]] = []

    for index, row in enumerate(read_jsonl(manifest_path)):
        if args.limit is not None and index >= args.limit:
            break
        episode = episodes[row["episode_id"]]
        image_paths = [(manifest_path.parent / path).resolve() for path in row["image_paths"]]
        images = load_images(image_paths)
        pixel_budget = sum(image.width * image.height for image in images)
        configure_image_budget(processor, max(image.width * image.height for image in images))
        inputs = prepare_inputs(processor, images, episode["question"])
        text_tokens = count_text_tokens(processor, episode["history"])
        vision_tokens, grid_vision_tokens = count_vision_tokens(processor, inputs)
        realized_tokens = vision_tokens or grid_vision_tokens
        output_rows.append(
            {
                **row,
                "tokenizer_model": args.model,
                "text_tokens": text_tokens,
                "vision_tokens": vision_tokens,
                "grid_vision_tokens": grid_vision_tokens,
                "realized_compression": (
                    round(text_tokens / realized_tokens, 6) if realized_tokens else None
                ),
                "pixel_budget": pixel_budget,
            }
        )

    safe_model = args.model.replace("/", "_").replace("\\", "_")
    output = project_path(
        args.output or manifest_path.parent / f"measured_manifest_{safe_model}.jsonl"
    )
    write_jsonl(output, output_rows)
    print(f"Wrote {len(output_rows)} measured rows to {output}")


if __name__ == "__main__":
    main()

