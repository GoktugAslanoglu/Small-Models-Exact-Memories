from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image


DEFAULT_PROMPT = (
    "The image contains a verbatim rendering of earlier agent history. "
    "Answer the memory question using only that history. Return only the requested "
    "value, without explanation.\n\nQuestion: {question}"
)


def load_images(paths: list[Path]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())
    return images


def configure_image_budget(processor: Any, pixel_budget: int) -> None:
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return
    if hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = pixel_budget
    if hasattr(image_processor, "min_pixels"):
        image_processor.min_pixels = min(int(image_processor.min_pixels), pixel_budget)


def prepare_inputs(processor: Any, images: list[Image.Image], question: str) -> Any:
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": DEFAULT_PROMPT.format(question=question)})
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


def count_text_tokens(processor: Any, text: str) -> int:
    encoded = processor.tokenizer(text, add_special_tokens=False)
    return len(encoded["input_ids"])


def count_vision_tokens(processor: Any, inputs: Any) -> tuple[int | None, int | None]:
    token_count: int | None = None
    tokenizer = processor.tokenizer
    image_token_ids: list[int] = []
    for token in ("<|image_pad|>", "<|vision_pad|>", "<|image|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0 and token_id != tokenizer.unk_token_id:
            image_token_ids.append(token_id)
    input_ids = inputs.get("input_ids")
    if input_ids is not None and image_token_ids:
        token_count = sum(int((input_ids == token_id).sum().item()) for token_id in set(image_token_ids))
        if token_count == 0:
            token_count = None

    grid_count: int | None = None
    grid = inputs.get("image_grid_thw")
    if grid is not None:
        merge_size = int(getattr(processor.image_processor, "merge_size", 1))
        grid_count = int(sum(int(row.prod().item()) for row in grid) / (merge_size**2))
    return token_count, grid_count

