from __future__ import annotations

from typing import Any


MODEL_ID = "zai-org/GLM-4.6V-Flash"


def model_device(model: Any) -> Any:
    for parameter in model.parameters():
        if str(parameter.device) != "meta":
            return parameter.device
    return model.device


def sync_cuda(torch: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_runtime(model_id: str = MODEL_ID) -> tuple[Any, Any, Any, str]:
    try:
        import torch
        from transformers import AutoProcessor, Glm4vForConditionalGeneration
    except ImportError as exc:
        raise SystemExit(
            "Install the GLM Colab dependencies from the supplied notebook."
        ) from exc
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU runtime is required.")

    major, _ = torch.cuda.get_device_capability()
    dtype = torch.bfloat16 if major >= 8 else torch.float16
    processor = AutoProcessor.from_pretrained(model_id)
    model = Glm4vForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return torch, processor, model, str(dtype).replace("torch.", "")


def sanitize_inputs(inputs: Any) -> Any:
    """Apply the GLM model-card input fix while preserving mm token types."""

    inputs.pop("token_type_ids", None)
    return inputs


def _scalar(value: Any) -> int:
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def _input_token_count(inputs: Any) -> int:
    input_ids = inputs.get("input_ids")
    if input_ids is None:
        raise RuntimeError("GLM processor did not return input_ids")
    return int(input_ids.shape[-1])


def measure_budget(
    processor: Any,
    inputs: Any,
    *,
    memory_text_tokens: int,
) -> dict[str, Any]:
    """Measure realized GLM input and visual tokens from processor outputs.

    The primary visual count is the number of GLM image placeholders actually
    present in input_ids. image_grid_thw provides an independent count and must
    agree after spatial merging. This deliberately avoids carrying Qwen token
    estimates into the GLM experiment.
    """

    input_ids = inputs.get("input_ids")
    if input_ids is None:
        raise RuntimeError("GLM processor did not return input_ids")

    image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is None:
        image_token_id = getattr(processor.tokenizer, "image_token_id", None)
    if image_token_id is None:
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image|>")
    if not isinstance(image_token_id, int) or image_token_id < 0:
        raise RuntimeError("Could not resolve GLM's image token ID")

    placeholder_tokens = _scalar((input_ids == image_token_id).sum())
    grid = inputs.get("image_grid_thw")
    per_image_tokens: list[int] = []
    grid_tokens = 0
    grid_rows: list[list[int]] = []
    if grid is not None:
        merge_size = int(getattr(processor.image_processor, "merge_size", 1))
        for row in grid:
            values = [_scalar(item) for item in row]
            grid_rows.append(values)
            tokens = 1
            for value in values:
                tokens *= value
            tokens //= merge_size**2
            per_image_tokens.append(tokens)
            grid_tokens += tokens

    if grid_tokens and placeholder_tokens != grid_tokens:
        raise RuntimeError(
            "GLM visual-token accounting mismatch: "
            f"input_ids={placeholder_tokens}, image_grid_thw={grid_tokens}"
        )
    vision_tokens = placeholder_tokens or grid_tokens or None
    realized_input_tokens = _input_token_count(inputs)
    nonvision_input_tokens = (
        realized_input_tokens - vision_tokens
        if vision_tokens is not None
        else realized_input_tokens
    )
    compression = (
        round(memory_text_tokens / vision_tokens, 6)
        if vision_tokens is not None and vision_tokens > 0
        else None
    )
    return {
        "realized_input_tokens": realized_input_tokens,
        "nonvision_input_tokens": nonvision_input_tokens,
        "vision_tokens": vision_tokens,
        "grid_vision_tokens": grid_tokens or None,
        "vision_tokens_per_image": per_image_tokens,
        "image_grid_thw": grid_rows,
        "image_token_id": image_token_id,
        "vision_token_source": "glm_input_ids_image_placeholders_crosschecked_with_grid",
        "model_specific_compression": compression,
        "realized_compression": compression,
        "processor_class": type(processor).__name__,
        "image_processor_class": type(processor.image_processor).__name__,
    }


def text_budget(
    processor: Any,
    inputs: Any,
    *,
    memory_text_tokens: int,
) -> dict[str, Any]:
    input_tokens = _input_token_count(inputs)
    return {
        "realized_input_tokens": input_tokens,
        "nonvision_input_tokens": input_tokens,
        "vision_tokens": None,
        "grid_vision_tokens": None,
        "vision_tokens_per_image": [],
        "image_grid_thw": [],
        "image_token_id": getattr(processor, "image_token_id", None),
        "vision_token_source": None,
        "model_specific_compression": None,
        "realized_compression": None,
        "processor_class": type(processor).__name__,
        "image_processor_class": type(processor.image_processor).__name__,
        "memory_text_tokens": memory_text_tokens,
    }
