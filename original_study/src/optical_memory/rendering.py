from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

from .common import load_json, project_path, read_jsonl, write_jsonl
from .exactspan import detect_exact_spans, format_protection_strip


FONT_CANDIDATES = (
    r"C:\Windows\Fonts\consola.ttf",
    r"C:\Windows\Fonts\cour.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
    "/System/Library/Fonts/Menlo.ttc",
)


@dataclass(frozen=True)
class TextLayout:
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont
    font_size: int
    lines: tuple[str, ...]
    line_height: int
    max_chars: int


def find_font_path(explicit: str | None = None) -> str | None:
    if explicit:
        candidate = Path(explicit)
        if not candidate.exists():
            raise FileNotFoundError(f"Font does not exist: {candidate}")
        return str(candidate)
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return path
    return None


def load_font(font_path: str | None, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if font_path:
        return ImageFont.truetype(font_path, size=size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _font_geometry(font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> tuple[int, int]:
    left, top, right, bottom = font.getbbox("M")
    glyph_width = max(1, right - left)
    left, top, right, bottom = font.getbbox("Ag")
    line_height = max(1, bottom - top + 2)
    return glyph_width, line_height


def wrap_text(text: str, max_chars: int) -> tuple[str, ...]:
    wrapped: list[str] = []
    for source_line in text.expandtabs(4).splitlines():
        if not source_line:
            wrapped.append("")
            continue
        remaining = source_line
        while len(remaining) > max_chars:
            wrapped.append(remaining[:max_chars])
            remaining = remaining[max_chars:]
        wrapped.append(remaining)
    return tuple(wrapped)


def fit_text(
    text: str,
    width: int,
    height: int,
    font_path: str | None,
    min_font_size: int,
    max_font_size: int,
) -> TextLayout:
    for size in range(max_font_size, min_font_size - 1, -1):
        font = load_font(font_path, size)
        glyph_width, line_height = _font_geometry(font)
        max_chars = max(1, width // glyph_width)
        lines = wrap_text(text, max_chars)
        if len(lines) * line_height <= height:
            return TextLayout(font, size, lines, line_height, max_chars)
    smallest = load_font(font_path, min_font_size)
    glyph_width, line_height = _font_geometry(smallest)
    max_chars = max(1, width // glyph_width)
    required_lines = len(wrap_text(text, max_chars))
    available_lines = height // line_height
    raise ValueError(
        "Text does not fit the configured canvas: "
        f"requires {required_lines} lines but only {available_lines} fit at "
        f"font size {min_font_size}. Reduce target_chars, enlarge the canvas, "
        "or lower min_font_size."
    )


def _draw_layout(
    draw: ImageDraw.ImageDraw,
    layout: TextLayout,
    origin: tuple[int, int],
    fill: tuple[int, int, int] = (15, 20, 28),
) -> None:
    x, y = origin
    for line in layout.lines:
        draw.text((x, y), line, font=layout.font, fill=fill)
        y += layout.line_height


def render_uniform(
    history: str,
    width: int,
    height: int,
    margin: int,
    font_path: str | None,
    min_font_size: int,
    max_font_size: int,
) -> tuple[Image.Image, dict[str, Any]]:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    layout = fit_text(
        history,
        width - 2 * margin,
        height - 2 * margin,
        font_path,
        min_font_size,
        max_font_size,
    )
    _draw_layout(draw, layout, (margin, margin))
    return image, {
        "dense_font_size": layout.font_size,
        "dense_line_count": len(layout.lines),
        "dense_max_chars": layout.max_chars,
        "protected_span_count": 0,
    }


def render_exactstrip(
    history: str,
    width: int,
    height: int,
    margin: int,
    strip_fraction: float,
    font_path: str | None,
    min_font_size: int,
    max_font_size: int,
    strip_min_font_size: int,
    strip_max_font_size: int,
) -> tuple[Image.Image, dict[str, Any]]:
    if not 0.10 <= strip_fraction <= 0.45:
        raise ValueError("strip_fraction must be between 0.10 and 0.45")

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    strip_top = round(height * (1.0 - strip_fraction))
    divider_y = strip_top - max(2, margin // 4)
    dense_height = divider_y - 2 * margin
    dense_layout = fit_text(
        history,
        width - 2 * margin,
        dense_height,
        font_path,
        min_font_size,
        max_font_size,
    )
    _draw_layout(draw, dense_layout, (margin, margin))

    spans = detect_exact_spans(history)
    strip_text = format_protection_strip(spans)
    strip_layout = fit_text(
        strip_text,
        width - 2 * margin,
        height - strip_top - margin,
        font_path,
        strip_min_font_size,
        strip_max_font_size,
    )
    draw.line((margin, divider_y, width - margin, divider_y), fill=(30, 74, 120), width=2)
    draw.rectangle((margin, strip_top, width - margin, height - margin), fill=(239, 247, 255))
    _draw_layout(draw, strip_layout, (margin, strip_top), fill=(8, 43, 77))
    return image, {
        "dense_font_size": dense_layout.font_size,
        "dense_line_count": len(dense_layout.lines),
        "dense_max_chars": dense_layout.max_chars,
        "strip_font_size": strip_layout.font_size,
        "strip_line_count": len(strip_layout.lines),
        "protected_span_count": len(spans),
    }


def _round_to_multiple(value: float, multiple: int, minimum: int, maximum: int) -> int:
    rounded = int(round(value / multiple) * multiple)
    return max(minimum, min(maximum, rounded))


def target_side_length(
    history_chars: int,
    target_compression: float,
    canvas_side: int,
    chars_per_text_token: float,
    pixels_per_vision_token: int,
) -> int:
    estimated_text_tokens = history_chars / chars_per_text_token
    target_vision_tokens = max(64.0, estimated_text_tokens / target_compression)
    target_pixels = target_vision_tokens * pixels_per_vision_token
    return _round_to_multiple(math.sqrt(target_pixels), 28, 224, canvas_side)


def _save_variant(source: Image.Image, path: Path, side: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if source.size != (side, side):
        source = source.resize((side, side), Image.Resampling.LANCZOS)
    source.save(path, format="PNG", optimize=True)


def render_dataset(config: dict[str, Any], dataset_path: Path) -> tuple[Path, list[dict[str, Any]]]:
    render_config = config["render"]
    output_dir = project_path(render_config["output_dir"])
    images_dir = output_dir / "images"
    manifest_path = output_dir / "render_manifest.jsonl"
    width = int(render_config["canvas_width"])
    height = int(render_config["canvas_height"])
    if width != height:
        raise ValueError("The pilot currently requires a square canvas.")
    font_path = find_font_path(render_config.get("font_path"))
    manifest: list[dict[str, Any]] = []

    for episode in read_jsonl(dataset_path):
        history = episode["history"]
        uniform, uniform_meta = render_uniform(
            history,
            width,
            height,
            int(render_config["margin"]),
            font_path,
            int(render_config["min_font_size"]),
            int(render_config["max_font_size"]),
        )
        exactstrip, exactstrip_meta = render_exactstrip(
            history,
            width,
            height,
            int(render_config["margin"]),
            float(render_config["strip_fraction"]),
            font_path,
            int(render_config["min_font_size"]),
            int(render_config["max_font_size"]),
            int(render_config["strip_min_font_size"]),
            int(render_config["strip_max_font_size"]),
        )

        conditions: list[tuple[str, int, Image.Image, dict[str, Any]]] = []
        for compression in render_config["compression_targets"]:
            conditions.append(("uniform", int(compression), uniform, uniform_meta))
        for compression in render_config["exactstrip_targets"]:
            conditions.append(("exactstrip", int(compression), exactstrip, exactstrip_meta))

        for kind, compression, source, metadata in conditions:
            side = target_side_length(
                len(history),
                compression,
                width,
                float(render_config["chars_per_text_token_estimate"]),
                int(render_config["pixels_per_vision_token"]),
            )
            relative_path = Path("images") / kind / f"c{compression}" / f"{episode['episode_id']}.png"
            _save_variant(source, output_dir / relative_path, side)
            manifest.append(
                {
                    "schema_version": 1,
                    "episode_id": episode["episode_id"],
                    "memory_type": episode["memory_type"],
                    "condition": f"{kind}_c{compression}",
                    "render_kind": kind,
                    "target_compression": compression,
                    "image_paths": [relative_path.as_posix()],
                    "image_width": side,
                    "image_height": side,
                    "native_canvas_width": width,
                    "native_canvas_height": height,
                    "font_path": font_path,
                    **metadata,
                }
            )

    write_jsonl(manifest_path, manifest)
    return manifest_path, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render uniform and fixed-budget ExactStrip images.")
    parser.add_argument("--config", default="configs/pilot.json")
    parser.add_argument("--dataset", help="Override the dataset path in the config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(project_path(args.config))
    dataset_path = project_path(args.dataset or config["output"])
    manifest_path, rows = render_dataset(config, dataset_path)
    print(f"Rendered {len(rows)} episode-condition pairs")
    print(f"Manifest: {manifest_path}")
    font_paths = sorted({str(row["font_path"]) for row in rows})
    print(f"Font: {font_paths[0] if font_paths else 'Pillow default'}")


if __name__ == "__main__":
    main()

