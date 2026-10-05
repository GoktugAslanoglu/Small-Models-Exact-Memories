from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .common import load_json, project_path, read_jsonl, write_jsonl
from .exactspan import detect_exact_spans, format_protection_strip
from .rendering import _font_geometry, find_font_path, load_font, wrap_text


@dataclass(frozen=True)
class PagedLayout:
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont
    font_size: int
    lines: tuple[str, ...]
    line_height: int
    max_chars: int
    lines_per_page: int
    page_count: int


def _layout(
    text: str,
    width: int,
    height: int,
    font_path: str | None,
    min_font_size: int,
    max_font_size: int,
    page_limit: int | None,
) -> PagedLayout:
    smallest: PagedLayout | None = None
    for size in range(max_font_size, min_font_size - 1, -1):
        font = load_font(font_path, size)
        glyph_width, line_height = _font_geometry(font)
        max_chars = max(1, width // glyph_width)
        lines = wrap_text(text, max_chars)
        lines_per_page = max(1, height // line_height)
        page_count = max(1, math.ceil(len(lines) / lines_per_page))
        candidate = PagedLayout(
            font,
            size,
            lines,
            line_height,
            max_chars,
            lines_per_page,
            page_count,
        )
        smallest = candidate
        if page_limit is None or page_count <= page_limit:
            return candidate
    assert smallest is not None
    return smallest


def _layout_for_exact_page_count(
    text: str,
    width: int,
    height: int,
    page_count: int,
    font_path: str | None,
    min_font_size: int,
    max_font_size: int,
) -> PagedLayout:
    candidate = _layout(
        text,
        width,
        height,
        font_path,
        min_font_size,
        max_font_size,
        page_count,
    )
    if candidate.page_count > page_count:
        raise ValueError(
            f"Content requires {candidate.page_count} pages but fixed-budget comparison "
            f"permits {page_count}. Lower min_font_size or increase preferred_max_pages."
        )
    return PagedLayout(
        candidate.font,
        candidate.font_size,
        candidate.lines,
        candidate.line_height,
        candidate.max_chars,
        candidate.lines_per_page,
        page_count,
    )


def _blank_pages(count: int, width: int, height: int) -> list[Image.Image]:
    return [Image.new("RGB", (width, height), "white") for _ in range(count)]


def _draw_page_slice(
    image: Image.Image,
    layout: PagedLayout,
    page_index: int,
    origin: tuple[int, int],
    fill: tuple[int, int, int],
) -> None:
    start = page_index * layout.lines_per_page
    stop = start + layout.lines_per_page
    lines = layout.lines[start:stop]
    draw = ImageDraw.Draw(image)
    x, y = origin
    for line in lines:
        draw.text((x, y), line, font=layout.font, fill=fill)
        y += layout.line_height


def render_uniform_pages(
    history: str,
    width: int,
    height: int,
    margin: int,
    preferred_max_pages: int,
    font_path: str | None,
    min_font_size: int,
    max_font_size: int,
) -> tuple[list[Image.Image], PagedLayout, dict[str, Any]]:
    layout = _layout(
        history,
        width - 2 * margin,
        height - 2 * margin,
        font_path,
        min_font_size,
        max_font_size,
        preferred_max_pages,
    )
    pages = _blank_pages(layout.page_count, width, height)
    for index, page in enumerate(pages):
        _draw_page_slice(page, layout, index, (margin, margin), (15, 20, 28))
    return pages, layout, {
        "dense_font_size": layout.font_size,
        "dense_line_count": len(layout.lines),
        "dense_max_chars": layout.max_chars,
        "protected_span_count": 0,
    }


def render_exactstrip_pages(
    history: str,
    page_count: int,
    width: int,
    height: int,
    margin: int,
    strip_fraction: float,
    font_path: str | None,
    min_font_size: int,
    max_font_size: int,
    strip_min_font_size: int,
    strip_max_font_size: int,
) -> tuple[list[Image.Image], dict[str, Any]]:
    if not 0.10 <= strip_fraction <= 0.45:
        raise ValueError("strip_fraction must be between 0.10 and 0.45")
    strip_top = round(height * (1.0 - strip_fraction))
    divider_y = strip_top - max(2, margin // 4)
    dense_height = divider_y - 2 * margin
    strip_height = height - strip_top - margin
    content_width = width - 2 * margin

    dense_layout = _layout_for_exact_page_count(
        history,
        content_width,
        dense_height,
        page_count,
        font_path,
        min_font_size,
        max_font_size,
    )
    spans = detect_exact_spans(history)
    strip_text = format_protection_strip(spans)
    strip_layout = _layout_for_exact_page_count(
        strip_text,
        content_width,
        strip_height,
        page_count,
        font_path,
        strip_min_font_size,
        strip_max_font_size,
    )

    pages = _blank_pages(page_count, width, height)
    for index, page in enumerate(pages):
        draw = ImageDraw.Draw(page)
        _draw_page_slice(page, dense_layout, index, (margin, margin), (15, 20, 28))
        draw.line((margin, divider_y, width - margin, divider_y), fill=(30, 74, 120), width=2)
        draw.rectangle((margin, strip_top, width - margin, height - margin), fill=(239, 247, 255))
        _draw_page_slice(page, strip_layout, index, (margin, strip_top), (8, 43, 77))

    return pages, {
        "dense_font_size": dense_layout.font_size,
        "dense_line_count": len(dense_layout.lines),
        "dense_max_chars": dense_layout.max_chars,
        "strip_font_size": strip_layout.font_size,
        "strip_line_count": len(strip_layout.lines),
        "protected_span_count": len(spans),
    }


def _round_side(value: float, maximum: int) -> int:
    return max(224, min(maximum, int(round(value / 28) * 28)))


def _side_per_page(
    history_chars: int,
    compression: int,
    page_count: int,
    canvas_side: int,
    chars_per_text_token: float,
    pixels_per_vision_token: int,
) -> int:
    estimated_text_tokens = history_chars / chars_per_text_token
    total_target_pixels = estimated_text_tokens / compression * pixels_per_vision_token
    return _round_side(math.sqrt(total_target_pixels / page_count), canvas_side)


def _save_pages(
    pages: list[Image.Image],
    output_dir: Path,
    relative_dir: Path,
    episode_id: str,
    side: int,
) -> list[str]:
    paths: list[str] = []
    for index, page in enumerate(pages):
        relative = relative_dir / f"{episode_id}-p{index:02d}.png"
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        resized = page if page.size == (side, side) else page.resize((side, side), Image.Resampling.LANCZOS)
        resized.save(destination, format="PNG", optimize=True)
        paths.append(relative.as_posix())
    return paths


def render_dataset(config: dict[str, Any], dataset_path: Path) -> tuple[Path, list[dict[str, Any]]]:
    render_config = config["render"]
    output_dir = project_path(render_config["output_dir"])
    manifest_path = output_dir / "render_manifest.jsonl"
    width = int(render_config["canvas_width"])
    height = int(render_config["canvas_height"])
    if width != height:
        raise ValueError("The pilot currently requires square pages.")
    font_path = find_font_path(render_config.get("font_path"))
    manifest: list[dict[str, Any]] = []

    for episode in read_jsonl(dataset_path):
        history = str(episode["history"])
        uniform_pages, uniform_layout, uniform_meta = render_uniform_pages(
            history,
            width,
            height,
            int(render_config["margin"]),
            int(render_config["preferred_max_pages"]),
            font_path,
            int(render_config["min_font_size"]),
            int(render_config["max_font_size"]),
        )
        exactstrip_pages, exactstrip_meta = render_exactstrip_pages(
            history,
            uniform_layout.page_count,
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
        conditions: list[tuple[str, int, list[Image.Image], dict[str, Any]]] = []
        conditions.extend(
            ("uniform", int(compression), uniform_pages, uniform_meta)
            for compression in render_config["compression_targets"]
        )
        conditions.extend(
            ("exactstrip", int(compression), exactstrip_pages, exactstrip_meta)
            for compression in render_config["exactstrip_targets"]
        )

        for kind, compression, pages, metadata in conditions:
            side = _side_per_page(
                len(history),
                compression,
                len(pages),
                width,
                float(render_config["chars_per_text_token_estimate"]),
                int(render_config["pixels_per_vision_token"]),
            )
            relative_dir = Path("images") / kind / f"c{compression}"
            image_paths = _save_pages(
                pages,
                output_dir,
                relative_dir,
                str(episode["episode_id"]),
                side,
            )
            manifest.append(
                {
                    "schema_version": 1,
                    "episode_id": episode["episode_id"],
                    "memory_type": episode["memory_type"],
                    "condition": f"{kind}_c{compression}",
                    "render_kind": kind,
                    "target_compression": compression,
                    "image_paths": image_paths,
                    "page_count": len(pages),
                    "image_width": side,
                    "image_height": side,
                    "total_pixel_budget": len(pages) * side * side,
                    "native_canvas_width": width,
                    "native_canvas_height": height,
                    "font_path": font_path,
                    **metadata,
                }
            )

    write_jsonl(manifest_path, manifest)
    return manifest_path, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render fixed-budget multi-page optical memory.")
    parser.add_argument("--config", default="configs/pilot_pages.json")
    parser.add_argument("--dataset", help="Override the dataset path in the config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(project_path(args.config))
    dataset_path = project_path(args.dataset or config["output"])
    manifest_path, rows = render_dataset(config, dataset_path)
    page_counts = sorted({row["page_count"] for row in rows})
    print(f"Rendered {len(rows)} episode-condition pairs")
    print(f"Page counts: {page_counts}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

