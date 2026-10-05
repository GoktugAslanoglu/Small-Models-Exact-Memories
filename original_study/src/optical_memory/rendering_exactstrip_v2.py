from __future__ import annotations

from typing import Any

from PIL import Image, ImageDraw

from .exactspan_v2 import format_protection_strip_v2, select_protected_spans_v2
from .rendering_pages import (
    _blank_pages,
    _draw_page_slice,
    _layout_for_exact_page_count,
)


def render_exactstrip_v2_pages(
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
    """Render the v2 development representation from history alone.

    V2 removes bracketed type labels, drops syntax matches contained inside a
    longer match, and permits a larger protected-strip font. It otherwise uses
    the same full-history and fixed-page-budget layout as v1.
    """

    if not 0.10 <= strip_fraction <= 0.45:
        raise ValueError("strip_fraction must be between 0.10 and 0.45")
    if page_count <= 0:
        raise ValueError("page_count must be positive")

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
    spans, detected_count = select_protected_spans_v2(history)
    strip_text = format_protection_strip_v2(spans)
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
        draw.line(
            (margin, divider_y, width - margin, divider_y),
            fill=(30, 74, 120),
            width=2,
        )
        draw.rectangle(
            (margin, strip_top, width - margin, height - margin),
            fill=(239, 247, 255),
        )
        _draw_page_slice(
            page,
            strip_layout,
            index,
            (margin, strip_top),
            (8, 43, 77),
        )

    return pages, {
        "dense_font_size": dense_layout.font_size,
        "dense_line_count": len(dense_layout.lines),
        "dense_max_chars": dense_layout.max_chars,
        "strip_font_size": strip_layout.font_size,
        "strip_line_count": len(strip_layout.lines),
        "detected_span_count": detected_count,
        "protected_span_count": len(spans),
        "strip_format": "raw_span_no_type_prefix",
        "contained_span_policy": "drop_fully_contained",
    }
