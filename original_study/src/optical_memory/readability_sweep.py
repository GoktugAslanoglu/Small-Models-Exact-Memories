from __future__ import annotations

import argparse
import hashlib
import math
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from .common import project_path, read_jsonl, write_jsonl


DEFAULT_SOURCE_CONDITION = "full_history_12px"
DEFAULT_TARGET_FONTS = (12, 10, 8, 6)
PROCESSOR_GRID_MULTIPLE = 28
_FONT_SUFFIX = re.compile(r"_\d+(?:\.\d+)?px$")
_UNSAFE_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")


def parse_target_fonts(value: str) -> tuple[int, ...]:
    """Parse an ordered comma-separated list of positive integer font sizes."""
    fonts: list[int] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        try:
            font = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"target font sizes must be integers; received {item!r}"
            ) from exc
        if font <= 0:
            raise argparse.ArgumentTypeError("target font sizes must be positive")
        if font in fonts:
            raise argparse.ArgumentTypeError(
                f"target font sizes must be unique; received {font} more than once"
            )
        fonts.append(font)
    if not fonts:
        raise argparse.ArgumentTypeError("at least one target font size is required")
    return tuple(fonts)


def _condition_name(
    source_condition: str,
    target_font: int,
    width: int,
    height: int,
) -> str:
    replacement = f"_{target_font}px"
    if _FONT_SUFFIX.search(source_condition):
        base = _FONT_SUFFIX.sub(replacement, source_condition)
    else:
        base = f"{source_condition}{replacement}"
    return f"{base}_{width}x{height}"


def _safe_episode_component(episode_id: str) -> str:
    cleaned = _UNSAFE_COMPONENT.sub("-", episode_id).strip("._-") or "episode"
    digest = hashlib.sha256(episode_id.encode("utf-8")).hexdigest()[:10]
    return f"{cleaned[:80]}-{digest}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ceil_to_grid(value: float) -> int:
    return max(
        PROCESSOR_GRID_MULTIPLE,
        math.ceil(value / PROCESSOR_GRID_MULTIPLE) * PROCESSOR_GRID_MULTIPLE,
    )


def _positive_number(row: dict[str, Any], field: str, episode_id: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{episode_id}: source row requires a numeric {field!r} value"
        ) from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{episode_id}: {field!r} must be finite and positive")
    return value


def _inspect_source_row(
    row: dict[str, Any],
    source_manifest: Path,
) -> tuple[str, float, list[Path], int, int]:
    if "episode_id" not in row:
        raise ValueError("source manifest row is missing 'episode_id'")
    episode_id = str(row["episode_id"])
    source_font = _positive_number(row, "dense_font_size", episode_id)
    image_values = row.get("image_paths")
    if not isinstance(image_values, list) or not image_values:
        raise ValueError(f"{episode_id}: 'image_paths' must be a non-empty list")

    try:
        page_count = int(row.get("page_count", len(image_values)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{episode_id}: 'page_count' must be an integer") from exc
    if page_count != len(image_values):
        raise ValueError(
            f"{episode_id}: page_count={page_count} but image_paths contains "
            f"{len(image_values)} entries"
        )

    paths: list[Path] = []
    source_size: tuple[int, int] | None = None
    for page_index, value in enumerate(image_values):
        path = (source_manifest.parent / str(value)).resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"{episode_id} page {page_index}: source image does not exist: {path}"
            )
        try:
            with Image.open(path) as image:
                image.load()
                size = image.size
        except Exception as exc:
            raise ValueError(
                f"{episode_id} page {page_index}: cannot read source image {path}: {exc}"
            ) from exc
        if source_size is None:
            source_size = size
        elif size != source_size:
            raise ValueError(
                f"{episode_id}: all pages must share one size; found "
                f"{source_size} and {size}"
            )
        paths.append(path)

    assert source_size is not None
    width, height = source_size
    for field, actual in (("image_width", width), ("image_height", height)):
        if row.get(field) is None:
            continue
        try:
            declared = int(row[field])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{episode_id}: {field!r} must be an integer") from exc
        if declared != actual:
            raise ValueError(
                f"{episode_id}: manifest {field}={declared}, but source image is "
                f"{width}x{height}"
            )
    return episode_id, source_font, paths, width, height


def build_readability_sweep(
    manifest: str | Path,
    output: str | Path,
    *,
    source_condition: str = DEFAULT_SOURCE_CONDITION,
    target_fonts: Sequence[int] = DEFAULT_TARGET_FONTS,
) -> list[dict[str, Any]]:
    """Resize readable-control pages without reflowing text or changing pages."""
    source_manifest = Path(manifest).resolve()
    output_manifest = Path(output).resolve()
    if not source_manifest.is_file():
        raise FileNotFoundError(f"source manifest does not exist: {source_manifest}")
    if source_manifest == output_manifest:
        raise ValueError("output manifest must differ from the source manifest")
    if not source_condition.strip():
        raise ValueError("source_condition must not be empty")

    targets = tuple(int(value) for value in target_fonts)
    if not targets:
        raise ValueError("at least one target font size is required")
    if any(value <= 0 for value in targets):
        raise ValueError("target font sizes must be positive")
    if len(set(targets)) != len(targets):
        raise ValueError("target font sizes must be unique")

    selected = [
        row
        for row in read_jsonl(source_manifest)
        if str(row.get("condition")) == source_condition
    ]
    if not selected:
        raise ValueError(
            f"source manifest contains no rows for condition {source_condition!r}"
        )

    inspected: list[tuple[dict[str, Any], str, float, list[Path], int, int]] = []
    episode_counts: Counter[str] = Counter()
    for row in selected:
        episode_id, source_font, paths, width, height = _inspect_source_row(
            row, source_manifest
        )
        episode_counts[episode_id] += 1
        inspected.append((row, episode_id, source_font, paths, width, height))
    duplicates = sorted(key for key, count in episode_counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"source condition contains duplicate episode rows: {duplicates}")

    source_fonts = {item[2] for item in inspected}
    source_sizes = {(item[4], item[5]) for item in inspected}
    if len(source_fonts) != 1 or len(source_sizes) != 1:
        raise ValueError(
            "all selected source rows must share one dense font size and image size"
        )
    source_font = next(iter(source_fonts))
    source_width, source_height = next(iter(source_sizes))
    if any(target > source_font for target in targets):
        raise ValueError(
            f"target fonts cannot exceed the source font ({source_font:g}px)"
        )

    rows: list[dict[str, Any]] = []
    for target_font in targets:
        is_anchor = math.isclose(float(target_font), source_font)
        requested_scale = target_font / source_font
        if is_anchor:
            target_width, target_height = source_width, source_height
        else:
            target_width = _ceil_to_grid(source_width * requested_scale)
            target_height = _ceil_to_grid(source_height * requested_scale)
        condition = _condition_name(
            source_condition, target_font, target_width, target_height
        )

        for row, episode_id, row_font, paths, row_width, row_height in inspected:
            actual_scale_x = target_width / row_width
            actual_scale_y = target_height / row_height
            actual_font_x = row_font * actual_scale_x
            actual_font_y = row_font * actual_scale_y
            actual_font = row_font * math.sqrt(actual_scale_x * actual_scale_y)
            page_count = len(paths)
            relative_paths: list[str] = []
            source_hashes: list[str] = []
            output_hashes: list[str] = []
            episode_component = _safe_episode_component(episode_id)

            for page_index, source_path in enumerate(paths):
                relative = (
                    Path("images")
                    / condition
                    / episode_component
                    / f"page-{page_index:02d}.png"
                )
                destination = output_manifest.parent / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                source_hash = _sha256(source_path)
                if is_anchor:
                    shutil.copyfile(source_path, destination)
                else:
                    with Image.open(source_path) as source_image:
                        resized = source_image.convert("RGB").resize(
                            (target_width, target_height), Image.Resampling.LANCZOS
                        )
                        resized.save(destination, format="PNG", optimize=True)
                relative_paths.append(relative.as_posix())
                source_hashes.append(source_hash)
                output_hashes.append(_sha256(destination))

            source_pixels = page_count * row_width * row_height
            requested_pixels = source_pixels * requested_scale**2
            actual_pixels = page_count * target_width * target_height
            rows.append(
                {
                    **row,
                    "condition": condition,
                    "image_paths": relative_paths,
                    "page_count": page_count,
                    "image_width": target_width,
                    "image_height": target_height,
                    "total_pixel_budget": actual_pixels,
                    "dense_font_size": round(actual_font, 6),
                    "source_condition": source_condition,
                    "source_dense_font_size_px": row_font,
                    "requested_effective_font_size_px": target_font,
                    "actual_effective_font_size_px": round(actual_font, 6),
                    "actual_effective_font_size_x_px": round(actual_font_x, 6),
                    "actual_effective_font_size_y_px": round(actual_font_y, 6),
                    "requested_linear_scale": round(requested_scale, 9),
                    "actual_linear_scale_x": round(actual_scale_x, 9),
                    "actual_linear_scale_y": round(actual_scale_y, 9),
                    "source_image_width_px": row_width,
                    "source_image_height_px": row_height,
                    "requested_image_width_px": round(row_width * requested_scale, 6),
                    "requested_image_height_px": round(row_height * requested_scale, 6),
                    "actual_image_width_px": target_width,
                    "actual_image_height_px": target_height,
                    "source_total_pixel_budget": source_pixels,
                    "requested_total_pixel_budget": round(requested_pixels, 6),
                    "actual_total_pixel_budget": actual_pixels,
                    "processor_grid_multiple_px": PROCESSOR_GRID_MULTIPLE,
                    "dimension_rounding_policy": (
                        "source_byte_copy" if is_anchor else "ceil_to_processor_grid"
                    ),
                    "resize_filter": "NONE" if is_anchor else "LANCZOS",
                    "layout_preserved": True,
                    "is_source_anchor": is_anchor,
                    "source_page_sha256": source_hashes,
                    "output_page_sha256": output_hashes,
                    "sweep_fingerprint": (
                        f"readability-v1-f{target_font}-{target_width}x{target_height}-"
                        f"{'copy' if is_anchor else 'ceil28-lanczos'}"
                    ),
                }
            )

    write_jsonl(output_manifest, rows)
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a grid-aligned effective-font sweep from readable full-history pages."
        )
    )
    parser.add_argument("--manifest", required=True, help="Readable-control manifest.")
    parser.add_argument("--output", required=True, help="Output sweep manifest.")
    parser.add_argument(
        "--source-condition",
        default=DEFAULT_SOURCE_CONDITION,
        help=f"Condition to resize (default: {DEFAULT_SOURCE_CONDITION}).",
    )
    parser.add_argument(
        "--target-fonts",
        type=parse_target_fonts,
        default=DEFAULT_TARGET_FONTS,
        help="Comma-separated effective font sizes (default: 12,10,8,6).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = project_path(args.output)
    rows = build_readability_sweep(
        project_path(args.manifest),
        output,
        source_condition=args.source_condition,
        target_fonts=args.target_fonts,
    )
    counts = Counter(str(row["condition"]) for row in rows)
    print(f"Wrote {len(rows)} sweep rows to {output}")
    for condition, count in counts.items():
        example = next(row for row in rows if row["condition"] == condition)
        print(
            f"{condition}: rows={count}, pages={example['page_count']}, "
            f"size={example['image_width']}x{example['image_height']}, "
            f"actual_font={example['actual_effective_font_size_px']:.3f}px"
        )


if __name__ == "__main__":
    main()
