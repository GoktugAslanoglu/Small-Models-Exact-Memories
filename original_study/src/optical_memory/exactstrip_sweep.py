from __future__ import annotations

import argparse
import hashlib
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from .common import project_path, read_jsonl, write_jsonl
from .rendering import find_font_path
from .rendering_pages import render_exactstrip_pages


DEFAULT_REFERENCE_CONDITIONS = (
    "full_history_8px_1372x1372",
    "full_history_6px_1036x1036",
)
DEFAULT_NATIVE_CANVAS_SIDE = 2044
DEFAULT_MARGIN = 28
DEFAULT_STRIP_FRACTION = 0.22
_UNSAFE_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")


def parse_conditions(value: str) -> tuple[str, ...]:
    conditions = tuple(item.strip() for item in value.split(",") if item.strip())
    if not conditions:
        raise argparse.ArgumentTypeError("at least one reference condition is required")
    if len(set(conditions)) != len(conditions):
        raise argparse.ArgumentTypeError("reference conditions must be unique")
    return conditions


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


def _positive_int(row: dict[str, Any], field: str, context: str) -> int:
    try:
        value = int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{context}: {field!r} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{context}: {field!r} must be positive")
    return value


def _inspect_reference_row(
    row: dict[str, Any],
    manifest_path: Path,
) -> tuple[str, int, int, int, list[Path]]:
    condition = str(row.get("condition", ""))
    episode_id = str(row.get("episode_id", ""))
    context = f"{episode_id or '<missing episode>'}/{condition or '<missing condition>'}"
    if not episode_id or not condition:
        raise ValueError("reference rows require episode_id and condition")

    width = _positive_int(row, "image_width", context)
    height = _positive_int(row, "image_height", context)
    page_count = _positive_int(row, "page_count", context)
    image_values = row.get("image_paths")
    if not isinstance(image_values, list) or len(image_values) != page_count:
        raise ValueError(
            f"{context}: image_paths must contain exactly {page_count} entries"
        )

    paths: list[Path] = []
    for page_index, value in enumerate(image_values):
        path = (manifest_path.parent / str(value)).resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"{context} page {page_index}: reference image does not exist: {path}"
            )
        with Image.open(path) as image:
            image.load()
            if image.size != (width, height):
                raise ValueError(
                    f"{context} page {page_index}: manifest declares {width}x{height}, "
                    f"but image is {image.width}x{image.height}"
                )
        paths.append(path)
    return episode_id, page_count, width, height, paths


def _requested_font(row: dict[str, Any], condition: str) -> float:
    value = row.get("requested_effective_font_size_px")
    if value is not None:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{condition}: requested_effective_font_size_px must be numeric"
            ) from exc
        if math.isfinite(result) and result > 0:
            return result
    match = re.search(r"_(\d+(?:\.\d+)?)px(?:_|$)", condition)
    if not match:
        raise ValueError(f"cannot infer effective font size from {condition!r}")
    return float(match.group(1))


def _exactstrip_condition(reference_row: dict[str, Any]) -> str:
    width = int(reference_row["image_width"])
    height = int(reference_row["image_height"])
    font = _requested_font(reference_row, str(reference_row["condition"]))
    label = str(int(font)) if font.is_integer() else f"{font:g}"
    return f"exactstrip_{label}px_{width}x{height}"


def build_exactstrip_sweep(
    dataset: str | Path,
    reference_manifest: str | Path,
    output_manifest: str | Path,
    *,
    reference_conditions: Sequence[str] = DEFAULT_REFERENCE_CONDITIONS,
    font_path: str | Path | None = None,
    native_canvas_side: int = DEFAULT_NATIVE_CANVAS_SIDE,
    margin: int = DEFAULT_MARGIN,
    strip_fraction: float = DEFAULT_STRIP_FRACTION,
    min_font_size: int = 3,
    max_font_size: int = 12,
    strip_min_font_size: int = 4,
    strip_max_font_size: int = 12,
) -> list[dict[str, Any]]:
    """Render ExactStrip at image budgets copied from uniform reference rows.

    The renderer receives only each episode's history. Reference rows determine
    page counts and output dimensions; questions and ground-truth answers never
    influence span detection, layout, or resizing.
    """

    dataset_path = Path(dataset).resolve()
    reference_path = Path(reference_manifest).resolve()
    output_path = Path(output_manifest).resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"dataset does not exist: {dataset_path}")
    if not reference_path.is_file():
        raise FileNotFoundError(f"reference manifest does not exist: {reference_path}")
    if output_path == reference_path:
        raise ValueError("output manifest must differ from the reference manifest")
    if native_canvas_side <= 0 or margin < 0:
        raise ValueError("native canvas must be positive and margin nonnegative")

    selected_conditions = tuple(str(value) for value in reference_conditions)
    if not selected_conditions or len(set(selected_conditions)) != len(selected_conditions):
        raise ValueError("reference conditions must be nonempty and unique")

    episodes = list(read_jsonl(dataset_path))
    episode_ids = [str(row.get("episode_id", "")) for row in episodes]
    if any(not value for value in episode_ids):
        raise ValueError("every dataset row requires a nonempty episode_id")
    duplicates = sorted(key for key, count in Counter(episode_ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"dataset contains duplicate episode IDs: {duplicates}")
    episode_lookup = {str(row["episode_id"]): row for row in episodes}

    references: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    reference_rows = [
        row
        for row in read_jsonl(reference_path)
        if str(row.get("condition")) in selected_conditions
    ]
    for row in reference_rows:
        episode_id, page_count, width, height, paths = _inspect_reference_row(
            row, reference_path
        )
        condition = str(row["condition"])
        if condition in references[episode_id]:
            raise ValueError(f"duplicate reference row: {episode_id}/{condition}")
        references[episode_id][condition] = {
            **row,
            "page_count": page_count,
            "image_width": width,
            "image_height": height,
            "resolved_image_paths": paths,
        }

    expected = {
        (episode_id, condition)
        for episode_id in episode_ids
        for condition in selected_conditions
    }
    actual = {
        (episode_id, condition)
        for episode_id, conditions in references.items()
        for condition in conditions
    }
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"reference matrix does not match the dataset; missing={missing}, extra={extra}"
        )

    resolved_font = find_font_path(str(font_path) if font_path is not None else None)
    if resolved_font is None:
        raise FileNotFoundError("a deterministic TrueType font is required")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    condition_names: set[str] = set()
    for episode_id in episode_ids:
        episode = episode_lookup[episode_id]
        per_condition = references[episode_id]
        page_counts = {int(row["page_count"]) for row in per_condition.values()}
        if len(page_counts) != 1:
            raise ValueError(
                f"{episode_id}: all paired reference conditions must share one page count"
            )
        page_count = next(iter(page_counts))
        pages, metadata = render_exactstrip_pages(
            history=str(episode["history"]),
            page_count=page_count,
            width=native_canvas_side,
            height=native_canvas_side,
            margin=margin,
            strip_fraction=strip_fraction,
            font_path=resolved_font,
            min_font_size=min_font_size,
            max_font_size=max_font_size,
            strip_min_font_size=strip_min_font_size,
            strip_max_font_size=strip_max_font_size,
        )

        for reference_condition in selected_conditions:
            reference = per_condition[reference_condition]
            width = int(reference["image_width"])
            height = int(reference["image_height"])
            condition = _exactstrip_condition(reference)
            condition_names.add(condition)
            episode_component = _safe_episode_component(episode_id)
            relative_paths: list[str] = []
            output_hashes: list[str] = []
            for page_index, page in enumerate(pages):
                relative = (
                    Path("images")
                    / condition
                    / episode_component
                    / f"page-{page_index:02d}.png"
                )
                destination = output_path.parent / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                resized = page.convert("RGB").resize(
                    (width, height), Image.Resampling.LANCZOS
                )
                resized.save(destination, format="PNG", optimize=True)
                relative_paths.append(relative.as_posix())
                output_hashes.append(_sha256(destination))

            scale_x = width / native_canvas_side
            scale_y = height / native_canvas_side
            scale = math.sqrt(scale_x * scale_y)
            requested_font = _requested_font(reference, reference_condition)
            rows.append(
                {
                    "schema_version": 1,
                    "episode_id": episode_id,
                    "memory_type": episode["memory_type"],
                    "position_bucket": episode["position_bucket"],
                    "condition": condition,
                    "render_kind": "exactstrip",
                    "target_compression": int(reference.get("target_compression", 0)),
                    "image_paths": relative_paths,
                    "page_count": page_count,
                    "image_width": width,
                    "image_height": height,
                    "total_pixel_budget": page_count * width * height,
                    "native_canvas_width": native_canvas_side,
                    "native_canvas_height": native_canvas_side,
                    "font_path": resolved_font,
                    "dense_font_size": round(float(metadata["dense_font_size"]) * scale, 6),
                    "strip_font_size": round(float(metadata["strip_font_size"]) * scale, 6),
                    "native_dense_font_size_px": metadata["dense_font_size"],
                    "native_strip_font_size_px": metadata["strip_font_size"],
                    "dense_line_count": metadata["dense_line_count"],
                    "dense_max_chars": metadata["dense_max_chars"],
                    "strip_line_count": metadata["strip_line_count"],
                    "protected_span_count": metadata["protected_span_count"],
                    "strip_fraction": strip_fraction,
                    "requested_effective_font_size_px": requested_font,
                    "actual_effective_dense_font_size_px": round(
                        float(metadata["dense_font_size"]) * scale, 6
                    ),
                    "actual_effective_strip_font_size_px": round(
                        float(metadata["strip_font_size"]) * scale, 6
                    ),
                    "actual_linear_scale_x": round(scale_x, 9),
                    "actual_linear_scale_y": round(scale_y, 9),
                    "resize_filter": "LANCZOS",
                    "layout_preserved": True,
                    "matched_uniform_condition": reference_condition,
                    "matched_uniform_total_pixel_budget": int(
                        reference.get("total_pixel_budget", page_count * width * height)
                    ),
                    "span_detector": "syntax_only_v1",
                    "detector_input": "history_only",
                    "output_page_sha256": output_hashes,
                    "sweep_fingerprint": (
                        f"exactstrip-v1-f{requested_font:g}-{width}x{height}-"
                        f"p{page_count}-strip{strip_fraction:g}"
                    ),
                }
            )

    if len(condition_names) != len(selected_conditions):
        raise ValueError(
            "reference conditions collapsed to duplicate ExactStrip condition names"
        )
    write_jsonl(output_path, rows)
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render ExactStrip at the calibrated uniform 8px/6px budgets."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--reference-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--reference-conditions",
        type=parse_conditions,
        default=DEFAULT_REFERENCE_CONDITIONS,
    )
    parser.add_argument("--font-path")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows = build_exactstrip_sweep(
        project_path(args.dataset),
        project_path(args.reference_manifest),
        project_path(args.output),
        reference_conditions=args.reference_conditions,
        font_path=project_path(args.font_path) if args.font_path else None,
    )
    conditions = sorted({str(row["condition"]) for row in rows})
    print(f"Rendered {len(rows)} ExactStrip episode-condition pairs")
    print(f"Conditions: {conditions}")
    print(f"Manifest: {project_path(args.output)}")


if __name__ == "__main__":
    main()
