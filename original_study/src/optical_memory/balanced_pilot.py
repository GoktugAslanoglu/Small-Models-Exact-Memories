from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from .common import project_path, read_jsonl, write_jsonl
from .generate import MEMORY_TYPES, POSITION_BUCKETS
from .rendering import find_font_path
from .rendering_pages import render_uniform_pages


SOURCE_CONDITION = "full_history_12px"


def select_balanced_episodes(
    episodes: Iterable[dict[str, Any]],
    *,
    memory_types: Sequence[str] = MEMORY_TYPES,
    position_buckets: Sequence[str] = POSITION_BUCKETS,
) -> list[dict[str, Any]]:
    """Select one deterministic episode for every memory-type/position pair."""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen_ids: set[str] = set()
    for episode in episodes:
        episode_id = str(episode.get("episode_id", ""))
        if not episode_id:
            raise ValueError("every episode requires a nonempty episode_id")
        if episode_id in seen_ids:
            raise ValueError(f"duplicate episode_id in source dataset: {episode_id}")
        seen_ids.add(episode_id)
        grouped[(str(episode.get("memory_type")), str(episode.get("position_bucket")))].append(
            episode
        )

    selected: list[dict[str, Any]] = []
    for memory_type in memory_types:
        for bucket in position_buckets:
            candidates = sorted(
                grouped.get((memory_type, bucket), []),
                key=lambda row: str(row["episode_id"]),
            )
            if not candidates:
                raise ValueError(
                    f"source dataset has no {memory_type!r}/{bucket!r} episode"
                )
            selected.append(candidates[0])

    expected = len(memory_types) * len(position_buckets)
    if len(selected) != expected or len({str(row["episode_id"]) for row in selected}) != expected:
        raise AssertionError("balanced selection did not produce a unique Cartesian grid")
    return selected


def render_readable_sources(
    episodes: Sequence[dict[str, Any]],
    output_dir: str | Path,
    *,
    font_path: str | Path,
    canvas_side: int = 2044,
    font_size: int = 12,
    margin: int = 28,
    preferred_max_pages: int = 12,
) -> tuple[Path, list[dict[str, Any]]]:
    """Render native-resolution, fixed-font full histories without localization."""

    if canvas_side <= 0 or font_size <= 0 or margin < 0 or preferred_max_pages <= 0:
        raise ValueError("render dimensions, font size, and page limit must be positive")
    explicit_font = find_font_path(str(font_path))
    if explicit_font is None:
        raise FileNotFoundError(f"font does not exist: {font_path}")

    root = Path(output_dir).resolve()
    manifest_path = root / "render_manifest.jsonl"
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        pages, layout, metadata = render_uniform_pages(
            history=str(episode["history"]),
            width=canvas_side,
            height=canvas_side,
            margin=margin,
            preferred_max_pages=preferred_max_pages,
            font_path=explicit_font,
            min_font_size=font_size,
            max_font_size=font_size,
        )
        relative_paths: list[str] = []
        for page_index, page in enumerate(pages):
            relative = (
                Path("images")
                / SOURCE_CONDITION
                / episode_id
                / f"page-{page_index:02d}.png"
            )
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            page.save(destination, format="PNG", optimize=True)
            relative_paths.append(relative.as_posix())

        rows.append(
            {
                "schema_version": 1,
                "episode_id": episode_id,
                "memory_type": episode["memory_type"],
                "position_bucket": episode["position_bucket"],
                "condition": SOURCE_CONDITION,
                "render_kind": "uniform",
                "target_compression": 0,
                "image_paths": relative_paths,
                "page_count": len(pages),
                "image_width": canvas_side,
                "image_height": canvas_side,
                "total_pixel_budget": len(pages) * canvas_side * canvas_side,
                "native_canvas_width": canvas_side,
                "native_canvas_height": canvas_side,
                "font_path": explicit_font,
                **metadata,
            }
        )

    write_jsonl(manifest_path, rows)
    return manifest_path, rows


def prepare_balanced_pilot(
    dataset: str | Path,
    output_dataset: str | Path,
    output_render_dir: str | Path,
    *,
    font_path: str | Path,
) -> tuple[Path, Path, list[dict[str, Any]]]:
    source = Path(dataset).resolve()
    selected = select_balanced_episodes(read_jsonl(source))
    dataset_output = Path(output_dataset).resolve()
    write_jsonl(dataset_output, selected)
    manifest, _ = render_readable_sources(
        selected,
        output_render_dir,
        font_path=font_path,
    )
    return dataset_output, manifest, selected


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the deterministic 18-episode readability pilot."
    )
    parser.add_argument("--dataset", required=True, help="Source 60-episode JSONL.")
    parser.add_argument("--output-dataset", required=True)
    parser.add_argument("--output-render-dir", required=True)
    parser.add_argument("--font-path", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dataset, manifest, selected = prepare_balanced_pilot(
        project_path(args.dataset),
        project_path(args.output_dataset),
        project_path(args.output_render_dir),
        font_path=project_path(args.font_path),
    )
    print(f"Wrote {len(selected)} balanced episodes to {dataset}")
    print(f"Memory types: {dict(sorted(Counter(str(row['memory_type']) for row in selected).items()))}")
    print(f"Position buckets: {dict(sorted(Counter(str(row['position_bucket']) for row in selected).items()))}")
    print(f"Readable source manifest: {manifest}")


if __name__ == "__main__":
    main()
