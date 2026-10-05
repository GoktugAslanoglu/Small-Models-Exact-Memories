from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from optical_memory.common import read_jsonl, write_jsonl
from optical_memory.readability_sweep import (
    build_readability_sweep,
    parse_target_fonts,
)


class ReadabilitySweepTests(unittest.TestCase):
    def _source_manifest(
        self,
        root: Path,
        *,
        size: tuple[int, int] = (336, 168),
        missing_second_page: bool = False,
    ) -> Path:
        source_root = root / "source"
        image_paths = [
            Path("images") / "full_history_12px" / "episode-1-p00.png",
            Path("images") / "full_history_12px" / "episode-1-p01.png",
        ]
        for index, relative in enumerate(image_paths):
            if missing_second_page and index == 1:
                continue
            path = source_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            color = (255, 0, 0) if index == 0 else (0, 0, 255)
            Image.new("RGB", size, color).save(path)

        rows = [
            {
                "schema_version": 1,
                "episode_id": "episode/1",
                "memory_type": "numeric_state",
                "condition": "full_history_12px",
                "render_kind": "uniform",
                "target_compression": 0,
                "image_paths": [path.as_posix() for path in image_paths],
                "page_count": 2,
                "image_width": size[0],
                "image_height": size[1],
                "total_pixel_budget": 2 * size[0] * size[1],
                "native_canvas_width": size[0],
                "native_canvas_height": size[1],
                "dense_font_size": 12,
                "dense_line_count": 42,
                "dense_max_chars": 80,
                "protected_span_count": 0,
            },
            {
                "schema_version": 1,
                "episode_id": "ignored",
                "condition": "oracle_window_24px",
                "dense_font_size": 24,
                "image_paths": ["missing-but-unselected.png"],
            },
        ]
        manifest = source_root / "render_manifest.jsonl"
        write_jsonl(manifest, rows)
        return manifest

    def test_builds_anchor_and_ordered_grid_aligned_variants(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source_manifest(root)
            output = root / "sweep" / "render_manifest.jsonl"

            rows = build_readability_sweep(source, output)

            self.assertEqual(len(rows), 4)
            self.assertEqual(
                [row["condition"] for row in rows],
                [
                    "full_history_12px_336x168",
                    "full_history_10px_280x140",
                    "full_history_8px_224x112",
                    "full_history_6px_168x84",
                ],
            )
            expected_sizes = [(336, 168), (280, 140), (224, 112), (168, 84)]
            for row, expected_size, requested_font in zip(
                rows, expected_sizes, (12, 10, 8, 6)
            ):
                self.assertEqual(row["page_count"], 2)
                self.assertEqual(
                    (row["image_width"], row["image_height"]), expected_size
                )
                self.assertEqual(row["requested_effective_font_size_px"], requested_font)
                self.assertAlmostEqual(row["actual_effective_font_size_px"], requested_font)
                self.assertEqual(
                    row["actual_total_pixel_budget"],
                    2 * expected_size[0] * expected_size[1],
                )
                self.assertEqual(row["total_pixel_budget"], row["actual_total_pixel_budget"])
                self.assertTrue(row["layout_preserved"])
                self.assertEqual(len(row["source_page_sha256"]), 2)
                self.assertEqual(len(row["output_page_sha256"]), 2)

                generated = [output.parent / value for value in row["image_paths"]]
                self.assertTrue(all(path.is_file() for path in generated))
                with Image.open(generated[0]) as first, Image.open(generated[1]) as second:
                    self.assertEqual(first.size, expected_size)
                    self.assertEqual(second.size, expected_size)
                    self.assertEqual(first.getpixel((0, 0)), (255, 0, 0))
                    self.assertEqual(second.getpixel((0, 0)), (0, 0, 255))

            self.assertTrue(rows[0]["is_source_anchor"])
            self.assertEqual(rows[0]["resize_filter"], "NONE")
            self.assertEqual(rows[0]["source_page_sha256"], rows[0]["output_page_sha256"])
            self.assertTrue(all(not row["is_source_anchor"] for row in rows[1:]))
            self.assertTrue(
                all(
                    row["dimension_rounding_policy"] == "ceil_to_processor_grid"
                    for row in rows[1:]
                )
            )
            self.assertEqual(list(read_jsonl(output)), rows)
            self.assertNotIn("oracle_window_24px", {row["condition"] for row in rows})

    def test_rounds_up_to_grid_and_records_actual_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source_manifest(root, size=(1001, 599))
            output = root / "sweep" / "render_manifest.jsonl"

            [row] = build_readability_sweep(source, output, target_fonts=(10,))

            self.assertEqual(row["condition"], "full_history_10px_840x504")
            self.assertEqual((row["image_width"], row["image_height"]), (840, 504))
            self.assertEqual(row["requested_image_width_px"], 834.166667)
            self.assertEqual(row["requested_image_height_px"], 499.166667)
            expected = 12 * ((840 / 1001) * (504 / 599)) ** 0.5
            self.assertAlmostEqual(row["actual_effective_font_size_px"], expected, places=6)
            self.assertGreater(row["actual_effective_font_size_px"], 10)
            self.assertEqual(row["actual_total_pixel_budget"], 2 * 840 * 504)
            self.assertEqual(row["processor_grid_multiple_px"], 28)

    def test_explicit_source_target_is_a_byte_identical_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source_manifest(root)
            output = root / "sweep" / "render_manifest.jsonl"

            [row] = build_readability_sweep(source, output, target_fonts=(12,))

            self.assertTrue(row["is_source_anchor"])
            self.assertEqual(row["source_page_sha256"], row["output_page_sha256"])
            self.assertEqual(row["dimension_rounding_policy"], "source_byte_copy")

    def test_rejects_invalid_targets_and_missing_source_images(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_target_fonts("10,8,8")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_target_fonts("10,zero")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source_manifest(root, missing_second_page=True)
            output = root / "sweep" / "render_manifest.jsonl"
            with self.assertRaisesRegex(FileNotFoundError, "source image does not exist"):
                build_readability_sweep(source, output)

    def test_rejects_in_place_manifest_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = self._source_manifest(Path(temporary))
            with self.assertRaisesRegex(ValueError, "must differ"):
                build_readability_sweep(source, source)


if __name__ == "__main__":
    unittest.main()
