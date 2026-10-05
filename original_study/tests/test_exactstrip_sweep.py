from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from optical_memory.common import read_jsonl, write_jsonl
from optical_memory.exactstrip_sweep import build_exactstrip_sweep, parse_conditions
from optical_memory.rendering import find_font_path


class ExactStripSweepTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        dataset = root / "data.jsonl"
        episodes = [
            {
                "episode_id": "case-000",
                "memory_type": "high_entropy_identifier",
                "position_bucket": "early",
                "history": (
                    "USER: inspect state\n"
                    "TOOL: registry\n"
                    "case_id = case-000\n"
                    "authoritative_sha = 8fe2926d8f64708e\n"
                    "status = pinned\n"
                )
                * 30,
                "question": "Return the final SHA.",
                "answer": "8fe2926d8f64708e",
                "scoring": "exact",
            },
            {
                "episode_id": "case-001",
                "memory_type": "symbol_path",
                "position_bucket": "late",
                "history": (
                    "USER: inspect state\n"
                    "TOOL: manifest\n"
                    "case_id = case-001\n"
                    "checkpoint = /srv/checkpoints/model-a1.safetensors\n"
                    "status = verified\n"
                )
                * 30,
                "question": "Return the final path.",
                "answer": "/srv/checkpoints/model-a1.safetensors",
                "scoring": "exact",
            },
        ]
        write_jsonl(dataset, episodes)

        manifest_root = root / "uniform"
        manifest = manifest_root / "render_manifest.jsonl"
        rows = []
        for condition, font, side in (
            ("full_history_8px_280x280", 8, 280),
            ("full_history_6px_224x224", 6, 224),
        ):
            for episode in episodes:
                paths = []
                for page_index in range(2):
                    relative = (
                        Path("images")
                        / condition
                        / episode["episode_id"]
                        / f"page-{page_index:02d}.png"
                    )
                    destination = manifest_root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (side, side), "white").save(destination)
                    paths.append(relative.as_posix())
                rows.append(
                    {
                        "episode_id": episode["episode_id"],
                        "memory_type": episode["memory_type"],
                        "position_bucket": episode["position_bucket"],
                        "condition": condition,
                        "render_kind": "uniform",
                        "target_compression": 0,
                        "image_paths": paths,
                        "page_count": 2,
                        "image_width": side,
                        "image_height": side,
                        "total_pixel_budget": 2 * side * side,
                        "requested_effective_font_size_px": font,
                    }
                )
        write_jsonl(manifest, rows)
        return dataset, manifest

    def test_builds_equal_budget_exactstrip_pairs(self) -> None:
        font = find_font_path(None)
        if font is None:
            self.skipTest("No deterministic TrueType font is available")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, reference = self._fixture(root)
            output = root / "exactstrip" / "render_manifest.jsonl"
            rows = build_exactstrip_sweep(
                dataset,
                reference,
                output,
                reference_conditions=(
                    "full_history_8px_280x280",
                    "full_history_6px_224x224",
                ),
                font_path=font,
                native_canvas_side=560,
                margin=14,
                min_font_size=3,
                max_font_size=12,
                strip_min_font_size=4,
                strip_max_font_size=12,
            )

            self.assertEqual(len(rows), 4)
            self.assertEqual(
                {row["condition"] for row in rows},
                {"exactstrip_8px_280x280", "exactstrip_6px_224x224"},
            )
            self.assertTrue(all(row["render_kind"] == "exactstrip" for row in rows))
            self.assertTrue(all(row["detector_input"] == "history_only" for row in rows))
            self.assertTrue(all(row["protected_span_count"] > 0 for row in rows))
            self.assertTrue(all(row["page_count"] == 2 for row in rows))

            for row in rows:
                expected_side = 280 if "_8px_" in row["condition"] else 224
                self.assertEqual(
                    (row["image_width"], row["image_height"]),
                    (expected_side, expected_side),
                )
                self.assertEqual(
                    row["total_pixel_budget"], 2 * expected_side * expected_side
                )
                self.assertEqual(
                    row["total_pixel_budget"], row["matched_uniform_total_pixel_budget"]
                )
                generated = [output.parent / value for value in row["image_paths"]]
                self.assertTrue(all(path.is_file() for path in generated))
                self.assertEqual(len(row["output_page_sha256"]), 2)
                with Image.open(generated[0]) as image:
                    self.assertEqual(image.size, (expected_side, expected_side))

            self.assertEqual(list(read_jsonl(output)), rows)

    def test_rejects_incomplete_reference_matrix(self) -> None:
        font = find_font_path(None)
        if font is None:
            self.skipTest("No deterministic TrueType font is available")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, reference = self._fixture(root)
            rows = list(read_jsonl(reference))[:-1]
            write_jsonl(reference, rows)
            with self.assertRaisesRegex(ValueError, "reference matrix"):
                build_exactstrip_sweep(
                    dataset,
                    reference,
                    root / "out" / "render_manifest.jsonl",
                    reference_conditions=(
                        "full_history_8px_280x280",
                        "full_history_6px_224x224",
                    ),
                    font_path=font,
                    native_canvas_side=560,
                )

    def test_parses_reference_conditions(self) -> None:
        self.assertEqual(parse_conditions("a,b"), ("a", "b"))
        with self.assertRaisesRegex(Exception, "unique"):
            parse_conditions("a,a")


if __name__ == "__main__":
    unittest.main()
