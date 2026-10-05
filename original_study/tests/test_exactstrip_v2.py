from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from optical_memory.common import read_jsonl, write_jsonl
from optical_memory.exactspan_v2 import (
    format_protection_strip_v2,
    select_protected_spans_v2,
)
from optical_memory.exactstrip_v2_sweep import build_exactstrip_v2_sweep
from optical_memory.rendering import find_font_path


class ExactStripV2Tests(unittest.TestCase):
    def test_removes_contained_spans_and_bracket_labels(self) -> None:
        history = (
            "TOOL: release_manifest\n"
            "case_id = sym-03-3np7ru\n"
            "checkpoint = /srv/checkpoints/sym-03/model-a1.safetensors\n"
            "rollback_sha = 8fe2926d8f64708e\n"
        )
        spans, detected_count = select_protected_spans_v2(history)
        strip = format_protection_strip_v2(spans)

        self.assertLess(len(spans), detected_count)
        self.assertIn(
            "checkpoint = /srv/checkpoints/sym-03/model-a1.safetensors", strip
        )
        self.assertIn("rollback_sha = 8fe2926d8f64708e", strip)
        self.assertNotIn("[path]", strip)
        self.assertNotIn("[identifier]", strip)
        self.assertEqual(strip.splitlines()[0], "EXACT STATE - COPY VALUES VERBATIM")

    def test_builds_one_equal_budget_v2_condition(self) -> None:
        font = find_font_path(None)
        if font is None:
            self.skipTest("No deterministic TrueType font is available")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
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
                    * 20,
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
                    * 20,
                    "question": "Return the final path.",
                    "answer": "/srv/checkpoints/model-a1.safetensors",
                    "scoring": "exact",
                },
            ]
            write_jsonl(dataset, episodes)

            reference_root = root / "uniform"
            reference_manifest = reference_root / "render_manifest.jsonl"
            reference_condition = "full_history_8px_280x280"
            reference_rows = []
            for episode in episodes:
                paths = []
                for page_index in range(2):
                    relative = (
                        Path("images")
                        / reference_condition
                        / episode["episode_id"]
                        / f"page-{page_index:02d}.png"
                    )
                    destination = reference_root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (280, 280), "white").save(destination)
                    paths.append(relative.as_posix())
                reference_rows.append(
                    {
                        "episode_id": episode["episode_id"],
                        "memory_type": episode["memory_type"],
                        "position_bucket": episode["position_bucket"],
                        "condition": reference_condition,
                        "render_kind": "uniform",
                        "target_compression": 0,
                        "image_paths": paths,
                        "page_count": 2,
                        "image_width": 280,
                        "image_height": 280,
                        "total_pixel_budget": 2 * 280 * 280,
                        "requested_effective_font_size_px": 8,
                    }
                )
            write_jsonl(reference_manifest, reference_rows)

            output = root / "v2" / "render_manifest.jsonl"
            rows = build_exactstrip_v2_sweep(
                dataset,
                reference_manifest,
                output,
                reference_condition=reference_condition,
                font_path=font,
                native_canvas_side=560,
                margin=14,
                strip_fraction=0.18,
                min_font_size=3,
                max_font_size=12,
                strip_min_font_size=8,
                strip_max_font_size=16,
            )

            self.assertEqual(len(rows), 2)
            self.assertEqual(
                {row["condition"] for row in rows},
                {"exactstrip_v2_8px_280x280"},
            )
            for row in rows:
                self.assertEqual(row["render_kind"], "exactstrip_v2")
                self.assertEqual(row["span_detector"], "syntax_only_v2")
                self.assertEqual(row["detector_input"], "history_only")
                self.assertEqual(row["development_status"], "posthoc_v2_ablation")
                self.assertEqual(row["strip_format"], "raw_span_no_type_prefix")
                self.assertLess(row["protected_span_count"], row["detected_span_count"])
                self.assertEqual(row["page_count"], 2)
                self.assertEqual((row["image_width"], row["image_height"]), (280, 280))
                self.assertEqual(row["total_pixel_budget"], 2 * 280 * 280)
                self.assertEqual(
                    row["total_pixel_budget"], row["matched_uniform_total_pixel_budget"]
                )
                self.assertGreater(
                    row["actual_effective_strip_font_size_px"],
                    row["actual_effective_dense_font_size_px"],
                )
                paths = [output.parent / value for value in row["image_paths"]]
                self.assertTrue(all(path.is_file() for path in paths))
                with Image.open(paths[0]) as image:
                    self.assertEqual(image.size, (280, 280))
            self.assertEqual(list(read_jsonl(output)), rows)


if __name__ == "__main__":
    unittest.main()
