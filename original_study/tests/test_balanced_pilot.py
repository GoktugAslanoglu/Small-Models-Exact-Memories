from __future__ import annotations

import tempfile
import unittest
from collections import Counter
from pathlib import Path

from optical_memory.balanced_pilot import (
    render_readable_sources,
    select_balanced_episodes,
)
from optical_memory.generate import MEMORY_TYPES, POSITION_BUCKETS
from optical_memory.generate_balanced import generate_balanced_dataset
from optical_memory.rendering import find_font_path


class BalancedPilotTests(unittest.TestCase):
    def test_selects_one_episode_per_type_and_position(self) -> None:
        config = {"seed": 20260826, "samples_per_type": 10, "target_chars": 2400}
        episodes = generate_balanced_dataset(config)

        first = select_balanced_episodes(episodes)
        second = select_balanced_episodes(reversed(episodes))

        self.assertEqual(
            [str(row["episode_id"]) for row in first],
            [str(row["episode_id"]) for row in second],
        )
        self.assertEqual(len(first), 18)
        self.assertEqual(
            Counter(str(row["memory_type"]) for row in first),
            Counter({memory_type: 3 for memory_type in MEMORY_TYPES}),
        )
        self.assertEqual(
            Counter(str(row["position_bucket"]) for row in first),
            Counter({bucket: 6 for bucket in POSITION_BUCKETS}),
        )
        self.assertEqual(
            {
                (str(row["memory_type"]), str(row["position_bucket"]))
                for row in first
            },
            {
                (memory_type, bucket)
                for memory_type in MEMORY_TYPES
                for bucket in POSITION_BUCKETS
            },
        )

    def test_renders_native_fixed_font_source_pages(self) -> None:
        font = find_font_path(None)
        if font is None:
            self.skipTest("No deterministic TrueType font is available")
        config = {"seed": 7, "samples_per_type": 10, "target_chars": 1800}
        selected = select_balanced_episodes(generate_balanced_dataset(config))[:2]

        with tempfile.TemporaryDirectory() as temporary:
            manifest, rows = render_readable_sources(
                selected,
                Path(temporary) / "renders",
                font_path=font,
                canvas_side=560,
                font_size=12,
                margin=14,
                preferred_max_pages=20,
            )

            self.assertTrue(manifest.is_file())
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertEqual(row["condition"], "full_history_12px")
                self.assertEqual(row["dense_font_size"], 12)
                self.assertEqual((row["image_width"], row["image_height"]), (560, 560))
                self.assertGreaterEqual(row["page_count"], 1)
                self.assertEqual(row["page_count"], len(row["image_paths"]))
                self.assertTrue(
                    all((manifest.parent / value).is_file() for value in row["image_paths"])
                )


if __name__ == "__main__":
    unittest.main()
