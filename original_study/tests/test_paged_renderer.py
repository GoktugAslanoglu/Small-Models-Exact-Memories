from __future__ import annotations

import unittest

from optical_memory.generate import generate_episode
from optical_memory.rendering_pages import render_exactstrip_pages, render_uniform_pages


class PagedRendererTests(unittest.TestCase):
    def test_fixed_budget_page_count(self) -> None:
        history = str(generate_episode("high_entropy_identifier", 0, 1234, 12000)["history"])
        uniform, layout, _ = render_uniform_pages(history, 1022, 1022, 20, 2, None, 3, 12)
        strip, metadata = render_exactstrip_pages(
            history,
            layout.page_count,
            1022,
            1022,
            20,
            0.22,
            None,
            3,
            12,
            4,
            12,
        )
        self.assertEqual(len(uniform), len(strip))
        self.assertGreaterEqual(len(uniform), 1)
        self.assertGreater(metadata["protected_span_count"], 0)
        self.assertTrue(all(left.size == right.size for left, right in zip(uniform, strip)))


if __name__ == "__main__":
    unittest.main()

