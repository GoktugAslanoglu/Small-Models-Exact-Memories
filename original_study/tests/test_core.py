from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from optical_memory.exactspan import detect_exact_spans
from optical_memory.generate import MEMORY_TYPES, generate_episode
from optical_memory.rendering import render_exactstrip, render_uniform
from optical_memory.score import clean_prediction, levenshtein, score_row


class GeneratorTests(unittest.TestCase):
    def test_all_types_are_deterministic_and_contain_answer(self) -> None:
        for memory_type in MEMORY_TYPES:
            first = generate_episode(memory_type, 0, 1234, 4000)
            second = generate_episode(memory_type, 0, 1234, 4000)
            self.assertEqual(first, second)
            self.assertIn(str(first["answer"]), str(first["history"]))
            self.assertGreaterEqual(len(str(first["history"])), 4000)

    def test_position_buckets_move_target(self) -> None:
        fractions = [
            float(generate_episode("numeric_state", index, 1234, 8000)["target_char_fraction"])
            for index in range(3)
        ]
        self.assertLess(fractions[0], fractions[1])
        self.assertLess(fractions[1], fractions[2])


class DetectorTests(unittest.TestCase):
    def test_detector_is_syntax_only_and_finds_technical_state(self) -> None:
        text = (
            "authoritative_port = 4817\n"
            "sha = 7f3a91c82e11abcd\n"
            "path = /srv/checkpoints/v17/model.bin\n"
        )
        spans = detect_exact_spans(text)
        joined = "\n".join(span.text for span in spans)
        self.assertIn("authoritative_port = 4817", joined)
        self.assertIn("7f3a91c82e11abcd", joined)
        self.assertIn("/srv/checkpoints/v17/model.bin", joined)


class RendererTests(unittest.TestCase):
    def test_uniform_and_exactstrip_share_canvas(self) -> None:
        history = str(generate_episode("high_entropy_identifier", 0, 1234, 5000)["history"])
        uniform, uniform_meta = render_uniform(history, 1022, 1022, 20, None, 4, 16)
        strip, strip_meta = render_exactstrip(
            history,
            1022,
            1022,
            20,
            0.22,
            None,
            4,
            16,
            4,
            12,
        )
        self.assertEqual(uniform.size, strip.size)
        self.assertGreater(strip_meta["protected_span_count"], 0)
        self.assertEqual(uniform_meta["protected_span_count"], 0)


class MetricTests(unittest.TestCase):
    def test_prediction_cleanup_and_exact_scoring(self) -> None:
        self.assertEqual(clean_prediction('{"answer": "7f3a"}'), "7f3a")
        self.assertEqual(
            clean_prediction('<|begin_of_box|>"7f3a"<|end_of_box|>'),
            "7f3a",
        )
        scored = score_row(
            {
                "prediction": "Answer: 7f3a",
                "answer": "7f3a",
                "scoring": "exact",
            }
        )
        self.assertEqual(scored["exact_match"], 1.0)
        self.assertEqual(levenshtein("4817", "4871"), 2)


if __name__ == "__main__":
    unittest.main()
