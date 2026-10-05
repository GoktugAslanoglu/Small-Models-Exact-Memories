from __future__ import annotations

import argparse
import unittest

from optical_memory.locomo_runner import (
    PROMPT_PROFILES,
    SHORT_ANSWER_INSTRUCTION,
    parse_conditions,
)


class LoCoMoShortAnswerTests(unittest.TestCase):
    def test_short_prompt_is_frozen(self) -> None:
        self.assertEqual(SHORT_ANSWER_INSTRUCTION, PROMPT_PROFILES["short32"])
        rendered = SHORT_ANSWER_INSTRUCTION.format(question="Where is the key?")
        self.assertIn("Return only the shortest answer needed", rendered)
        self.assertIn("reply exactly: No information available", rendered)
        self.assertTrue(rendered.endswith("Question: Where is the key?"))

    def test_single_condition_subset(self) -> None:
        self.assertEqual(
            ("full_optical_8px",),
            parse_conditions("full_optical_8px"),
        )

    def test_unknown_or_duplicate_conditions_fail(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_conditions("unknown")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_conditions("full_optical_8px,full_optical_8px")


if __name__ == "__main__":
    unittest.main()
