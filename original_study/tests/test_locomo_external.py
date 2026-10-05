from __future__ import annotations

import unittest
from collections import Counter

try:
    import nltk  # noqa: F401
except ImportError:
    nltk = None

from optical_memory.locomo_external import (
    build_subset,
    normalize_answer,
    normalize_evidence_ids,
    official_f1_score,
    official_multihop_f1,
    score_prediction,
)


def _fixture() -> list[dict]:
    source: list[dict] = []
    for conversation_index in range(10):
        turns = []
        qa = []
        qa_index = 0
        for category, count in ((4, 7), (2, 5), (1, 4)):
            for item_index in range(count):
                dia_id = f"D1:{qa_index + 1}"
                turns.append(
                    {
                        "speaker": "Alex",
                        "dia_id": dia_id,
                        "text": f"fact {conversation_index}-{category}-{item_index}",
                    }
                )
                qa.append(
                    {
                        "question": f"question {conversation_index}-{category}-{item_index}",
                        "answer": f"answer {item_index}",
                        "evidence": [dia_id],
                        "category": category,
                    }
                )
                qa_index += 1
        source.append(
            {
                "sample_id": f"conv-{conversation_index}",
                "conversation": {
                    "session_1_date_time": "1 January 2026",
                    "session_1": turns,
                },
                "session_summary": {"session_1_summary": "A compact summary."},
                "qa": qa,
            }
        )
    return source


class LoCoMoExternalTests(unittest.TestCase):
    def test_frozen_subset_is_balanced_and_deterministic(self) -> None:
        conversations, questions = build_subset(_fixture())
        _, repeated = build_subset(_fixture())
        self.assertEqual(10, len(conversations))
        self.assertEqual(100, len(questions))
        self.assertEqual(
            Counter({4: 50, 2: 30, 1: 20}),
            Counter(row["category"] for row in questions),
        )
        self.assertEqual(
            Counter({f"conv-{index}": 10 for index in range(10)}),
            Counter(row["conversation_id"] for row in questions),
        )
        self.assertEqual(
            [row["question_id"] for row in questions],
            [row["question_id"] for row in repeated],
        )

    def test_evidence_id_normalization_handles_combined_annotation(self) -> None:
        self.assertEqual(
            ["D8:6", "D9:17", "D10:2"],
            normalize_evidence_ids(["D8:6; D9:17", "D10:2", "D8:6"]),
        )

    def test_official_normalization(self) -> None:
        self.assertEqual("artist selfexpression", normalize_answer("The artist, and self-expression!"))

    def test_boxed_glm_answer_is_scored_as_answer_text(self) -> None:
        score, exact = score_prediction(
            "<|begin_of_box|>art and self-expression<|end_of_box|>",
            "art and self-expression",
            4,
        )
        self.assertEqual(1.0, score)
        self.assertTrue(exact)

    def test_normalized_exact_match_preserves_word_order(self) -> None:
        _, exact = score_prediction("tea books", "books tea", 4)
        self.assertFalse(exact)

    @unittest.skipIf(nltk is None, "LoCoMo scoring extra is not installed")
    def test_official_f1_stems_tokens(self) -> None:
        self.assertEqual(1.0, official_f1_score("painted pictures", "painting picture"))
        self.assertEqual(1.0, official_multihop_f1("tea, books", "books, tea"))


if __name__ == "__main__":
    unittest.main()
