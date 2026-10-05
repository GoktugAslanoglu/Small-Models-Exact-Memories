from __future__ import annotations

import unittest

from optical_memory.locomo_controls import (
    EXPECTED_CALLS_PER_MODEL,
    MODEL_IDS,
    build_control_work,
    corrected_selected_history,
    expected_control_keys,
    pack_bm25_documents,
    rank_bm25,
)


class LoCoMoControlTests(unittest.TestCase):
    def test_corrected_selected_history_adds_source_dates(self) -> None:
        lookup = {
            "D1:1": {
                "session_number": 1,
                "session_heading": "=== SESSION 1 | 5 May 2024 ===",
                "rendered_turn": "[D1:1] Alex: First fact",
            },
            "D1:2": {
                "session_number": 1,
                "session_heading": "=== SESSION 1 | 5 May 2024 ===",
                "rendered_turn": "[D1:2] Sam: Second fact",
            },
            "D2:1": {
                "session_number": 2,
                "session_heading": "=== SESSION 2 | 9 May 2024 ===",
                "rendered_turn": "[D2:1] Alex: Later fact",
            },
        }
        rendered = corrected_selected_history(["D1:1", "D1:2", "D2:1"], lookup)
        self.assertEqual(1, rendered.count("SESSION 1 | 5 May 2024"))
        self.assertEqual(1, rendered.count("SESSION 2 | 9 May 2024"))
        self.assertLess(rendered.index("D1:1"), rendered.index("D1:2"))
        self.assertLess(rendered.index("D1:2"), rendered.index("D2:1"))

    def test_control_matrix_has_430_calls_per_model(self) -> None:
        questions = [
            {
                "question_id": f"q{index}",
                "category": 2 if index < 30 else 4,
            }
            for index in range(100)
        ]
        work = build_control_work(questions)
        keys = expected_control_keys(questions, MODEL_IDS["2b"])
        self.assertEqual(EXPECTED_CALLS_PER_MODEL, len(work))
        self.assertEqual(EXPECTED_CALLS_PER_MODEL, len(keys))

    def test_bm25_ranking_prefers_question_overlap(self) -> None:
        documents = [
            {
                "document_index": 0,
                "dia_id": "D1:1",
                "document_text": "=== SESSION 1 | Monday ===\nAlex discussed painting.",
            },
            {
                "document_index": 1,
                "dia_id": "D1:2",
                "document_text": "=== SESSION 1 | Monday ===\nSam adopted a puppy named Luna.",
            },
        ]
        ranked = rank_bm25("What was the puppy named?", documents)
        self.assertEqual("D1:2", ranked[0][1]["dia_id"])

    def test_bm25_packing_never_exceeds_budget(self) -> None:
        documents = [
            {
                "document_index": index,
                "dia_id": f"D1:{index}",
                "document_text": text,
            }
            for index, text in enumerate(
                (
                    "puppy named Luna",
                    "travel plans for June",
                    "favorite tea is jasmine",
                )
            )
        ]
        token_count = lambda text: len(text.split())
        packed, metadata = pack_bm25_documents(
            "What was the puppy named?", documents, token_count, 7
        )
        self.assertLessEqual(token_count(packed), 7)
        self.assertEqual(token_count(packed), metadata["packed_text_tokens"])
        self.assertEqual(7, metadata["target_memory_tokens"])
        self.assertIn("puppy named Luna", packed)


if __name__ == "__main__":
    unittest.main()
