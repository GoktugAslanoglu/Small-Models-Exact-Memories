from __future__ import annotations

import unittest

from optical_memory.glm46v import measure_budget, sanitize_inputs
from optical_memory.glm46v_locomo_runner import CONDITIONS, EXPECTED_CALLS


class _Scalar:
    def __init__(self, value: int) -> None:
        self.value = value

    def item(self) -> int:
        return self.value


class _Mask:
    def __init__(self, count: int) -> None:
        self.count = count

    def sum(self) -> _Scalar:
        return _Scalar(self.count)


class _InputIds:
    def __init__(self, values: list[int]) -> None:
        self.values = values
        self.shape = (1, len(values))

    def __eq__(self, value: object) -> _Mask:
        return _Mask(sum(item == value for item in self.values))


class _ImageProcessor:
    merge_size = 2


class _Tokenizer:
    image_token_id = 99

    @staticmethod
    def convert_tokens_to_ids(token: str) -> int:
        return 99 if token == "<|image|>" else -1


class _Processor:
    image_token_id = 99
    image_processor = _ImageProcessor()
    tokenizer = _Tokenizer()


class GlmBudgetTests(unittest.TestCase):
    def test_glm_placeholder_and_grid_counts_must_agree(self) -> None:
        inputs = {
            "input_ids": _InputIds([1, 99, 99, 99, 99, 2]),
            "image_grid_thw": [[_Scalar(1), _Scalar(4), _Scalar(4)]],
        }
        result = measure_budget(_Processor(), inputs, memory_text_tokens=12)
        self.assertEqual(6, result["realized_input_tokens"])
        self.assertEqual(4, result["vision_tokens"])
        self.assertEqual(4, result["grid_vision_tokens"])
        self.assertEqual([4], result["vision_tokens_per_image"])
        self.assertEqual(3.0, result["model_specific_compression"])
        self.assertIn("crosschecked", result["vision_token_source"])

    def test_glm_budget_rejects_mismatched_qwen_style_estimate(self) -> None:
        inputs = {
            "input_ids": _InputIds([1, 99, 99, 2]),
            "image_grid_thw": [[_Scalar(1), _Scalar(4), _Scalar(4)]],
        }
        with self.assertRaisesRegex(RuntimeError, "accounting mismatch"):
            measure_budget(_Processor(), inputs, memory_text_tokens=12)

    def test_glm_model_card_input_cleanup_preserves_mm_types(self) -> None:
        inputs = {"token_type_ids": [0], "mm_token_type_ids": [1]}
        sanitize_inputs(inputs)
        self.assertNotIn("token_type_ids", inputs)
        self.assertIn("mm_token_type_ids", inputs)

    def test_locomo_replication_is_exactly_four_hundred_calls(self) -> None:
        self.assertEqual(
            (
                "full_raw_text",
                "full_optical_8px",
                "selected_evidence_8px",
                "bm25_equal_vision_budget_text",
            ),
            CONDITIONS,
        )
        self.assertEqual(400, EXPECTED_CALLS)


if __name__ == "__main__":
    unittest.main()
