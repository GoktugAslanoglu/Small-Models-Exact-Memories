from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_qwen35_9b_main_colab.py"
SPEC = importlib.util.spec_from_file_location("run_qwen35_9b_main_colab", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class Qwen35NineBMainTests(unittest.TestCase):
    def test_frozen_model_and_outputs_are_9b_specific(self) -> None:
        self.assertEqual(RUNNER.MODEL, "Qwen/Qwen3.5-9B")
        self.assertEqual(RUNNER.SEED, 20260826)
        self.assertEqual(RUNNER.TARGET_FONTS, (12, 10, 8, 6))
        for path in (
            RUNNER.OPTICAL_RESULTS,
            RUNNER.RAW_RESULTS,
            RUNNER.COMBINED_RESULTS,
            RUNNER.RUN_METADATA,
            RUNNER.RESULT_ZIP,
        ):
            self.assertIn("qwen35_9b", path.as_posix())
            self.assertNotIn("qwen35_4b", path.as_posix())

    def test_parser_rejects_nonfrozen_settings_in_main(self) -> None:
        arguments = RUNNER.parse_args([])
        self.assertEqual(arguments.model, RUNNER.MODEL)
        self.assertEqual(arguments.seed, RUNNER.SEED)
        self.assertEqual(arguments.max_new_tokens, 64)
        self.assertEqual(arguments.required_gpu, "A100")


if __name__ == "__main__":
    unittest.main()
