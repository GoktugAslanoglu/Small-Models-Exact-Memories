from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from optical_memory.common import write_jsonl
from optical_memory.generate import MEMORY_TYPES, POSITION_BUCKETS


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_qwen35_4b_main_colab.py"
SPEC = importlib.util.spec_from_file_location("run_qwen35_4b_main_colab", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def balanced_rows() -> list[dict]:
    rows = []
    for memory_type in MEMORY_TYPES:
        for bucket in POSITION_BUCKETS:
            rows.append(
                {
                    "episode_id": f"{memory_type}-{bucket}",
                    "memory_type": memory_type,
                    "position_bucket": bucket,
                }
            )
    return rows


class Qwen35FourBMainTests(unittest.TestCase):
    def test_validates_complete_a100_result_matrix(self) -> None:
        dataset = balanced_rows()
        conditions = [
            "full_history_12px_2044x2044",
            "full_history_10px_1708x1708",
            "full_history_8px_1372x1372",
            "full_history_6px_1036x1036",
        ]
        sides = dict(zip(conditions, (2044, 1708, 1372, 1036)))
        optical = []
        raw = []
        for episode in dataset:
            for index, condition in enumerate(conditions):
                optical.append(
                    {
                        "episode_id": episode["episode_id"],
                        "condition": condition,
                        "model": RUNNER.MODEL,
                        "gpu_name": "NVIDIA A100-SXM4-40GB",
                        "text_tokens": 7000,
                        "vision_tokens": 16000 - 3000 * index,
                        "grid_vision_tokens": 16000 - 3000 * index,
                        "realized_compression": 0.5 + 0.2 * index,
                        "image_width": sides[condition],
                        "image_height": sides[condition],
                    }
                )
            raw.append(
                {
                    "episode_id": episode["episode_id"],
                    "condition": "raw_text",
                    "model": RUNNER.MODEL,
                    "gpu_name": "NVIDIA A100-SXM4-40GB",
                }
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            optical_path = root / "optical.jsonl"
            raw_path = root / "raw.jsonl"
            write_jsonl(optical_path, optical)
            write_jsonl(raw_path, raw)
            combined = RUNNER.validate_result_matrix(
                dataset,
                optical_path,
                raw_path,
                conditions=conditions,
                sides=sides,
                model=RUNNER.MODEL,
                required_gpu="A100",
            )
            self.assertEqual(len(combined), 90)

            optical[0]["gpu_name"] = "NVIDIA L4"
            write_jsonl(optical_path, optical)
            with self.assertRaisesRegex(SystemExit, "GPU provenance failed"):
                RUNNER.validate_result_matrix(
                    dataset,
                    optical_path,
                    raw_path,
                    conditions=conditions,
                    sides=sides,
                    model=RUNNER.MODEL,
                    required_gpu="A100",
                )

    def test_resume_rejects_wrong_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "partial.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "episode_id": "episode-1",
                        "condition": "raw_text",
                        "model": RUNNER.MODEL,
                        "gpu_name": "NVIDIA L4",
                    }
                ],
            )
            with self.assertRaisesRegex(SystemExit, "Unsafe/incompatible resume"):
                RUNNER.check_resume(
                    path,
                    allowed_conditions=["raw_text"],
                    model=RUNNER.MODEL,
                    required_gpu="A100",
                )

    def test_inspects_frozen_four_condition_manifest(self) -> None:
        dataset = balanced_rows()
        condition_sides = {
            "full_history_12px_2044x2044": 2044,
            "full_history_10px_1708x1708": 1708,
            "full_history_8px_1372x1372": 1372,
            "full_history_6px_1036x1036": 1036,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "images" / "placeholder.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"placeholder")
            manifest = root / "render_manifest.jsonl"
            rows = []
            for condition, side in condition_sides.items():
                for episode in dataset:
                    rows.append(
                        {
                            "episode_id": episode["episode_id"],
                            "condition": condition,
                            "image_width": side,
                            "image_height": side,
                            "page_count": 1,
                            "image_paths": ["images/placeholder.png"],
                        }
                    )
            write_jsonl(manifest, rows)
            conditions, sides = RUNNER.inspect_sweep_manifest(manifest, dataset)
            self.assertEqual(conditions, list(condition_sides))
            self.assertEqual(sides, condition_sides)


if __name__ == "__main__":
    unittest.main()
