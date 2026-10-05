from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from optical_memory.common import read_jsonl
from optical_memory.locomo_external import (
    SEED,
    package_results,
    prepare_protocol,
    render_protocol,
    validate_stage,
)


MODELS = {
    "2b": "Qwen/Qwen3.5-2B",
    "9b": "Qwen/Qwen3.5-9B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage-based Colab driver: prepare, 2B, then 9B on frozen LoCoMo-100."
    )
    parser.add_argument("stage", choices=("prepare", "2b", "9b", "finalize", "all"))
    parser.add_argument("--root", default=".")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--require-a100", action="store_true")
    return parser.parse_args()


def _gpu_report(require_a100: bool) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU runtime is required.")
    name = torch.cuda.get_device_name(0)
    if require_a100 and "A100" not in name.upper():
        raise SystemExit(f"This protocol requested an A100, but Colab assigned: {name}")
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": name,
    }


def _render_ready(root: Path) -> bool:
    render_dir = root / "renders" / "generated" / "locomo_external100"
    full_path = render_dir / "full_manifest.jsonl"
    oracle_path = render_dir / "oracle_manifest.jsonl"
    if not full_path.is_file() or not oracle_path.is_file():
        return False
    full = list(read_jsonl(full_path))
    oracle = list(read_jsonl(oracle_path))
    if len(full) != 10 or len(oracle) != 100:
        return False
    return all(
        (render_dir / relative).is_file()
        for row in [*full, *oracle]
        for relative in row["image_paths"]
    )


def prepare(root: Path, seed: int) -> None:
    from matplotlib import font_manager

    source_path = root / "data" / "source" / "locomo10.json"
    paths = prepare_protocol(root, source_path, seed=seed)
    print("Frozen protocol:")
    for label, path in paths.items():
        print(f"  {label}: {path}")
    if _render_ready(root):
        print("Render manifests and all referenced PNGs already exist; reusing them.")
        return
    font_path = font_manager.findfont("DejaVu Sans Mono", fallback_to_default=True)
    print(f"Rendering with deterministic font: {font_path}")
    rendered = render_protocol(root, font_path)
    for label, path in rendered.items():
        print(f"  {label}: {path}")


def run_model(root: Path, stage: str, max_new_tokens: int, require_a100: bool) -> Path:
    report = _gpu_report(require_a100)
    results_dir = root / "results" / "locomo_external100"
    results_dir.mkdir(parents=True, exist_ok=True)
    environment_path = results_dir / "environment.json"
    environment_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    model = MODELS[stage]
    output = results_dir / f"qwen35_{stage}.jsonl"
    command = [
        sys.executable,
        "-m",
        "optical_memory.locomo_runner",
        "--root",
        str(root),
        "--model",
        model,
        "--output",
        str(output),
        "--max-new-tokens",
        str(max_new_tokens),
        "--seed",
        str(SEED),
        "--resume",
    ]
    print("Launching isolated model process:", " ".join(command))
    subprocess.run(command, cwd=root, check=True)
    models = [MODELS["2b"]] if stage == "2b" else [MODELS["2b"], MODELS["9b"]]
    validate_stage(root, models)
    bundle = package_results(root, models)
    print(f"Validated bundle: {bundle}")
    return bundle


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    if args.seed != SEED:
        raise SystemExit(f"Protocol seed is frozen at {SEED}; received {args.seed}.")
    if args.stage in ("prepare", "all"):
        prepare(root, args.seed)
    if args.stage in ("2b", "all"):
        run_model(root, "2b", args.max_new_tokens, args.require_a100)
    if args.stage in ("9b", "all"):
        run_model(root, "9b", args.max_new_tokens, args.require_a100)
    if args.stage in ("finalize",):
        validate_stage(root, [MODELS["2b"], MODELS["9b"]])
        print(package_results(root, [MODELS["2b"], MODELS["9b"]]))


if __name__ == "__main__":
    main()
