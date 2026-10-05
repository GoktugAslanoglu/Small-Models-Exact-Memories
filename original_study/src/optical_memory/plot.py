from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from .common import project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot type-specific fidelity curves.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", default="plots/fidelity_curves.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Install plotting support with: pip install -e '.[plots]'") from exc

    summary_path = project_path(args.summary)
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("Summary CSV is empty")

    models = sorted({row["model"] for row in rows})
    figure, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4.5), squeeze=False)
    for axis, model in zip(axes[0], models):
        groups = defaultdict(list)
        for row in rows:
            if row["model"] == model and row["condition"].startswith("uniform"):
                groups[row["memory_type"]].append(row)
        for memory_type, group in sorted(groups.items()):
            group.sort(key=lambda row: float(row["mean_vision_tokens"] or 0), reverse=True)
            x = [float(row["mean_vision_tokens"]) for row in group]
            y = [float(row["exact_match"]) for row in group]
            axis.plot(x, y, marker="o", label=memory_type)
        axis.set_xscale("log")
        axis.invert_xaxis()
        axis.set_ylim(-0.03, 1.03)
        axis.set_title(model.split("/")[-1])
        axis.set_xlabel("Realized vision tokens (lower is denser)")
        axis.set_ylabel("Exact match")
        axis.grid(alpha=0.25)
    axes[0][-1].legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

