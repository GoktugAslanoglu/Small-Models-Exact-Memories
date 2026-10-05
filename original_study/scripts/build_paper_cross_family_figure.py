"""Build the paper's headline cross-family LoCoMo figure from validated data."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


INPUT = Path("outputs/glm46v_cross_family_analysis/locomo_cross_family.csv")
OUTPUT = Path("outputs/paper/figures/figure_1_locomo_cross_family.png")

MODELS = ("Qwen3.5-2B", "Qwen3.5-9B", "GLM-4.6V-Flash")
CONDITIONS = (
    ("full_raw_text", "Raw text"),
    ("full_optical_8px", "Full optical"),
    ("selected_evidence_8px", "Selected optical"),
    ("bm25_equal_vision_budget_text", "Input-count-matched BM25"),
)
COLORS = ("#4C78A8", "#E45756", "#72B7B2", "#F2CF5B")
HATCHES = ("", "///", "..", "xx")


def main() -> None:
    with INPUT.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    values = {
        (row["family_model"], row["condition"]): float(row["official_token_f1"])
        for row in rows
    }
    expected = {(model, condition) for model in MODELS for condition, _ in CONDITIONS}
    if set(values) != expected:
        missing = sorted(expected - set(values))
        extra = sorted(set(values) - expected)
        raise ValueError(f"Unexpected cross-family matrix; missing={missing}, extra={extra}")

    plt.rcParams.update(
        {
            "font.size": 9.2,
            "axes.labelsize": 9.2,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.4,
            "legend.fontsize": 7.8,
        }
    )
    fig, ax = plt.subplots(figsize=(6.3, 3.15), constrained_layout=True)
    x = np.arange(len(MODELS))
    width = 0.19
    offsets = (np.arange(len(CONDITIONS)) - 1.5) * width

    for offset, (condition, label), color, hatch in zip(
        offsets, CONDITIONS, COLORS, HATCHES, strict=True
    ):
        heights = [values[(model, condition)] for model in MODELS]
        bars = ax.bar(
            x + offset,
            heights,
            width=width,
            label=label,
            color=color,
            edgecolor="#333333",
            linewidth=0.45,
            hatch=hatch,
            zorder=3,
        )
        for bar, height in zip(bars, heights, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.012,
                f"{height:.3f}".lstrip("0"),
                ha="center",
                va="bottom",
                fontsize=7.2,
                rotation=90,
            )

    ax.set_ylabel("LoCoMo-100 token F1")
    ax.set_xticks(x, MODELS)
    ax.set_ylim(0, 0.58)
    ax.set_yticks(np.arange(0, 0.6, 0.1))
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=4,
        frameon=False,
        columnspacing=1.0,
        handlelength=1.6,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Created {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
