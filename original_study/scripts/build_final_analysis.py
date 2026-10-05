from __future__ import annotations

import csv
import json
import re
import shutil
import string
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs" / "paper"
FIGURES = OUT / "figures"
ANALYSIS = OUT / "analysis"
CORE_PATH = ROOT / "outputs" / "qwen35_four_model_analysis" / "analysis_results.json"
LOCOMO_PATH = ROOT / "work" / "locomo_external_analysis.json"
LOCOMO_SUMMARY = (
    ROOT
    / "work"
    / "locomo_results_ingest_20260828"
    / "results"
    / "locomo_external100"
    / "analysis"
    / "summary.csv"
)
QUESTIONS = (
    ROOT
    / "work"
    / "locomo_results_ingest_20260828"
    / "data"
    / "generated"
    / "locomo_external100"
    / "questions.jsonl"
)
SCORED_RESULTS = (
    ROOT
    / "work"
    / "locomo_results_ingest_20260828"
    / "results"
    / "locomo_external100"
    / "analysis"
    / "scored_results.jsonl"
)
LOCOMO_CONTROL_SUMMARY = (
    ROOT
    / "work"
    / "locomo_controls_results_20260828"
    / "results"
    / "locomo_controls100"
    / "analysis"
    / "summary.csv"
)
LOCOMO_CONTROL_SCORED = (
    ROOT
    / "work"
    / "locomo_controls_results_20260828"
    / "results"
    / "locomo_controls100"
    / "analysis"
    / "scored_results.jsonl"
)

MODELS = ["0.8B", "2B", "4B", "9B"]
MODEL_IDS = {
    "2B": "Qwen/Qwen3.5-2B",
    "9B": "Qwen/Qwen3.5-9B",
}
CONDITIONS = [
    "full_raw_text",
    "session_summary_text",
    "full_optical_8px",
    "oracle_evidence_24px",
]
CONDITION_LABELS = {
    "full_raw_text": "Full raw text",
    "session_summary_text": "Session summary",
    "full_optical_8px": "Full optical, 8 px",
    "oracle_evidence_24px": "Localized oracle, 24 px*",
}
COLORS = {
    "2B": "#2D6A9F",
    "9B": "#D16D3C",
    "raw": "#314A67",
    "summary": "#8596A8",
    "optical": "#D46A3A",
    "oracle": "#2A8C82",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_answer(value: str) -> str:
    value = value.lower().replace(",", "")
    value = "".join(character for character in value if character not in set(string.punctuation))
    value = re.sub(r"\b(a|an|the|and)\b", " ", value)
    return " ".join(value.split())


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size=size)


def centered(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    face: ImageFont.FreeTypeFont,
    fill: str,
) -> None:
    box = draw.multiline_textbbox((0, 0), text, font=face, spacing=3, align="center")
    draw.multiline_text(
        (xy[0] - (box[2] - box[0]) / 2, xy[1]),
        text,
        font=face,
        fill=fill,
        spacing=3,
        align="center",
    )


def grouped_bar_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    title: str,
    categories: list[str],
    series: list[tuple[str, str, list[float]]],
    ymax: float,
    percent: bool = False,
) -> None:
    x0, y0, x1, y1 = box
    plot_left = x0 + 90
    plot_right = x1 - 24
    plot_top = y0 + 82
    plot_bottom = y1 - 142
    draw.text((x0, y0), title, font=font(30, bold=True), fill="#172432")
    draw.text(
        (plot_left, y0 + 42),
        "Share of questions (%)" if percent else "Official token F1",
        font=font(18),
        fill="#5D6873",
    )
    for step in range(6):
        value = ymax * step / 5
        y = plot_bottom - (plot_bottom - plot_top) * step / 5
        draw.line((plot_left, y, plot_right, y), fill="#D9E0E6", width=2)
        label = f"{value:.0f}" if percent else f"{value:.1f}"
        text_box = draw.textbbox((0, 0), label, font=font(18))
        draw.text(
            (plot_left - 16 - (text_box[2] - text_box[0]), y - 11),
            label,
            font=font(18),
            fill="#5D6873",
        )
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#66727D", width=3)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#66727D", width=3)

    group_width = (plot_right - plot_left) / len(categories)
    bar_gap = 10
    bar_width = min(74, (group_width - 28 - bar_gap * (len(series) - 1)) / len(series))
    total_bars = len(series) * bar_width + (len(series) - 1) * bar_gap
    for category_index, category in enumerate(categories):
        group_center = plot_left + group_width * (category_index + 0.5)
        start = group_center - total_bars / 2
        for series_index, (_, color, values) in enumerate(series):
            value = values[category_index]
            left = start + series_index * (bar_width + bar_gap)
            right = left + bar_width
            top = plot_bottom - (plot_bottom - plot_top) * value / ymax
            draw.rounded_rectangle((left, top, right, plot_bottom), radius=4, fill=color)
            value_text = f"{value:.0f}%" if percent else f"{value:.3f}"
            tbox = draw.textbbox((0, 0), value_text, font=font(16, bold=True))
            draw.text(
                ((left + right - (tbox[2] - tbox[0])) / 2, top - 25),
                value_text,
                font=font(16, bold=True),
                fill="#263442",
            )
        centered(draw, (group_center, plot_bottom + 17), category, font(17), "#34424F")

    legend_x = plot_right - (len(series) * 120)
    for index, (name, color, _) in enumerate(series):
        ly = y0 + 44
        lx = legend_x + index * 120
        draw.rounded_rectangle((lx, ly, lx + 22, ly + 17), radius=3, fill=color)
        draw.text((lx + 29, ly - 3), name, font=font(17), fill="#34424F")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    core = load_json(CORE_PATH)
    locomo = load_json(LOCOMO_PATH)
    control_scored_rows = load_jsonl(LOCOMO_CONTROL_SCORED)

    assert core["integrity"]["passed"] is True
    assert locomo["integrity"]["rows"] == 800
    assert not locomo["integrity"]["hash_failures"]
    assert len(control_scored_rows) == 860

    # Preserve the already verified controlled-study plots under paper-facing names.
    shutil.copy2(
        ROOT / "outputs" / "qwen35_four_model_analysis" / "figure_1_accuracy_by_budget.png",
        FIGURES / "figure_1_controlled_frontier.png",
    )
    shutil.copy2(
        ROOT / "outputs" / "qwen35_four_model_analysis" / "figure_3_memory_type_heatmap.png",
        FIGURES / "figure_2_memory_type_heatmap.png",
    )

    # Main controlled-study figure: use realized token ratios and mark C=1 explicitly.
    frontier = Image.new("RGB", (2100, 1120), "white")
    frontier_draw = ImageDraw.Draw(frontier)
    frontier_draw.text(
        (70, 35),
        "Exact recall drops at the compressed point; larger Qwen readers recover more",
        font=font(40, bold=True),
        fill="#172432",
    )
    frontier_draw.text(
        (70, 92),
        "Text-to-vision token ratio C; only C > 1 reduces the measured input-token count",
        font=font(23),
        fill="#5D6873",
    )
    plot_left, plot_top, plot_right, plot_bottom = 165, 185, 1990, 925
    x_min, x_max = 0.35, 1.82
    x_to_px = lambda value: plot_left + (value - x_min) / (x_max - x_min) * (plot_right - plot_left)
    y_to_px = lambda value: plot_bottom - value * (plot_bottom - plot_top)
    boundary = x_to_px(1.0)
    frontier_draw.rectangle((plot_left, plot_top, boundary, plot_bottom), fill="#F3F5F7")
    frontier_draw.rectangle((boundary, plot_top, plot_right, plot_bottom), fill="#EEF7F2")
    frontier_draw.text(
        (plot_left + 18, plot_top + 16),
        "Token expansion / near parity",
        font=font(20, bold=True),
        fill="#697681",
    )
    frontier_draw.text(
        (boundary + 18, plot_top + 16),
        "Measured token reduction",
        font=font(20, bold=True),
        fill="#37765E",
    )
    for step in range(6):
        value = step / 5
        y = y_to_px(value)
        frontier_draw.line((plot_left, y, plot_right, y), fill="#D5DDE4", width=2)
        label = f"{value:.1f}"
        tbox = frontier_draw.textbbox((0, 0), label, font=font(20))
        frontier_draw.text(
            (plot_left - 22 - (tbox[2] - tbox[0]), y - 12),
            label,
            font=font(20),
            fill="#5D6873",
        )
    frontier_draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#65717D", width=3)
    frontier_draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#65717D", width=3)
    frontier_draw.line((boundary, plot_top, boundary, plot_bottom), fill="#4C8B6F", width=4)
    frontier_draw.text((boundary + 10, plot_bottom - 35), "C = 1", font=font(18, bold=True), fill="#37765E")
    frontier_draw.text((plot_left, 135), "Exact-match accuracy", font=font(21), fill="#5D6873")

    controlled_conditions = [
        ("12 px\n16,384 vision", "full_history_12px_2044x2044"),
        ("10 px\n11,236 vision", "full_history_10px_1708x1708"),
        ("8 px\n7,396 vision", "full_history_8px_1372x1372"),
        ("6 px\n4,096 vision", "full_history_6px_1036x1036"),
    ]
    ratios = [
        core["condition_results"][condition]["2B"]["mean_realized_compression"]
        for _, condition in controlled_conditions
    ]
    line_colors = {
        "0.8B": "#5C6770",
        "2B": "#2D6A9F",
        "4B": "#2A8C82",
        "9B": "#D16D3C",
    }
    for model in MODELS:
        points = []
        for (_, condition), ratio in zip(controlled_conditions, ratios):
            result = core["condition_results"][condition][model]
            x = x_to_px(ratio)
            y = y_to_px(result["accuracy"])
            low, high = result["wilson_95ci"]
            frontier_draw.line((x, y_to_px(high), x, y_to_px(low)), fill=line_colors[model], width=3)
            frontier_draw.line((x - 9, y_to_px(high), x + 9, y_to_px(high)), fill=line_colors[model], width=3)
            frontier_draw.line((x - 9, y_to_px(low), x + 9, y_to_px(low)), fill=line_colors[model], width=3)
            points.append((x, y))
        frontier_draw.line(points, fill=line_colors[model], width=6)
        for x, y in points:
            frontier_draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=line_colors[model], outline="white", width=3)

    for (label, _), ratio in zip(controlled_conditions, ratios):
        x = x_to_px(ratio)
        frontier_draw.line((x, plot_bottom, x, plot_bottom + 10), fill="#65717D", width=3)
        centered(frontier_draw, (x, plot_bottom + 20), f"{ratio:.2f}x\n{label}", font(18), "#34424F")
    frontier_draw.text(
        (plot_left + 570, 1060),
        "Realized C = mean text tokens / total vision tokens across four pages",
        font=font(20),
        fill="#5D6873",
    )
    legend_x, legend_y = 1390, 128
    for index, model in enumerate(MODELS):
        lx = legend_x + (index % 2) * 215
        ly = legend_y + (index // 2) * 42
        frontier_draw.line((lx, ly + 10, lx + 42, ly + 10), fill=line_colors[model], width=6)
        frontier_draw.ellipse((lx + 14, ly, lx + 34, ly + 20), fill=line_colors[model])
        frontier_draw.text((lx + 52, ly - 3), model, font=font(19), fill="#34424F")
    frontier.save(FIGURES / "figure_1_controlled_compression_frontier.png", quality=95)

    cells = {(row["model"], row["condition"]): row for row in locomo["cells"]}
    scored_rows = load_jsonl(SCORED_RESULTS)
    string_em_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in scored_rows:
        key = (str(row["model"]), str(row["condition"]))
        string_em_groups[key].append(
            int(normalize_answer(str(row["prediction"])) == normalize_answer(str(row["answer"])))
        )
    normalized_string_em = {
        key: sum(values) / len(values) for key, values in string_em_groups.items()
    }

    # Figure 3: retrieval, prior-only, modality, and repaired temporal controls.
    with LOCOMO_SUMMARY.open("r", encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    summary = {
        (row["model"], row["condition"], row["category_name"]): float(
            row["mean_official_token_f1"]
        )
        for row in summary_rows
    }

    with LOCOMO_CONTROL_SUMMARY.open("r", encoding="utf-8", newline="") as handle:
        control_summary_rows = list(csv.DictReader(handle))
    control_summary = {
        (row["model"], row["condition"], row["category_name"]): float(
            row["mean_official_token_f1"]
        )
        for row in control_summary_rows
    }

    non_temporal = {}
    for model in ("2B", "9B"):
        model_id = MODEL_IDS[model]
        for condition in ("full_raw_text", "full_optical_8px", "oracle_evidence_24px"):
            single = summary[(model_id, condition, "single_hop")]
            multi = summary[(model_id, condition, "multi_hop")]
            non_temporal[(model, condition)] = (50 * single + 20 * multi) / 70

    canvas = Image.new("RGB", (2600, 1050), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (50, 25),
        "Retrieval controls separate dense representation failure from reader failure",
        font=font(42, bold=True),
        fill="#172432",
    )
    overall_categories = [
        "Full raw\ntext",
        "Input-count-matched\nBM25 text",
        "Full optical\n8 px",
        "Selected optical\n8 px",
        "Selected\ntext",
        "Question\nonly",
    ]
    overall_series = [
        (
            "2B",
            COLORS["2B"],
            [
                cells[(MODEL_IDS["2B"], "full_raw_text")]["mean_f1"],
                control_summary[(MODEL_IDS["2B"], "bm25_equal_vision_budget_text", "ALL")],
                cells[(MODEL_IDS["2B"], "full_optical_8px")]["mean_f1"],
                control_summary[(MODEL_IDS["2B"], "selected_evidence_8px", "ALL")],
                control_summary[(MODEL_IDS["2B"], "selected_evidence_text", "ALL")],
                control_summary[(MODEL_IDS["2B"], "question_only", "ALL")],
            ],
        ),
        (
            "9B",
            COLORS["9B"],
            [
                cells[(MODEL_IDS["9B"], "full_raw_text")]["mean_f1"],
                control_summary[(MODEL_IDS["9B"], "bm25_equal_vision_budget_text", "ALL")],
                cells[(MODEL_IDS["9B"], "full_optical_8px")]["mean_f1"],
                control_summary[(MODEL_IDS["9B"], "selected_evidence_8px", "ALL")],
                control_summary[(MODEL_IDS["9B"], "selected_evidence_text", "ALL")],
                control_summary[(MODEL_IDS["9B"], "question_only", "ALL")],
            ],
        ),
    ]
    grouped_bar_panel(
        draw,
        (50, 105, 1500, 925),
        title="A. Overall LoCoMo-100 (n = 100)",
        categories=overall_categories,
        series=overall_series,
        ymax=0.6,
    )
    diagnostic_categories = [
        "Old selected\n24 px*",
        "Date-aware\n24 px",
        "Selected\n8 px",
        "Selected\ntext",
    ]
    diagnostic_series = [
        (
            "2B",
            COLORS["2B"],
            [
                summary[(MODEL_IDS["2B"], "oracle_evidence_24px", "temporal")],
                control_summary[(MODEL_IDS["2B"], "oracle_evidence_24px_temporal_fixed", "temporal")],
                control_summary[(MODEL_IDS["2B"], "selected_evidence_8px", "temporal")],
                control_summary[(MODEL_IDS["2B"], "selected_evidence_text", "temporal")],
            ],
        ),
        (
            "9B",
            COLORS["9B"],
            [
                summary[(MODEL_IDS["9B"], "oracle_evidence_24px", "temporal")],
                control_summary[(MODEL_IDS["9B"], "oracle_evidence_24px_temporal_fixed", "temporal")],
                control_summary[(MODEL_IDS["9B"], "selected_evidence_8px", "temporal")],
                control_summary[(MODEL_IDS["9B"], "selected_evidence_text", "temporal")],
            ],
        ),
    ]
    grouped_bar_panel(
        draw,
        (1555, 105, 2550, 925),
        title="B. Temporal date repair (n = 30)",
        categories=diagnostic_categories,
        series=diagnostic_series,
        ymax=0.4,
    )
    draw.text(
        (50, 990),
        "* Original 24 px temporal control omitted session dates; the date-aware control repairs that defect.",
        font=font(20),
        fill="#4A5560",
    )
    canvas.save(FIGURES / "figure_3_locomo_validation.png", quality=95)

    # Figure 4: model size changes the dominant optical failure mode.
    failure_metrics = ["No-information response", "Hit 96-token limit"]
    values = {}
    for model in ("2B", "9B"):
        row = cells[(MODEL_IDS[model], "full_optical_8px")]
        values[model] = [
            100 * row["no_information_predictions"] / row["n"],
            100 * row["truncated_at_limit"] / row["n"],
        ]
    failure_canvas = Image.new("RGB", (1700, 850), "white")
    failure_draw = ImageDraw.Draw(failure_canvas)
    grouped_bar_panel(
        failure_draw,
        (60, 40, 1640, 820),
        title="Scaling changes the dominant full-optical failure behavior",
        categories=["No-information\nresponse", "Hit 96-token\nlimit"],
        series=[
            ("2B", COLORS["2B"], values["2B"]),
            ("9B", COLORS["9B"], values["9B"]),
        ],
        ymax=80,
        percent=True,
    )
    failure_canvas.save(FIGURES / "figure_4_optical_failure_modes.png", quality=95)

    # Paper tables.
    controlled_rows = []
    for model in MODELS:
        row = {"model": model}
        for label, condition in [
            ("12px", "full_history_12px_2044x2044"),
            ("10px", "full_history_10px_1708x1708"),
            ("8px", "full_history_8px_1372x1372"),
            ("6px", "full_history_6px_1036x1036"),
            ("raw", "raw_text"),
        ]:
            result = core["condition_results"][condition][model]
            row[label] = f"{result['correct']}/{result['n']} ({100 * result['accuracy']:.1f}%)"
        row["optical_aggregate"] = (
            f"{core['optical_results'][model]['correct']}/{core['optical_results'][model]['n']} "
            f"({100 * core['optical_results'][model]['accuracy']:.1f}%)"
        )
        controlled_rows.append(row)
    write_csv(
        ANALYSIS / "table_controlled_accuracy.csv",
        controlled_rows,
        ["model", "12px", "10px", "8px", "6px", "raw", "optical_aggregate"],
    )

    locomo_rows = []
    for model in ("2B", "9B"):
        for condition in CONDITIONS:
            cell = cells[(MODEL_IDS[model], condition)]
            locomo_rows.append(
                {
                    "model": model,
                    "condition": CONDITION_LABELS[condition].replace("*", ""),
                    "n": cell["n"],
                    "f1": f"{cell['mean_f1']:.3f}",
                    "normalized_string_exact_match": f"{normalized_string_em[(MODEL_IDS[model], condition)]:.2f}",
                    "latency_seconds": f"{cell['mean_latency_seconds']:.3f}",
                    "mean_text_tokens": f"{cell['mean_text_tokens']:.0f}",
                    "mean_vision_tokens": f"{cell['mean_vision_tokens']:.0f}",
                }
            )
    write_csv(
        ANALYSIS / "table_locomo_overall.csv",
        locomo_rows,
        [
            "model",
            "condition",
            "n",
            "f1",
            "normalized_string_exact_match",
            "latency_seconds",
            "mean_text_tokens",
            "mean_vision_tokens",
        ],
    )

    # Audit the temporal oracle rather than silently averaging over a defective control.
    temporal_rows = [row for row in load_jsonl(QUESTIONS) if row["category_name"] == "temporal"]
    temporal_audit = []
    for row in temporal_rows:
        oracle = row["oracle_history"]
        answer = str(row["answer"])
        temporal_audit.append(
            {
                "question_id": row["question_id"],
                "question": row["question"],
                "answer": answer,
                "answer_verbatim_in_oracle": answer.casefold() in oracle.casefold(),
                "has_session_heading": "SESSION" in oracle.upper(),
                "oracle_history": oracle,
            }
        )
    assert len(temporal_audit) == 30
    assert sum(row["has_session_heading"] for row in temporal_audit) == 0
    assert sum(row["answer_verbatim_in_oracle"] for row in temporal_audit) == 0
    write_csv(
        ANALYSIS / "temporal_oracle_audit.csv",
        temporal_audit,
        [
            "question_id",
            "question",
            "answer",
            "answer_verbatim_in_oracle",
            "has_session_heading",
            "oracle_history",
        ],
    )

    derived = {
        "source_integrity": {
            "controlled_passed": core["integrity"]["passed"],
            "controlled_rows": sum(core["integrity"]["row_counts"].values()),
            "locomo_rows": locomo["integrity"]["rows"],
            "locomo_control_rows": len(control_scored_rows),
            "locomo_hash_failures": locomo["integrity"]["hash_failures"],
        },
        "controlled": {
            "raw_correct_total": sum(core["condition_results"]["raw_text"][m]["correct"] for m in MODELS),
            "raw_n_total": sum(core["condition_results"]["raw_text"][m]["n"] for m in MODELS),
            "optical_accuracy_by_model": {m: core["optical_results"][m]["accuracy"] for m in MODELS},
            "accuracy_slope_per_parameter_doubling": core["scaling"]["accuracy_slope_per_log2_parameter_doubling"],
            "accuracy_slope_cluster_bootstrap_95ci": core["scaling"]["accuracy_slope_cluster_bootstrap_95ci"],
            "ordered_permutation_p": core["scaling"]["accuracy_ordered_cluster_permutation_one_sided_p_monte_carlo"],
            "monotonic_episode_trajectories": core["scaling"]["monotonic_episode_trajectories"],
            "episode_count": core["scaling"]["episode_count"],
        },
        "locomo": {
            "auxiliary_metric_correction": {
                "old_field": "normalized_exact_match used unordered token-set equality",
                "paper_field": "normalized_string_exact_match uses equality after LoCoMo normalization",
            },
            "overall": {
                model: {
                    condition: cells[(MODEL_IDS[model], condition)]["mean_f1"]
                    for condition in CONDITIONS
                }
                for model in ("2B", "9B")
            },
            "reviewer_controls": {
                model: {
                    condition: control_summary[(MODEL_IDS[model], condition, "ALL")]
                    for condition in (
                        "bm25_equal_vision_budget_text",
                        "question_only",
                        "selected_evidence_8px",
                        "selected_evidence_text",
                    )
                }
                for model in ("2B", "9B")
            },
            "non_temporal_f1": {
                model: {
                    condition: non_temporal[(model, condition)]
                    for condition in ("full_raw_text", "full_optical_8px", "oracle_evidence_24px")
                }
                for model in ("2B", "9B")
            },
            "temporal_oracle_audit": {
                "n": len(temporal_audit),
                "with_session_heading": sum(row["has_session_heading"] for row in temporal_audit),
                "with_answer_verbatim": sum(row["answer_verbatim_in_oracle"] for row in temporal_audit),
            },
        },
    }
    (ANALYSIS / "derived_results.json").write_text(
        json.dumps(derived, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(derived, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
