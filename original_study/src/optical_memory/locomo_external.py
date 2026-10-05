from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import string
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from .common import project_path, read_jsonl, write_jsonl
from .rendering_pages import render_uniform_pages
from .score import clean_prediction


LOCOMO_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/"
    "main/data/locomo10.json"
)
LOCOMO_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
SEED = 20260826
CATEGORY_NAMES = {1: "multi_hop", 2: "temporal", 4: "single_hop"}
PER_CONVERSATION_QUOTAS = {4: 5, 2: 3, 1: 2}
CONDITIONS = (
    "full_raw_text",
    "session_summary_text",
    "full_optical_8px",
    "oracle_evidence_24px",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_locomo(destination: Path, *, url: str = LOCOMO_URL) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    urllib.request.urlretrieve(url, temporary)
    actual = sha256_file(temporary)
    if actual != LOCOMO_SHA256:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"LoCoMo checksum mismatch: expected {LOCOMO_SHA256}, got {actual}"
        )
    temporary.replace(destination)


def ensure_locomo(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        download_locomo(path)
    actual = sha256_file(path)
    if actual != LOCOMO_SHA256:
        raise ValueError(f"Unexpected LoCoMo checksum at {path}: {actual}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) != 10:
        raise ValueError("Expected the official 10-conversation LoCoMo JSON file")
    return data


def _session_number(key: str) -> int:
    match = re.fullmatch(r"session_(\d+)", key)
    if not match:
        raise ValueError(f"Not a session key: {key}")
    return int(match.group(1))


def session_keys(conversation: dict[str, Any]) -> list[str]:
    keys = [
        key
        for key, value in conversation.items()
        if re.fullmatch(r"session_\d+", key) and isinstance(value, list)
    ]
    return sorted(keys, key=_session_number)


def _turn_text(turn: dict[str, Any]) -> str:
    text = str(turn.get("text", "")).strip()
    caption = str(turn.get("blip_caption", "")).strip()
    if caption:
        text = f"{text} [Shared-image caption: {caption}]"
    return text


def serialize_conversation(item: dict[str, Any]) -> tuple[str, str, dict[str, str]]:
    conversation = item["conversation"]
    full_parts: list[str] = []
    summary_parts: list[str] = []
    turn_lookup: dict[str, str] = {}
    summaries = item.get("session_summary") or {}
    for key in session_keys(conversation):
        number = _session_number(key)
        date = str(conversation.get(f"{key}_date_time", "unknown date"))
        heading = f"=== SESSION {number} | {date} ==="
        full_parts.append(heading)
        for turn in conversation[key]:
            dia_id = str(turn.get("dia_id", "")).strip()
            speaker = str(turn.get("speaker", "Unknown")).strip()
            rendered = f"[{dia_id}] {speaker}: {_turn_text(turn)}"
            full_parts.append(rendered)
            if dia_id:
                turn_lookup[dia_id] = rendered
        summary = str(summaries.get(f"{key}_summary", "")).strip()
        if summary:
            summary_parts.extend((heading, summary))
    return "\n".join(full_parts), "\n".join(summary_parts), turn_lookup


def normalize_evidence_ids(values: Iterable[Any]) -> list[str]:
    ids: list[str] = []
    for value in values:
        ids.extend(part.strip() for part in str(value).split(";") if part.strip())
    return list(dict.fromkeys(ids))


def _rank(seed: int, sample_id: str, qa_index: int, question: str) -> str:
    value = f"{seed}|{sample_id}|{qa_index}|{question}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def build_subset(
    source: list[dict[str, Any]], seed: int = SEED
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    conversations: list[dict[str, Any]] = []
    questions: list[dict[str, Any]] = []
    for item in source:
        sample_id = str(item["sample_id"])
        full_text, summary_text, turn_lookup = serialize_conversation(item)
        conversations.append(
            {
                "schema_version": 1,
                "conversation_id": sample_id,
                "full_history": full_text,
                "session_summary": summary_text,
                "session_count": len(session_keys(item["conversation"])),
                "turn_count": len(turn_lookup),
            }
        )
        eligible: dict[int, list[tuple[str, int, dict[str, Any], list[str]]]] = defaultdict(list)
        for qa_index, qa in enumerate(item["qa"]):
            category = int(qa.get("category", 0))
            answer = qa.get("answer")
            question = str(qa.get("question", "")).strip()
            evidence_ids = normalize_evidence_ids(qa.get("evidence") or [])
            if category not in PER_CONVERSATION_QUOTAS:
                continue
            if not question or answer is None or not str(answer).strip() or not evidence_ids:
                continue
            if any(evidence_id not in turn_lookup for evidence_id in evidence_ids):
                continue
            eligible[category].append(
                (_rank(seed, sample_id, qa_index, question), qa_index, qa, evidence_ids)
            )
        for category in (4, 2, 1):
            quota = PER_CONVERSATION_QUOTAS[category]
            ranked = sorted(eligible[category])
            if len(ranked) < quota:
                raise ValueError(
                    f"{sample_id} has only {len(ranked)} eligible category-{category} "
                    f"items; {quota} required"
                )
            for _, qa_index, qa, evidence_ids in ranked[:quota]:
                oracle_lines = [turn_lookup[evidence_id] for evidence_id in evidence_ids]
                question_id = f"{sample_id}-c{category}-q{qa_index:03d}"
                questions.append(
                    {
                        "schema_version": 1,
                        "question_id": question_id,
                        "conversation_id": sample_id,
                        "source_qa_index": qa_index,
                        "category": category,
                        "category_name": CATEGORY_NAMES[category],
                        "question": str(qa["question"]).strip(),
                        "answer": str(qa["answer"]).strip(),
                        "evidence_ids": evidence_ids,
                        "oracle_history": "\n".join(oracle_lines),
                    }
                )
    questions.sort(key=lambda row: (row["conversation_id"], -row["category"], row["question_id"]))
    if len(conversations) != 10 or len(questions) != 100:
        raise ValueError(
            f"Protocol requires 10 conversations and 100 questions; got "
            f"{len(conversations)} and {len(questions)}"
        )
    counts = Counter(int(row["category"]) for row in questions)
    if counts != Counter({4: 50, 2: 30, 1: 20}):
        raise ValueError(f"Unexpected category counts: {dict(counts)}")
    return conversations, questions


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_protocol(root: Path, source_path: Path, *, seed: int = SEED) -> dict[str, Path]:
    source = ensure_locomo(source_path)
    conversations, questions = build_subset(source, seed)
    data_dir = root / "data" / "generated" / "locomo_external100"
    conversations_path = data_dir / "conversations.jsonl"
    questions_path = data_dir / "questions.jsonl"
    audit_path = data_dir / "question_audit.csv"
    protocol_path = data_dir / "protocol.json"
    write_jsonl(conversations_path, conversations)
    write_jsonl(questions_path, questions)
    _write_csv(
        audit_path,
        [
            {
                "question_id": row["question_id"],
                "conversation_id": row["conversation_id"],
                "category": row["category"],
                "category_name": row["category_name"],
                "question": row["question"],
                "answer": row["answer"],
                "evidence_ids": "; ".join(row["evidence_ids"]),
                "oracle_history": row["oracle_history"],
            }
            for row in questions
        ],
    )
    protocol = {
        "schema_version": 1,
        "benchmark": "LoCoMo",
        "source_url": LOCOMO_URL,
        "source_sha256": LOCOMO_SHA256,
        "license": "CC BY-NC 4.0",
        "selection_seed": seed,
        "question_count": 100,
        "conversation_count": 10,
        "categories": {"single_hop": 50, "temporal": 30, "multi_hop": 20},
        "per_conversation": {"single_hop": 5, "temporal": 3, "multi_hop": 2},
        "excluded_categories": ["open_domain", "adversarial"],
        "conditions": list(CONDITIONS),
        "models_in_order": ["Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-9B"],
        "generation": {"do_sample": False, "max_new_tokens": 96},
        "render": {
            "canvas": [1372, 1372],
            "margin": 24,
            "full_font_px": 8,
            "oracle_font_px": 24,
        },
    }
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    return {
        "conversations": conversations_path,
        "questions": questions_path,
        "audit": audit_path,
        "protocol": protocol_path,
    }


def _render_text(
    text: str,
    destination: Path,
    relative_prefix: Path,
    item_id: str,
    font_path: str,
    font_size: int,
    preferred_max_pages: int,
) -> tuple[list[str], dict[str, Any]]:
    pages, layout, metadata = render_uniform_pages(
        text,
        width=1372,
        height=1372,
        margin=24,
        preferred_max_pages=preferred_max_pages,
        font_path=font_path,
        min_font_size=font_size,
        max_font_size=font_size,
    )
    image_paths: list[str] = []
    for page_index, image in enumerate(pages):
        relative = relative_prefix / f"{item_id}-p{page_index:02d}.png"
        output = destination / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output, format="PNG", optimize=True)
        image_paths.append(relative.as_posix())
    return image_paths, {
        **metadata,
        "page_count": layout.page_count,
        "font_size": layout.font_size,
        "font_path": font_path,
        "font_sha256": sha256_file(Path(font_path)),
        "image_width": 1372,
        "image_height": 1372,
    }


def render_protocol(root: Path, font_path: str) -> dict[str, Path]:
    data_dir = root / "data" / "generated" / "locomo_external100"
    render_dir = root / "renders" / "generated" / "locomo_external100"
    conversations = list(read_jsonl(data_dir / "conversations.jsonl"))
    questions = list(read_jsonl(data_dir / "questions.jsonl"))
    full_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    for ordinal, conversation in enumerate(conversations, start=1):
        image_paths, metadata = _render_text(
            str(conversation["full_history"]),
            render_dir,
            Path("images") / "full_8px",
            str(conversation["conversation_id"]),
            font_path,
            8,
            24,
        )
        full_rows.append(
            {
                "schema_version": 1,
                "conversation_id": conversation["conversation_id"],
                "condition": "full_optical_8px",
                "render_kind": "uniform",
                "image_paths": image_paths,
                **metadata,
            }
        )
        print(f"Rendered full conversation {ordinal}/{len(conversations)}: {len(image_paths)} pages")
    for ordinal, question in enumerate(questions, start=1):
        image_paths, metadata = _render_text(
            str(question["oracle_history"]),
            render_dir,
            Path("images") / "oracle_24px",
            str(question["question_id"]),
            font_path,
            24,
            8,
        )
        oracle_rows.append(
            {
                "schema_version": 1,
                "question_id": question["question_id"],
                "conversation_id": question["conversation_id"],
                "condition": "oracle_evidence_24px",
                "render_kind": "oracle_uniform",
                "image_paths": image_paths,
                **metadata,
            }
        )
        if ordinal % 20 == 0:
            print(f"Rendered oracle evidence {ordinal}/{len(questions)}")
    full_manifest = render_dir / "full_manifest.jsonl"
    oracle_manifest = render_dir / "oracle_manifest.jsonl"
    write_jsonl(full_manifest, full_rows)
    write_jsonl(oracle_manifest, oracle_rows)
    return {"full_manifest": full_manifest, "oracle_manifest": oracle_manifest}


def normalize_answer(value: str) -> str:
    value = value.lower().replace(",", "")
    value = "".join(character for character in value if character not in set(string.punctuation))
    value = re.sub(r"\b(a|an|the|and)\b", " ", value)
    return " ".join(value.split())


def _stemmed_tokens(value: str) -> list[str]:
    try:
        from nltk.stem import PorterStemmer
    except ImportError as exc:
        raise RuntimeError("Scoring LoCoMo requires nltk (pip install nltk)") from exc
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in normalize_answer(value).split()]


def official_f1_score(prediction: str, answer: str) -> float:
    prediction_tokens = _stemmed_tokens(prediction)
    answer_tokens = _stemmed_tokens(answer)
    if not prediction_tokens or not answer_tokens:
        return float(prediction_tokens == answer_tokens)
    overlap = Counter(prediction_tokens) & Counter(answer_tokens)
    same = sum(overlap.values())
    if same == 0:
        return 0.0
    precision = same / len(prediction_tokens)
    recall = same / len(answer_tokens)
    return 2 * precision * recall / (precision + recall)


def official_multihop_f1(prediction: str, answer: str) -> float:
    predictions = [part.strip() for part in prediction.split(",")]
    answers = [part.strip() for part in answer.split(",")]
    return mean(max(official_f1_score(candidate, answer_part) for candidate in predictions) for answer_part in answers)


def score_prediction(prediction: str, answer: str, category: int) -> tuple[float, bool]:
    prediction = clean_prediction(prediction)
    if category == 1:
        score = official_multihop_f1(prediction, answer)
    elif category in (2, 4):
        score = official_f1_score(prediction, answer)
    else:
        raise ValueError(f"Unsupported external-validation category: {category}")
    exact = normalize_answer(prediction) == normalize_answer(answer)
    return score, exact


def score_results(results_path: Path, output_dir: Path) -> dict[str, Path]:
    rows = list(read_jsonl(results_path))
    scored: list[dict[str, Any]] = []
    for row in rows:
        f1_value, exact = score_prediction(
            str(row["prediction"]), str(row["answer"]), int(row["category"])
        )
        scored.append(
            {
                **row,
                "official_token_f1": round(f1_value, 8),
                "normalized_exact_match": int(exact),
            }
        )
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        groups[(str(row["model"]), str(row["condition"]), str(row["category_name"]))].append(row)
        groups[(str(row["model"]), str(row["condition"]), "ALL")].append(row)
    summary: list[dict[str, Any]] = []
    for (model, condition, category_name), group in sorted(groups.items()):
        summary.append(
            {
                "model": model,
                "condition": condition,
                "category_name": category_name,
                "n": len(group),
                "mean_official_token_f1": round(mean(float(row["official_token_f1"]) for row in group), 6),
                "normalized_exact_match_rate": round(mean(int(row["normalized_exact_match"]) for row in group), 6),
                "mean_latency_seconds": round(mean(float(row["latency_seconds"]) for row in group), 6),
                "total_latency_seconds": round(sum(float(row["latency_seconds"]) for row in group), 3),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    scored_path = output_dir / "scored_results.jsonl"
    summary_csv = output_dir / "summary.csv"
    summary_json = output_dir / "summary.json"
    write_jsonl(scored_path, scored)
    _write_csv(summary_csv, summary)
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {"scored": scored_path, "summary_csv": summary_csv, "summary_json": summary_json}


def validate_stage(root: Path, models: list[str]) -> None:
    data_dir = root / "data" / "generated" / "locomo_external100"
    render_dir = root / "renders" / "generated" / "locomo_external100"
    questions = list(read_jsonl(data_dir / "questions.jsonl"))
    full_manifest = list(read_jsonl(render_dir / "full_manifest.jsonl"))
    oracle_manifest = list(read_jsonl(render_dir / "oracle_manifest.jsonl"))
    if len(questions) != 100 or len(full_manifest) != 10 or len(oracle_manifest) != 100:
        raise ValueError("Prepared LoCoMo protocol is incomplete")
    for manifest_path, manifest in (
        (render_dir / "full_manifest.jsonl", full_manifest),
        (render_dir / "oracle_manifest.jsonl", oracle_manifest),
    ):
        for row in manifest:
            for relative in row["image_paths"]:
                if not (manifest_path.parent / relative).is_file():
                    raise FileNotFoundError(manifest_path.parent / relative)
    for model in models:
        slug = "qwen35_2b" if model.endswith("2B") else "qwen35_9b"
        result_path = root / "results" / "locomo_external100" / f"{slug}.jsonl"
        rows = list(read_jsonl(result_path))
        expected = {(q["question_id"], condition) for q in questions for condition in CONDITIONS}
        actual = {(row["question_id"], row["condition"]) for row in rows}
        unique = {
            (row["question_id"], row["condition"], row["model"] ) for row in rows
        }
        if (
            actual != expected
            or len(rows) != 400
            or len(unique) != 400
            or any(row["model"] != model for row in rows)
            or any(int(row["seed"] ) != SEED for row in rows)
            or any("A100" not in str(row["gpu_name"] ).upper() for row in rows)
        ):
            raise ValueError(
                f"{model} must have exactly 400 unique question-condition rows; "
                f"found {len(rows)}"
            )


def package_results(root: Path, include_models: list[str]) -> Path:
    results_dir = root / "results" / "locomo_external100"
    model_paths: list[Path] = []
    for model in include_models:
        slug = "qwen35_2b" if model.endswith("2B") else "qwen35_9b"
        path = results_dir / f"{slug}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        model_paths.append(path)
    combined_path = results_dir / "combined.jsonl"
    combined: list[dict[str, Any]] = []
    for path in model_paths:
        combined.extend(read_jsonl(path))
    write_jsonl(combined_path, combined)
    score_paths = score_results(combined_path, results_dir / "analysis")
    suffix = "2b" if len(include_models) == 1 else "2b_9b"
    bundle_path = results_dir / f"locomo_qwen35_{suffix}_external_bundle.zip"
    candidates = [
        *model_paths,
        combined_path,
        root / "data" / "generated" / "locomo_external100" / "protocol.json",
        root / "data" / "generated" / "locomo_external100" / "questions.jsonl",
        root / "data" / "generated" / "locomo_external100" / "question_audit.csv",
        root / "renders" / "generated" / "locomo_external100" / "full_manifest.jsonl",
        root / "renders" / "generated" / "locomo_external100" / "oracle_manifest.jsonl",
        results_dir / "environment.json",
        *score_paths.values(),
    ]
    manifest: list[dict[str, Any]] = []
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in candidates:
            arcname = path.relative_to(root).as_posix()
            archive.write(path, arcname)
            manifest.append({"path": arcname, "sha256": sha256_file(path), "bytes": path.stat().st_size})
        archive.writestr("bundle_manifest.json", json.dumps(manifest, indent=2) + "\n")
    return bundle_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare, render, score, and validate LoCoMo external validation.")
    parser.add_argument("command", choices=("prepare", "render", "score", "validate", "package"))
    parser.add_argument("--root", default=".")
    parser.add_argument("--source", default="data/source/locomo10.json")
    parser.add_argument("--font")
    parser.add_argument("--results")
    parser.add_argument("--output-dir")
    parser.add_argument("--models", default="Qwen/Qwen3.5-2B,Qwen/Qwen3.5-9B")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_path(args.root)
    models = [part.strip() for part in args.models.split(",") if part.strip()]
    if args.command == "prepare":
        paths = prepare_protocol(root, project_path(args.source, root))
        print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    elif args.command == "render":
        if not args.font:
            raise SystemExit("--font is required for render")
        paths = render_protocol(root, args.font)
        print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    elif args.command == "score":
        if not args.results or not args.output_dir:
            raise SystemExit("--results and --output-dir are required for score")
        paths = score_results(project_path(args.results, root), project_path(args.output_dir, root))
        print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    elif args.command == "validate":
        validate_stage(root, models)
        print(f"Validated {len(models)} model stage(s)")
    else:
        bundle = package_results(root, models)
        print(bundle)


if __name__ == "__main__":
    main()
