from __future__ import annotations

import hashlib
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .common import read_jsonl, write_jsonl
from .locomo_external import (
    SEED,
    _render_text,
    _session_number,
    _turn_text,
    build_subset,
    ensure_locomo,
    score_results,
    session_keys,
    sha256_file,
)


PROTOCOL_VERSION = "locomo-controls-v1"
CONTROL_CONDITIONS = (
    "oracle_evidence_24px_temporal_fixed",
    "selected_evidence_8px",
    "selected_evidence_text",
    "question_only",
    "bm25_equal_vision_budget_text",
)
ALL_QUESTION_CONDITIONS = CONTROL_CONDITIONS[1:]
TEMPORAL_CONDITION = CONTROL_CONDITIONS[0]
EXPECTED_CALLS_PER_MODEL = 30 + 4 * 100
MODEL_IDS = {
    "08b": "Qwen/Qwen3.5-0.8B",
    "2b": "Qwen/Qwen3.5-2B",
    "4b": "Qwen/Qwen3.5-4B",
    "9b": "Qwen/Qwen3.5-9B",
}
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def model_slug(model_id: str) -> str:
    for slug, expected in MODEL_IDS.items():
        if model_id == expected:
            return slug
    raise ValueError(f"Unsupported frozen model ID: {model_id}")


def expected_control_keys(
    questions: Sequence[dict[str, Any]], model_id: str
) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    for question in questions:
        question_id = str(question["question_id"])
        for condition in ALL_QUESTION_CONDITIONS:
            keys.add((question_id, condition, model_id))
        if int(question["category"]) == 2:
            keys.add((question_id, TEMPORAL_CONDITION, model_id))
    return keys


def build_control_work(
    questions: Sequence[dict[str, Any]],
) -> list[tuple[dict[str, Any], str]]:
    work: list[tuple[dict[str, Any], str]] = []
    for question in questions:
        if int(question["category"]) == 2:
            work.append((question, TEMPORAL_CONDITION))
        work.extend((question, condition) for condition in ALL_QUESTION_CONDITIONS)
    return work


def _source_turn_lookup(
    item: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    conversation = item["conversation"]
    lookup: dict[str, dict[str, Any]] = {}
    retrieval_documents: list[dict[str, Any]] = []
    document_index = 0
    for key in session_keys(conversation):
        session_number = _session_number(key)
        date = str(conversation.get(f"{key}_date_time", "unknown date"))
        heading = f"=== SESSION {session_number} | {date} ==="
        for turn in conversation[key]:
            dia_id = str(turn.get("dia_id", "")).strip()
            speaker = str(turn.get("speaker", "Unknown")).strip()
            rendered_turn = f"[{dia_id}] {speaker}: {_turn_text(turn)}"
            if not dia_id:
                continue
            record = {
                "dia_id": dia_id,
                "session_number": session_number,
                "session_heading": heading,
                "rendered_turn": rendered_turn,
            }
            lookup[dia_id] = record
            retrieval_documents.append(
                {
                    "document_index": document_index,
                    **record,
                    "document_text": f"{heading}\n{rendered_turn}",
                }
            )
            document_index += 1
    return lookup, retrieval_documents


def corrected_selected_history(
    evidence_ids: Iterable[str], turn_lookup: dict[str, dict[str, Any]]
) -> str:
    """Serialize selected evidence with every relevant session/date heading.

    The evidence IDs are frozen LoCoMo annotations. No question or answer text is
    used to select turns. A heading is emitted once before the first retained
    turn from that session, preserving evidence order.
    """

    lines: list[str] = []
    active_session: int | None = None
    for evidence_id in evidence_ids:
        if evidence_id not in turn_lookup:
            raise KeyError(f"Unknown evidence turn: {evidence_id}")
        record = turn_lookup[evidence_id]
        session_number = int(record["session_number"])
        if session_number != active_session:
            lines.append(str(record["session_heading"]))
            active_session = session_number
        lines.append(str(record["rendered_turn"]))
    return "\n".join(lines)


def prepare_controls_protocol(
    root: Path, source_path: Path, *, seed: int = SEED
) -> dict[str, Path]:
    source = ensure_locomo(source_path)
    conversations, frozen_questions = build_subset(source, seed)
    source_by_id = {str(item["sample_id"]): item for item in source}

    questions: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    for conversation in conversations:
        conversation_id = str(conversation["conversation_id"])
        _, documents = _source_turn_lookup(source_by_id[conversation_id])
        for document in documents:
            retrieval_rows.append(
                {
                    "schema_version": 1,
                    "conversation_id": conversation_id,
                    **document,
                }
            )

    lookups = {
        conversation_id: _source_turn_lookup(item)[0]
        for conversation_id, item in source_by_id.items()
    }
    for question in frozen_questions:
        selected_history = corrected_selected_history(
            [str(value) for value in question["evidence_ids"]],
            lookups[str(question["conversation_id"])],
        )
        questions.append(
            {
                **question,
                "original_oracle_history_without_dates": question["oracle_history"],
                "selected_history": selected_history,
                "selected_history_sha256": hashlib.sha256(
                    selected_history.encode("utf-8")
                ).hexdigest(),
                "selected_history_has_session_date": int(
                    bool(re.search(r"(?m)^=== SESSION \d+ \| .+ ===$", selected_history))
                ),
                "protocol_version": PROTOCOL_VERSION,
            }
        )

    if len(questions) != 100 or len(retrieval_rows) == 0:
        raise ValueError("The versioned LoCoMo controls protocol is incomplete")
    if any(
        not int(row["selected_history_has_session_date"])
        for row in questions
        if int(row["category"]) == 2
    ):
        raise ValueError("Every repaired temporal oracle must include a session date")

    data_dir = root / "data" / "generated" / "locomo_controls100"
    questions_path = data_dir / "questions.jsonl"
    conversations_path = data_dir / "conversations.jsonl"
    retrieval_path = data_dir / "retrieval_documents.jsonl"
    protocol_path = data_dir / "protocol.json"
    write_jsonl(questions_path, questions)
    write_jsonl(conversations_path, conversations)
    write_jsonl(retrieval_path, retrieval_rows)
    protocol = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "selection_seed": seed,
        "question_count": 100,
        "temporal_question_count": 30,
        "control_conditions": list(CONTROL_CONDITIONS),
        "expected_calls_per_model": EXPECTED_CALLS_PER_MODEL,
        "default_models": [MODEL_IDS["2b"], MODEL_IDS["9b"]],
        "optional_comprehensive_models": [MODEL_IDS["08b"], MODEL_IDS["4b"]],
        "selected_evidence_policy": (
            "Frozen LoCoMo evidence IDs with the source session/date heading added "
            "before retained turns; identical UTF-8 text feeds the 8px and text controls."
        ),
        "bm25": {
            "unit": "individual dialogue turn with its source session/date heading",
            "query": "question_only",
            "k1": 1.5,
            "b": 0.75,
            "packing": "descending BM25 score until the per-model measured full-optical vision-token budget",
        },
        "generation": {"do_sample": False, "max_new_tokens": 96},
        "render": {
            "canvas": [1372, 1372],
            "margin": 24,
            "selected_font_px": 8,
            "temporal_oracle_font_px": 24,
        },
    }
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    return {
        "questions": questions_path,
        "conversations": conversations_path,
        "retrieval_documents": retrieval_path,
        "protocol": protocol_path,
    }


def render_controls_protocol(root: Path, font_path: str) -> dict[str, Path]:
    data_dir = root / "data" / "generated" / "locomo_controls100"
    render_dir = root / "renders" / "generated" / "locomo_controls100"
    questions = list(read_jsonl(data_dir / "questions.jsonl"))
    selected_rows: list[dict[str, Any]] = []
    temporal_rows: list[dict[str, Any]] = []

    for ordinal, question in enumerate(questions, start=1):
        question_id = str(question["question_id"])
        selected_history = str(question["selected_history"])
        selected_hash = str(question["selected_history_sha256"])
        image_paths, metadata = _render_text(
            selected_history,
            render_dir,
            Path("images") / "selected_8px",
            question_id,
            font_path,
            8,
            8,
        )
        selected_rows.append(
            {
                "schema_version": 1,
                "protocol_version": PROTOCOL_VERSION,
                "question_id": question_id,
                "conversation_id": question["conversation_id"],
                "condition": "selected_evidence_8px",
                "render_kind": "selected_uniform_8px",
                "source_text_sha256": selected_hash,
                "image_paths": image_paths,
                **metadata,
            }
        )
        if int(question["category"]) == 2:
            image_paths, metadata = _render_text(
                selected_history,
                render_dir,
                Path("images") / "temporal_oracle_fixed_24px",
                question_id,
                font_path,
                24,
                8,
            )
            temporal_rows.append(
                {
                    "schema_version": 1,
                    "protocol_version": PROTOCOL_VERSION,
                    "question_id": question_id,
                    "conversation_id": question["conversation_id"],
                    "condition": TEMPORAL_CONDITION,
                    "render_kind": "temporal_oracle_fixed_24px",
                    "source_text_sha256": selected_hash,
                    "image_paths": image_paths,
                    **metadata,
                }
            )
        if ordinal % 20 == 0:
            print(f"Rendered control evidence {ordinal}/{len(questions)}")

    selected_manifest = render_dir / "selected_8px_manifest.jsonl"
    temporal_manifest = render_dir / "temporal_oracle24_fixed_manifest.jsonl"
    write_jsonl(selected_manifest, selected_rows)
    write_jsonl(temporal_manifest, temporal_rows)
    return {
        "selected_8px_manifest": selected_manifest,
        "temporal_oracle24_fixed_manifest": temporal_manifest,
    }


def _bm25_tokens(value: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(value)]


def rank_bm25(
    question: str,
    documents: Sequence[dict[str, Any]],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[tuple[float, dict[str, Any]]]:
    if not documents:
        return []
    tokenized = [_bm25_tokens(str(document["document_text"])) for document in documents]
    query_terms = list(dict.fromkeys(_bm25_tokens(question)))
    average_length = sum(len(tokens) for tokens in tokenized) / len(tokenized)
    document_frequency: Counter[str] = Counter()
    for tokens in tokenized:
        document_frequency.update(set(tokens))
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for document, tokens in zip(documents, tokenized):
        counts = Counter(tokens)
        length = len(tokens)
        score = 0.0
        for term in query_terms:
            frequency = counts[term]
            if frequency == 0:
                continue
            df = document_frequency[term]
            inverse_frequency = math.log(1.0 + (len(documents) - df + 0.5) / (df + 0.5))
            denominator = frequency + k1 * (
                1.0 - b + b * length / max(average_length, 1.0)
            )
            score += inverse_frequency * frequency * (k1 + 1.0) / denominator
        scored.append((score, int(document["document_index"]), document))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [(score, document) for score, _, document in scored]


def pack_bm25_documents(
    question: str,
    documents: Sequence[dict[str, Any]],
    token_count: Callable[[str], int],
    token_budget: int,
) -> tuple[str, dict[str, Any]]:
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    ranked = rank_bm25(question, documents)
    retained: list[dict[str, Any]] = []
    packed = ""
    for score, document in ranked:
        candidate_text = str(document["document_text"])
        candidate = candidate_text if not packed else f"{packed}\n\n{candidate_text}"
        if token_count(candidate) <= token_budget:
            retained.append({"score": round(score, 8), **document})
            packed = candidate
    if not retained:
        raise ValueError("No retrieval document fits the measured token budget")
    return packed, {
        "retrieval_document_count": len(retained),
        "retrieved_document_indices": [row["document_index"] for row in retained],
        "retrieved_dia_ids": [row["dia_id"] for row in retained],
        "bm25_scores": [row["score"] for row in retained],
        "packed_text_tokens": token_count(packed),
        "target_memory_tokens": token_budget,
    }


def validate_prepared_controls(root: Path) -> None:
    data_dir = root / "data" / "generated" / "locomo_controls100"
    render_dir = root / "renders" / "generated" / "locomo_controls100"
    questions = list(read_jsonl(data_dir / "questions.jsonl"))
    selected = list(read_jsonl(render_dir / "selected_8px_manifest.jsonl"))
    temporal = list(read_jsonl(render_dir / "temporal_oracle24_fixed_manifest.jsonl"))
    if len(questions) != 100 or len(selected) != 100 or len(temporal) != 30:
        raise ValueError(
            f"Expected 100 questions, 100 selected renders, and 30 temporal renders; "
            f"found {len(questions)}, {len(selected)}, and {len(temporal)}"
        )
    question_by_id = {str(row["question_id"]): row for row in questions}
    if len(question_by_id) != 100:
        raise ValueError("Control question IDs must be unique")
    for manifest_path, rows in (
        (render_dir / "selected_8px_manifest.jsonl", selected),
        (render_dir / "temporal_oracle24_fixed_manifest.jsonl", temporal),
    ):
        for row in rows:
            question = question_by_id[str(row["question_id"])]
            if row["source_text_sha256"] != question["selected_history_sha256"]:
                raise ValueError(f"Selected-text hash mismatch: {row['question_id']}")
            for relative in row["image_paths"]:
                if not (manifest_path.parent / str(relative)).is_file():
                    raise FileNotFoundError(manifest_path.parent / str(relative))
    if any(
        int(question_by_id[str(row["question_id"])]["category"]) != 2
        for row in temporal
    ):
        raise ValueError("The fixed 24px manifest must contain temporal questions only")


def validate_control_results(
    root: Path, model_ids: Sequence[str], *, require_a100: bool = True
) -> None:
    questions = list(
        read_jsonl(root / "data" / "generated" / "locomo_controls100" / "questions.jsonl")
    )
    validate_prepared_controls(root)
    for model_id in model_ids:
        slug = model_slug(model_id)
        result_path = root / "results" / "locomo_controls100" / f"qwen35_{slug}.jsonl"
        rows = list(read_jsonl(result_path))
        expected = expected_control_keys(questions, model_id)
        actual = {
            (str(row["question_id"]), str(row["condition"]), str(row["model"]))
            for row in rows
        }
        if len(rows) != EXPECTED_CALLS_PER_MODEL or actual != expected:
            raise ValueError(
                f"{model_id} requires {EXPECTED_CALLS_PER_MODEL} unique control rows; "
                f"found {len(rows)}"
            )
        if any(int(row["seed"]) != SEED for row in rows):
            raise ValueError(f"{model_id}: result seed differs from frozen seed {SEED}")
        if require_a100 and any("A100" not in str(row["gpu_name"]).upper() for row in rows):
            raise ValueError(f"{model_id}: all confirmatory rows must come from an A100")


def package_control_results(
    root: Path,
    model_ids: Sequence[str],
    *,
    original_model_ids: Sequence[str] = (),
) -> Path:
    validate_control_results(root, model_ids)
    results_dir = root / "results" / "locomo_controls100"
    control_paths = [
        results_dir / f"qwen35_{model_slug(model_id)}.jsonl" for model_id in model_ids
    ]
    combined_path = results_dir / "combined.jsonl"
    combined: list[dict[str, Any]] = []
    for path in control_paths:
        combined.extend(read_jsonl(path))
    write_jsonl(combined_path, combined)
    score_paths = score_results(combined_path, results_dir / "analysis")

    slugs = "_".join(model_slug(model_id) for model_id in model_ids)
    bundle_path = results_dir / f"locomo_controls_qwen35_{slugs}_bundle.zip"
    candidates = [
        *control_paths,
        combined_path,
        root / "data" / "generated" / "locomo_controls100" / "protocol.json",
        root / "data" / "generated" / "locomo_controls100" / "questions.jsonl",
        root / "data" / "generated" / "locomo_controls100" / "retrieval_documents.jsonl",
        root / "renders" / "generated" / "locomo_controls100" / "selected_8px_manifest.jsonl",
        root / "renders" / "generated" / "locomo_controls100" / "temporal_oracle24_fixed_manifest.jsonl",
        *score_paths.values(),
    ]
    for model_id in model_ids:
        environment = results_dir / f"environment_{model_slug(model_id)}.json"
        if environment.is_file():
            candidates.append(environment)
    for model_id in original_model_ids:
        original = (
            root
            / "results"
            / "locomo_external100"
            / f"qwen35_{model_slug(model_id)}.jsonl"
        )
        if not original.is_file():
            raise FileNotFoundError(original)
        candidates.append(original)

    manifest: list[dict[str, Any]] = []
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        seen: set[Path] = set()
        for path in candidates:
            if path in seen:
                continue
            seen.add(path)
            arcname = path.relative_to(root).as_posix()
            archive.write(path, arcname)
            manifest.append(
                {
                    "path": arcname,
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
        archive.writestr("bundle_manifest.json", json.dumps(manifest, indent=2) + "\n")
    return bundle_path
