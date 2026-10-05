from __future__ import annotations

import argparse
import hashlib
import math
import random
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .common import load_json, project_path, write_jsonl


MEMORY_TYPES = (
    "semantic_decision",
    "numeric_state",
    "high_entropy_identifier",
    "symbol_path",
    "structured_config",
    "code_tool_state",
)

POSITION_BUCKETS = ("early", "middle", "late")
POSITION_RATIOS = {"early": 0.15, "middle": 0.50, "late": 0.85}


@dataclass(frozen=True)
class TargetEvent:
    text: str
    question: str
    answer: str
    scoring: str
    entropy_bits: float | None
    distractors: tuple[str, ...]


def _hex(rng: random.Random, length: int) -> str:
    return "".join(rng.choice("0123456789abcdef") for _ in range(length))


def _mutate(value: str, rng: random.Random) -> str:
    index = rng.randrange(len(value))
    alphabet = "0123456789abcdef"
    replacement = rng.choice(alphabet.replace(value[index].lower(), ""))
    return value[:index] + replacement + value[index + 1 :]


def _semantic_event(case_id: str, rng: random.Random) -> TargetEvent:
    choices = ["blue-green", "canary", "rolling", "shadow"]
    answer = rng.choice(choices)
    distractors = tuple(choice for choice in choices if choice != answer)
    text = f"""USER: For experiment {case_id}, finalize the deployment strategy after the review.
ASSISTANT: Earlier notes considered {distractors[0]} and {distractors[1]}, but they are rejected.
TOOL: decision_registry
case_id = {case_id}
authoritative_strategy = {answer}
status = approved
ASSISTANT: The final deployment strategy for {case_id} is {answer}."""
    return TargetEvent(
        text=text,
        question=f"What final deployment strategy was approved for experiment {case_id}? Return only the strategy name.",
        answer=answer,
        scoring="semantic",
        entropy_bits=None,
        distractors=distractors,
    )


def _numeric_event(case_id: str, rng: random.Random) -> TargetEvent:
    answer_int = rng.randint(3000, 8999)
    distractors = (str(answer_int - 6), str(answer_int + 54), str(int(str(answer_int)[::-1])))
    answer = str(answer_int)
    text = f"""USER: Lock the service port for experiment {case_id}.
ASSISTANT: Candidate ports {distractors[0]}, {distractors[1]}, and {distractors[2]} were tested and rejected.
TOOL: config_writer
case_id = {case_id}
authoritative_deployment_port = {answer}
status = committed
ASSISTANT: Use exactly port {answer} for this experiment."""
    return TargetEvent(
        text=text,
        question=f"Return the exact authoritative_deployment_port for experiment {case_id}.",
        answer=answer,
        scoring="exact",
        entropy_bits=math.log2(6000),
        distractors=distractors,
    )


def _identifier_event(case_id: str, rng: random.Random) -> TargetEvent:
    answer = _hex(rng, 16)
    distractors = tuple(_mutate(answer, rng) for _ in range(3))
    text = f"""USER: Select the rollback commit for experiment {case_id}.
TOOL: git candidates
candidate_sha = {distractors[0]}
candidate_sha = {distractors[1]}
candidate_sha = {distractors[2]}
ASSISTANT: The candidates above are not authoritative.
TOOL: release_registry
case_id = {case_id}
authoritative_rollback_sha = {answer}
status = pinned
ASSISTANT: The rollback SHA has been pinned verbatim."""
    return TargetEvent(
        text=text,
        question=f"Return the exact authoritative_rollback_sha for experiment {case_id}.",
        answer=answer,
        scoring="exact",
        entropy_bits=64.0,
        distractors=distractors,
    )


def _path_event(case_id: str, rng: random.Random) -> TargetEvent:
    version = rng.randint(11, 98)
    shard = _hex(rng, 6)
    answer = f"/srv/checkpoints/{case_id}/v{version}/model-{shard}.safetensors"
    distractors = (
        answer.replace(f"v{version}", f"v{version - 1}"),
        answer.replace(shard, _mutate(shard, rng)),
        f"/srv/archive/{case_id}/v{version}/model-{shard}.safetensors",
    )
    text = f"""USER: Confirm the checkpoint path for experiment {case_id}.
TOOL: artifact_search
candidate_path = {distractors[0]}
candidate_path = {distractors[1]}
candidate_path = {distractors[2]}
ASSISTANT: Ignore the candidate paths; the release manifest is authoritative.
TOOL: release_manifest
case_id = {case_id}
authoritative_checkpoint_path = {answer}
status = verified"""
    return TargetEvent(
        text=text,
        question=f"Return the exact authoritative_checkpoint_path for experiment {case_id}.",
        answer=answer,
        scoring="exact",
        entropy_bits=round(len(answer) * math.log2(40), 2),
        distractors=distractors,
    )


def _config_event(case_id: str, rng: random.Random) -> TargetEvent:
    values = ["3.7e-5", "7.5e-4", "0.03125", "1.25e-3"]
    answer = rng.choice(values)
    distractors = tuple(value for value in values if value != answer)
    text = f"""USER: Freeze the optimizer configuration for experiment {case_id}.
ASSISTANT: Drafts used {distractors[0]}, {distractors[1]}, and {distractors[2]}; none is final.
TOOL: yaml_commit
experiment_id: {case_id}
authoritative_learning_rate: \"{answer}\"
config_status: frozen
ASSISTANT: Preserve the learning-rate string exactly as written."""
    return TargetEvent(
        text=text,
        question=f"Return the exact authoritative_learning_rate string for experiment {case_id}.",
        answer=answer,
        scoring="exact",
        entropy_bits=round(len(answer) * math.log2(16), 2),
        distractors=distractors,
    )


def _code_event(case_id: str, rng: random.Random) -> TargetEvent:
    answer = f"ERR_CACHE_SYNC_{_hex(rng, 8).upper()}"
    distractors = tuple(f"ERR_CACHE_SYNC_{_hex(rng, 8).upper()}" for _ in range(3))
    text = f"""USER: Record the terminal failure code for experiment {case_id}.
TOOL: retry_log
previous_error = {distractors[0]}
previous_error = {distractors[1]}
previous_error = {distractors[2]}
TOOL: final_attempt
case_id = {case_id}
authoritative_terminal_error = {answer}
exit_code = 73
ASSISTANT: The final error symbol must be reproduced exactly."""
    return TargetEvent(
        text=text,
        question=f"Return the exact authoritative_terminal_error for experiment {case_id}.",
        answer=answer,
        scoring="exact",
        entropy_bits=32.0,
        distractors=distractors,
    )


EVENT_BUILDERS: dict[str, Callable[[str, random.Random], TargetEvent]] = {
    "semantic_decision": _semantic_event,
    "numeric_state": _numeric_event,
    "high_entropy_identifier": _identifier_event,
    "symbol_path": _path_event,
    "structured_config": _config_event,
    "code_tool_state": _code_event,
}


def _filler_turn(index: int, rng: random.Random) -> str:
    subjects = [
        "dependency audit",
        "retrieval evaluation",
        "cache invalidation",
        "schema migration",
        "unit-test triage",
        "latency profiling",
        "release documentation",
        "dataset validation",
    ]
    subject = subjects[index % len(subjects)]
    verbs = ["reviewed", "checked", "compared", "replayed", "summarized", "validated"]
    if index % 5 == 0:
        run_id = _hex(rng, 10)
        return f"""USER: Continue the {subject} without changing the approved release decision.
ASSISTANT: I {rng.choice(verbs)} the evidence and kept the current plan unchanged.
TOOL: shell
run_id = {run_id}
artifact_path = /srv/work/runs/batch-{index:04d}/report.txt
exit_code = 0
ASSISTANT: The diagnostic completed normally; this is routine context, not a final configuration record."""
    return f"""USER: Continue the {subject} and report only material issues.
ASSISTANT: I {rng.choice(verbs)} the available notes. The evidence remains internally consistent, no blocking issue was found, and no deployment choice was changed. The next action is to retain the current experimental protocol and continue collecting reproducible measurements.
TOOL: notebook
Observation: the routine check completed and produced no authoritative state update."""


def _episode_seed(global_seed: int, memory_type: str, sample_index: int) -> int:
    digest = hashlib.sha256(f"{global_seed}:{memory_type}:{sample_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def generate_episode(
    memory_type: str,
    sample_index: int,
    global_seed: int,
    target_chars: int,
) -> dict[str, object]:
    seed = _episode_seed(global_seed, memory_type, sample_index)
    rng = random.Random(seed)
    position_bucket = POSITION_BUCKETS[sample_index % len(POSITION_BUCKETS)]
    case_suffix = "".join(rng.choice(string.ascii_lowercase + string.digits) for _ in range(6))
    case_id = f"{memory_type[:3]}-{sample_index:02d}-{case_suffix}"
    event = EVENT_BUILDERS[memory_type](case_id, rng)

    header = (
        "SYSTEM: You are a research-and-coding agent. Preserve prior decisions and exact "
        "technical state. Later questions may require verbatim recovery.\n"
    )
    fillers: list[str] = []
    current_chars = len(header) + len(event.text)
    turn = 0
    while current_chars < target_chars:
        block = _filler_turn(turn, rng)
        fillers.append(block)
        current_chars += len(block) + 2
        turn += 1

    insertion_index = round(POSITION_RATIOS[position_bucket] * len(fillers))
    blocks = fillers[:insertion_index] + [event.text] + fillers[insertion_index:]
    history = header + "\n\n".join(blocks)
    target_offset = history.index(event.text)
    target_fraction = target_offset / max(1, len(history))

    return {
        "schema_version": 1,
        "episode_id": f"{memory_type}-{sample_index:03d}",
        "memory_type": memory_type,
        "sample_index": sample_index,
        "seed": seed,
        "position_bucket": position_bucket,
        "target_char_fraction": round(target_fraction, 4),
        "source_entropy_bits": event.entropy_bits,
        "question": event.question,
        "answer": event.answer,
        "scoring": event.scoring,
        "distractors": list(event.distractors),
        "history": history,
        "history_chars": len(history),
    }


def generate_dataset(config: dict[str, object]) -> list[dict[str, object]]:
    seed = int(config["seed"])
    samples_per_type = int(config["samples_per_type"])
    target_chars = int(config["target_chars"])
    rows: list[dict[str, object]] = []
    for memory_type in MEMORY_TYPES:
        for sample_index in range(samples_per_type):
            rows.append(generate_episode(memory_type, sample_index, seed, target_chars))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic synthetic agent histories.")
    parser.add_argument("--config", default="configs/pilot.json", help="Experiment JSON config.")
    parser.add_argument("--output", help="Override output JSONL path from the config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_json(config_path)
    rows = generate_dataset(config)
    output = project_path(args.output or str(config["output"]))
    write_jsonl(output, rows)
    counts = {kind: sum(row["memory_type"] == kind for row in rows) for kind in MEMORY_TYPES}
    print(f"Wrote {len(rows)} episodes to {output}")
    print("Counts:", counts)


if __name__ == "__main__":
    main()

