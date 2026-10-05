from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ExactSpan:
    kind: str
    text: str
    start: int


PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "key_value",
        re.compile(
            r"(?im)\b[A-Za-z_][A-Za-z0-9_.-]{1,48}\s*[:=]\s*"
            r"(?:\"[^\"\r\n]{1,160}\"|'[^'\r\n]{1,160}'|[^\s,}\]\r\n]{1,160})"
        ),
    ),
    (
        "uuid",
        re.compile(
            r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
            r"[89ab][0-9a-f]{3}-[0-9a-f]{12}\b"
        ),
    ),
    ("url", re.compile(r"https?://[^\s<>\"']{4,180}")),
    (
        "path",
        re.compile(
            r"(?<!\w)(?:[A-Za-z]:\\(?:[^\\\r\n:*?\"<>|]+\\)*[^\\\r\n:*?\"<>|]+|"
            r"/(?:[A-Za-z0-9._-]+/)+[A-Za-z0-9._-]+)"
        ),
    ),
    ("ip_port", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}:\d{2,5}\b")),
    (
        "identifier",
        re.compile(r"\b(?:[0-9a-f]{10,64}|ERR_[A-Z0-9_]{6,80})\b"),
    ),
)


def detect_exact_spans(text: str) -> list[ExactSpan]:
    """Return deterministic, syntax-only exact spans in source order.

    Overlapping matches are deduplicated by their verbatim text. No learned
    importance model or ground-truth answer is consulted.
    """

    matches: list[ExactSpan] = []
    for kind, pattern in PATTERNS:
        matches.extend(ExactSpan(kind, match.group(0), match.start()) for match in pattern.finditer(text))

    matches.sort(key=lambda item: (item.start, -len(item.text), item.kind))
    seen: set[str] = set()
    deduplicated: list[ExactSpan] = []
    for match in matches:
        normalized = match.text.strip()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduplicated.append(ExactSpan(match.kind, normalized, match.start))
    return deduplicated


def format_protection_strip(spans: list[ExactSpan]) -> str:
    lines = ["EXACT STATE — VERBATIM DUPLICATION"]
    lines.extend(f"[{span.kind}] {span.text}" for span in spans)
    return "\n".join(lines)

