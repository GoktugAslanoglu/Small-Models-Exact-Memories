from __future__ import annotations

from .exactspan import ExactSpan, detect_exact_spans


def select_protected_spans_v2(text: str) -> tuple[list[ExactSpan], int]:
    """Select compact syntax-only spans and report the pre-filter count.

    Equal-start matches are already ordered longest first by
    :func:`detect_exact_spans`. A path or identifier wholly contained by a
    key/value expression is therefore removed without consulting the question,
    answer, memory type, or target position.
    """

    detected = detect_exact_spans(text)
    ordered = sorted(detected, key=lambda item: (item.start, -len(item.text), item.kind))
    selected: list[ExactSpan] = []
    for candidate in ordered:
        candidate_end = candidate.start + len(candidate.text)
        if any(
            existing.start <= candidate.start
            and candidate_end <= existing.start + len(existing.text)
            for existing in selected
        ):
            continue
        selected.append(candidate)
    return selected, len(detected)


def format_protection_strip_v2(spans: list[ExactSpan]) -> str:
    """Use raw syntax spans without bracket labels adjacent to exact values."""

    lines = ["EXACT STATE - COPY VALUES VERBATIM"]
    lines.extend(span.text for span in spans)
    return "\n".join(lines)
