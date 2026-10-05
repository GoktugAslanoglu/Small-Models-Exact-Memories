"""Frozen BM25 plus a distinct complete-ranked-prefix budget control."""
from iclr.retrieval import canonical, rank, documents, select, sha_text, pack_matched


def pack_prefix(question, docs, processed_chat_count, target):
    """Evaluate every prefix: tokenizer counts need not be monotone.

    The counter must count the full processed chat, including the question and
    template overhead. It is supplied by the pinned processor adapter, never a
    text-token proxy. Nothing here accesses answers or gold annotations.
    """
    if type(target) is not int or target < 1:
        raise ValueError("Target must be a positive processed-input count")
    ranked = rank(question, docs)
    candidates = []
    best = None
    for k in range(len(ranked) + 1):
        evidence = canonical(ranked[:k])
        count = processed_chat_count(evidence)
        if type(count) is not int or count < 1:
            raise ValueError("Processor counter must return a positive integer")
        candidates.append({"prefix_length": k, "total_input_tokens": count})
        if count <= target:
            best = (k, evidence, count)
    if candidates[0]["total_input_tokens"] > target:
        return {"status": "infeasible", "reason": "empty_envelope_exceeds_target",
                "target": target, "candidate_counts": candidates}
    k, evidence, count = best
    selected = sorted(ranked[:k], key=lambda d: d["index"])
    return {"status": "planned", "rule": "longest_complete_ranked_prefix",
            "prefix_length": k, "evidence": evidence, "evidence_sha256": sha_text(evidence),
            "selected_turn_ids": [d["turn_id"] for d in selected],
            "total_input_tokens": count, "target": target, "unused_budget": target - count,
            "candidate_counts": candidates}


def support_coverage(selected_turn_ids, annotated_support, known_turn_ids=None):
    """Evaluation-only diagnostic; empty support is unknown, not perfect recall."""
    selected, support = set(selected_turn_ids), set(annotated_support)
    hits = sorted(selected & support)
    unresolved = sorted(support - set(known_turn_ids)) if known_turn_ids is not None else []
    valid = bool(support) and not unresolved
    return {"annotated_support_count": len(support), "retrieved_support_count": len(hits),
            "support_recall": len(hits) / len(support) if valid else None,
            "all_annotated_support": support <= selected if valid else None,
            "unresolved_support_ids": unresolved,
            "annotation_validation": "unresolved" if unresolved else ("resolved" if known_turn_ids is not None else "not_checked"),
            "matched_support_ids": hits,
            "interpretation": "annotation coverage; annotations may be incomplete"}
