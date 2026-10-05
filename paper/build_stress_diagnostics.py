"""Rebuild the two added diagnostics from unchanged archived answers.

--native-workspace additionally exports compact checks from the larger local
input archives. Normal numerical replay uses the included, hash-bound export;
it does not claim to recreate or execute native model inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from build_camera_ready_assets import rows, single_f1, tex_table

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"
PROTOCOL = ROOT / "stress_protocol"
READERS = ("Qwen2B", "Qwen9B", "GLM")
ARMS = ("full_raw", "original_c2_p4", "efficient_c2_p4", "readable_p4")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def canonical(raw):
    value = raw.strip()
    left, right = "<|begin_of_box|>", "<|end_of_box|>"
    if value.startswith(left) and value.endswith(right) and value.count(left) == value.count(right) == 1:
        value = value[len(left):-len(right)].strip()
    return value


def locomo_f1(prediction, reference, category):
    answer = canonical(prediction)
    if category == 1:
        return statistics.mean(max(single_f1(p.strip(), a.strip()) for p in answer.split(","))
                               for a in reference.split(","))
    return single_f1(answer, reference)


def export_native_checks(workspace):
    prepared = workspace / "final_iclr/checkpoints/cpu_eval_1789993635114781407/prepared"
    all_rows = rows(EVIDENCE / "stress_layout_rows.jsonl")
    native = {}
    for row in all_rows:
        folder = prepared / row["path"]
        request_path, measurement_path = folder / "REQUEST.json", folder / "MEASUREMENT.json"
        assert sha(request_path) == row["request_sha256"]
        assert sha(measurement_path) == row["measurement_sha256"]
        request, measurement = read(request_path), read(measurement_path)
        case = read(PROTOCOL / "cases" / (row["case_id"] + ".json"))
        source_sha = hashlib.sha256(case["memory"].encode("utf-8")).hexdigest()
        assert request["source_sha256"] == source_sha
        assert request["question"] == case["question"]
        native[row["reader"], row["case_id"], row["arm"]] = (request, measurement, row)
    pairs, counts = [], []
    for (reader, case_id, arm), (request, measurement, row) in sorted(native.items()):
        counts.append({"reader": reader, "case_id": case_id, "arm": arm,
                       "request_sha256": row["request_sha256"],
                       "measurement_sha256": row["measurement_sha256"],
                       **{key: measurement[key] for key in ("source_text_tokens", "vision_tokens",
                          "nonvision_input_tokens", "total_input_tokens", "source_to_vision_ratio")}})
        if arm != "original_c2_p4":
            continue
        efficient_request, efficient_measurement, efficient_row = native[reader, case_id, "efficient_c2_p4"]
        keys = ("source_text_tokens", "vision_tokens", "nonvision_input_tokens", "total_input_tokens",
                "image_grid_thw", "input_token_ids", "patch_size", "merge_size")
        assert all(measurement[key] == efficient_measurement[key] for key in keys)
        assert request["source_sha256"] == efficient_request["source_sha256"]
        original_tensor, efficient_tensor = measurement["tensor_inventory"], efficient_measurement["tensor_inventory"]
        assert original_tensor["input_ids"] == efficient_tensor["input_ids"]
        assert original_tensor["pixel_values"]["sha256"] != efficient_tensor["pixel_values"]["sha256"]
        pairs.append({"reader": reader, "case_id": case_id, "source_sha256": request["source_sha256"],
                      "original_measurement_sha256": row["measurement_sha256"],
                      "fullwidth_measurement_sha256": efficient_row["measurement_sha256"],
                      "matched_fields": list(keys), "input_ids": original_tensor["input_ids"],
                      "original_pixels_sha256": original_tensor["pixel_values"]["sha256"],
                      "fullwidth_pixels_sha256": efficient_tensor["pixel_values"]["sha256"],
                      "image_grid_thw": measurement["image_grid_thw"],
                      **{key: measurement[key] for key in keys[:4]}})
    assert len(pairs) == 144 and len(counts) == 576
    put(PROTOCOL / "layout_native_pairs.json", {"role": "Export checked against archived native measurements; not fresh processor execution",
                                              "pairs": pairs, "counts": counts})
    prefixes = []
    for row in read(EVIDENCE / "stress_cap_rows.json"):
        original_path = workspace / "final_iclr/locomo_cap_followup/prepared" / row["path"] / "ORIGINAL.json"
        old = read(original_path)
        assert len(old["generated_token_ids"]) == 96
        assert row["generated_token_ids"][:96] == old["generated_token_ids"]
        prefixes.append({"reader": row["reader"], "case_id": row["case_id"],
                         "original_sha256": sha(original_path),
                         "original_generated_token_ids": old["generated_token_ids"]})
    assert len(prefixes) == 247
    put(PROTOCOL / "cap_prefixes.json", prefixes)


def verify_layout():
    archived = read(EVIDENCE / "stress_layout_analysis.json")["readers"]
    protocol = read(PROTOCOL / "layout_protocol.json")
    selection = read(PROTOCOL / "layout_selection.json")
    assert selection["selected_layout"] == "L3_fullwidth_wordwrap" and selection["verified_rows"] == 36
    cases = {p.stem: read(p) for p in sorted((PROTOCOL / "cases").glob("*.json"))}
    assert len(cases) == 48
    ids = sorted(cases)
    cell_keys = [tuple(cases[i]["factors"][k] for k in ("length", "alphabet", "entropy", "position")) for i in ids]
    assert len(set(cell_keys)) == 24 and all(v == 2 for v in Counter(cell_keys).values())
    assert all(cell_keys[i] == cell_keys[i + 1] for i in range(0, 48, 2))
    source = rows(EVIDENCE / "stress_layout_rows.jsonl")
    lookup = {(r["reader"], r["case_id"], r["arm"]): r for r in source}
    assert len(source) == len(lookup) == 576
    assert set(lookup) == {(reader, case_id, arm) for reader in READERS for case_id in ids for arm in ARMS}
    native = read(PROTOCOL / "layout_native_pairs.json")
    assert len(native["pairs"]) == 144 and len(native["counts"]) == 576
    assert {(n["reader"], n["case_id"], n["arm"]) for n in native["counts"]} == set(lookup)
    assert {(n["reader"], n["case_id"]) for n in native["pairs"]} == {(reader, case_id) for reader in READERS for case_id in ids}
    for n in native["counts"]:
        row = lookup[n["reader"], n["case_id"], n["arm"]]
        assert n["measurement_sha256"] == row["measurement_sha256"]
        assert n["request_sha256"] == row["request_sha256"]
    for n in native["pairs"]:
        old = lookup[n["reader"], n["case_id"], "original_c2_p4"]
        new = lookup[n["reader"], n["case_id"], "efficient_c2_p4"]
        assert n["original_measurement_sha256"] == old["measurement_sha256"]
        assert n["fullwidth_measurement_sha256"] == new["measurement_sha256"]
        assert n["original_pixels_sha256"] != n["fullwidth_pixels_sha256"]
        assert n["source_sha256"] == hashlib.sha256(cases[n["case_id"]]["memory"].encode("utf-8")).hexdigest()
    rng = np.random.default_rng(protocol["statistics_seed"])
    results, contrasts = {}, []
    for reader in READERS:
        vectors = {}
        for arm in ARMS:
            values = []
            for case_id in ids:
                row = lookup[reader, case_id, arm]
                answer = canonical(row["prediction"])
                assert answer == row["scores"]["canonical_answer"]
                correct = int(" ".join(answer.split()) == " ".join(cases[case_id]["reference"].split()))
                assert correct == row["scores"]["canonical"]["exact_match"]
                assert len(row["generated_token_ids"]) == row["generated_token_count"] <= 64
                assert row["cap_hit"] == (row["generated_token_count"] == 64)
                values.append(correct)
            vectors[arm] = np.array(values)
            assert sum(values) == archived[reader]["arms"][arm]["correct"]
        difference = vectors["efficient_c2_p4"] - vectors["original_c2_p4"]
        gains, losses = int(np.sum(difference == 1)), int(np.sum(difference == -1))
        discordant = gains + losses
        p = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(gains, losses) + 1)) / 2**discordant) if discordant else 1.
        weights = rng.binomial(2, .5, size=(50000, 24))
        boot = (weights * difference[0::2] + (2 - weights) * difference[1::2]).sum(axis=1) / 48
        ci = np.quantile(boot, [.025, .975])
        original = archived[reader]["primary"]
        assert p == original["p"] and np.max(np.abs(ci - original["paired_ci95"])) < 1e-12
        result = {"reader": reader, "gains": gains, "losses": losses, "effect": float(difference.mean()),
                  "ci95": list(map(float, ci)), "p": p}
        contrasts.append(result)
        results[reader] = {arm: int(v.sum()) for arm, v in vectors.items()}
    prior = 0.
    for rank, result in enumerate(sorted(contrasts, key=lambda r: r["p"])):
        prior = max(prior, min(1., (3 - rank) * result["p"]))
        result["holm_p"] = prior
        assert prior == archived[result["reader"]]["primary"]["holm_p"]
    labels = ("Full text", "Original layout, $C\\approx2$", "Full-width layout, $C\\approx2$", "Enlarged-image reference")
    body = [[label] + [f"{results[reader][arm]}/48" for reader in READERS] for label, arm in zip(labels, ARMS)]
    (ROOT / "table_stress_layout.tex").write_text(tex_table(["Representation", *READERS], body, "lrrr"), encoding="utf-8")
    (ROOT / "table_stress_layout_contrasts.tex").write_text(tex_table(
        ["Reader", "Gains/losses", "$\\Delta$ (pp)", "95\\% interval (pp)", "Holm $p$"],
        [[r["reader"], f"{r['gains']}/{r['losses']}", f"{100*r['effect']:.2f}",
          "[" + ", ".join(f"{100*v:.2f}" for v in r["ci95"]) + "]",
          f"{r['holm_p']:.8f}".rstrip("0")] for r in contrasts], "llrrr"), encoding="utf-8")
    ranges = {}
    for reader in READERS:
        ranges[reader] = {}
        for arm in ARMS:
            matching = [n for n in native["counts"] if n["reader"] == reader and n["arm"] == arm]
            ranges[reader][arm] = {}
            for field in ("source_text_tokens", "vision_tokens", "total_input_tokens", "source_to_vision_ratio"):
                values = [n[field] for n in matching if n[field] is not None]
                ranges[reader][arm][field] = [min(values), max(values)] if values else None
    return {"rows": 576, "cases": 48, "matched_native_pairs": 144, "counts": results,
            "contrasts": contrasts, "input_ranges": ranges,
            "fresh_processor_reexecution": False, "development_rows_excluded": 36}


def verify_cap():
    baseline = [r for r in rows(EVIDENCE / "followup_scored_rows.jsonl")
                if r["study"] == "LoCoMo" and r["reader"] in READERS[:2] and r["condition"] == "full_optical_c2"]
    assert len(baseline) == 1000
    baseline_lookup = {(r["reader"], r["item_id"]): r for r in baseline}
    updated = read(EVIDENCE / "stress_cap_rows.json")
    updated_lookup = {(r["reader"], r["case_id"]): r for r in updated}
    assert len(updated) == len(updated_lookup) == 247
    prefix_lookup = {(r["reader"], r["case_id"]): r for r in read(PROTOCOL / "cap_prefixes.json")}
    assert prefix_lookup.keys() == updated_lookup.keys()
    expected = read(EVIDENCE / "stress_cap_analysis.json")["readers"]
    rng = np.random.default_rng(2026092106)
    results = {}
    for reader in READERS[:2]:
        base = [r for r in baseline if r["reader"] == reader]
        # An answer can emit EOS on token 96. The frozen rerun selection uses
        # termination=cap, not length alone, and excludes those natural stops.
        capped = {r["item_id"] for r in base if r["termination"] == "cap"}
        assert all(r["generated_token_count"] == 96 for r in base if r["item_id"] in capped)
        assert {case_id for r, case_id in updated_lookup if r == reader} == capped
        assert len(capped) == expected[reader]["fixed_capped_n"]
        new_scores = {}
        for item_id in sorted(capped):
            key = reader, item_id
            row, old = updated_lookup[key], baseline_lookup[key]
            assert row["cluster"] == old["cluster"] and row["arm"] == "full_optical_c2"
            assert len(row["generated_token_ids"]) == row["generated_token_count"] <= 256
            assert row["cap_hit"] == (row["generated_token_count"] == 256)
            assert row["generated_token_ids"][:96] == prefix_lookup[key]["original_generated_token_ids"]
            assert row["prefix96_matches"]
            score = locomo_f1(row["prediction"], old["reference"], old["category"])
            assert abs(score - row["new_f1"]) < 1e-12
            assert abs(row["old_f1"] - old["score"]["canonical"]["f1"]) < 1e-12
            new_scores[item_id] = score
        clusters = sorted({r["cluster"] for r in base})
        assert len(clusters) == 10
        old_means, new_means = [], []
        for cluster in clusters:
            group = [r for r in base if r["cluster"] == cluster]
            old_means.append(statistics.mean(r["score"]["canonical"]["f1"] for r in group))
            new_means.append(statistics.mean(new_scores.get(r["item_id"], r["score"]["canonical"]["f1"]) for r in group))
        difference = np.array(new_means) - old_means
        boot = difference[rng.integers(0, 10, size=(50000, 10))].mean(axis=1)
        ci = np.quantile(boot, [.025, .975])
        exp = expected[reader]
        assert abs(statistics.mean(old_means) - exp["full_500_conversation_macro_old"]) < 1e-12
        assert abs(statistics.mean(new_means) - exp["full_500_conversation_macro_sensitivity"]) < 1e-12
        assert np.max(np.abs(ci - exp["paired_conversation_bootstrap_ci95"])) < 1e-12
        still_capped = sum(updated_lookup[reader, i]["cap_hit"] for i in capped)
        assert still_capped == exp["still_capped_256"]
        results[reader] = {"old": statistics.mean(old_means), "new": statistics.mean(new_means),
                           "change": float(difference.mean()), "ci95": list(map(float, ci)),
                           "reruns": len(capped), "still_capped": still_capped,
                           "prefix_mismatches": 0}
    (ROOT / "table_stress_cap.tex").write_text(tex_table(
        ["Reader", "F1, cap 96", "F1, cap 256", "$\\Delta$ (pp)", "95\\% interval (pp)", "Still capped"],
        [[reader, f"{r['old']:.4f}", f"{r['new']:.4f}", f"{100*r['change']:+.4f}",
          "[" + ", ".join(f"{100*v:+.4f}" for v in r["ci95"]) + "]", str(r["still_capped"])]
         for reader, r in results.items()], "lrrrrr"), encoding="utf-8")
    return {"rows": 247, "baseline_rows_accounted_for": 1000,
            "prefix_mismatches": 0, "readers": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-workspace", type=Path)
    args = parser.parse_args()
    if args.native_workspace:
        export_native_checks(args.native_workspace)
    report = {"status": "passed", "layout": verify_layout(), "cap": verify_cap(),
              "scope": "Separate diagnostic samples and bounded cap sensitivity; original results unchanged",
              "input_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for folder in (PROTOCOL, EVIDENCE)
                               for p in folder.rglob("*") if p.is_file() and (folder == PROTOCOL or p.name.startswith("stress_"))}}
    put(ROOT / "STRESS_DIAGNOSTICS_VERIFICATION.json", report)
    print(json.dumps({"status": "passed", "layout_rows": 576, "matched_native_pairs": 144,
                      "cap_rows": 247, "prefix_mismatches": 0, "layout_counts": report["layout"]["counts"],
                      "cap_results": report["cap"]["readers"]}, indent=2))


if __name__ == "__main__":
    main()
