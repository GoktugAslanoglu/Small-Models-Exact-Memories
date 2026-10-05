"""Verify immutable public inputs and the documented LoCoMo-only selection."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / "ARTIFACT_MANIFEST.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["immutable_files"].items():
        path = (ROOT / relative).resolve()
        assert path.is_relative_to(ROOT), relative
        assert path.is_file(), f"Missing immutable file: {relative}"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == expected, f"Immutable-file hash differs: {relative}"
    with (ROOT / "paper/evidence/followup_scored_rows.jsonl").open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert len(rows) == 9000
    assert {row["study"] for row in rows} == {"LoCoMo"}
    assert len({(r["reader"], r["condition"], r["item_id"]) for r in rows}) == 9000
    analysis = json.loads((ROOT / "paper/evidence/followup_analysis.json").read_text())
    assert set(analysis) == {"LoCoMo", "statistics_seed", "bootstrap_replicates"}
    assert (ROOT / "LICENSE").is_file()
    print(json.dumps({"status": "passed", "immutable_files": len(manifest["immutable_files"]),
                      "paper_specific_retrieval_rows": len(rows),
                      "generated_files": len(manifest["generated_files"]),
                      "note": "Generated outputs are not immutable inputs; run the reconstruction script to check scores and statistics."}, indent=2))


if __name__ == "__main__":
    main()
