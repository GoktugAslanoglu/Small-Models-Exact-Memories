# Anonymous supplement: Small Models, Exact Memories

This archive supports the anonymous SLM-Agents submission. It contains the
frozen protocols, raw model predictions, corrected per-example scores,
processor-budget metadata, analysis outputs, code, and regression tests used
for the paper. It does not contain author identities, model weights, or API
keys.

## Reported response matrix

- Qwen controlled study: 360 responses (four readers x 90 conditions/examples).
- Qwen LoCoMo study and controls: 1,660 responses.
- GLM controlled study: 90 responses.
- GLM LoCoMo second-family diagnostic: 400 responses.
- Qwen short32 full-optical sensitivity: 200 responses.
- Total used by the paper: 2,710 responses.

The controlled pilot contains 18 synthetic histories. LoCoMo-100 is a frozen
100-question diagnostic subset from ten conversations, with 50 single-hop, 30
temporal, and 20 multi-hop questions. It is not official full-benchmark
performance.

## Important scoring note

Archived GLM generations often enclose answers in
<|begin_of_box|>...<|end_of_box|>. The final model-agnostic cleaner removes one
enclosing pair before the ordinary answer normalization. Raw generations and
processor accounting are unchanged. Regression tests also enforce
normalized-string exact match rather than unordered token-set equality.

## Layout

- src/optical_memory: experiment and scoring implementation.
- scripts: final Colab runners and cross-family analysis.
- tests: deterministic unit and regression tests.
- artifacts/qwen_controlled: final 0.8B, 2B, 4B, and 9B controlled outputs.
- artifacts/qwen_locomo: original four-condition and reviewer-control outputs.
- artifacts/qwen_locomo/short32: completed 2B/9B short-answer sensitivity.
- artifacts/glm: controlled and LoCoMo cross-family outputs.
- artifacts/development_exactstrip_v2: the failed post-hoc exploratory
  rendering intervention retained only as a development diagnostic.
- derived: final tables, figures, clustered inference, and integrity reports.
- protocol_inputs: frozen generated datasets and render manifests. PNG pages
  are omitted to reduce the archive; the included runners regenerate them.

## Verification

From the archive root, install the package and run:

    python -m pip install -e ".[locomo,plots]"
    python -m unittest discover -s tests -p "test_*.py" -v
    python scripts/analyze_glm46v_cross_family.py

GPU inference additionally requires the hf extra and the public model
checkpoints named in the paper. The original Qwen environments did not record
exact Hugging Face snapshot revisions; this is disclosed as a limitation. GLM
metadata records its model and processor revisions.

The short32 bundle's protocol metadata was frozen before execution and labels
the run as excluded from claims until completed. The paper integrates it only
after validating all 200 rows and independently reproducing its paired scores.

## Data provenance and terms

The LoCoMo protocol metadata records source URL
https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json,
SHA-256
79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4,
and CC BY-NC 4.0. Controlled histories are synthetic. Qwen3.5 and
GLM-4.6V-Flash remain subject to their model-card licenses and are not
redistributed here.
