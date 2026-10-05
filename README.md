# Small Models, Exact Memories

**Stress-Testing Optical Memory for Small Multimodal Agents**

Göktuğ Aslanoğlu, Viswanadh Vadlamani, and Şuayp Talha Kocabay.
Accepted at the SLM-Agents Workshop at NeurIPS 2026.

This repository contains the paper's LaTeX source, paper-specific
evaluation evidence, and numerical reconstruction scripts. It studies fixed,
zero-shot multimodal readers, not a new compression architecture or an
end-to-end agent system.

## Paper and experiments

[Read the paper on OpenReview](https://openreview.net/forum?id=J6WelHzPEa).

The reported studies comprise:

- An 18-episode controlled exact-state test across four Qwen3.5 sizes and GLM.
- A frozen 100-question LoCoMo comparison and its diagnostic controls.
- A separate 500-question retrieve-then-render study on the same ten
  conversations, with disjoint question IDs.
- A 48-case held-out identifier-layout diagnostic, selected on 12 development
  cases, with 144 paired native-input checks.
- A 96-to-256-token cap sensitivity using 247 originally capped Qwen outputs.

These studies retain separate samples, protocols, interfaces, and analyses.
Their results are not pooled. The repository excludes unrelated experiment
results from the larger research workspace.

## Reconstruct the reported numbers (CPU only)

Use Python 3.10 or newer. From the repository root:

```sh
python -m pip install -r paper/requirements-analysis.txt
python verify_artifact.py
python paper/build_camera_ready_assets.py
```

The last command reconstructs the figures and tables, recomputes LoCoMo F1
from decoded predictions, checks group coverage and question-ID separation,
and reproduces the reported retrieval bootstrap intervals. It also checks
all 576 held-out identifier responses and the 247 cap extensions, including
paired statistics and original token prefixes. It writes
`paper/NUMERICAL_VERIFICATION.json` and
`paper/STRESS_DIAGNOSTICS_VERIFICATION.json`.

No GPU, model weights, model downloads, or NLTK corpus downloads are needed.
Controlled figures use the archived count tables; original raw outputs and
implementation are provided under `original_study/`.

To run the original CPU unit tests after installing the reconstruction
dependencies:

```sh
python -m pip install -e original_study
python -m unittest discover -s original_study/tests -p "test_*.py" -v
```

The publication check passed all 42 tests. This does not validate a fresh
GPU inference replay.

Generated files may change binary hashes when rebuilt, particularly PDF
figure metadata. `verify_artifact.py` verifies immutable inputs, source code,
and documents; generated numerical/figure/table outputs are identified
separately in the artifact manifest.

## Compile the paper

Upload the contents of `paper/` to Overleaf, choose `main.tex` as the main
document, and use PDFLaTeX. All figures and tables are already included.
Python and a GPU are unnecessary for compilation. Locally:

```sh
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## Repository layout

- `paper/`: final manuscript, figures, tables, reconstruction code, and
  paper-specific evidence.
- `paper/followup_protocol/`: archived retrieval configuration, environment,
  asset pins, scoring/retrieval code, and request bindings.
- `paper/stress_protocol/`: diagnostic protocols, held-out source cases,
  layout selection, and compact checked native-input/token-prefix exports.
- `original_study/`: original experiment code, tests, datasets, render
  manifests, raw results, and protocol inputs extracted from the original
  supplement. Historical scores and runbooks do not override the final
  manuscript's complete-answer scoring and reconstructed displays.
- `third_party/`: upstream license texts and attribution notices.
- `PROVENANCE.json`: source-to-public hashes and documented filtering.
- `ARTIFACT_MANIFEST.json`: immutable-file hashes and generated-file inventory.

The selected 9,000 retrieval result lines are preserved byte-for-byte from
the original export. Its public analysis file retains only the LoCoMo
section and shared statistical settings; `PROVENANCE.json` records that
selection. No original archive was rewritten.

## Reproducibility boundary

This compact artifact supports manuscript compilation and score/table
reconstruction. It is **not a one-command fresh GPU replay of all studies**.
It does not redistribute every rendered image, native processor tensor,
execution journal, model weight, or original input archive. Some historical
runbooks require restoring their original directory layout and reacquiring
upstream assets.

Original Qwen runs did not record exact model revision hashes. The later
diagnostics include pinned snapshots; those pins do not repair the earlier
gap. Historical manifests retain their original planned/sealed status and
paths, including old project labels. Those strings are provenance, not
evidence that an omitted archive is included or that a study was independently
preregistered. The matched-input exports are checked records, not a fresh
processor run.

## Rights

The existing repository MIT license applies to original software only.
Benchmark-derived data, manuscript, figures, fonts, templates, model assets,
and third-party code are not automatically relicensed under MIT.
See [LICENSE_SCOPE.md](LICENSE_SCOPE.md) and
[third-party notices](third_party/NOTICES.md). LoCoMo-derived material retains
its Attribution-NonCommercial 4.0 terms. Model weights are not distributed.

## Citation

The paper is available on [OpenReview](https://openreview.net/forum?id=J6WelHzPEa).
Please cite it using the entry below or [`citation.bib`](citation.bib).
GitHub's citation button uses the paper citation in `CITATION.cff`.
When identifying the software artifact, also specify the commit or release
used. The `v1.0.0` tag preserves the initial artifact; current `main` links to
OpenReview instead of distributing a compiled paper PDF.

```bibtex
@inproceedings{aslanoglu2026smallmodels,
  title     = {{Small Models, Exact Memories}: Stress-Testing Optical Memory for Small Multimodal Agents},
  author    = {Aslano{\u{g}}lu, G{\"o}ktu{\u{g}} and Vadlamani, Viswanadh and Kocabay, {\c{S}}uayp Talha},
  booktitle = {SLM-Agents: Workshop on SLMs for Agentic Systems at NeurIPS},
  year      = {2026},
  url       = {https://openreview.net/forum?id=J6WelHzPEa}
}
```
