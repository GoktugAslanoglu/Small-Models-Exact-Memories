"""Recompute camera-ready displays from preserved evidence; no model inference."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import itertools
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from nltk.stem import PorterStemmer

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"
MODELS = ("Qwen3.5-2B", "Qwen3.5-9B", "GLM-4.6V-Flash")
ALIASES = {"Qwen/Qwen3.5-2B": MODELS[0], "Qwen/Qwen3.5-9B": MODELS[1],
           "zai-org/GLM-4.6V-Flash": MODELS[2]}
CONDITIONS = (
    ("full_raw_text", "Raw text"),
    ("full_optical_8px", "Full optical"),
    ("selected_evidence_8px", "Gold-selected optical"),
    ("bm25_equal_vision_budget_text", "Input-count-matched BM25"),
    ("selected_evidence_text", "Gold-selected text"),
)
EXPECTED = {
    MODELS[0]: (.3837133196, .0270164365, .383489, .41506, .491846),
    MODELS[1]: (.4104149797, .0985308615, .509593, .422219, .374619),
    MODELS[2]: (.4030452971, .1822173287, .4739814108, .4016098, None),
}
STEM = PorterStemmer()


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def clean(value, boxes=True):
    value = re.sub(r"<think>.*?</think>", "", value, flags=re.S | re.I).strip()
    if boxes:
        match = re.fullmatch(r"\s*<\|begin_of_box\|>(.*?)<\|end_of_box\|>\s*", value, re.S | re.I)
        if match:
            value = match.group(1).strip()
    value = value.replace("```json", "").replace("```text", "").replace("```", "").strip()
    try:
        parsed = json.loads(value)
        if isinstance(parsed, dict) and "answer" in parsed:
            value = str(parsed["answer"])
        elif isinstance(parsed, str):
            value = parsed
    except (ValueError, TypeError):
        pass
    value = re.sub(r"(?i)^answer\s*[:=]\s*", "", value).strip()
    # LoCoMo scores complete answers, including multiline lists. Do not apply
    # the controlled literal-answer scorer's first-line truncation here.
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"', "`"}:
        value = value[1:-1]
    return value.strip()


def normalize(value):
    import string
    value = value.lower().replace(",", "")
    value = "".join(c for c in value if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the|and)\b", " ", value).split())


def single_f1(pred, answer):
    p = [STEM.stem(t) for t in normalize(pred).split()]
    a = [STEM.stem(t) for t in normalize(answer).split()]
    if not p or not a:
        return float(p == a)
    n = sum((Counter(p) & Counter(a)).values())
    return 2 * n / (len(p) + len(a)) if n else 0.0


def f1(pred, answer, category, boxes=True):
    pred = clean(pred, boxes)
    if int(category) == 1:
        return statistics.mean(max(single_f1(p.strip(), a.strip()) for p in pred.split(","))
                               for a in answer.split(","))
    return single_f1(pred, answer)


def tex_table(headers, body, columns):
    return ("\\begin{tabular}{" + columns + "}\n\\toprule\n" + " & ".join(headers)
            + " \\\\\n\\midrule\n" + "\n".join(" & ".join(r) + " \\\\" for r in body)
            + "\n\\bottomrule\n\\end{tabular}\n")


def fmt(v):
    return f"{v:.3f}".lstrip("0")


def controlled_figure():
    with (EVIDENCE/"controlled_qwen.csv").open(encoding="utf-8") as h:
        q=list(csv.DictReader(h))
    with (EVIDENCE/"controlled_glm.csv").open(encoding="utf-8") as h:
        g=list(csv.DictReader(h))
    def wilson(k,n=18):
        z=1.959963984540054; p=k/n; den=1+z*z/n
        center=(p+z*z/(2*n))/den
        half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
        return center-half,center+half
    plt.rcParams.update({"font.size":9,"legend.fontsize":8,"xtick.labelsize":8})
    fig,axs=plt.subplots(1,2,figsize=(6.5,3.4),sharey=True,
                         gridspec_kw={"width_ratios":[1.55,1]},constrained_layout=True)
    colors=["#687584","#277DA8","#20968B","#DD713C"]
    xs=np.array([.437,.637,.967,1.746])
    for row,color in zip(q,colors):
        ks=np.array([int(row[f"{f}px"].split('/')[0]) for f in [12,10,8,6]])
        assert row["raw"].startswith("18/18")
        ps=ks/18; bounds=np.array([wilson(k) for k in ks])
        axs[0].errorbar(xs,ps,yerr=np.stack([ps-bounds[:,0],bounds[:,1]-ps]),
                        color=color,marker="o",markersize=4,linewidth=1.4,
                        capsize=2,label=row["model"])
    gg=[r for r in g if r["condition"]!="raw_text"]
    gx=np.array([float(r["mean_source_to_vision_ratio_glm"]) for r in gg])
    gp=np.array([int(r["correct"])/18 for r in gg]);gb=np.array([wilson(int(r["correct"])) for r in gg])
    axs[1].errorbar(gx,gp,yerr=np.stack([gp-gb[:,0],gb[:,1]-gp]),
                    color="#785A96",marker="s",markersize=4,linewidth=1.4,capsize=2)
    axs[0].set_title("Qwen3.5 size sweep",fontsize=10,loc="left")
    axs[1].set_title("GLM-4.6V-Flash",fontsize=10,loc="left")
    axs[0].set_ylabel("Exact-match accuracy")
    for ax,xlim,ticks in [(axs[0],(.35,1.82),xs),(axs[1],(.26,1.32),gx)]:
        ax.set_xlim(xlim);ax.set_ylim(0,1.08)
        ax.axvspan(1,xlim[1],color="#EAF3EF",zorder=-1)
        ax.axvline(1,color="#5E927F",linewidth=.8)
        ax.axhline(1,color="#999999",linewidth=.7,linestyle=":")
        ax.set_xticks(ticks,[f"{x:.2f}\n{f} px" for x,f in zip(ticks,[12,10,8,6])])
        ax.set_xlabel("Within-processor ratio $C$",fontsize=9)
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",linewidth=.5,color="#dddddd")
    axs[0].legend(ncol=2,frameon=False,loc="lower left",
                  handlelength=1.1,columnspacing=.6)
    axs[1].set_xticks(gx,[f"{x:.2f}" for x in gx])
    fig.savefig(ROOT/"figures/figure_2_controlled.pdf",bbox_inches="tight")
    fig.savefig(ROOT/"figures/figure_2_controlled.png",dpi=300,bbox_inches="tight")
    plt.close(fig)


def main():
    controlled_figure()
    inputs = sorted(EVIDENCE.glob("legacy_*.jsonl"))
    raw = [r for p in inputs for r in rows(p)]
    groups = defaultdict(list)
    seen = set()
    for r in raw:
        if r["condition"] not in dict(CONDITIONS):
            continue
        key = (ALIASES[r["model"]], r["condition"], r["question_id"])
        assert key not in seen, key
        seen.add(key)
        r = dict(r)
        r["recomputed_f1"] = f1(r["prediction"], str(r["answer"]), r["category"])
        r["abstains"] = normalize(clean(r["prediction"])).startswith("no information available")
        groups[key[:2]].append(r)
    assert len(seen) == 1400, len(seen)
    summary = []
    qwen_box_changed = 0
    for model in MODELS:
        for j, (condition, _) in enumerate(CONDITIONS):
            if EXPECTED[model][j] is None:
                assert (model, condition) not in groups
                continue
            g = groups[model, condition]
            assert len(g) == 100
            avg = statistics.mean(r["recomputed_f1"] for r in g)
            # Original Qwen used complete answers. The historical GLM cleanup
            # additionally truncated multiline answers; audit that separately.
            if model.startswith("Qwen"):
                assert abs(avg - EXPECTED[model][j]) < 1e-6, (model, condition, avg)
            else:
                assert abs(avg - EXPECTED[model][j]) < .01, (model, condition, avg)
            answered = [r for r in g if not r["abstains"]]
            box_changed = sum(clean(r["prediction"]) != clean(r["prediction"], False) for r in g)
            if model.startswith("Qwen"):
                qwen_box_changed += box_changed
            summary.append({"model": model, "condition": condition, "n": len(g),
                "f1": avg, "historical_f1":EXPECTED[model][j], "abstentions": len(g)-len(answered), "answered_n": len(answered),
                "answer_conditional_f1": statistics.mean(r["recomputed_f1"] for r in answered) if answered else None,
                "cap_hits": sum(r["generated_tokens"] >= r["max_new_tokens"] for r in g),
                "mean_latency_s": statistics.mean(r["latency_seconds"] for r in g),
                "total_generation_s": sum(r["latency_seconds"] for r in g),
                "box_cleanup_changed_n": box_changed})
    assert qwen_box_changed == 0
    rng = np.random.default_rng(20260826)
    contrasts = []
    for model in MODELS:
        for left,right in [("bm25_equal_vision_budget_text","full_optical_8px"),
                           ("full_raw_text","full_optical_8px"),
                           ("selected_evidence_8px","full_optical_8px"),
                           ("full_raw_text","bm25_equal_vision_budget_text")]:
            cg = []
            for condition in [left,right]:
                cs=defaultdict(list)
                for r in groups[model,condition]:
                    cs[r["conversation_id"]].append(r["recomputed_f1"])
                assert len(cs)==10 and all(len(v)==10 for v in cs.values())
                cg.append({k:statistics.mean(v) for k,v in cs.items()})
            assert cg[0].keys()==cg[1].keys()
            d=np.array([cg[0][k]-cg[1][k] for k in sorted(cg[0])])
            effect=float(d.mean())
            boot=d[rng.integers(0,10,size=(50000,10))].mean(axis=1)
            null=np.array([np.mean(d*np.array(s)) for s in itertools.product([-1,1],repeat=10)])
            p=float(np.mean(np.abs(null)>=abs(effect)-1e-12))
            contrasts.append({"model":model,"left":left,"right":right,"effect":effect,
                              "ci95":np.quantile(boot,[.025,.975]).tolist(),"p_two_sided":p})
    (ROOT/"legacy_contrasts.json").write_text(json.dumps(contrasts,indent=2)+"\n",encoding="utf-8")
    with (ROOT / "legacy_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    (ROOT / "legacy_summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    display = {s["model"]:{c:None for c,_ in CONDITIONS} for s in summary}
    for s in summary:
        display[s["model"]][s["condition"]]=s["f1"]
    body = [[m] + [fmt(display[m][c]) if display[m][c] is not None else "---" for c,_ in CONDITIONS] for m in MODELS]
    (ROOT / "table_legacy.tex").write_text(tex_table(
        ["Reader", "Raw", "Full optical", "Gold optical", "BM25", "Gold text"], body, "lrrrrr"), encoding="utf-8")
    latency = [[m] + [f"{next(x for x in summary if x['model']==m and x['condition']==c)['mean_latency_s']:.3f}"
                   if EXPECTED[m][j] is not None else "---" for j,(c,_) in enumerate(CONDITIONS)] for m in MODELS]
    (ROOT / "table_latency.tex").write_text(tex_table(
        ["Reader", "Raw", "Full optical", "Gold optical", "BM25", "Gold text"], latency, "lrrrrr"), encoding="utf-8")
    diag = [s for s in summary if s["condition"] == "full_optical_8px"]
    (ROOT / "table_output_diagnostics.tex").write_text(tex_table(
        ["Reader", "F1 (all)", "Abstentions", "Answering $n$", "F1 (answering)", "Cap hits"],
        [[s["model"], fmt(s["f1"]), str(s["abstentions"]), str(s["answered_n"]),
          fmt(s["answer_conditional_f1"]), str(s["cap_hits"])] for s in diag], "lrrrrr"), encoding="utf-8")
    plt.rcParams.update({"font.size":9,"legend.fontsize":7.5,"xtick.labelsize":8.5})
    fig, ax = plt.subplots(figsize=(6.5,3.35), constrained_layout=True)
    x = np.arange(3); width=.155
    colors = ["#4C78A8", "#E45756", "#72B7B2", "#E9BC39", "#9A7CB5"]
    hatches = ["", "///", "..", "xx", "\\\\"]
    for j, ((_, label), color, hatch) in enumerate(zip(CONDITIONS, colors, hatches)):
        for i, model in enumerate(MODELS):
            v = display[model][CONDITIONS[j][0]]
            pos = x[i] + (j-2)*width
            if v is None:
                ax.text(pos, .025, "N/A", rotation=90, ha="center", va="bottom", fontsize=7, color="#666666")
                continue
            ax.bar(pos, v, width, color=color, edgecolor="#333333", linewidth=.45,
                   hatch=hatch, label=label if i==0 else None, zorder=3)
            ax.text(pos, v+.012, fmt(v), ha="center", va="bottom", fontsize=7.4, rotation=90)
    ax.set(xticks=x, xticklabels=MODELS, ylim=(0,.59), ylabel="LoCoMo-100 token F1")
    ax.grid(axis="y", color="#dddddd", linewidth=.6, zorder=0)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(.5,1.23), ncol=3, frameon=False)
    fig.savefig(ROOT/"figures/figure_1_locomo_cross_family.pdf", bbox_inches="tight")
    fig.savefig(ROOT/"figures/figure_1_locomo_cross_family.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    authoritative = json.loads((EVIDENCE/"followup_analysis.json").read_text(encoding="utf-8"))["LoCoMo"]
    follow = [r for r in rows(EVIDENCE/"followup_scored_rows.jsonl") if r["study"]=="LoCoMo"]
    assert len(follow)==9000
    fg = defaultdict(list); fs = set()
    for r in follow:
        key = (r["reader"], r["condition"], r["item_id"])
        assert key not in fs, key
        fs.add(key); fg[key[:2]].append(r)
        rawanswer=r["prediction"].strip()
        boxed=re.fullmatch(r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>",rawanswer,re.S)
        canonical=boxed.group(1).strip() if boxed and rawanswer.count("<|begin_of_box|>")==rawanswer.count("<|end_of_box|>")==1 else rawanswer
        assert canonical==r["score"]["canonical_answer"]
        val=(statistics.mean(max(single_f1(p.strip(),a.strip()) for p in canonical.split(','))
                             for a in r["reference"].split(',')) if r["category"]==1 else single_f1(canonical,r["reference"]))
        assert abs(val-r["score"]["canonical"]["f1"])<1e-12
    assert {r["conversation_id"] for r in raw}=={r["cluster"] for r in follow}
    assert not ({r["question_id"] for r in raw} & {r["item_id"] for r in follow})
    fsummary = []
    for (reader, arm), g in sorted(fg.items()):
        assert len(g)==500
        clusters = defaultdict(list)
        for r in g:
            clusters[r["cluster"]].append(r["score"]["canonical"]["f1"])
        assert len(clusters)==10
        macro = statistics.mean(statistics.mean(v) for v in clusters.values())
        micro = statistics.mean(r["score"]["canonical"]["f1"] for r in g)
        a = authoritative[reader]["arms"][arm]
        assert abs(macro-a["conversation_macro_f1"]) < 1e-12
        assert abs(micro-a["question_micro_f1"]) < 1e-12
        fsummary.append({"reader":reader,"condition":arm,"n":500,"clusters":10,"macro_f1":macro,"micro_f1":micro,
                         "mean_input_tokens":a["mean_input_tokens"],"cap_hits":a["cap"]})
    with (ROOT/"followup_summary.csv").open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=list(fsummary[0]));w.writeheader();w.writerows(fsummary)
    followarms = [("Full text","full_raw"),("Full optical ($C\\approx2$)","full_optical_c2"),
                 ("BM25-selected text","retrieved_text"),("BM25-selected optical, 8 px","retrieved_optical_8px"),
                 ("Text at full-optical budget","retrieved_text_matched"),
                 ("Text at selected-optical budget","retrieved_text_matched_selected")]
    readers = ["Qwen2B","Qwen9B","GLM"]
    clusters=sorted({r["cluster"] for r in follow})
    weights=np.random.default_rng(2026090505).multinomial(10,[.1]*10,size=50000)/10
    for reader in readers:
        for name,left,right in [("H1","retrieved_optical_8px","full_optical_c2"),
                                ("H2","retrieved_optical_8px","retrieved_text")]:
            dif=np.array([statistics.mean(r["score"]["canonical"]["f1"] for r in fg[reader,left] if r["cluster"]==c)
                          -statistics.mean(r["score"]["canonical"]["f1"] for r in fg[reader,right] if r["cluster"]==c)
                          for c in clusters])
            ci=np.quantile(weights@dif,[.025,.975])
            assert abs(float(dif.mean())-authoritative[reader]["contrasts"][name]["effect"])<1e-12
            assert np.max(np.abs(ci-authoritative[reader]["contrasts"][name]["ci95"]))<1e-12
    (ROOT/"table_followup.tex").write_text(tex_table(
        ["Representation","Qwen2B","Qwen9B","GLM"],
        [[label]+[fmt(authoritative[r]["arms"][a]["conversation_macro_f1"]) for r in readers]
         for label,a in followarms],"lrrr"),encoding="utf-8")
    (ROOT/"table_followup_contrasts.tex").write_text(tex_table(
        ["Reader","Contrast","$\\Delta$ F1","95\\% cluster interval"],
        [[reader,label,fmt(authoritative[reader]["contrasts"][arm]["effect"]),
          "["+", ".join(fmt(v) for v in authoritative[reader]["contrasts"][arm]["ci95"])+"]"]
         for reader in readers for arm,label in [("H1","Retrieved optical $-$ full optical"),
                                                 ("H2","Retrieved optical $-$ same-evidence text")]],"llrr"),encoding="utf-8")
    report = {"legacy_display_rows":1400,"followup_rows":9000,"qwen_box_cleanup_changes":qwen_box_changed,
              "glm_full_answer_audit":[s for s in summary if s["model"].startswith("GLM")],
              "verified_legacy_groups":len(summary),"verified_followup_groups":len(fsummary),
              "method":"Legacy and follow-up F1 independently recomputed from decoded answers; aggregates and follow-up bootstrap intervals reconciled to frozen analyses. Original and follow-up conversation sets verified identical.",
              "input_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in EVIDENCE.glob("*.*")},
              "output_diagnostics":diag}
    (ROOT/"NUMERICAL_VERIFICATION.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ["input_sha256"]},indent=2))


if __name__ == "__main__":
    main()
    import subprocess
    import sys
    subprocess.run([sys.executable, str(ROOT / "build_stress_diagnostics.py")], check=True)
