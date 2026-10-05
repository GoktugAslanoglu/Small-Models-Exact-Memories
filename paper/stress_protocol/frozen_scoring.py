"""Model-agnostic scoring v2: raw-native and explicitly canonicalized views."""
from collections import Counter
from statistics import mean
import regex as re
import string
from nltk.stem import PorterStemmer

STEMMER = PorterStemmer()
BOX = re.compile(r"^<\|begin_of_box\|>(.*?)<\|end_of_box\|>$", re.DOTALL)
RULER_TASKS = {"niah_single_1", "niah_single_2", "niah_single_3",
               "niah_multikey_3", "niah_multivalue", "vt"}
SINGLE_TASKS = RULER_TASKS - {"niah_multivalue", "vt"}


def canonical_answer(raw):
    if not isinstance(raw, str):
        raise TypeError("Raw output must be a string")
    value = raw.strip()
    m = BOX.fullmatch(value)
    if m and value.count("<|begin_of_box|>") == value.count("<|end_of_box|>") == 1:
        return m.group(1).strip()
    return value


def normalize(value):
    value = value.lower().replace(",", "")
    value = "".join(c for c in value if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the|and)\b", " ", value).split())


def token_f1(prediction, answer):
    p = [STEMMER.stem(t) for t in normalize(prediction).split()]
    a = [STEMMER.stem(t) for t in normalize(answer).split()]
    overlap = sum((Counter(p) & Counter(a)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(a)
    return 2 * precision * recall / (precision + recall)


def locomo_f1(prediction, answer, category):
    if category not in (1, 2, 4):
        raise ValueError("LoCoMo v2 scope is categories 1, 2, 4")
    if category == 1:
        return mean(max(token_f1(p.strip(), a.strip()) for p in prediction.split(","))
                    for a in answer.split(","))
    return token_f1(prediction, answer)


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def exact_scores(prediction, reference):
    p, a = " ".join(prediction.split()), " ".join(reference.split())
    return {"exact_match": float(p == a),
            "literal_exact_match": float(prediction == reference),
            "cer": edit_distance(a, p) / max(1, len(a))}


def ruler_fraction(prediction, references):
    if not isinstance(references, list) or not references or any(
            not isinstance(r, str) or not r for r in references):
        raise ValueError("RULER references must be a nonempty list of nonempty strings")
    return sum(r.lower() in prediction.lower() for r in references) / len(references)


def ruler_native_aggregate(predictions, references):
    if not predictions or len(predictions) != len(references):
        raise ValueError("Prediction/reference lengths must agree and be nonzero")
    return round(mean(ruler_fraction(p, r) for p, r in zip(predictions, references)) * 100, 2)


def score(raw, reference, task, category=None):
    canonical = canonical_answer(raw)
    if task == "locomo":
        if not isinstance(reference, str):
            raise TypeError("LoCoMo reference must be a string")
        return {"canonical_answer": canonical,
                "native": {"f1": locomo_f1(raw, reference, category)},
                "canonical": {"f1": locomo_f1(canonical, reference, category),
                              "normalized_string_em": float(normalize(canonical) == normalize(reference))}}
    if task == "controlled":
        return {"canonical_answer": canonical,
                "native": exact_scores(raw, reference),
                "canonical": exact_scores(canonical, reference)}
    if task == "ruler":
        if not isinstance(reference, dict) or reference.get("task_id") not in RULER_TASKS:
            raise ValueError("RULER reference requires a scoped task_id and answers")
        refs = reference["answers"]
        native = {"answer_containment_fraction": ruler_fraction(raw, refs)}
        canonical_scores = {"answer_containment_fraction": ruler_fraction(canonical, refs)}
        if reference["task_id"] in SINGLE_TASKS:
            if len(refs) != 1:
                raise ValueError("Single-answer RULER task has multiple references")
            native.update(exact_scores(raw, refs[0]))
            canonical_scores.update(exact_scores(canonical, refs[0]))
        return {"canonical_answer": canonical, "native": native, "canonical": canonical_scores}
    raise ValueError(f"Unsupported task: {task}")


# Compatibility with the frozen validator's call signature, never its cleanup.
legacy_score = score
