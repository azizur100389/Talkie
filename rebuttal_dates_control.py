"""
Rebuttal control (Reviewer question, Probe B / temporal provenance):
does Probe B rely on explicit dates in the input?

Partitions the Probe B statements into those containing an explicit year
vs. those with none, and reports 5-fold cross-validated accuracy on each
subset. If date-free statements still classify well above chance, the
temporal signal is not reducible to reading date tokens.

Requires the Probe B hidden states produced by experiment3_probing.py:
  results/hidden_states_<model>_probe_b_temporal.npz

Run:  python rebuttal_dates_control.py
"""
import json
import re

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config

YEAR = re.compile(r"\b(1\d{3}|20\d{2})\b")


def make_pipe():
    # Matches the probe in experiment3_probing.py
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")),
    ])


def probe_b_texts_labels():
    """Rebuild Probe B texts/labels in the same order as experiment3_probing.py."""
    with open(config.DATA_DIR / "temporal_facts.json", encoding="utf-8") as f:
        data = json.load(f)
    pre_true = [it["text"] for it in data["pre_1930_true"]]
    post_true = [it["text"] for it in data["post_1930_true"]]
    texts = pre_true + post_true
    labels = np.array([0] * len(pre_true) + [1] * len(post_true))
    return texts, labels, len(pre_true)


def main():
    texts, labels, n_pre = probe_b_texts_labels()
    has_year = np.array([bool(YEAR.search(t)) for t in texts])
    print(f"Probe B items: {len(texts)}  (pre-1930={n_pre}, post-1930={len(texts)-n_pre})")
    print(f"  with explicit year: {has_year.sum()}   without: {(~has_year).sum()}\n")

    skf = StratifiedKFold(n_splits=config.PROBE_CV_FOLDS, shuffle=True, random_state=42)

    for model, layer in [("Talkie-1930", 16), ("Talkie-Web", 20)]:
        npz = config.RESULTS_DIR / f"hidden_states_{model}_probe_b_temporal.npz"
        d = np.load(npz, allow_pickle=True)
        assert np.array_equal(d["labels"], labels), "label/text order mismatch"
        X = d[f"layer_{layer}"]

        full = cross_val_score(make_pipe(), X, labels, cv=skf, scoring="accuracy")
        preds = cross_val_predict(make_pipe(), X, labels, cv=skf)
        acc_year = (preds[has_year] == labels[has_year]).mean()
        acc_nodate = (preds[~has_year] == labels[~has_year]).mean()

        print(f"{model} (layer {layer}):")
        print(f"  full 5-fold accuracy : {full.mean():.3f} (+/-{full.std():.3f})")
        print(f"  with explicit year   : {acc_year:.3f}  (n={has_year.sum()})")
        print(f"  no date at all       : {acc_nodate:.3f}  (n={(~has_year).sum()})")
        print()


if __name__ == "__main__":
    main()
