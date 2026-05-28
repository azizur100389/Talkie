"""
Extended Probe Analysis — Reviewer-Requested Additions

Generates:
1. Per-class precision/recall/F1 + confusion matrix for Probe C
2. Bag-of-words lexical baseline for all probes (controls for surface features)
3. Bootstrap confidence intervals for BLiMP and ICL results
4. Behavioural knowledge boundary test (factual completion accuracy pre/post 1930)
5. MLP probe robustness check (nonlinear probe to verify linear probe findings)
6. Permutation baseline (shuffled labels to establish proper chance floor)

Run: python probe_analysis.py
Requires: results/ directory with saved hidden states and experiment JSONs.
"""

import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
from experiment3_probing import load_temporal_facts, prepare_probe_datasets


# ── 1. Per-class metrics and confusion matrix for Probe C ─────────────

def probe_c_detailed_metrics():
    """Run Probe C with per-class precision/recall/F1 and confusion matrix.

    Uses saved hidden states from experiment3 run. Reports metrics at the
    best-performing layer (24) and prints full classification report.
    """
    print("=" * 60)
    print("1. PROBE C — Per-Class Metrics & Confusion Matrix")
    print("=" * 60)

    data = load_temporal_facts()
    probe_datasets = prepare_probe_datasets(data)
    texts, labels = probe_datasets["probe_c_knowledge_boundary"]

    class_names = ["should-know (pre-1930)", "should-not-know (post-1930)", "leaked"]

    # Try to load saved hidden states
    hs_path = config.RESULTS_DIR / "hidden_states_Talkie-1930_probe_c_knowledge_boundary.npz"
    if not hs_path.exists():
        print(f"  ERROR: {hs_path} not found. Run experiment3 first.")
        return None

    npz = np.load(str(hs_path))
    saved_labels = npz["labels"]

    results = {}
    # Check layers around the best (20, 24, 28)
    for layer in [16, 20, 24, 28, 32]:
        layer_key = f"layer_{layer}"
        if layer_key not in npz:
            continue

        X = npz[layer_key]

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")),
        ])

        n_folds = min(5, min(np.bincount(saved_labels.astype(int))))
        if n_folds < 2:
            n_folds = 2

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        y_pred = cross_val_predict(pipe, X, saved_labels, cv=skf)

        print(f"\n  Layer {layer}:")
        report = classification_report(
            saved_labels, y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        print(classification_report(
            saved_labels, y_pred,
            target_names=class_names,
            zero_division=0,
        ))

        cm = confusion_matrix(saved_labels, y_pred)
        print("  Confusion Matrix:")
        print(f"  {'':>30} Predicted")
        print(f"  {'':>30} {'should-know':>12} {'shouldnt-know':>14} {'leaked':>8}")
        for i, name in enumerate(class_names):
            row = "  " + f"{name:>28}  "
            row += "  ".join(f"{cm[i, j]:>8}" for j in range(cm.shape[1]))
            print(row)

        results[layer] = {
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "per_class_f1": {
                class_names[i]: report[class_names[i]]["f1-score"]
                for i in range(len(class_names))
            },
        }

    return results


# ── 2. Bag-of-Words lexical baseline ──────────────────────────────────

def lexical_baselines():
    """Train the same probes on TF-IDF features instead of hidden states.

    If the BoW baseline matches the neural probe, then the probe is just
    detecting surface lexical features. If the neural probe substantially
    exceeds BoW, the model encodes something beyond word identity.
    """
    print("\n" + "=" * 60)
    print("2. LEXICAL BASELINES (TF-IDF Bag-of-Words)")
    print("=" * 60)

    data = load_temporal_facts()
    probe_datasets = prepare_probe_datasets(data)

    results = {}

    for probe_name, (texts, labels) in probe_datasets.items():
        print(f"\n  {probe_name} (n={len(texts)}, classes={len(np.unique(labels))})")

        # TF-IDF features
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_tfidf = vectorizer.fit_transform(texts).toarray()

        clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")

        n_classes = len(np.unique(labels))
        min_class = min(np.bincount(labels.astype(int)))
        n_folds = min(5, min_class)
        if n_folds < 2:
            n_folds = 2

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X_tfidf, labels, cv=skf, scoring="accuracy")

        print(f"    TF-IDF baseline: {scores.mean():.3f} +/- {scores.std():.3f}")

        # Compare with best neural probe layer if available
        for model_name in ["Talkie-1930", "Talkie-Web"]:
            hs_path = config.RESULTS_DIR / f"hidden_states_{model_name}_{probe_name}.npz"
            if not hs_path.exists():
                continue

            npz = np.load(str(hs_path))
            best_acc = 0
            best_layer = -1
            for key in npz.files:
                if not key.startswith("layer_"):
                    continue
                layer = int(key.split("_")[1])
                X_hs = npz[key]
                hs_pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")),
                ])
                hs_scores = cross_val_score(
                    hs_pipe, X_hs, labels, cv=skf, scoring="accuracy",
                )
                if hs_scores.mean() > best_acc:
                    best_acc = hs_scores.mean()
                    best_layer = layer

            delta = best_acc - scores.mean()
            print(f"    {model_name} best neural probe (layer {best_layer}): "
                  f"{best_acc:.3f}  (delta = +{delta:.3f})")

        results[probe_name] = {
            "tfidf_mean": float(scores.mean()),
            "tfidf_std": float(scores.std()),
        }

    return results


# ── 3. Bootstrap CIs for BLiMP and ICL ───────────────────────────────

def bootstrap_confidence_intervals():
    """Compute 95% bootstrap CIs for BLiMP aggregate and ICL accuracies."""
    print("\n" + "=" * 60)
    print("3. BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 60)

    results = {}

    # BLiMP CIs
    try:
        blimp = json.load(open(config.RESULTS_DIR / "experiment1_syntactic.json"))
        print("\n  BLiMP aggregate CIs:")

        for model_name, data in blimp.items():
            # Collect per-phenomenon accuracies and total counts
            accs = []
            for phenom, vals in data["blimp"].items():
                if phenom == "__aggregate__":
                    continue
                accs.append(vals["accuracy"])

            accs = np.array(accs)
            rng = np.random.RandomState(42)
            boot_means = []
            for _ in range(10000):
                sample = rng.choice(accs, size=len(accs), replace=True)
                boot_means.append(sample.mean())
            boot_means = np.array(boot_means)
            ci_low = np.percentile(boot_means, 2.5)
            ci_high = np.percentile(boot_means, 97.5)

            print(f"    {model_name}: {accs.mean():.3f} "
                  f"[{ci_low:.3f}, {ci_high:.3f}] 95% CI")
            results[f"blimp_{model_name}"] = {
                "mean": float(accs.mean()),
                "ci_95": [float(ci_low), float(ci_high)],
            }

        # Paired bootstrap for the gap
        v_accs = np.array([blimp["Talkie-1930"]["blimp"][p]["accuracy"]
                           for p in config.BLIMP_PHENOMENA])
        m_accs = np.array([blimp["Talkie-Web"]["blimp"][p]["accuracy"]
                           for p in config.BLIMP_PHENOMENA])
        diffs = m_accs - v_accs
        boot_diffs = []
        rng = np.random.RandomState(42)
        for _ in range(10000):
            idx = rng.choice(len(diffs), size=len(diffs), replace=True)
            boot_diffs.append(diffs[idx].mean())
        boot_diffs = np.array(boot_diffs)
        p_value = np.mean(boot_diffs <= 0)  # one-sided: is modern significantly better?
        print(f"    Gap (Modern - Vintage): {diffs.mean():.3f} "
              f"[{np.percentile(boot_diffs, 2.5):.3f}, {np.percentile(boot_diffs, 97.5):.3f}] "
              f"p={p_value:.4f}")
        results["blimp_gap"] = {
            "mean_diff": float(diffs.mean()),
            "ci_95": [float(np.percentile(boot_diffs, 2.5)),
                      float(np.percentile(boot_diffs, 97.5))],
            "p_value": float(p_value),
        }

    except FileNotFoundError:
        print("  BLiMP results not found — skipping")

    # ICL CIs (from seed variation)
    try:
        icl = json.load(open(config.RESULTS_DIR / "experiment2_icl.json"))
        print("\n  ICL slope CIs (from seed variation):")

        for model_name, tasks in icl.items():
            slopes = []
            for task_name, task_data in tasks.items():
                metrics = task_data.get("metrics", {})
                if "learning_rate_slope" in metrics:
                    slopes.append(metrics["learning_rate_slope"])

            if slopes:
                slopes = np.array(slopes)
                rng = np.random.RandomState(42)
                boot_means = [rng.choice(slopes, len(slopes), replace=True).mean()
                              for _ in range(10000)]
                boot_means = np.array(boot_means)
                print(f"    {model_name} mean slope: {slopes.mean():.4f} "
                      f"[{np.percentile(boot_means, 2.5):.4f}, "
                      f"{np.percentile(boot_means, 97.5):.4f}]")
                results[f"icl_slope_{model_name}"] = {
                    "mean": float(slopes.mean()),
                    "ci_95": [float(np.percentile(boot_means, 2.5)),
                              float(np.percentile(boot_means, 97.5))],
                }

    except FileNotFoundError:
        print("  ICL results not found — skipping")

    return results


# ── 4. Behavioural knowledge boundary test ────────────────────────────

def behavioural_knowledge_test():
    """Test whether Talkie-1930 behaviourally lacks post-1930 knowledge.

    Uses factual completion: give the model a factual prompt and check
    whether its greedy completion contains the correct answer.
    This validates the temporal cutoff as a genuine epistemic barrier.
    """
    print("\n" + "=" * 60)
    print("4. BEHAVIOURAL KNOWLEDGE BOUNDARY TEST")
    print("=" * 60)
    print("  (Requires GPU — run on RunPod)")

    # Knowledge test items: prompt → expected answer
    # Pre-1930 facts the vintage model SHOULD know
    pre_1930_prompts = [
        ("The inventor of the telephone was", "Alexander Graham Bell"),
        ("The Titanic sank in the year", "1912"),
        ("The theory of relativity was proposed by", "Einstein"),
        ("The first modern Olympic Games were held in", "Athens"),
        ("Charles Darwin is famous for the theory of", "evolution"),
        ("The American Civil War ended in", "1865"),
        ("The Eiffel Tower is located in", "Paris"),
        ("Queen Victoria ruled over the", "British"),
        ("The Suez Canal connects the Mediterranean to the", "Red"),
        ("Napoleon was exiled to the island of", "Saint Helena"),
        ("Marie Curie discovered the element", "radium"),
        ("World War I began in the year", "1914"),
        ("The Wright brothers are known for inventing the", "airplane"),
        ("The Russian Revolution took place in", "1917"),
        ("Abraham Lincoln was the president during the", "Civil War"),
    ]

    # Post-1930 facts the vintage model SHOULD NOT know
    post_1930_prompts = [
        ("The first person to walk on the moon was", "Neil Armstrong"),
        ("The Berlin Wall fell in the year", "1989"),
        ("The structure of DNA was discovered by Watson and", "Crick"),
        ("The first atomic bomb was dropped on", "Hiroshima"),
        ("Nelson Mandela became president of", "South Africa"),
        ("The World Wide Web was invented by", "Tim Berners-Lee"),
        ("The Soviet Union collapsed in", "1991"),
        ("The first iPhone was released in", "2007"),
        ("Martin Luther King Jr. is known for his speech I Have a", "Dream"),
        ("The Chernobyl nuclear disaster occurred in", "1986"),
        ("Yuri Gagarin was the first human in", "space"),
        ("The European Union was established in", "1993"),
        ("The Hubble Space Telescope was launched in", "1990"),
        ("GPS technology was developed by the", "military"),
        ("The Cold War was between the United States and the", "Soviet Union"),
    ]

    test_items = {
        "pre_1930": pre_1930_prompts,
        "post_1930": post_1930_prompts,
    }

    # Save the test definition for reproducibility
    out_path = config.RESULTS_DIR / "knowledge_boundary_test.json"
    with open(out_path, "w") as f:
        json.dump({
            "pre_1930": [{"prompt": p, "expected": e} for p, e in pre_1930_prompts],
            "post_1930": [{"prompt": p, "expected": e} for p, e in post_1930_prompts],
        }, f, indent=2)
    print(f"  Test definition saved to {out_path}")

    # Attempt to run if model is available
    try:
        from model_loader import load_model
        import torch

        for model_id in [config.VINTAGE_MODEL_ID, config.MODERN_MODEL_ID]:
            model_name = config.MODEL_NAMES.get(model_id, model_id)
            print(f"\n  Testing: {model_name}")
            model = load_model(model_id)

            model_results = {}
            for period, prompts in test_items.items():
                correct = 0
                for prompt_text, expected in prompts:
                    completion = model.generate(prompt_text, max_new_tokens=20, temperature=0.0)
                    # Check if expected answer appears in first 20 generated tokens
                    hit = expected.lower() in completion.lower()
                    correct += hit
                    mark = "Y" if hit else "N"
                    print(f"    [{mark}] \"{prompt_text}\" -> \"{completion.strip()[:60]}\"")

                acc = correct / len(prompts) if prompts else 0
                print(f"    {period}: {correct}/{len(prompts)} = {acc:.1%}")
                model_results[period] = {
                    "correct": correct,
                    "total": len(prompts),
                    "accuracy": acc,
                }

            # Save results
            results_path = config.RESULTS_DIR / f"knowledge_boundary_{model_name}.json"
            with open(results_path, "w") as f:
                json.dump(model_results, f, indent=2)
            print(f"  Results saved to {results_path}")

            del model
            if config.DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    except Exception as e:
        print(f"  Could not load models (expected on CPU-only machine): {e}")
        print("  Run this script on RunPod to get behavioural results.")


# ── 5. MLP probe robustness ──────────────────────────────────────────

def mlp_probe_robustness():
    """Run a 2-layer MLP probe alongside the linear probe on all probes.

    If both linear and nonlinear probes agree, the finding is robust to
    probe architecture. If the MLP substantially exceeds the linear probe,
    the representation may encode the information nonlinearly.
    """
    print("\n" + "=" * 60)
    print("5. MLP PROBE ROBUSTNESS CHECK")
    print("=" * 60)

    data = load_temporal_facts()
    probe_datasets = prepare_probe_datasets(data)

    results = {}

    for probe_name, (texts, labels) in probe_datasets.items():
        print(f"\n  {probe_name} (n={len(texts)}, classes={len(np.unique(labels))})")

        for model_name in ["Talkie-1930", "Talkie-Web"]:
            if probe_name == "probe_c_knowledge_boundary" and model_name != "Talkie-1930":
                continue

            hs_path = config.RESULTS_DIR / f"hidden_states_{model_name}_{probe_name}.npz"
            if not hs_path.exists():
                continue

            npz = np.load(str(hs_path))
            saved_labels = npz["labels"]

            n_classes = len(np.unique(saved_labels))
            min_class = min(np.bincount(saved_labels.astype(int)))
            n_folds = min(5, min_class)
            if n_folds < 2:
                n_folds = 2
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

            best_linear = 0
            best_mlp = 0
            best_layer_lin = -1
            best_layer_mlp = -1

            for key in sorted(npz.files):
                if not key.startswith("layer_"):
                    continue
                layer = int(key.split("_")[1])
                X = npz[key]

                lin_pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")),
                ])
                mlp_pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", MLPClassifier(
                        hidden_layer_sizes=(256, 128),
                        max_iter=500,
                        early_stopping=True,
                        random_state=42,
                    )),
                ])

                lin_scores = cross_val_score(
                    lin_pipe, X, saved_labels, cv=skf, scoring="accuracy",
                )
                mlp_scores = cross_val_score(
                    mlp_pipe, X, saved_labels, cv=skf, scoring="accuracy",
                )

                if lin_scores.mean() > best_linear:
                    best_linear = lin_scores.mean()
                    best_layer_lin = layer
                if mlp_scores.mean() > best_mlp:
                    best_mlp = mlp_scores.mean()
                    best_layer_mlp = layer

            delta = best_mlp - best_linear
            print(f"    {model_name}:")
            print(f"      Linear (layer {best_layer_lin}): {best_linear:.3f}")
            print(f"      MLP    (layer {best_layer_mlp}): {best_mlp:.3f}")
            print(f"      Delta (MLP - Linear): {delta:+.3f}")

            key = f"{probe_name}_{model_name}"
            results[key] = {
                "linear_acc": float(best_linear),
                "linear_layer": best_layer_lin,
                "mlp_acc": float(best_mlp),
                "mlp_layer": best_layer_mlp,
                "delta": float(delta),
            }

    return results


# ── 6. Permutation baseline ─────────────────────────────────────────

def permutation_baseline():
    """Train probes on shuffled labels to establish a proper chance floor.

    A permutation test is the gold-standard control: if the probe cannot
    exceed permuted-label accuracy, then the real accuracy is uninformative.
    We run 20 permutations per probe and report mean +/- std.
    """
    print("\n" + "=" * 60)
    print("6. PERMUTATION BASELINE (SHUFFLED LABELS)")
    print("=" * 60)

    data = load_temporal_facts()
    probe_datasets = prepare_probe_datasets(data)

    n_permutations = 20
    rng = np.random.RandomState(42)

    results = {}

    for probe_name, (texts, labels) in probe_datasets.items():
        print(f"\n  {probe_name} (n={len(texts)}, classes={len(np.unique(labels))})")

        for model_name in ["Talkie-1930", "Talkie-Web"]:
            if probe_name == "probe_c_knowledge_boundary" and model_name != "Talkie-1930":
                continue

            hs_path = config.RESULTS_DIR / f"hidden_states_{model_name}_{probe_name}.npz"
            if not hs_path.exists():
                continue

            npz = np.load(str(hs_path))
            saved_labels = npz["labels"]

            # Find the best layer from the real probe
            best_layer = -1
            best_real_acc = 0
            for key in npz.files:
                if not key.startswith("layer_"):
                    continue
                layer = int(key.split("_")[1])
                X = npz[key]

                n_classes = len(np.unique(saved_labels))
                min_class = min(np.bincount(saved_labels.astype(int)))
                n_folds = min(5, min_class)
                if n_folds < 2:
                    n_folds = 2
                skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")),
                ])
                scores = cross_val_score(
                    pipe, X, saved_labels, cv=skf, scoring="accuracy",
                )
                if scores.mean() > best_real_acc:
                    best_real_acc = scores.mean()
                    best_layer = layer

            if best_layer < 0:
                continue

            X = npz[f"layer_{best_layer}"]

            perm_accs = []
            for i in range(n_permutations):
                shuffled = rng.permutation(saved_labels)
                n_classes = len(np.unique(shuffled))
                min_class = min(np.bincount(shuffled.astype(int)))
                n_folds = min(5, min_class)
                if n_folds < 2:
                    n_folds = 2
                skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42 + i)
                perm_pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")),
                ])
                scores = cross_val_score(
                    perm_pipe, X, shuffled, cv=skf, scoring="accuracy",
                )
                perm_accs.append(scores.mean())

            perm_accs = np.array(perm_accs)
            p_value = np.mean(perm_accs >= best_real_acc)

            print(f"    {model_name} (layer {best_layer}):")
            print(f"      Real accuracy:       {best_real_acc:.3f}")
            print(f"      Permuted accuracy:   {perm_accs.mean():.3f} +/- {perm_accs.std():.3f}")
            print(f"      Permutation p-value: {p_value:.4f}")

            key = f"{probe_name}_{model_name}"
            results[key] = {
                "real_acc": float(best_real_acc),
                "perm_mean": float(perm_accs.mean()),
                "perm_std": float(perm_accs.std()),
                "p_value": float(p_value),
                "n_permutations": n_permutations,
                "layer": best_layer,
            }

    return results


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_results = {}

    # These run without GPU (use saved hidden states / result JSONs)
    probe_c_results = probe_c_detailed_metrics()
    if probe_c_results:
        all_results["probe_c_detailed"] = {
            str(k): {
                "per_class_f1": v["per_class_f1"],
                "confusion_matrix": v["confusion_matrix"],
            } for k, v in probe_c_results.items()
        }

    lexical_results = lexical_baselines()
    if lexical_results:
        all_results["lexical_baselines"] = lexical_results

    ci_results = bootstrap_confidence_intervals()
    if ci_results:
        all_results["bootstrap_cis"] = ci_results

    mlp_results = mlp_probe_robustness()
    if mlp_results:
        all_results["mlp_robustness"] = mlp_results

    perm_results = permutation_baseline()
    if perm_results:
        all_results["permutation_baseline"] = perm_results

    # This requires GPU
    behavioural_knowledge_test()

    # Save all CPU-side results
    out_path = config.RESULTS_DIR / "probe_analysis_extended.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n{'='*60}")
    print(f"Extended analysis saved to {out_path}")