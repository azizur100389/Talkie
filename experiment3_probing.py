"""
Experiment 3 — Temporal Knowledge Probing

Probes whether LMs develop internal representations that distinguish
"known" from "unknown" factual claims, and whether the knowledge
boundary is detectable in representation space.
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import config
from model_loader import ModelWrapper, load_model


# ── Data loading ───────────────────────────────────────────────────────

def load_temporal_facts() -> dict[str, list[dict]]:
    """Load the temporal factual statements dataset."""
    path = config.DATA_DIR / "temporal_facts.json"
    with open(path) as f:
        data = json.load(f)
    return data


def prepare_probe_datasets(data: dict) -> dict[str, tuple[list[str], np.ndarray]]:
    """Prepare labelled datasets for each probe type.

    Returns dict mapping probe name -> (texts, labels).
    """
    pre_true = [(item["text"], 1) for item in data["pre_1930_true"]]
    pre_false = [(item["text"], 0) for item in data["pre_1930_false"]]
    post_true = [(item["text"], 1) for item in data["post_1930_true"]]
    post_false = [(item["text"], 0) for item in data["post_1930_false"]]
    leaked = [(item["text"], 2) for item in data["leaked_knowledge"]]

    # Probe A: true vs false (all statements)
    probe_a_texts = [t for t, _ in pre_true + pre_false + post_true + post_false]
    probe_a_labels = np.array([l for _, l in pre_true + pre_false + post_true + post_false])

    # Probe B: pre-1930 vs post-1930 (true statements only, to control for veracity)
    probe_b_texts = [t for t, _ in pre_true + post_true]
    probe_b_labels = np.array(
        [0] * len(pre_true) + [1] * len(post_true)
    )

    # Probe C (vintage model only): should-know vs shouldn't-know vs leaked
    #   0 = should know (pre-1930 true)
    #   1 = shouldn't know (post-1930 true)
    #   2 = leaked knowledge
    probe_c_texts = (
        [t for t, _ in pre_true]
        + [t for t, _ in post_true]
        + [item["text"] for item in data["leaked_knowledge"]]
    )
    probe_c_labels = np.array(
        [0] * len(pre_true) + [1] * len(post_true) + [2] * len(data["leaked_knowledge"])
    )

    return {
        "probe_a_veracity": (probe_a_texts, probe_a_labels),
        "probe_b_temporal": (probe_b_texts, probe_b_labels),
        "probe_c_knowledge_boundary": (probe_c_texts, probe_c_labels),
    }


# ── Hidden state extraction ───────────────────────────────────────────

def extract_all_hidden_states(
    model: ModelWrapper,
    texts: list[str],
    layer_indices: list[int] | None = None,
) -> dict[int, np.ndarray]:
    """Extract hidden states for all texts at specified layers.

    Returns: {layer_idx: array of shape (n_texts, hidden_dim)}
    """
    if layer_indices is None:
        layer_indices = config.PROBE_LAYERS

    all_states = {layer: [] for layer in layer_indices}

    for text in tqdm(texts, desc="    Extracting hidden states"):
        states = model.extract_hidden_states(text, layer_indices)
        for layer in layer_indices:
            if layer in states:
                all_states[layer].append(states[layer].numpy())

    return {
        layer: np.stack(vectors)
        for layer, vectors in all_states.items()
        if vectors
    }


# ── Linear probing ────────────────────────────────────────────────────

def train_probe(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    n_folds: int = config.PROBE_CV_FOLDS,
) -> dict:
    """Train a linear probe with cross-validation.

    Uses a Pipeline to ensure StandardScaler is fit only on training
    folds, avoiding train/test leakage.

    Returns accuracy stats (mean, std, per-fold).
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")),
    ])

    n_classes = len(np.unique(labels))
    if n_classes < 2:
        return {"mean": 0.0, "std": 0.0, "per_fold": [], "n_classes": n_classes}

    min_class_count = min(np.bincount(labels.astype(int)))
    effective_folds = min(n_folds, min_class_count)
    if effective_folds < 2:
        pipe.fit(hidden_states, labels)
        train_acc = pipe.score(hidden_states, labels)
        return {"mean": train_acc, "std": 0.0, "per_fold": [train_acc], "n_classes": n_classes}

    skf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, hidden_states, labels, cv=skf, scoring="accuracy")

    return {
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "per_fold": scores.tolist(),
        "n_classes": n_classes,
    }


def run_probes_all_layers(
    hidden_states_by_layer: dict[int, np.ndarray],
    labels: np.ndarray,
) -> dict[int, dict]:
    """Run probing at each layer. Returns {layer: probe_result}."""
    results = {}
    for layer_idx in sorted(hidden_states_by_layer.keys()):
        hs = hidden_states_by_layer[layer_idx]
        result = train_probe(hs, labels)
        results[layer_idx] = result
        print(f"      Layer {layer_idx:>2}: {result['mean']:.3f} ± {result['std']:.3f}")
    return results


# ── Main runner ────────────────────────────────────────────────────────

def run_experiment3(model_ids: list[str] | None = None) -> dict:
    """Run full Experiment 3 for all specified models."""
    if model_ids is None:
        model_ids = [config.VINTAGE_MODEL_ID, config.MODERN_MODEL_ID]

    data = load_temporal_facts()
    probe_datasets = prepare_probe_datasets(data)
    all_results = {}

    for model_id in model_ids:
        model_name = config.MODEL_NAMES.get(model_id, model_id)
        is_vintage = "1930" in model_id
        print(f"\n{'='*60}")
        print(f"Probing: {model_name}")
        print(f"{'='*60}")

        model = load_model(model_id)
        model_results = {}

        probes_to_run = ["probe_a_veracity", "probe_b_temporal"]
        if is_vintage:
            probes_to_run.append("probe_c_knowledge_boundary")

        for probe_name in probes_to_run:
            texts, labels = probe_datasets[probe_name]
            print(f"\n  {probe_name} (n={len(texts)}, classes={len(np.unique(labels))})")

            print("    Extracting hidden states...")
            hidden_states = extract_all_hidden_states(model, texts)

            # Save hidden states for downstream analysis (t-SNE/UMAP)
            hs_path = config.RESULTS_DIR / f"hidden_states_{model_name}_{probe_name}.npz"
            np.savez_compressed(
                str(hs_path),
                **{f"layer_{k}": v for k, v in hidden_states.items()},
                labels=labels,
            )

            print("    Running probes...")
            probe_results = run_probes_all_layers(hidden_states, labels)
            model_results[probe_name] = {
                "layer_results": {str(k): v for k, v in probe_results.items()},
                "n_samples": len(texts),
                "n_classes": len(np.unique(labels)),
                "label_distribution": {
                    str(l): int(c)
                    for l, c in zip(*np.unique(labels, return_counts=True))
                },
            }

        all_results[model_name] = model_results

        # Save after each model so partial results survive crashes
        out_path = config.RESULTS_DIR / "experiment3_probing.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {out_path}")

        del model
        if config.DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    return all_results


if __name__ == "__main__":
    run_experiment3()