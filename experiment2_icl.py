"""
Experiment 2 — Contamination-Free In-Context Learning

Measures how much of few-shot ICL performance is attributable to
pretraining data overlap vs. general linguistic capability, by testing
both models on tasks that cannot exist in pre-1931 text.
"""

import json
import random
from collections import defaultdict

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

import config
from model_loader import ModelWrapper, load_model


# ── Prompt templates ───────────────────────────────────────────────────

def _format_sst2_example(text: str, label: int | None = None) -> str:
    label_str = config.ICL_TASKS["sst2"]["label_names"][label] if label is not None else ""
    if label is not None:
        return f"Review: {text}\nSentiment: {label_str}"
    return f"Review: {text}\nSentiment:"


def _format_mnli_example(premise: str, hypothesis: str, label: int | None = None) -> str:
    label_str = config.ICL_TASKS["mnli"]["label_names"][label] if label is not None else ""
    if label is not None:
        return f"Premise: {premise}\nHypothesis: {hypothesis}\nRelation: {label_str}"
    return f"Premise: {premise}\nHypothesis: {hypothesis}\nRelation:"


def _format_tweet_example(text: str, label: int | None = None, task: str = "tweet_sentiment") -> str:
    label_str = config.ICL_TASKS[task]["label_names"][label] if label is not None else ""
    if label is not None:
        return f"Tweet: {text}\nLabel: {label_str}"
    return f"Tweet: {text}\nLabel:"


_UNSET = object()

TASK_FORMATTERS = {
    "sst2": lambda row, label=_UNSET: _format_sst2_example(
        row["sentence"], row.get("label") if label is _UNSET else label
    ),
    "mnli": lambda row, label=_UNSET: _format_mnli_example(
        row["premise"], row["hypothesis"], row.get("label") if label is _UNSET else label
    ),
    "tweet_sentiment": lambda row, label=_UNSET: _format_tweet_example(
        row["text"], row.get("label") if label is _UNSET else label, "tweet_sentiment"
    ),
    "tweet_emotion": lambda row, label=_UNSET: _format_tweet_example(
        row["text"], row.get("label") if label is _UNSET else label, "tweet_emotion"
    ),
}

TASK_INSTRUCTIONS = {
    "sst2": "Classify the sentiment of the following movie review as positive or negative.\n\n",
    "mnli": "Determine the relationship between the premise and hypothesis. Answer entailment, neutral, or contradiction.\n\n",
    "tweet_sentiment": "Classify the sentiment of the following tweet as negative, neutral, or positive.\n\n",
    "tweet_emotion": "Classify the emotion of the following tweet as anger, joy, optimism, or sadness.\n\n",
}


# ── Data loading ───────────────────────────────────────────────────────

def load_task_data(task_name: str) -> tuple[list[dict], list[dict]]:
    """Load a task and split into demonstration pool and evaluation set.

    Returns (demo_pool, eval_set).  Stratified shuffle ensures both splits
    contain all labels in roughly equal proportion.
    """
    task_cfg = config.ICL_TASKS[task_name]
    kwargs = {"split": task_cfg["split"]}
    if "config" in task_cfg:
        kwargs["name"] = task_cfg["config"]

    ds = load_dataset(task_cfg["dataset"], **kwargs)
    rows = list(ds)

    label_key = task_cfg["label_key"]
    by_label = defaultdict(list)
    for row in rows:
        by_label[row[label_key]].append(row)

    rng = random.Random(42)
    demo_pool, eval_set = [], []
    for label, examples in sorted(by_label.items()):
        rng.shuffle(examples)
        mid = len(examples) // 2
        demo_pool.extend(examples[:mid])
        eval_set.extend(examples[mid:])

    rng.shuffle(demo_pool)
    rng.shuffle(eval_set)
    eval_set = eval_set[:config.ICL_MAX_EVAL_SAMPLES]

    labels = [r[label_key] for r in eval_set]
    dist = {l: labels.count(l) for l in set(labels)}
    print(f"    Eval set: {len(eval_set)} samples, label dist: {dist}")

    return demo_pool, eval_set


def select_demonstrations(
    demo_pool: list[dict],
    k: int,
    seed: int,
    label_key: str = "label",
    n_labels: int | None = None,
) -> list[dict]:
    """Select k demonstrations, balanced across labels where possible."""
    rng = random.Random(seed)

    if k == 0:
        return []

    if n_labels is None:
        n_labels = len(set(row[label_key] for row in demo_pool))

    by_label = defaultdict(list)
    for row in demo_pool:
        by_label[row[label_key]].append(row)

    per_label = k // n_labels
    remainder = k % n_labels

    selected = []
    for i, (label, examples) in enumerate(sorted(by_label.items())):
        n = per_label + (1 if i < remainder else 0)
        sampled = rng.sample(examples, min(n, len(examples)))
        selected.extend(sampled)

    rng.shuffle(selected)
    return selected[:k]


# ── Evaluation ─────────────────────────────────────────────────────────

def build_prompt(
    task_name: str,
    demonstrations: list[dict],
    test_example: dict,
) -> str:
    """Build the full prompt: instruction + demonstrations + test query."""
    formatter = TASK_FORMATTERS[task_name]
    instruction = TASK_INSTRUCTIONS[task_name]

    parts = [instruction]
    for demo in demonstrations:
        parts.append(formatter(demo))
        parts.append("")  # blank line separator

    # Test example without label
    parts.append(formatter(test_example, label=None))
    return "\n".join(parts)


def _resolve_label_token_ids(model: ModelWrapper, label_names: list[str]) -> tuple[list[str], list[int], dict[str, list[int]]]:
    """Build all label variants and their first-token IDs.

    Checks with and without leading space (the model predicts " negative"
    after "Sentiment:", not "negative"), plus case variants.
    """
    variants = []
    token_ids = []
    label_to_indices: dict[str, list[int]] = {l: [] for l in label_names}

    for label in label_names:
        seen_ids: set[int] = set()
        for variant_fn in [str.lower, str.capitalize, str.upper]:
            for prefix in [" ", ""]:
                var = prefix + variant_fn(label)
                ids = model.tokeniser.encode(var)
                if ids and ids[0] not in seen_ids:
                    seen_ids.add(ids[0])
                    label_to_indices[label].append(len(variants))
                    variants.append(var)
                    token_ids.append(ids[0])

    return variants, token_ids, label_to_indices


def evaluate_task(
    model: ModelWrapper,
    task_name: str,
    demo_pool: list[dict],
    eval_set: list[dict],
    k: int,
    seed: int,
) -> dict:
    """Evaluate via completion scoring: pick the label whose continuation
    has the highest length-normalised conditional log-likelihood."""
    task_cfg = config.ICL_TASKS[task_name]
    label_names = task_cfg["label_names"]
    label_key = task_cfg["label_key"]
    n_labels = len(label_names)

    demonstrations = select_demonstrations(
        demo_pool, k, seed, label_key, n_labels
    )

    prompts = [build_prompt(task_name, demonstrations, ex) for ex in eval_set]

    all_prompts = []
    all_completions = []
    for prompt in prompts:
        for label in label_names:
            all_prompts.append(prompt)
            all_completions.append(" " + label)

    batch_size = max(2, 128 // max(1, k + 1))
    ll_results = model.batch_conditional_ll(
        all_prompts, all_completions, batch_size=batch_size
    )

    correct = 0
    predictions = []

    for i, example in enumerate(eval_set):
        label_scores = {}
        for j, label in enumerate(label_names):
            cond_ll, n_tokens = ll_results[i * n_labels + j]
            label_scores[label] = cond_ll / max(n_tokens, 1)

        predicted_label = max(label_names, key=lambda l: label_scores[l])
        predicted_idx = label_names.index(predicted_label)
        true_idx = example[label_key]
        is_correct = predicted_idx == true_idx
        if is_correct:
            correct += 1

        predictions.append({
            "true": true_idx,
            "predicted": predicted_idx,
            "correct": is_correct,
            "scores": label_scores,
        })

    accuracy = correct / len(eval_set) if eval_set else 0.0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(eval_set),
        "predictions": predictions,
    }


def compute_icl_metrics(task_results: dict) -> dict:
    """Compute learning rate (slope) and ceiling gap from per-k results."""
    k_values = sorted(int(k) for k in task_results.keys() if k.isdigit())
    accuracies = [task_results[str(k)]["mean_accuracy"] for k in k_values]

    # Learning rate: slope of accuracy vs log2(k+1)
    log_k = np.log2(np.array(k_values) + 1)
    if len(log_k) >= 2:
        slope, intercept = np.polyfit(log_k, accuracies, 1)
    else:
        slope, intercept = 0.0, accuracies[0] if accuracies else 0.0

    return {
        "learning_rate_slope": float(slope),
        "learning_rate_intercept": float(intercept),
        "zero_shot_accuracy": accuracies[0] if accuracies else 0.0,
        "max_k_accuracy": accuracies[-1] if accuracies else 0.0,
        "accuracy_gain": (accuracies[-1] - accuracies[0]) if len(accuracies) >= 2 else 0.0,
    }


# ── Main runner ────────────────────────────────────────────────────────

def run_experiment2(
    model_ids: list[str] | None = None,
    tasks: list[str] | None = None,
) -> dict:
    """Run full Experiment 2 for all models and tasks."""
    if model_ids is None:
        model_ids = [config.VINTAGE_MODEL_ID, config.MODERN_MODEL_ID]
    if tasks is None:
        tasks = list(config.ICL_TASKS.keys())

    # Pre-load all task data
    task_data = {}
    for task_name in tasks:
        print(f"[exp2] Loading {task_name}...")
        task_data[task_name] = load_task_data(task_name)

    all_results = {}

    for model_id in model_ids:
        model_name = config.MODEL_NAMES.get(model_id, model_id)
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")

        model = load_model(model_id)
        model_results = {}

        for task_name in tasks:
            print(f"\n  Task: {task_name}")
            demo_pool, eval_set = task_data[task_name]
            task_results = {}

            for k in config.ICL_K_VALUES:
                seed_results = []
                for seed in config.ICL_SEEDS:
                    result = evaluate_task(model, task_name, demo_pool, eval_set, k, seed)
                    seed_results.append(result["accuracy"])

                task_results[str(k)] = {
                    "mean_accuracy": float(np.mean(seed_results)),
                    "std_accuracy": float(np.std(seed_results)),
                    "per_seed": seed_results,
                }
                print(f"    k={k:>2}: {np.mean(seed_results):.3f} ± {np.std(seed_results):.3f}")

            task_results["metrics"] = compute_icl_metrics(task_results)
            model_results[task_name] = task_results

        all_results[model_name] = model_results

        # Save after each model so partial results survive crashes
        out_path = config.RESULTS_DIR / "experiment2_icl.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=float)
        print(f"\nResults saved to {out_path}")

        del model
        if config.DEVICE.type == "cuda":
            import torch
            torch.cuda.empty_cache()

    return all_results


# ── Statistical tests ─────────────────────────────────────────────────

def paired_bootstrap_test(
    scores_a: list[float],
    scores_b: list[float],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap test for whether two score vectors differ significantly.

    Tests H0: mean(scores_a) == mean(scores_b).
    """
    rng = np.random.RandomState(seed)
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)
    observed_diff = np.mean(scores_a) - np.mean(scores_b)

    n = len(scores_a)
    boot_diffs = []
    combined = np.concatenate([scores_a, scores_b])

    for _ in range(n_bootstrap):
        idx = rng.choice(2 * n, size=2 * n, replace=True)
        boot_a = combined[idx[:n]]
        boot_b = combined[idx[n:]]
        boot_diffs.append(np.mean(boot_a) - np.mean(boot_b))

    boot_diffs = np.array(boot_diffs)
    p_value = np.mean(np.abs(boot_diffs) >= np.abs(observed_diff))

    return {
        "observed_diff": float(observed_diff),
        "p_value": float(p_value),
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
        "ci_95": (float(np.percentile(boot_diffs, 2.5)), float(np.percentile(boot_diffs, 97.5))),
    }


if __name__ == "__main__":
    run_experiment2()