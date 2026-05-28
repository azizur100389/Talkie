"""
Experiment 1 — Syntactic Competence (BLiMP + SyntaxGym)

Evaluates whether core grammatical knowledge emerges regardless of
training distribution by testing both models on minimal pair benchmarks.
"""

import json
from collections import defaultdict

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

import config
from model_loader import ModelWrapper, load_model


# ── BLiMP evaluation ──────────────────────────────────────────────────

BLIMP_CONFIGS = [
    "adjunct_island",
    "anaphor_gender_agreement",
    "anaphor_number_agreement",
    "animate_subject_passive",
    "animate_subject_trans",
    "causative",
    "complex_NP_island",
    "coordinate_structure_constraint_complex_left_branch",
    "coordinate_structure_constraint_object_extraction",
    "determiner_noun_agreement_1",
    "determiner_noun_agreement_2",
    "determiner_noun_agreement_irregular_1",
    "determiner_noun_agreement_irregular_2",
    "determiner_noun_agreement_with_adj_2",
    "determiner_noun_agreement_with_adj_irregular_1",
    "determiner_noun_agreement_with_adj_irregular_2",
    "determiner_noun_agreement_with_adjective_1",
    "distractor_agreement_relational_noun",
    "distractor_agreement_relative_clause",
    "drop_argument",
    "ellipsis_n_bar_1",
    "ellipsis_n_bar_2",
    "existential_there_object_raising",
    "existential_there_quantifiers_1",
    "existential_there_quantifiers_2",
    "existential_there_subject_raising",
    "expletive_it_object_raising",
    "inchoative",
    "intransitive",
    "irregular_past_participle_adjectives",
    "irregular_past_participle_verbs",
    "irregular_plural_subject_verb_agreement_1",
    "irregular_plural_subject_verb_agreement_2",
    "left_branch_island_echo_question",
    "left_branch_island_simple_question",
    "matrix_question_npi_licensor_present",
    "npi_present_1",
    "npi_present_2",
    "only_npi_licensor_present",
    "only_npi_scope",
    "passive_1",
    "passive_2",
    "principle_A_c_command",
    "principle_A_case_1",
    "principle_A_case_2",
    "principle_A_domain_1",
    "principle_A_domain_2",
    "principle_A_domain_3",
    "principle_A_reconstruction",
    "regular_plural_subject_verb_agreement_1",
    "regular_plural_subject_verb_agreement_2",
    "sentential_negation_npi_licensor_present",
    "sentential_negation_npi_scope",
    "sentential_subject_island",
    "superlative_quantifiers_1",
    "superlative_quantifiers_2",
    "tough_vs_raising_1",
    "tough_vs_raising_2",
    "transitive",
    "wh_island",
    "wh_questions_object_gap",
    "wh_questions_subject_gap",
    "wh_questions_subject_gap_long_distance",
    "wh_vs_that_no_gap",
    "wh_vs_that_no_gap_long_distance",
    "wh_vs_that_with_gap",
    "wh_vs_that_with_gap_long_distance",
]


def load_blimp() -> dict[str, list[dict]]:
    """Load all 67 BLiMP subtasks, grouped by linguistic phenomenon."""
    print(f"[exp1] Loading BLiMP dataset ({len(BLIMP_CONFIGS)} configs)...")

    by_phenomenon = defaultdict(list)
    for cfg_name in tqdm(BLIMP_CONFIGS, desc="Loading BLiMP configs"):
        try:
            ds = load_dataset("nyu-mll/blimp", cfg_name, split="train")
        except Exception as e:
            print(f"  WARNING: failed to load {cfg_name}: {e}")
            continue

        phenomenon = _map_to_phenomenon(cfg_name, cfg_name)
        for row in ds:
            by_phenomenon[phenomenon].append({
                "uid": cfg_name,
                "sentence_good": row["sentence_good"],
                "sentence_bad": row["sentence_bad"],
                "linguistics_term": cfg_name,
                "field": row.get("field", ""),
            })

    print(f"[exp1] Loaded {sum(len(v) for v in by_phenomenon.values())} pairs "
          f"across {len(by_phenomenon)} phenomena")
    return dict(by_phenomenon)


CONFIG_TO_PHENOMENON = {
    # anaphor_agreement
    "anaphor_gender_agreement": "anaphor_agreement",
    "anaphor_number_agreement": "anaphor_agreement",
    # argument_structure
    "animate_subject_passive": "argument_structure",
    "animate_subject_trans": "argument_structure",
    "causative": "argument_structure",
    "drop_argument": "argument_structure",
    "inchoative": "argument_structure",
    "intransitive": "argument_structure",
    "passive_1": "argument_structure",
    "passive_2": "argument_structure",
    "transitive": "argument_structure",
    # binding
    "principle_A_c_command": "binding",
    "principle_A_case_1": "binding",
    "principle_A_case_2": "binding",
    "principle_A_domain_1": "binding",
    "principle_A_domain_2": "binding",
    "principle_A_domain_3": "binding",
    "principle_A_reconstruction": "binding",
    # control_raising
    "existential_there_object_raising": "control_raising",
    "existential_there_subject_raising": "control_raising",
    "expletive_it_object_raising": "control_raising",
    "tough_vs_raising_1": "control_raising",
    "tough_vs_raising_2": "control_raising",
    # determiner_noun_agreement
    "determiner_noun_agreement_1": "determiner_noun_agreement",
    "determiner_noun_agreement_2": "determiner_noun_agreement",
    "determiner_noun_agreement_irregular_1": "determiner_noun_agreement",
    "determiner_noun_agreement_irregular_2": "determiner_noun_agreement",
    "determiner_noun_agreement_with_adj_2": "determiner_noun_agreement",
    "determiner_noun_agreement_with_adj_irregular_1": "determiner_noun_agreement",
    "determiner_noun_agreement_with_adj_irregular_2": "determiner_noun_agreement",
    "determiner_noun_agreement_with_adjective_1": "determiner_noun_agreement",
    # ellipsis
    "ellipsis_n_bar_1": "ellipsis",
    "ellipsis_n_bar_2": "ellipsis",
    # filler_gap
    "wh_questions_object_gap": "filler_gap",
    "wh_questions_subject_gap": "filler_gap",
    "wh_questions_subject_gap_long_distance": "filler_gap",
    "wh_vs_that_no_gap": "filler_gap",
    "wh_vs_that_no_gap_long_distance": "filler_gap",
    "wh_vs_that_with_gap": "filler_gap",
    "wh_vs_that_with_gap_long_distance": "filler_gap",
    # irregular_forms
    "irregular_past_participle_adjectives": "irregular_forms",
    "irregular_past_participle_verbs": "irregular_forms",
    # island_effects
    "adjunct_island": "island_effects",
    "complex_NP_island": "island_effects",
    "coordinate_structure_constraint_complex_left_branch": "island_effects",
    "coordinate_structure_constraint_object_extraction": "island_effects",
    "left_branch_island_echo_question": "island_effects",
    "left_branch_island_simple_question": "island_effects",
    "sentential_subject_island": "island_effects",
    "wh_island": "island_effects",
    # npi_licensing
    "matrix_question_npi_licensor_present": "npi_licensing",
    "npi_present_1": "npi_licensing",
    "npi_present_2": "npi_licensing",
    "only_npi_licensor_present": "npi_licensing",
    "only_npi_scope": "npi_licensing",
    "sentential_negation_npi_licensor_present": "npi_licensing",
    "sentential_negation_npi_scope": "npi_licensing",
    # quantifiers
    "existential_there_quantifiers_1": "quantifiers",
    "existential_there_quantifiers_2": "quantifiers",
    "superlative_quantifiers_1": "quantifiers",
    "superlative_quantifiers_2": "quantifiers",
    # subject_verb_agreement
    "distractor_agreement_relational_noun": "subject_verb_agreement",
    "distractor_agreement_relative_clause": "subject_verb_agreement",
    "irregular_plural_subject_verb_agreement_1": "subject_verb_agreement",
    "irregular_plural_subject_verb_agreement_2": "subject_verb_agreement",
    "regular_plural_subject_verb_agreement_1": "subject_verb_agreement",
    "regular_plural_subject_verb_agreement_2": "subject_verb_agreement",
}


def _map_to_phenomenon(linguistics_term: str, uid: str) -> str:
    """Map a BLiMP config name to a high-level phenomenon."""
    if uid in CONFIG_TO_PHENOMENON:
        return CONFIG_TO_PHENOMENON[uid]
    if linguistics_term in CONFIG_TO_PHENOMENON:
        return CONFIG_TO_PHENOMENON[linguistics_term]
    return uid


def evaluate_blimp(
    model: ModelWrapper,
    blimp_data: dict[str, list[dict]],
    batch_size: int = 256,
) -> dict:
    """Run BLiMP evaluation for a single model (batched for speed)."""
    results = {}
    all_correct = 0
    all_total = 0

    for phenomenon, pairs in sorted(blimp_data.items()):
        goods = [p["sentence_good"] for p in pairs]
        bads = [p["sentence_bad"] for p in pairs]

        good_lls = model.batch_log_likelihood(goods, batch_size=batch_size)
        bad_lls = model.batch_log_likelihood(bads, batch_size=batch_size)

        correct = 0
        pair_results = []
        for i, pair in enumerate(pairs):
            ll_g, n_g = good_lls[i]
            ll_b, n_b = bad_lls[i]
            is_correct = ll_g > ll_b
            pair_results.append({
                "uid": pair["uid"],
                "ll_good": ll_g, "ll_bad": ll_b,
                "n_tokens_good": n_g, "n_tokens_bad": n_b,
                "ll_good_norm": ll_g / n_g, "ll_bad_norm": ll_b / n_b,
                "correct": is_correct,
                "correct_norm": (ll_g / n_g) > (ll_b / n_b),
            })
            if is_correct:
                correct += 1

        total = len(pairs)
        acc = correct / total if total > 0 else 0.0
        results[phenomenon] = {
            "accuracy": acc,
            "correct": correct,
            "total": total,
            "pairs": pair_results,
        }
        all_correct += correct
        all_total += total
        print(f"  {phenomenon}: {acc:.3f} ({correct}/{total})")

    results["__aggregate__"] = {
        "accuracy": all_correct / all_total if all_total > 0 else 0.0,
        "correct": all_correct,
        "total": all_total,
    }
    return results


# ── SyntaxGym-style targeted tests ────────────────────────────────────

SYNTAXGYM_TESTS = [
    {
        "name": "SVA_simple",
        "category": "subject_verb_agreement",
        "description": "Simple subject-verb agreement",
        "pairs": [
            ("The boy runs quickly.", "The boy run quickly."),
            ("The dogs bark loudly.", "The dogs barks loudly."),
            ("She walks to the store.", "She walk to the store."),
            ("They have finished.", "They has finished."),
            ("The cat sleeps all day.", "The cat sleep all day."),
        ],
    },
    {
        "name": "SVA_attractor_1",
        "category": "subject_verb_agreement",
        "description": "SVA with 1 attractor noun",
        "pairs": [
            ("The key to the cabinets is here.", "The key to the cabinets are here."),
            ("The boys near the car run fast.", "The boys near the car runs fast."),
            ("The dog behind the cats was barking.", "The dog behind the cats were barking."),
            ("The teacher of the students was absent.", "The teacher of the students were absent."),
            ("The flowers in the vase are wilting.", "The flowers in the vase is wilting."),
        ],
    },
    {
        "name": "SVA_attractor_2",
        "category": "subject_verb_agreement",
        "description": "SVA with 2 attractor nouns",
        "pairs": [
            ("The key to the cabinets near the door is rusty.", "The key to the cabinets near the door are rusty."),
            ("The dog behind the cats near the fence was barking.", "The dog behind the cats near the fence were barking."),
            ("The boys near the car beside the house run fast.", "The boys near the car beside the house runs fast."),
        ],
    },
    {
        "name": "NPI_simple",
        "category": "npi_licensing",
        "description": "Negative polarity item licensing",
        "pairs": [
            ("No student has ever failed this exam.", "Some student has ever failed this exam."),
            ("Nobody has ever been to that island.", "Somebody has ever been to that island."),
            ("The man who no one liked has ever complained.", "The man who someone liked has ever complained."),
        ],
    },
    {
        "name": "filler_gap_simple",
        "category": "filler_gap",
        "description": "Filler-gap dependencies",
        "pairs": [
            ("What did the boy buy at the store?", "What did the boy buy the toy at the store?"),
            ("Who did the teacher praise in class?", "Who did the teacher praise the student in class?"),
        ],
    },
    {
        "name": "reflexive_binding",
        "category": "binding",
        "description": "Reflexive pronoun binding",
        "pairs": [
            ("The boy hurt himself.", "The boy hurt herself."),
            ("The queen defended herself.", "The queen defended himself."),
            ("The men introduced themselves.", "The men introduced himself."),
        ],
    },
    {
        "name": "garden_path",
        "category": "garden_path",
        "description": "Garden-path disambiguation",
        "pairs": [
            ("The horse raced past the barn fell.", "The horse raced past the barn and fell."),
            ("The old man the boats.", "The old men man the boats."),
        ],
    },
    {
        "name": "RC_attachment",
        "category": "relative_clause",
        "description": "Relative clause attachment",
        "pairs": [
            ("The servant of the actress who was on the balcony was crying.",
             "The servant of the actress who were on the balcony was crying."),
        ],
    },
]


def evaluate_syntaxgym(model: ModelWrapper) -> dict:
    """Run SyntaxGym-style targeted syntactic tests."""
    results = {}

    for test in SYNTAXGYM_TESTS:
        correct = 0
        total = 0
        pair_details = []

        for good, bad in test["pairs"]:
            score = model.score_pair(good, bad)
            pair_details.append({
                "good": good,
                "bad": bad,
                **score,
            })
            if score["correct"]:
                correct += 1
            total += 1

        acc = correct / total if total > 0 else 0.0
        results[test["name"]] = {
            "category": test["category"],
            "description": test["description"],
            "accuracy": acc,
            "correct": correct,
            "total": total,
            "pairs": pair_details,
        }
        print(f"  {test['name']}: {acc:.3f} ({correct}/{total})")

    return results


# ── Main runner ────────────────────────────────────────────────────────

def run_experiment1(model_ids: list[str] | None = None) -> dict:
    """Run full Experiment 1 for all specified models."""
    if model_ids is None:
        model_ids = [config.VINTAGE_MODEL_ID, config.MODERN_MODEL_ID]

    blimp_data = load_blimp()
    all_results = {}

    for model_id in model_ids:
        print(f"\n{'='*60}")
        print(f"Evaluating: {config.MODEL_NAMES.get(model_id, model_id)}")
        print(f"{'='*60}")

        model = load_model(model_id)

        print("\n[BLiMP Evaluation]")
        blimp_results = evaluate_blimp(model, blimp_data)

        print("\n[SyntaxGym Evaluation]")
        syntaxgym_results = evaluate_syntaxgym(model)

        all_results[model_id] = {
            "blimp": blimp_results,
            "syntaxgym": syntaxgym_results,
        }

        # Save after each model so partial results survive crashes
        _save_results(all_results)

        # Free GPU memory before loading next model
        del model
        if config.DEVICE.type == "cuda":
            import torch
            torch.cuda.empty_cache()

    return all_results


def _save_results(all_results: dict):
    summary = {}
    for mid, res in all_results.items():
        name = config.MODEL_NAMES.get(mid, mid)
        summary[name] = {
            "blimp": {
                k: {"accuracy": v["accuracy"], "total": v["total"]}
                for k, v in res["blimp"].items()
            },
            "syntaxgym": {
                k: {"accuracy": v["accuracy"], "total": v["total"], "category": v.get("category", "")}
                for k, v in res["syntaxgym"].items()
            },
        }

    out_path = config.RESULTS_DIR / "experiment1_syntactic.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")

    full_path = config.RESULTS_DIR / "experiment1_full.json"
    serialisable = _make_serialisable(all_results)
    with open(full_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"Full results saved to {full_path}")


def _make_serialisable(obj):
    """Convert numpy/torch types to JSON-serialisable Python types."""
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serialisable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, bool):
        return obj
    return obj


if __name__ == "__main__":
    run_experiment1()