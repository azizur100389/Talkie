"""
OCR Noise Ablation Study

Addresses the confound that Talkie was trained on OCR'd text.
Injects character-level noise into the modern twin's inputs at rates
matching estimated OCR error, and re-runs key experiments to determine
whether performance gaps reflect data quality vs. data content.
"""

import json
import random
import string

import numpy as np
from tqdm import tqdm

import config
from experiment1_syntactic import load_blimp, evaluate_blimp
from model_loader import ModelWrapper, load_model


# ── OCR noise injection ───────────────────────────────────────────────

OCR_CONFUSION_PAIRS = {
    "c": ["e", "o"],
    "e": ["c", "o"],
    "o": ["c", "e", "0"],
    "l": ["1", "i", "|"],
    "1": ["l", "i"],
    "i": ["l", "1", "!"],
    "0": ["o", "O"],
    "O": ["0", "o"],
    "m": ["rn", "nn"],
    "n": ["ri", "h"],
    "h": ["b", "n"],
    "b": ["h", "d"],
    "d": ["cl", "b"],
    "rn": ["m"],
    "fi": ["fi"],
    "f": ["t"],
    "t": ["f"],
    "s": ["5"],
    "5": ["s"],
    "S": ["5"],
    "a": ["o", "e"],
    "g": ["q", "9"],
    "q": ["g"],
    "u": ["v", "n"],
    "v": ["u", "y"],
    "w": ["vv"],
    "B": ["8"],
    "8": ["B"],
}


def inject_ocr_noise(text: str, error_rate: float, seed: int = 42) -> str:
    """Inject character-level OCR-like noise into text.

    Args:
        text: Input text.
        error_rate: Probability of corrupting each character.
        seed: Random seed for reproducibility.

    Returns: Corrupted text.
    """
    rng = random.Random(seed)
    chars = list(text)
    result = []

    for ch in chars:
        if rng.random() < error_rate:
            if ch in OCR_CONFUSION_PAIRS:
                replacement = rng.choice(OCR_CONFUSION_PAIRS[ch])
                result.append(replacement)
            elif ch.isalpha():
                if rng.random() < 0.3:
                    continue  # deletion
                elif rng.random() < 0.5:
                    result.append(rng.choice(string.ascii_lowercase))  # substitution
                else:
                    result.append(ch)
                    result.append(rng.choice(string.ascii_lowercase))  # insertion
            else:
                result.append(ch)
        else:
            result.append(ch)

    return "".join(result)


# ── Ablation on BLiMP ─────────────────────────────────────────────────

class NoisyModelWrapper:
    """Wraps a ModelWrapper to inject OCR noise before scoring."""

    def __init__(self, base_model: ModelWrapper, error_rate: float, seed: int = 42):
        self.base = base_model
        self.error_rate = error_rate
        self.seed = seed
        self.name = f"{base_model.name}_ocr{error_rate:.2f}"

    def batch_log_likelihood(self, texts: list[str], batch_size: int = 256):
        noisy = [
            inject_ocr_noise(t, self.error_rate, self.seed + i)
            for i, t in enumerate(texts)
        ]
        return self.base.batch_log_likelihood(noisy, batch_size)


def run_ocr_ablation(error_rates: list[float] | None = None) -> dict:
    """Run BLiMP on the modern twin with various OCR noise levels.

    Compares clean modern performance vs. degraded modern performance
    to isolate the effect of OCR noise from data content.
    """
    if error_rates is None:
        error_rates = config.OCR_ERROR_RATES

    blimp_data = load_blimp()
    model = load_model(config.MODERN_MODEL_ID)
    all_results = {}

    # Clean baseline
    print("\n[OCR Ablation] Clean baseline...")
    clean_results = evaluate_blimp(model, blimp_data)
    all_results["clean"] = {
        k: {"accuracy": v["accuracy"], "total": v["total"]}
        for k, v in clean_results.items()
    }

    # Noisy runs (batched via evaluate_blimp)
    for rate in error_rates:
        print(f"\n[OCR Ablation] Error rate = {rate:.2f}")
        noisy = NoisyModelWrapper(model, rate)
        noisy_results = evaluate_blimp(noisy, blimp_data)
        all_results[f"ocr_{rate:.2f}"] = {
            k: {"accuracy": v["accuracy"], "total": v["total"]}
            for k, v in noisy_results.items()
        }

    del model
    if config.DEVICE.type == "cuda":
        import torch
        torch.cuda.empty_cache()

    out_path = config.RESULTS_DIR / "ocr_ablation.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return all_results


if __name__ == "__main__":
    run_ocr_ablation()