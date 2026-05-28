"""Central configuration for all experiments."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "model_cache"

for d in [RESULTS_DIR, FIGURES_DIR, DATA_DIR, CACHE_DIR]:
    d.mkdir(exist_ok=True)

# ── Model IDs ──────────────────────────────────────────────────────────
VINTAGE_MODEL_ID = "talkie-lm/talkie-1930-13b-base"
MODERN_MODEL_ID = "talkie-lm/talkie-web-13b-base"

MODEL_NAMES = {
    VINTAGE_MODEL_ID: "Talkie-1930",
    MODERN_MODEL_ID: "Talkie-Web",
}

# ── Architecture constants ─────────────────────────────────────────────
N_LAYERS = 40
HIDDEN_DIM = 5120
N_HEADS = 40
HEAD_DIM = 128
VOCAB_SIZE = 65536

# ── Experiment 1: BLiMP ────────────────────────────────────────────────
BLIMP_BATCH_SIZE = 16

BLIMP_PHENOMENA = [
    "anaphor_agreement",
    "argument_structure",
    "binding",
    "control_raising",
    "determiner_noun_agreement",
    "ellipsis",
    "filler_gap",
    "irregular_forms",
    "island_effects",
    "npi_licensing",
    "quantifiers",
    "subject_verb_agreement",
]

# ── Experiment 2: ICL ──────────────────────────────────────────────────
ICL_K_VALUES = [0, 1, 2, 4, 8, 16, 32]
ICL_SEEDS = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
ICL_MAX_EVAL_SAMPLES = 500

ICL_TASKS = {
    "sst2": {
        "dataset": "stanfordnlp/sst2",
        "split": "validation",
        "input_key": "sentence",
        "label_key": "label",
        "label_names": ["negative", "positive"],
        "description": "Sentiment analysis (movie reviews)",
    },
    "mnli": {
        "dataset": "nyu-mll/multi_nli",
        "split": "validation_matched",
        "input_keys": ["premise", "hypothesis"],
        "label_key": "label",
        "label_names": ["entailment", "neutral", "contradiction"],
        "description": "Natural language inference",
    },
    "tweet_sentiment": {
        "dataset": "cardiffnlp/tweet_eval",
        "config": "sentiment",
        "split": "test",
        "input_key": "text",
        "label_key": "label",
        "label_names": ["negative", "neutral", "positive"],
        "description": "Tweet sentiment classification",
    },
    "tweet_emotion": {
        "dataset": "cardiffnlp/tweet_eval",
        "config": "emotion",
        "split": "test",
        "input_key": "text",
        "label_key": "label",
        "label_names": ["anger", "joy", "optimism", "sadness"],
        "description": "Tweet emotion classification",
    },
}

# ── Experiment 3: Probing ──────────────────────────────────────────────
PROBE_LAYERS = list(range(0, N_LAYERS + 1, 4))
PROBE_CV_FOLDS = 5
PROBE_MAX_SEQ_LEN = 128

# ── OCR ablation ───────────────────────────────────────────────────────
OCR_ERROR_RATES = [0.01, 0.02, 0.05, 0.10]

# ── Device ─────────────────────────────────────────────────────────────
import torch

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DTYPE = torch.float32  # fp32 required - bf16 breaks logits
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    DTYPE = torch.float16
else:
    DEVICE = torch.device("cpu")
    DTYPE = torch.float32
