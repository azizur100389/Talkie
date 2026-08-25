# Talkie: Temporal Isolation as a Natural Experiment in Knowledge Attribution

Code and data for the paper *"What Does a Language Model Know If It Has Never Seen the Future? Temporal Isolation as a Natural Experiment in Knowledge Attribution"* (EMNLP 2026 submission).

## Overview

We study a 13B-parameter GPT trained exclusively on English text published before 1931 (**Talkie-1930**) alongside its architecture-matched twin trained on modern web text (**Talkie-Web**). The 1930 cutoff creates a hard epistemic boundary that enables controlled experiments separating linguistic competence from memorised knowledge.

**Key findings:**
- Syntactic competence transfers across eras (BLiMP: 76.9% vs 82.0%)
- In-context learning works on tasks structurally absent from pre-1931 text, with comparable learning-rate slopes
- Linear probes decode temporal provenance (98.2%) and epistemic status (96.0%, 3-class) from hidden representations
- A representational dissociation: the model's hidden states encode what it doesn't know, yet it confabulates confidently

## Requirements

- Python 3.10+
- A CUDA GPU with **80 GB VRAM** (A100 recommended). Inference runs in **float32** — see note below.
- Packages listed in [`requirements.txt`](requirements.txt)

```bash
pip install -r requirements.txt
```

The models are downloaded automatically from the Hugging Face Hub on first use
(`talkie-lm/talkie-1930-13b-base`, `talkie-lm/talkie-web-13b-base`). If the
repositories are gated, authenticate first:

```bash
huggingface-cli login
```

> **Inference precision.** This architecture must be run in float32. Its hidden-state
> norms collapse and recover across layers, and bfloat16 rounding errors amplify
> catastrophically, producing degenerate (whitespace-only) output. `config.py` selects
> float32 on CUDA automatically — do not override this.

## Quick Start

The dataset ships pre-built in [`data/temporal_facts.json`](data/temporal_facts.json)
and pre-computed results are in [`results/`](results/), so you can regenerate every
figure without a GPU:

```bash
python generate_figures.py        # writes figures/figure1..7 (pdf + png)
```

To re-run the evaluations from scratch (requires the GPU), use the orchestrator:

```bash
python run_all.py                 # runs Exp 1, 2, 3, OCR ablation, then figures/tables
```

or a single stage:

```bash
python run_all.py --exp 1         # Syntactic competence (BLiMP)   ~2 A100-h
python run_all.py --exp 2         # In-context learning            ~70 A100-h
python run_all.py --exp 3         # Temporal knowledge probing     ~3 A100-h
python run_all.py --exp ocr       # OCR noise ablation             ~1 A100-h
python run_all.py --exp figures   # Figures + LaTeX/CSV tables (analysis.py)
```

## Full Reproduction (including steps not in `run_all.py`)

`run_all.py` covers the three core experiments, the OCR ablation, and figure/table
generation. The following steps are run separately (all are also chained, in order,
in [`run_all_experiments.ipynb`](run_all_experiments.ipynb), the notebook used to
produce the paper results):

```bash
# 1. (optional) Re-mine post-1930 facts that leaked into the pre-1931 corpus.
#    data/temporal_facts.json already ships with all 819 items, including the
#    22 leaked ones (15 documented + 7 recovered by mining), so this step is
#    only needed to rebuild the dataset from scratch.
python mine_leaked_knowledge.py
python expand_dataset.py --leaked-file results/leaked_candidates.json

# 2. Core experiments
python run_all.py --exp 1
python run_all.py --exp 2          # saves results/experiment2_icl.json
python run_all.py --exp 3          # extracts hidden states + linear probes

# 3. Extended probe analysis: MLP probes, permutation baseline, bootstrap CIs,
#    lexical (TF-IDF) baseline, and the behavioural knowledge-boundary test.
python probe_analysis.py

# 4. Side-by-side qualitative completions (Table in the paper)
python qualitative_generations.py

# 5. OCR ablation + all figures
python run_all.py --exp ocr
python generate_figures.py
```

> **Note on ICL seeds.** The paper reports 5–10 seeds per condition (10 seeds for
> Talkie-1930 SST-2/MNLI, 5 for the rest) due to compute limits. The ICL cell in
> `run_all_experiments.ipynb` runs the full 10-seed grid for every condition and
> copies its output to `results/experiment2_icl_final.json`, which the figures and
> paper tables consume.

## Repository Structure

```
run_all.py                   # Orchestrator: --exp 1|2|3|ocr|figures|all
run_all_experiments.ipynb    # End-to-end notebook (the pipeline used for the paper)
config.py                    # Central configuration (model IDs, seeds, hyperparams)
model_loader.py              # Downloads checkpoints from HF; float32 inference

write_dataset.py             # Temporal facts dataset definition (imported, not run)
mine_leaked_knowledge.py     # Identify post-1930 facts leaked into the pre-1931 corpus
expand_dataset.py            # Fold leaked candidates into the dataset

experiment1_syntactic.py     # Exp 1: BLiMP syntactic evaluation
experiment2_icl.py           # Exp 2: in-context learning
experiment3_probing.py       # Exp 3: hidden-state extraction + linear probes
probe_analysis.py            # Extended probes: MLP, permutation, bootstrap, lexical,
                             #   and the behavioural knowledge-boundary test
ocr_ablation.py              # OCR noise ablation on BLiMP
qualitative_generations.py   # Side-by-side model completions

generate_figures.py          # All 7 paper figures (pdf + png)
analysis.py                  # Summary statistics + LaTeX/CSV tables

data/
  temporal_facts.json        # 819-item temporal facts dataset (pre-built)
results/                     # All experiment outputs (JSON, CSV, LaTeX)
figures/                     # Generated figures (pdf + png)
```

### Key result files
- `results/experiment2_icl_final.json` — ICL results (5–10 seeds per condition)
- `results/experiment1_syntactic.json` — BLiMP per-phenomenon accuracy
- `results/experiment3_probing.json` — probe accuracy by layer
- `results/probe_analysis_extended.json` — MLP robustness, permutation baselines, confusion matrices

## Compute Budget

All evaluation was conducted on a single NVIDIA A100 80 GB GPU (~77 A100-hours total).
The models are pre-existing artefacts obtained from Hugging Face; no pretraining was
conducted by the authors.

## License

Code will be released under the MIT License and the temporal facts dataset under
CC-BY-4.0 upon publication.
