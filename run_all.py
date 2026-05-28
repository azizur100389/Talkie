"""
Main experiment runner.

Orchestrates all three experiments + ablation + figure generation.
Run the full pipeline:
    python run_all.py

Or run individual experiments:
    python run_all.py --exp 1        # Syntactic competence (BLiMP)
    python run_all.py --exp 2        # In-context learning
    python run_all.py --exp 3        # Temporal knowledge probing
    python run_all.py --exp ocr      # OCR noise ablation
    python run_all.py --exp figures  # Generate all figures/tables
"""

import argparse
import time

import config


def main():
    parser = argparse.ArgumentParser(description="Talkie evaluation pipeline")
    parser.add_argument(
        "--exp",
        type=str,
        default="all",
        choices=["all", "1", "2", "3", "ocr", "figures"],
        help="Which experiment to run (default: all)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model IDs to evaluate (default: both vintage and modern)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="ICL task names for exp 2 (default: all)",
    )
    args = parser.parse_args()

    model_ids = args.models or [config.VINTAGE_MODEL_ID, config.MODERN_MODEL_ID]
    start = time.time()

    print(f"Device: {config.DEVICE}")
    print(f"Dtype: {config.DTYPE}")
    print(f"Models: {model_ids}")

    if args.exp in ("all", "1"):
        print("\n" + "=" * 60)
        print("EXPERIMENT 1: Syntactic Competence")
        print("=" * 60)
        from experiment1_syntactic import run_experiment1
        run_experiment1(model_ids)

    if args.exp in ("all", "2"):
        print("\n" + "=" * 60)
        print("EXPERIMENT 2: In-Context Learning")
        print("=" * 60)
        from experiment2_icl import run_experiment2
        run_experiment2(model_ids, args.tasks)

    if args.exp in ("all", "3"):
        print("\n" + "=" * 60)
        print("EXPERIMENT 3: Temporal Knowledge Probing")
        print("=" * 60)
        from experiment3_probing import run_experiment3
        run_experiment3(model_ids)

    if args.exp in ("all", "ocr"):
        print("\n" + "=" * 60)
        print("OCR NOISE ABLATION")
        print("=" * 60)
        from ocr_ablation import run_ocr_ablation
        run_ocr_ablation()

    if args.exp in ("all", "figures"):
        print("\n" + "=" * 60)
        print("GENERATING FIGURES & TABLES")
        print("=" * 60)
        from analysis import generate_all_figures
        generate_all_figures()

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
