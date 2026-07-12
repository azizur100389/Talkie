"""
Rebuttal control (Reviewer, OCR ablation):
is the Talkie-1930 vs Talkie-Web BLiMP gap explained by OCR noise?

Compares the *shape* of two per-phenomenon patterns:
  (a) OCR-induced degradation of Talkie-Web (clean -> noised), and
  (b) the real Talkie-1930 vs Talkie-Web gap (both clean).

If OCR noise explained the gap, (a) and (b) would be positively correlated
and similarly shaped. We report the coefficient of variation (uniformity)
of each pattern and their correlation.

Requires:
  results/ocr_ablation.json          (from ocr_ablation.py)
  results/experiment1_syntactic.json (from experiment1_syntactic.py)

Run:  python rebuttal_ocr_shape.py
"""
import json

import numpy as np

import config


def load(name):
    with open(config.RESULTS_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def main():
    ocr = load("ocr_ablation.json")
    syn = load("experiment1_syntactic.json")

    phen = [p for p in ocr["clean"] if p != "__aggregate__"]

    real_gap = np.array([
        syn["Talkie-Web"]["blimp"][p]["accuracy"] - syn["Talkie-1930"]["blimp"][p]["accuracy"]
        for p in phen
    ])
    cv = lambda x: x.std() / abs(x.mean())

    print(f"Phenomena: {len(phen)}\n")
    print("Real 1930-vs-Web gap (both clean):")
    print(f"  mean={real_gap.mean():.4f}  std={real_gap.std():.4f}  CV={cv(real_gap):.2f}"
          f"  min={real_gap.min():.3f}  max={real_gap.max():.3f}")
    print(f"  phenomena where Talkie-1930 wins (gap<0): {(real_gap < 0).sum()}/{len(phen)}\n")

    print("OCR-induced degradation of Talkie-Web (uniform if CV low; "
          "unrelated to real gap if |r| low):")
    for lvl in ["ocr_0.02", "ocr_0.05", "ocr_0.10"]:
        deg = np.array([ocr["clean"][p]["accuracy"] - ocr[lvl][p]["accuracy"] for p in phen])
        r = np.corrcoef(real_gap, deg)[0, 1]
        print(f"  {lvl}: mean={deg.mean():.4f}  std={deg.std():.4f}  "
              f"CV={cv(deg):.2f}  corr(real_gap, deg) r={r:+.3f}")


if __name__ == "__main__":
    main()
