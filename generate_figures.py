"""
Generate all paper figures from experiment result JSON files.
Run: python generate_figures.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

COLORS = {
    "Talkie-1930": "#D45E00",   # burnt orange
    "Talkie-Web":  "#0072B2",   # steel blue
}
MARKERS = {"Talkie-1930": "o", "Talkie-Web": "s"}


def save(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(FIGURES / f"{name}.{ext}")
    plt.close(fig)
    print(f"  Saved {name}.pdf/png")


# ══════════════════════════════════════════════════════════════════════
#  Figure 1 — Experimental setup / Knowledge boundary
# ══════════════════════════════════════════════════════════════════════

def figure1_knowledge_boundary():
    kb_1930 = json.load(open(RESULTS / "knowledge_boundary_Talkie-1930.json"))
    kb_web  = json.load(open(RESULTS / "knowledge_boundary_Talkie-Web.json"))

    categories = ["Pre-1930\nFacts", "Post-1930\nFacts"]
    vals_1930 = [kb_1930["pre_1930"]["accuracy"] * 100,
                 kb_1930["post_1930"]["accuracy"] * 100]
    vals_web  = [kb_web["pre_1930"]["accuracy"] * 100,
                 kb_web["post_1930"]["accuracy"] * 100]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(categories))
    w = 0.32

    bars1 = ax.bar(x - w/2, vals_1930, w, label="Talkie-1930",
                   color=COLORS["Talkie-1930"], edgecolor="white")
    bars2 = ax.bar(x + w/2, vals_web, w, label="Talkie-Web",
                   color=COLORS["Talkie-Web"], edgecolor="white")

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 1.5,
                    f"{h:.0f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Behavioural Knowledge Boundary Test")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add annotation for the key finding
    ax.annotate("0% — complete\ntemporal containment",
                xy=(1 - w/2, 1), fontsize=9, color=COLORS["Talkie-1930"],
                ha="center", va="bottom", style="italic")

    save(fig, "figure1_knowledge_boundary")


# ══════════════════════════════════════════════════════════════════════
#  Figure 2 — BLiMP syntactic evaluation
# ══════════════════════════════════════════════════════════════════════

def figure2_blimp():
    data = json.load(open(RESULTS / "experiment1_syntactic.json"))

    phenomena = list(data["Talkie-1930"]["blimp"].keys())
    phenomena = [p for p in phenomena if p != "__aggregate__"]

    # Sort by Talkie-Web accuracy for visual clarity
    phenomena.sort(key=lambda p: data["Talkie-Web"]["blimp"][p]["accuracy"])

    labels = [p.replace("_", " ").title() for p in phenomena]
    vals_1930 = [data["Talkie-1930"]["blimp"][p]["accuracy"] * 100 for p in phenomena]
    vals_web  = [data["Talkie-Web"]["blimp"][p]["accuracy"] * 100 for p in phenomena]

    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(phenomena))
    h = 0.35

    ax.barh(y - h/2, vals_1930, h, label="Talkie-1930",
            color=COLORS["Talkie-1930"], edgecolor="white")
    ax.barh(y + h/2, vals_web, h, label="Talkie-Web",
            color=COLORS["Talkie-Web"], edgecolor="white")

    ax.axvline(50, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("BLiMP Syntactic Evaluation")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(45, 100)
    ax.legend(loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add aggregate scores
    agg_1930 = data["Talkie-1930"]["blimp"].get("__aggregate__", {}).get("accuracy", 0)
    agg_web  = data["Talkie-Web"]["blimp"].get("__aggregate__", {}).get("accuracy", 0)
    if agg_1930 == 0:
        agg_1930 = np.mean(vals_1930) / 100
    if agg_web == 0:
        agg_web = np.mean(vals_web) / 100

    ax.text(0.98, 0.02,
            f"Aggregate: {agg_1930*100:.1f}% vs {agg_web*100:.1f}%",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    save(fig, "figure2_blimp")


# ══════════════════════════════════════════════════════════════════════
#  Figure 3 — ICL learning curves
# ══════════════════════════════════════════════════════════════════════

def figure3_icl():
    data = json.load(open(RESULTS / "experiment2_icl_final.json"))

    tasks = ["sst2", "mnli", "tweet_sentiment", "tweet_emotion"]
    task_titles = {
        "sst2": "SST-2 (Sentiment)",
        "mnli": "MNLI (NLI)",
        "tweet_sentiment": "Tweet Sentiment",
        "tweet_emotion": "Tweet Emotion",
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    axes = axes.flatten()

    for idx, task in enumerate(tasks):
        ax = axes[idx]

        for model_name in ["Talkie-1930", "Talkie-Web"]:
            if model_name not in data:
                continue
            task_data = data[model_name].get(task, {})
            k_vals = sorted(int(k) for k in task_data if k.isdigit())
            means = [task_data[str(k)]["mean_accuracy"] for k in k_vals]
            stds  = [task_data[str(k)]["std_accuracy"] for k in k_vals]

            ax.errorbar(k_vals, means, yerr=stds,
                       marker=MARKERS[model_name], color=COLORS[model_name],
                       label=model_name, capsize=3, linewidth=1.5, markersize=5)

        ax.set_title(task_titles.get(task, task))
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("k (# examples)")
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xticks(k_vals)
        ax.set_xticklabels([str(k) for k in k_vals], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if idx == 0:
            ax.legend(loc="lower right", fontsize=9)

    fig.suptitle("In-Context Learning Curves", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "figure3_icl_curves")


# ══════════════════════════════════════════════════════════════════════
#  Figure 4 — Probing heatmap
# ══════════════════════════════════════════════════════════════════════

def figure4_probing():
    data = json.load(open(RESULTS / "experiment3_probing.json"))

    probes = ["probe_a_veracity", "probe_b_temporal", "probe_c_knowledge_boundary"]
    probe_labels = {
        "probe_a_veracity": "A: Veracity\n(true vs false)",
        "probe_b_temporal": "B: Temporal\n(pre vs post 1930)",
        "probe_c_knowledge_boundary": "C: Knowledge\nBoundary (3-way)",
    }
    models = ["Talkie-1930", "Talkie-Web"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    im = None
    for m_idx, model in enumerate(models):
        ax = axes[m_idx]
        model_data = data.get(model, {})

        # Only use probes that exist for this model
        available_probes = [p for p in probes if p in model_data]
        available_labels = [probe_labels[p] for p in available_probes]

        if not available_probes:
            continue

        # Use consistent layer set from first available probe
        first_probe_data = model_data[available_probes[0]]["layer_results"]
        layers = sorted(int(l) for l in first_probe_data)

        matrix = []
        for probe in available_probes:
            probe_data = model_data[probe]["layer_results"]
            probe_layers = sorted(int(l) for l in probe_data)
            row = [probe_data[str(l)]["mean"] for l in probe_layers]
            matrix.append(row)

        # Pad rows to same length if needed
        max_len = max(len(r) for r in matrix)
        for i in range(len(matrix)):
            while len(matrix[i]) < max_len:
                matrix[i].append(0.5)

        matrix = np.array(matrix)
        im = ax.imshow(matrix, cmap="YlOrRd", vmin=0.4, vmax=1.0, aspect="auto")

        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, fontsize=8)
        ax.set_xlabel("Layer")
        ax.set_title(model, fontweight="bold", color=COLORS[model])

        if m_idx == 0:
            ax.set_yticks(range(len(available_probes)))
            ax.set_yticklabels(available_labels, fontsize=9)
        else:
            ax.set_yticks(range(len(available_probes)))
            ax.set_yticklabels(available_labels, fontsize=9)

        # Annotate cells
        for i in range(len(available_probes)):
            for j in range(min(len(layers), matrix.shape[1])):
                val = matrix[i, j]
                color = "white" if val > 0.8 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                       fontsize=7, color=color)

    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.8, label="Accuracy")
    fig.suptitle("Linear Probing Accuracy by Layer", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.92, 0.94])
    save(fig, "figure4_probing_heatmap")


# ══════════════════════════════════════════════════════════════════════
#  Figure 5 — Probe C confusion matrix + knowledge boundary
# ══════════════════════════════════════════════════════════════════════

def figure5_probe_c():
    ext = json.load(open(RESULTS / "probe_analysis_extended.json"))
    cm_data = ext.get("probe_c_detailed", {})

    # Use layer 16 (best for Probe C)
    best_layer = "16"
    if best_layer not in cm_data:
        best_layer = list(cm_data.keys())[0]

    cm = np.array(cm_data[best_layer]["confusion_matrix"])
    labels = ["Should-Know\n(pre-1930)", "Shouldn't-Know\n(post-1930)", "Leaked"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: Confusion matrix
    ax = axes[0]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
    ax.set_yticks(range(3))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Probe C Confusion Matrix (Layer {best_layer})")

    for i in range(3):
        for j in range(3):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.0%})",
                   ha="center", va="center", fontsize=9, color=color)

    # Right: Per-class F1 across layers
    ax2 = axes[1]
    class_names = list(cm_data[best_layer]["per_class_f1"].keys())
    layer_keys = sorted(cm_data.keys(), key=int)

    class_colors = ["#2ca02c", "#d62728", "#9467bd"]
    for c_idx, cls in enumerate(class_names):
        f1s = [cm_data[lk]["per_class_f1"][cls] for lk in layer_keys]
        short_name = cls.split("(")[0].strip() if "(" in cls else cls
        ax2.plot([int(l) for l in layer_keys], f1s,
                marker="o", label=short_name, color=class_colors[c_idx],
                linewidth=1.5, markersize=5)

    ax2.set_xlabel("Layer")
    ax2.set_ylabel("F1 Score")
    ax2.set_title("Per-Class F1 by Layer")
    ax2.legend(fontsize=8, loc="lower left")
    ax2.set_ylim(0.4, 1.05)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    save(fig, "figure5_probe_c_detail")


# ══════════════════════════════════════════════════════════════════════
#  Figure 6 — OCR ablation
# ══════════════════════════════════════════════════════════════════════

def figure6_ocr_ablation():
    data = json.load(open(RESULTS / "ocr_ablation.json"))

    conditions = ["clean", "ocr_0.01", "ocr_0.02", "ocr_0.05", "ocr_0.10"]
    x_labels = ["0%\n(clean)", "1%", "2%", "5%", "10%"]
    agg_accs = [data[c]["__aggregate__"]["accuracy"] * 100 for c in conditions]

    # Per-phenomenon breakdown
    phenomena = [p for p in data["clean"] if p != "__aggregate__"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: aggregate
    ax1.plot(range(len(conditions)), agg_accs, "o-",
             color=COLORS["Talkie-Web"], linewidth=2, markersize=8)
    ax1.axhline(50, color="gray", linestyle="--", alpha=0.4, label="Chance")
    ax1.set_xticks(range(len(conditions)))
    ax1.set_xticklabels(x_labels)
    ax1.set_xlabel("OCR Error Rate")
    ax1.set_ylabel("BLiMP Aggregate Accuracy (%)")
    ax1.set_title("Effect of OCR Noise on Syntactic Evaluation")
    ax1.set_ylim(50, 90)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    for i, v in enumerate(agg_accs):
        ax1.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=9)

    # Right: per-phenomenon heatmap
    matrix = []
    for p in phenomena:
        row = [data[c][p]["accuracy"] * 100 for c in conditions]
        matrix.append(row)
    matrix = np.array(matrix)

    im = ax2.imshow(matrix, cmap="RdYlGn", vmin=50, vmax=100, aspect="auto")
    ax2.set_xticks(range(len(conditions)))
    ax2.set_xticklabels(x_labels, fontsize=8)
    ax2.set_yticks(range(len(phenomena)))
    ax2.set_yticklabels([p.replace("_", " ").title()[:20] for p in phenomena], fontsize=7)
    ax2.set_xlabel("OCR Error Rate")
    ax2.set_title("Per-Phenomenon Accuracy (%)")
    fig.colorbar(im, ax=ax2, shrink=0.8)

    fig.tight_layout()
    save(fig, "figure6_ocr_ablation")


# ══════════════════════════════════════════════════════════════════════
#  Figure 7 — Summary dashboard
# ══════════════════════════════════════════════════════════════════════

def figure7_summary():
    """Compact summary of all key results."""
    # Load all data
    blimp = json.load(open(RESULTS / "experiment1_syntactic.json"))
    icl = json.load(open(RESULTS / "experiment2_icl_final.json"))
    probing = json.load(open(RESULTS / "experiment3_probing.json"))
    kb_1930 = json.load(open(RESULTS / "knowledge_boundary_Talkie-1930.json"))
    kb_web = json.load(open(RESULTS / "knowledge_boundary_Talkie-Web.json"))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Panel 1: Knowledge boundary
    ax = axes[0]
    x = np.arange(2)
    w = 0.32
    vals_1930 = [kb_1930["pre_1930"]["accuracy"]*100, kb_1930["post_1930"]["accuracy"]*100]
    vals_web = [kb_web["pre_1930"]["accuracy"]*100, kb_web["post_1930"]["accuracy"]*100]
    ax.bar(x-w/2, vals_1930, w, color=COLORS["Talkie-1930"], label="Talkie-1930")
    ax.bar(x+w/2, vals_web, w, color=COLORS["Talkie-Web"], label="Talkie-Web")
    ax.set_xticks(x)
    ax.set_xticklabels(["Pre-1930", "Post-1930"], fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Knowledge Boundary", fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel 2: BLiMP aggregate
    ax = axes[1]
    agg_1930 = blimp["Talkie-1930"]["blimp"]["__aggregate__"]["accuracy"] * 100 \
        if "__aggregate__" in blimp["Talkie-1930"]["blimp"] \
        else np.mean([v["accuracy"] for v in blimp["Talkie-1930"]["blimp"].values()]) * 100
    agg_web = blimp["Talkie-Web"]["blimp"]["__aggregate__"]["accuracy"] * 100 \
        if "__aggregate__" in blimp["Talkie-Web"]["blimp"] \
        else np.mean([v["accuracy"] for v in blimp["Talkie-Web"]["blimp"].values()]) * 100

    bars = ax.bar(["Talkie-1930", "Talkie-Web"], [agg_1930, agg_web],
                  color=[COLORS["Talkie-1930"], COLORS["Talkie-Web"]])
    ax.axhline(50, color="gray", linestyle="--", alpha=0.4)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("BLiMP (Syntax)", fontweight="bold")
    ax.set_ylim(45, 95)
    for bar, val in zip(bars, [agg_1930, agg_web]):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1, f"{val:.1f}",
               ha="center", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel 3: ICL best task (SST-2)
    ax = axes[2]
    for model in ["Talkie-1930", "Talkie-Web"]:
        task_data = icl[model]["sst2"]
        k_vals = sorted(int(k) for k in task_data if k.isdigit())
        means = [task_data[str(k)]["mean_accuracy"] for k in k_vals]
        ax.plot(k_vals, means, marker=MARKERS[model], color=COLORS[model],
               label=model, linewidth=1.5, markersize=5)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("k")
    ax.set_ylabel("Accuracy")
    ax.set_title("ICL (SST-2)", fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel 4: Probing peak accuracies
    ax = axes[3]
    probe_names = ["probe_a_veracity", "probe_b_temporal"]
    probe_short = ["Veracity", "Temporal"]
    peaks_1930, peaks_web = [], []
    for p in probe_names:
        lr_1930 = probing["Talkie-1930"][p]["layer_results"]
        peaks_1930.append(max(v["mean"] for v in lr_1930.values()) * 100)
        lr_web = probing["Talkie-Web"][p]["layer_results"]
        peaks_web.append(max(v["mean"] for v in lr_web.values()) * 100)

    x = np.arange(len(probe_short))
    ax.bar(x-w/2, peaks_1930, w, color=COLORS["Talkie-1930"], label="Talkie-1930")
    ax.bar(x+w/2, peaks_web, w, color=COLORS["Talkie-Web"], label="Talkie-Web")
    ax.set_xticks(x)
    ax.set_xticklabels(probe_short, fontsize=9)
    ax.set_ylabel("Peak Accuracy (%)")
    ax.set_title("Probing", fontweight="bold")
    ax.set_ylim(60, 105)
    ax.legend(fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Talkie: Temporal Knowledge in Language Models — Key Results",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "figure7_summary_dashboard")


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating figures from fp32 results...\n")

    print("Figure 1: Knowledge Boundary")
    figure1_knowledge_boundary()

    print("Figure 2: BLiMP")
    figure2_blimp()

    print("Figure 3: ICL Curves")
    figure3_icl()

    print("Figure 4: Probing Heatmap")
    figure4_probing()

    print("Figure 5: Probe C Detail")
    figure5_probe_c()

    print("Figure 6: OCR Ablation")
    figure6_ocr_ablation()

    print("Figure 7: Summary Dashboard")
    figure7_summary()

    print(f"\nAll figures saved to {FIGURES}/")
