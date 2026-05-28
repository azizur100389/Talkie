"""
Analysis & Visualization

Generates all figures and tables for the paper from saved experiment results.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

import config

sns.set_theme(style="whitegrid", font_scale=1.2)

VINTAGE_COLOR = "#2E86AB"
MODERN_COLOR = "#E8625C"
MODEL_COLORS = {"Talkie-1930": VINTAGE_COLOR, "Talkie-Web": MODERN_COLOR}


# ── Load results ───────────────────────────────────────────────────────

def load_results(name: str) -> dict:
    path = config.RESULTS_DIR / f"{name}.json"
    with open(path) as f:
        return json.load(f)


# ── Figure 1: Experimental setup diagram ──────────────────────────────

def figure1_setup_diagram():
    """Create a schematic of the experimental setup."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    boxes = [
        (0.05, 0.6, 0.25, 0.25, "Pre-1931 Text\n(260B tokens)", "#D4E6F1"),
        (0.05, 0.15, 0.25, 0.25, "FineWeb\n(260B tokens)", "#FADBD8"),
        (0.45, 0.6, 0.2, 0.25, "Talkie-1930\n13B params", VINTAGE_COLOR),
        (0.45, 0.15, 0.2, 0.25, "Talkie-Web\n13B params", MODERN_COLOR),
        (0.8, 0.55, 0.17, 0.12, "BLiMP\n(Syntax)", "#F9E79F"),
        (0.8, 0.38, 0.17, 0.12, "ICL\n(Few-shot)", "#F9E79F"),
        (0.8, 0.21, 0.17, 0.12, "Probing\n(Knowledge)", "#F9E79F"),
    ]

    for x, y, w, h, text, color in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="black",
                              linewidth=1.5, alpha=0.85, transform=ax.transAxes)
        ax.add_patch(rect)
        text_color = "white" if color in [VINTAGE_COLOR, MODERN_COLOR] else "black"
        ax.text(x + w / 2, y + h / 2, text, transform=ax.transAxes,
                ha="center", va="center", fontsize=10, fontweight="bold",
                color=text_color)

    arrow_props = dict(arrowstyle="->", color="black", lw=2)
    ax.annotate("", xy=(0.45, 0.72), xytext=(0.30, 0.72),
                arrowprops=arrow_props, xycoords="axes fraction")
    ax.annotate("", xy=(0.45, 0.27), xytext=(0.30, 0.27),
                arrowprops=arrow_props, xycoords="axes fraction")
    ax.annotate("", xy=(0.80, 0.61), xytext=(0.65, 0.65),
                arrowprops=arrow_props, xycoords="axes fraction")
    ax.annotate("", xy=(0.80, 0.44), xytext=(0.65, 0.50),
                arrowprops=arrow_props, xycoords="axes fraction")
    ax.annotate("", xy=(0.80, 0.27), xytext=(0.65, 0.35),
                arrowprops=arrow_props, xycoords="axes fraction")
    ax.annotate("", xy=(0.80, 0.61), xytext=(0.65, 0.30),
                arrowprops={**arrow_props, "color": "gray", "lw": 1.5, "ls": "--"},
                xycoords="axes fraction")
    ax.annotate("", xy=(0.80, 0.44), xytext=(0.65, 0.24),
                arrowprops={**arrow_props, "color": "gray", "lw": 1.5, "ls": "--"},
                xycoords="axes fraction")
    ax.annotate("", xy=(0.80, 0.27), xytext=(0.65, 0.18),
                arrowprops={**arrow_props, "color": "gray", "lw": 1.5, "ls": "--"},
                xycoords="axes fraction")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Experimental Setup: Vintage vs. Modern Twin Comparison", fontsize=14, pad=15)

    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "figure1_setup.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(config.FIGURES_DIR / "figure1_setup.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved figure1_setup")


# ── Table 1 + Figure 2: BLiMP results ────────────────────────────────

def table1_and_figure2_blimp():
    """BLiMP results table and grouped bar chart by phenomenon."""
    results = load_results("experiment1_syntactic")

    phenomena = [p for p in config.BLIMP_PHENOMENA if p != "__aggregate__"]
    models = list(results.keys())

    # Build DataFrame
    rows = []
    for phenomenon in phenomena:
        row = {"Phenomenon": phenomenon.replace("_", " ").title()}
        for model_name in models:
            blimp = results[model_name]["blimp"]
            if phenomenon in blimp:
                row[model_name] = blimp[phenomenon]["accuracy"]
            else:
                row[model_name] = np.nan
        rows.append(row)

    # Add aggregate
    for model_name in models:
        agg = results[model_name]["blimp"].get("__aggregate__", {})
        rows.append({
            "Phenomenon": "Aggregate",
            **{m: results[m]["blimp"].get("__aggregate__", {}).get("accuracy", np.nan) for m in models},
        })

    df = pd.DataFrame(rows)

    # Save table as CSV
    df.to_csv(config.RESULTS_DIR / "table1_blimp.csv", index=False, float_format="%.3f")

    # LaTeX table
    latex = df.to_latex(index=False, float_format="%.3f", caption="BLiMP accuracy by phenomenon.",
                        label="tab:blimp")
    with open(config.RESULTS_DIR / "table1_blimp.tex", "w") as f:
        f.write(latex)

    # Grouped bar chart
    plot_df = df[df["Phenomenon"] != "Aggregate"].copy()
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(plot_df))
    width = 0.35

    for i, model_name in enumerate(models):
        offset = (i - 0.5) * width
        vals = plot_df[model_name].values
        color = MODEL_COLORS.get(model_name, f"C{i}")
        ax.bar(x + offset, vals, width, label=model_name, color=color, alpha=0.85)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance")
    ax.set_ylabel("Accuracy")
    ax.set_title("BLiMP: Syntactic Competence by Phenomenon")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["Phenomenon"], rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "figure2_blimp.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(config.FIGURES_DIR / "figure2_blimp.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved table1_blimp and figure2_blimp")


# ── Figure 3 + Table 2: ICL learning curves ──────────────────────────

def figure3_and_table2_icl():
    """ICL learning curves and slope analysis."""
    results = load_results("experiment2_icl")
    models = list(results.keys())
    tasks = list(results[models[0]].keys())

    # ── Learning curves figure ──
    n_tasks = len(tasks)
    fig, axes = plt.subplots(1, n_tasks, figsize=(5 * n_tasks, 5), sharey=True)
    if n_tasks == 1:
        axes = [axes]

    for ax, task_name in zip(axes, tasks):
        for model_name in models:
            task_data = results[model_name][task_name]
            k_vals = sorted(
                [int(k) for k in task_data.keys() if k.isdigit()]
            )
            means = [task_data[str(k)]["mean_accuracy"] for k in k_vals]
            stds = [task_data[str(k)]["std_accuracy"] for k in k_vals]
            color = MODEL_COLORS.get(model_name, None)

            ax.errorbar(
                k_vals, means, yerr=stds,
                marker="o", label=model_name, color=color,
                capsize=3, linewidth=2, markersize=6,
            )

        task_cfg = config.ICL_TASKS.get(task_name, {})
        n_labels = len(task_cfg.get("label_names", [2]))
        ax.axhline(1.0 / n_labels, color="gray", linestyle="--", linewidth=1, alpha=0.5)

        ax.set_xlabel("k (demonstrations)")
        ax.set_title(task_cfg.get("description", task_name))
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xticks(config.ICL_K_VALUES)
        ax.set_xticklabels([str(k) for k in config.ICL_K_VALUES])

    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    plt.suptitle("In-Context Learning Curves", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "figure3_icl_curves.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(config.FIGURES_DIR / "figure3_icl_curves.png", bbox_inches="tight", dpi=300)
    plt.close()

    # ── Slopes table ──
    rows = []
    for task_name in tasks:
        row = {"Task": task_name}
        for model_name in models:
            metrics = results[model_name][task_name].get("metrics", {})
            row[f"{model_name} slope"] = metrics.get("learning_rate_slope", np.nan)
            row[f"{model_name} 0-shot"] = metrics.get("zero_shot_accuracy", np.nan)
            row[f"{model_name} 32-shot"] = metrics.get("max_k_accuracy", np.nan)
            row[f"{model_name} gain"] = metrics.get("accuracy_gain", np.nan)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS_DIR / "table2_icl_slopes.csv", index=False, float_format="%.3f")

    latex = df.to_latex(index=False, float_format="%.3f",
                        caption="ICL learning rates and accuracy gains.",
                        label="tab:icl_slopes")
    with open(config.RESULTS_DIR / "table2_icl_slopes.tex", "w") as f:
        f.write(latex)

    print("Saved figure3_icl_curves and table2_icl_slopes")


# ── Figure 4: Probing heatmap ─────────────────────────────────────────

def figure4_probing_heatmap():
    """Probe accuracy heatmap by layer and probe type."""
    results = load_results("experiment3_probing")
    models = list(results.keys())

    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 6), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        model_data = results[model_name]
        probe_names = sorted(model_data.keys())
        layers = sorted(
            int(l) for l in model_data[probe_names[0]]["layer_results"].keys()
        )

        matrix = np.zeros((len(probe_names), len(layers)))
        for i, probe_name in enumerate(probe_names):
            for j, layer in enumerate(layers):
                layer_result = model_data[probe_name]["layer_results"].get(str(layer), {})
                matrix[i, j] = layer_result.get("mean", 0.0)

        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0.4, vmax=1.0)

        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, fontsize=9)
        ax.set_yticks(range(len(probe_names)))
        ax.set_yticklabels([p.replace("_", " ").title() for p in probe_names], fontsize=10)
        ax.set_xlabel("Layer")
        ax.set_title(model_name)

        for i in range(len(probe_names)):
            for j in range(len(layers)):
                val = matrix[i, j]
                color = "white" if val > 0.7 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    fig.colorbar(im, ax=axes, label="Probe Accuracy", shrink=0.8)
    plt.suptitle("Linear Probe Accuracy by Layer", fontsize=14)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "figure4_probing_heatmap.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(config.FIGURES_DIR / "figure4_probing_heatmap.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved figure4_probing_heatmap")


# ── Figure 5: t-SNE / UMAP of representations ────────────────────────

def figure5_representation_space():
    """t-SNE/UMAP visualisation of hidden states coloured by temporal provenance."""
    try:
        import umap
    except ImportError:
        from sklearn.manifold import TSNE
        reducer_cls = TSNE
        reducer_name = "t-SNE"
    else:
        reducer_cls = umap.UMAP
        reducer_name = "UMAP"

    models_to_plot = ["Talkie-1930", "Talkie-Web"]
    probe_name = "probe_b_temporal"
    layer_idx = config.N_LAYERS  # final layer

    fig, axes = plt.subplots(1, len(models_to_plot), figsize=(7 * len(models_to_plot), 6))
    if len(models_to_plot) == 1:
        axes = [axes]

    temporal_labels = {0: "Pre-1930", 1: "Post-1930"}
    temporal_colors = {0: VINTAGE_COLOR, 1: MODERN_COLOR}

    for ax, model_name in zip(axes, models_to_plot):
        hs_path = config.RESULTS_DIR / f"hidden_states_{model_name}_{probe_name}.npz"
        if not hs_path.exists():
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(model_name)
            continue

        data = np.load(str(hs_path))
        layer_key = f"layer_{layer_idx}"
        if layer_key not in data:
            available = [k for k in data.files if k.startswith("layer_")]
            layer_key = available[-1] if available else None

        if layer_key is None:
            ax.text(0.5, 0.5, "No layer data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(model_name)
            continue

        X = data[layer_key]
        labels = data["labels"]

        reducer = reducer_cls(n_components=2, random_state=42)
        if reducer_name == "t-SNE":
            embedding = reducer.fit_transform(X)
        else:
            embedding = reducer.fit_transform(X)

        for label_val, label_name in temporal_labels.items():
            mask = labels == label_val
            ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                c=temporal_colors[label_val], label=label_name,
                alpha=0.6, s=40, edgecolors="white", linewidth=0.5,
            )

        ax.legend()
        ax.set_title(f"{model_name} — Layer {layer_key.split('_')[1]}")
        ax.set_xlabel(f"{reducer_name} 1")
        ax.set_ylabel(f"{reducer_name} 2")

    plt.suptitle("Temporal Provenance in Representation Space", fontsize=14)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "figure5_representation_space.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(config.FIGURES_DIR / "figure5_representation_space.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved figure5_representation_space")


# ── Table 3: Summary of findings ─────────────────────────────────────

def table3_summary():
    """Generate a summary table of headline findings across experiments."""
    rows = []

    # Exp 1 summary
    try:
        exp1 = load_results("experiment1_syntactic")
        for model_name, data in exp1.items():
            agg = data["blimp"].get("__aggregate__", {})
            rows.append({
                "Experiment": "Syntactic (BLiMP)",
                "Model": model_name,
                "Key Metric": "Aggregate Accuracy",
                "Value": f"{agg.get('accuracy', 0):.3f}",
            })
    except FileNotFoundError:
        pass

    # Exp 2 summary
    try:
        exp2 = load_results("experiment2_icl")
        for model_name, tasks in exp2.items():
            slopes = []
            for task_name, task_data in tasks.items():
                m = task_data.get("metrics", {})
                if "learning_rate_slope" in m:
                    slopes.append(m["learning_rate_slope"])
            if slopes:
                rows.append({
                    "Experiment": "ICL",
                    "Model": model_name,
                    "Key Metric": "Mean Learning Rate (slope)",
                    "Value": f"{np.mean(slopes):.3f}",
                })
    except FileNotFoundError:
        pass

    # Exp 3 summary
    try:
        exp3 = load_results("experiment3_probing")
        for model_name, probes in exp3.items():
            for probe_name, probe_data in probes.items():
                layer_results = probe_data.get("layer_results", {})
                if layer_results:
                    best_layer = max(layer_results, key=lambda l: layer_results[l].get("mean", 0))
                    best_acc = layer_results[best_layer]["mean"]
                    rows.append({
                        "Experiment": f"Probing ({probe_name})",
                        "Model": model_name,
                        "Key Metric": f"Best Layer ({best_layer}) Accuracy",
                        "Value": f"{best_acc:.3f}",
                    })
    except FileNotFoundError:
        pass

    if not rows:
        print("No results found to summarise.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS_DIR / "table3_summary.csv", index=False)

    latex = df.to_latex(index=False, caption="Summary of headline findings.",
                        label="tab:summary")
    with open(config.RESULTS_DIR / "table3_summary.tex", "w") as f:
        f.write(latex)
    print("Saved table3_summary")


# ── OCR ablation figure ───────────────────────────────────────────────

def figure_ocr_ablation():
    """Plot BLiMP accuracy vs. OCR noise level for the modern twin."""
    try:
        results = load_results("ocr_ablation")
    except FileNotFoundError:
        print("No OCR ablation results found.")
        return

    error_rates = [0.0] + config.OCR_ERROR_RATES
    labels = ["Clean"] + [f"{r:.0%}" for r in config.OCR_ERROR_RATES]

    agg_accuracies = []
    for key in ["clean"] + [f"ocr_{r:.2f}" for r in config.OCR_ERROR_RATES]:
        agg = results.get(key, {}).get("__aggregate__", {})
        agg_accuracies.append(agg.get("accuracy", np.nan))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(error_rates)), agg_accuracies, "o-", color=MODERN_COLOR,
            linewidth=2, markersize=8, label="Talkie-Web (degraded)")
    ax.set_xticks(range(len(error_rates)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("OCR Error Rate")
    ax.set_ylabel("BLiMP Aggregate Accuracy")
    ax.set_title("Effect of OCR Noise on Syntactic Evaluation")
    ax.legend()

    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "figure_ocr_ablation.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(config.FIGURES_DIR / "figure_ocr_ablation.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved figure_ocr_ablation")


# ── Run all ───────────────────────────────────────────────────────────

def generate_all_figures():
    """Generate all figures and tables from experiment results."""
    print("\n" + "=" * 60)
    print("Generating all figures and tables")
    print("=" * 60)

    figure1_setup_diagram()

    try:
        table1_and_figure2_blimp()
    except FileNotFoundError:
        print("Skipping BLiMP figures (no results found)")

    try:
        figure3_and_table2_icl()
    except FileNotFoundError:
        print("Skipping ICL figures (no results found)")

    try:
        figure4_probing_heatmap()
    except FileNotFoundError:
        print("Skipping probing heatmap (no results found)")

    try:
        figure5_representation_space()
    except Exception as e:
        print(f"Skipping representation space plot: {e}")

    table3_summary()

    try:
        figure_ocr_ablation()
    except FileNotFoundError:
        print("Skipping OCR ablation figure (no results found)")

    print("\nAll figures saved to:", config.FIGURES_DIR)


if __name__ == "__main__":
    generate_all_figures()
