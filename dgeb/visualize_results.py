"""
Visualize DGEB evaluation results from the results/ directory.

Generates:
  1. Heatmap of primary metric per model × task
  2. Grouped bar chart per task
  3. Grouped bar chart per task type (averaged)
  4. Saved to results/figures/ as PNG files
"""

import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Friendly display names for models (keyed by hf_name in JSON)
MODEL_LABELS = {
    "bioseqfeat-blosum": "BioSeqFeat (BLOSUM)",
    "facebook/esm2_t6_8M_UR50D": "ESM2-8M",
    "facebook/esm2_t30_150M_UR50D": "ESM2-150M",
    "facebook/esm2_t33_650M_UR50D": "ESM2-650M",
    "random-model": "Random",
}

# Desired display order for models in bar plots
MODEL_ORDER = [
    "random-model",
    "bioseqfeat-blosum",
    "facebook/esm2_t6_8M_UR50D",
    "facebook/esm2_t30_150M_UR50D",
    "facebook/esm2_t33_650M_UR50D",
]


def _sort_models(model_set) -> list:
    """Sort models according to MODEL_ORDER; unknown models go at the end (alphabetically)."""
    known = [m for m in MODEL_ORDER if m in model_set]
    unknown = sorted(m for m in model_set if m not in MODEL_ORDER)
    return known + unknown

TASK_TYPE_COLORS = {
    "retrieval": "#4C72B0",
    "pair_classification": "#DD8452",
    "classification": "#55A868",
    "eds": "#C44E52",
    "clustering": "#8172B2",
    "bigene_mining": "#937860",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(results_dir: Path) -> list[dict]:
    """Return list of parsed result dicts from all JSON files under results_dir."""
    records = []
    for json_file in sorted(results_dir.rglob("*.json")):
        with open(json_file) as f:
            data = json.load(f)

        task = data["task"]
        model = data["model"]
        primary_id = task["primary_metric_id"]

        # Take the best layer (or only layer) by primary metric
        best_value = None
        best_layer = None
        for layer_result in data["results"]:
            metric_map = {m["id"]: m["value"] for m in layer_result["metrics"]}
            value = metric_map.get(primary_id)
            if value is not None and (best_value is None or value > best_value):
                best_value = value
                best_layer = layer_result["layer_display_name"]

        records.append(
            {
                "task_id": task["id"],
                "task_name": task["display_name"],
                "task_type": task["type"],
                "model": model["hf_name"],
                "primary_metric": primary_id,
                "value": best_value,
                "best_layer": best_layer,
            }
        )
    return records


def build_pivot(records: list[dict]) -> tuple[list, list, np.ndarray]:
    """Return (models, tasks, matrix) where matrix[i,j] = score for model i, task j."""
    models = _sort_models({r["model"] for r in records})
    tasks = sorted({r["task_id"] for r in records})

    matrix = np.full((len(models), len(tasks)), np.nan)
    for r in records:
        i = models.index(r["model"])
        j = tasks.index(r["task_id"])
        matrix[i, j] = r["value"] if r["value"] is not None else np.nan

    return models, tasks, matrix


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _model_label(hf_name: str) -> str:
    return MODEL_LABELS.get(hf_name, hf_name)


def plot_heatmap(models, tasks, matrix, records, out_path: Path):
    task_display = {r["task_id"]: r["task_name"] for r in records}
    xlabels = [task_display.get(t, t) for t in tasks]
    ylabels = [_model_label(m) for m in models]

    # Normalize each column to [0, 1] within its own min/max
    normed = np.full_like(matrix, np.nan)
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) == 0:
            continue
        col_min, col_max = valid.min(), valid.max()
        if col_max > col_min:
            normed[:, j] = (col - col_min) / (col_max - col_min)
        else:
            normed[:, j] = 0.5

    cmap = plt.cm.YlOrRd
    cmap.set_bad(color="#f0f0f0")

    fig, ax = plt.subplots(figsize=(max(10, len(tasks) * 0.9), max(4, len(models) * 1.2)))
    im = ax.imshow(normed, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    # Annotate cells with raw values; color text by normalized brightness
    for i in range(len(models)):
        for j in range(len(tasks)):
            val = matrix[i, j]
            nval = normed[i, j]
            if not np.isnan(val):
                color = "black" if np.isnan(nval) or nval < 0.6 else "white"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=7.5, color=color)
            else:
                ax.text(j, i, "N/A", ha="center", va="center",
                        fontsize=7.5, color="#aaaaaa")

    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(xlabels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(ylabels, fontsize=10)
    ax.set_title("DGEB Benchmark: Primary Metric per Model × Task\n(color normalized per column)", fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_grouped_bar_by_task(records, out_path: Path):
    # Group by task
    task_ids = sorted({r["task_id"] for r in records})
    models = _sort_models({r["model"] for r in records})
    task_display = {r["task_id"]: r["task_name"] for r in records}
    task_type = {r["task_id"]: r["task_type"] for r in records}
    task_metric = {r["task_id"]: r["primary_metric"] for r in records}

    # score lookup
    scores: dict[tuple, float] = {(r["task_id"], r["model"]): r["value"] for r in records}

    n_tasks = len(task_ids)
    n_models = len(models)
    bar_w = 0.8 / n_models
    x = np.arange(n_tasks)

    colors = plt.cm.Set2(np.linspace(0, 1, n_models))
    fig, ax = plt.subplots(figsize=(max(14, n_tasks * 1.4), 5))

    for k, model in enumerate(models):
        vals = [v if (v := scores.get((t, model), np.nan)) is not None else np.nan for t in task_ids]
        offset = (k - n_models / 2 + 0.5) * bar_w
        bars = ax.bar(x + offset, vals, width=bar_w * 0.9,
                      color=colors[k], label=_model_label(model))

    # Color x-axis tick labels by task type
    ax.set_xticks(x)
    xlabels = ax.set_xticklabels(
        [task_display.get(t, t) for t in task_ids],
        rotation=35, ha="right", fontsize=9,
    )
    for lbl, tid in zip(xlabels, task_ids):
        lbl.set_color(TASK_TYPE_COLORS.get(task_type[tid], "black"))

    ax.set_ylabel("Primary Metric Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("DGEB Benchmark: Per-Task Scores by Model", fontsize=13)
    ax.legend(loc="upper right", fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Task type legend
    type_patches = [
        mpatches.Patch(color=c, label=tt.replace("_", " ").title())
        for tt, c in TASK_TYPE_COLORS.items()
        if tt in task_type.values()
    ]
    ax.legend(handles=ax.get_legend_handles_labels()[0] + type_patches,
              labels=ax.get_legend_handles_labels()[1] + [p.get_label() for p in type_patches],
              loc="upper right", fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_grouped_bar_by_task_type(records, out_path: Path):
    """Average primary metric per (model, task_type)."""
    models = _sort_models({r["model"] for r in records})
    task_types = sorted({r["task_type"] for r in records})

    # Average per (model, task_type)
    sums: dict[tuple, list] = defaultdict(list)
    for r in records:
        if r["value"] is not None:
            sums[(r["model"], r["task_type"])].append(r["value"])

    n_types = len(task_types)
    n_models = len(models)
    bar_w = 0.8 / n_models
    x = np.arange(n_types)

    colors = plt.cm.Set2(np.linspace(0, 1, n_models))
    fig, ax = plt.subplots(figsize=(max(8, n_types * 1.5), 5))

    for k, model in enumerate(models):
        vals = [
            float(np.mean(sums[(model, tt)])) if sums[(model, tt)] else np.nan
            for tt in task_types
        ]
        offset = (k - n_models / 2 + 0.5) * bar_w
        ax.bar(x + offset, vals, width=bar_w * 0.9,
               color=colors[k], label=_model_label(model))

    ax.set_xticks(x)
    ax.set_xticklabels(
        [tt.replace("_", " ").title() for tt in task_types],
        rotation=20, ha="right", fontsize=10,
    )
    ax.set_ylabel("Mean Primary Metric Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("DGEB Benchmark: Average Score per Task Type", fontsize=13)
    ax.legend(loc="upper right", fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_overall_summary(records, out_path: Path):
    """Bar chart of mean score across all tasks per model."""
    models = _sort_models({r["model"] for r in records})
    sums: dict[str, list] = defaultdict(list)
    for r in records:
        if r["value"] is not None:
            sums[r["model"]].append(r["value"])

    means = [float(np.mean(sums[m])) if sums[m] else 0.0 for m in models]
    stds = [float(np.std(sums[m])) if len(sums[m]) > 1 else 0.0 for m in models]
    labels = [_model_label(m) for m in models]

    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, means, yerr=stds, capsize=5,
                  color=colors, edgecolor="white", linewidth=0.5)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.01,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Mean Primary Metric Score")
    ax.set_ylim(0, 1.0)
    ax.set_title("DGEB Benchmark: Overall Model Comparison\n(mean ± std across tasks)", fontsize=12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.xticks(rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_best_model_per_task(records: list[dict]):
    """Print the best model (highest primary metric) for each task."""
    # Group records by task
    task_records: dict[str, list] = defaultdict(list)
    for r in records:
        if r["value"] is not None:
            task_records[r["task_id"]].append(r)

    print("\nBest model per task:")
    print(f"  {'Task':<45} {'Best Model':<25} {'Score':>7}  Metric")
    print("  " + "-" * 90)
    for task_id in sorted(task_records):
        best = max(task_records[task_id], key=lambda r: r["value"])
        task_label = best["task_name"]
        model_label = _model_label(best["model"])
        print(f"  {task_label:<45} {model_label:<25} {best['value']:>7.4f}  {best['primary_metric']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    records = load_results(RESULTS_DIR)
    if not records:
        print("No result JSON files found in", RESULTS_DIR)
        return

    print(f"Loaded {len(records)} result records across "
          f"{len({r['model'] for r in records})} models and "
          f"{len({r['task_id'] for r in records})} tasks.")

    models, tasks, matrix = build_pivot(records)

    plot_heatmap(models, tasks, matrix, records,
                 FIGURES_DIR / "heatmap_primary_metric.png")

    plot_grouped_bar_by_task(records,
                             FIGURES_DIR / "bar_by_task.png")

    plot_grouped_bar_by_task_type(records,
                                  FIGURES_DIR / "bar_by_task_type.png")

    plot_overall_summary(records,
                         FIGURES_DIR / "overall_summary.png")

    print_best_model_per_task(records)

    print("\nAll figures saved to:", FIGURES_DIR)


if __name__ == "__main__":
    main()
