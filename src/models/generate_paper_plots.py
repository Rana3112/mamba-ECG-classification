"""
Publication-quality plots for research paper.
Loads all 4 model summaries and generates plots in paper_plots/
"""

import json
import numpy as np
import matplotlib
import matplotlib.colors

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ===== Publication Style =====
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)

MODELS_DIR = "models_export"
OUT = "paper_plots"
Path(OUT).mkdir(exist_ok=True)

# Load all summaries
results = {}
for f in sorted(Path(MODELS_DIR).glob("*_summary.json")):
    with open(f) as fh:
        data = json.load(fh)
    results[data["model"].lower()] = data

print(f"Loaded: {', '.join(results.keys())}")

COLORS = {
    "mamba": "#E74C3C",
    "transformer": "#3498DB",
    "lstm": "#2ECC71",
    "ssm": "#F39C12",
}
NAMES = {"mamba": "Mamba", "transformer": "Transformer", "lstm": "LSTM", "ssm": "SSM"}


# ===== Figure 1: Training Loss Curves =====
fig, ax = plt.subplots(figsize=(7, 4))
for name, data in results.items():
    h = data["history"]
    epochs = range(1, len(h["train_loss"]) + 1)
    test_key = "val_loss" if "val_loss" in h else "test_loss"
    ax.plot(
        epochs,
        h["train_loss"],
        "o-",
        color=COLORS[name],
        label=f"{NAMES[name]} Train",
        lw=2,
        markersize=5,
    )
    ax.plot(
        epochs,
        h[test_key],
        "s--",
        color=COLORS[name],
        alpha=0.5,
        label=f"{NAMES[name]} Val",
        lw=1.5,
        markersize=4,
    )
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("(a) Training and Validation Loss Across Models")
ax.legend(loc="upper right", ncol=2, framealpha=0.9)
ax.set_xlim(0.5, 6.5)
plt.tight_layout()
plt.savefig(f"{OUT}/fig1_loss_curves.pdf")
plt.savefig(f"{OUT}/fig1_loss_curves.png")
plt.close()
print(f"Saved: fig1_loss_curves")


# ===== Figure 2: Validation Macro F1 Curves =====
fig, ax = plt.subplots(figsize=(7, 4))
for name, data in results.items():
    h = data["history"]
    epochs = range(1, len(h["val_macro_f1"]) + 1)
    f1_key = "val_macro_f1" if "val_macro_f1" in h else "macro_f1"
    ax.plot(
        epochs,
        h[f1_key],
        "o-",
        color=COLORS[name],
        label=NAMES[name],
        lw=2,
        markersize=5,
    )
ax.set_xlabel("Epoch")
ax.set_ylabel("Macro F1 Score (%)")
ax.set_title("(b) Validation Macro F1 Score Across Models")
ax.legend(loc="lower right", framealpha=0.9)
ax.set_xlim(0.5, 6.5)
ax.set_ylim(15, 50)
plt.tight_layout()
plt.savefig(f"{OUT}/fig2_f1_curves.pdf")
plt.savefig(f"{OUT}/fig2_f1_curves.png")
plt.close()
print(f"Saved: fig2_f1_curves")


# ===== Figure 3: Model Comparison Bar Chart =====
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
model_order = ["mamba", "ssm", "lstm", "transformer"]
bar_colors = [COLORS[m] for m in model_order]
labels = [NAMES[m] for m in model_order]

# Best F1
vals = [results[m]["best_macro_f1"] for m in model_order]
bars = axes[0].bar(labels, vals, color=bar_colors, edgecolor="black", linewidth=0.5)
for bar, val in zip(bars, vals):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{val:.1f}%",
        ha="center",
        fontsize=9,
        fontweight="bold",
    )
axes[0].set_ylabel("Macro F1 (%)")
axes[0].set_title("(a) Best Macro F1")
axes[0].set_ylim(0, 50)

# Training Time (hours)
times = [results[m]["total_time_sec"] / 3600 for m in model_order]
bars = axes[1].bar(labels, times, color=bar_colors, edgecolor="black", linewidth=0.5)
for bar, val in zip(bars, times):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.15,
        f"{val:.1f}h",
        ha="center",
        fontsize=9,
        fontweight="bold",
    )
axes[1].set_ylabel("Training Time (hours)")
axes[1].set_title("(b) Total Training Time")

# Final Train Accuracy
accs = [results[m]["history"]["train_acc"][-1] for m in model_order]
bars = axes[2].bar(labels, accs, color=bar_colors, edgecolor="black", linewidth=0.5)
for bar, val in zip(bars, accs):
    axes[2].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{val:.1f}%",
        ha="center",
        fontsize=9,
        fontweight="bold",
    )
axes[2].set_ylabel("Accuracy (%)")
axes[2].set_title("(c) Final Training Accuracy")
axes[2].set_ylim(0, 70)

plt.tight_layout()
plt.savefig(f"{OUT}/fig3_model_comparison.pdf")
plt.savefig(f"{OUT}/fig3_model_comparison.png")
plt.close()
print(f"Saved: fig3_model_comparison")


# ===== Figure 4: Training Accuracy Curves =====
fig, ax = plt.subplots(figsize=(7, 4))
for name, data in results.items():
    h = data["history"]
    epochs = range(1, len(h["train_acc"]) + 1)
    ax.plot(
        epochs,
        h["train_acc"],
        "o-",
        color=COLORS[name],
        label=f"{NAMES[name]}",
        lw=2,
        markersize=5,
    )
ax.set_xlabel("Epoch")
ax.set_ylabel("Training Accuracy (%)")
ax.set_title("(c) Training Accuracy Across Models")
ax.legend(loc="lower right", framealpha=0.9)
ax.set_xlim(0.5, 6.5)
ax.set_ylim(45, 65)
plt.tight_layout()
plt.savefig(f"{OUT}/fig4_train_accuracy.pdf")
plt.savefig(f"{OUT}/fig4_train_accuracy.png")
plt.close()
print(f"Saved: fig4_train_accuracy")


# ===== Figure 5: Summary Table as Figure =====
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis("off")
table_data = [
    [
        "Model",
        "Best Macro F1",
        "Epochs",
        "Train Time (h)",
        "Final Train Acc",
        "Final Val Loss",
    ]
]
for m in model_order:
    d = results[m]
    h = d["history"]
    test_key = "val_loss" if "val_loss" in h else "test_loss"
    table_data.append(
        [
            NAMES[m],
            f"{d['best_macro_f1']:.2f}%",
            str(d["epochs_completed"]),
            f"{d['total_time_sec'] / 3600:.1f}",
            f"{h['train_acc'][-1]:.2f}%",
            f"{h[test_key][-1]:.4f}",
        ]
    )
table = ax.table(cellText=table_data, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)

# Style header row
for j in range(6):
    table[0, j].set_facecolor("#2C3E50")
    table[0, j].set_text_props(color="white", fontweight="bold")

# Style data rows
colors_row = [COLORS[m] for m in model_order]
for i, m in enumerate(model_order):
    for j in range(6):
        table[i + 1, j].set_facecolor((*matplotlib.colors.to_rgb(COLORS[m]), 0.15))

ax.set_title(
    "(d) Summary of Model Performance on CinC 2017 Half 1",
    fontsize=13,
    fontweight="bold",
    pad=20,
)
plt.tight_layout()
plt.savefig(f"{OUT}/fig5_summary_table.pdf")
plt.savefig(f"{OUT}/fig5_summary_table.png")
plt.close()
print(f"Saved: fig5_summary_table")


# ===== Figure 6: Class Distribution =====
ref_path = "training2017/REFERENCE.csv"
with open(ref_path) as f:
    labels_raw = [l.strip().split(",")[1] for l in f if len(l.strip().split(",")) == 2]

from collections import Counter

counts = Counter(labels_raw)
class_order = ["N", "O", "A", "~"]
class_labels = ["Normal (N)", "Other (O)", "AFib (A)", "Noise (~)"]
class_colors = ["#2ECC71", "#3498DB", "#E74C3C", "#F39C12"]
vals = [counts[c] for c in class_order]
pcts = [v / sum(vals) * 100 for v in vals]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Bar chart
bars = axes[0].bar(
    class_labels, vals, color=class_colors, edgecolor="black", linewidth=0.5
)
for bar, val, pct in zip(bars, vals, pcts):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 50,
        f"{val}\n({pct:.1f}%)",
        ha="center",
        fontsize=9,
    )
axes[0].set_ylabel("Number of Samples")
axes[0].set_title("(a) Class Distribution")

# Pie chart
wedges, texts, autotexts = axes[1].pie(
    vals,
    labels=class_labels,
    colors=class_colors,
    autopct="%1.1f%%",
    startangle=90,
    pctdistance=0.85,
)
for t in autotexts:
    t.set_fontsize(9)
axes[1].set_title("(b) Class Proportion")

plt.tight_layout()
plt.savefig(f"{OUT}/fig6_class_distribution.pdf")
plt.savefig(f"{OUT}/fig6_class_distribution.png")
plt.close()
print(f"Saved: fig6_class_distribution")


# ===== Figure 7: Loss Convergence Rate =====
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for idx, (name, data) in enumerate(results.items()):
    ax = axes[idx // 2][idx % 2]
    h = data["history"]
    epochs = range(1, len(h["train_loss"]) + 1)
    test_key = "val_loss" if "val_loss" in h else "test_loss"
    ax.plot(epochs, h["train_loss"], "o-", color=COLORS[name], label="Train Loss", lw=2)
    ax.plot(
        epochs,
        h[test_key],
        "s--",
        color=COLORS[name],
        alpha=0.5,
        label="Val Loss",
        lw=1.5,
    )
    ax.fill_between(epochs, h["train_loss"], h[test_key], alpha=0.1, color=COLORS[name])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(NAMES[name], fontweight="bold")
    ax.legend(framealpha=0.9)
fig.suptitle(
    "(e) Training vs Validation Loss — Per Model", fontsize=13, fontweight="bold"
)
plt.tight_layout()
plt.savefig(f"{OUT}/fig7_loss_per_model.pdf")
plt.savefig(f"{OUT}/fig7_loss_per_model.png")
plt.close()
print(f"Saved: fig7_loss_per_model")


# ===== Figure 8: Efficiency Scatter (F1 vs Time) =====
fig, ax = plt.subplots(figsize=(7, 5))
for name, data in results.items():
    f1 = data["best_macro_f1"]
    time_h = data["total_time_sec"] / 3600
    ax.scatter(
        time_h, f1, color=COLORS[name], s=200, edgecolors="black", linewidth=1, zorder=5
    )
    ax.annotate(
        NAMES[name],
        (time_h, f1),
        textcoords="offset points",
        xytext=(10, 5),
        fontsize=10,
        fontweight="bold",
    )
ax.set_xlabel("Training Time (hours)")
ax.set_ylabel("Best Macro F1 (%)")
ax.set_title("(f) Model Efficiency: F1 Score vs Training Time")
ax.set_ylim(15, 50)
plt.tight_layout()
plt.savefig(f"{OUT}/fig8_efficiency_scatter.pdf")
plt.savefig(f"{OUT}/fig8_efficiency_scatter.png")
plt.close()
print(f"Saved: fig8_efficiency_scatter")


print(f"\nAll 8 figures saved to: {OUT}/")
print("Formats: .png (300 DPI) and .pdf (vector)")
