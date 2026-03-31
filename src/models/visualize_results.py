"""
Visualize training results from kaggle_cinc17_half1.py
Usage: python visualize_results.py --results results/
"""

import json
import argparse
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_results(path):
    results = {}
    p = Path(path)
    if p.is_dir():
        for f in sorted(p.glob("*_summary.json")):
            with open(f) as fh:
                data = json.load(fh)
            results[data["model"].lower()] = data
    elif p.is_file():
        with open(p) as f:
            data = json.load(f)
        if "model" in data:
            results[data["model"].lower()] = data
    return results


def safe_get(h, *keys):
    for k in keys:
        if k in h:
            return h[k]
    return None


def plot_training_curves(results, out):
    n = len(results)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows))
    fig.suptitle("Training Loss — Half 1", fontsize=16, fontweight="bold")
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    for idx, name in enumerate(results):
        ax = axes[idx]
        h = results[name]["history"]
        epochs = range(1, len(h["train_loss"]) + 1)
        test_key = "test_loss" if "test_loss" in h else "val_loss"
        ax.plot(
            epochs, h["train_loss"], "o-", color=colors[idx], label="Train Loss", lw=2
        )
        ax.plot(
            epochs,
            h[test_key],
            "s--",
            color=colors[idx],
            alpha=0.6,
            label="Val/Test Loss",
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(name.upper(), fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)
    plt.tight_layout()
    path = Path(out) / "training_curves_loss.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_accuracy_f1(results, out):
    n = len(results)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows))
    fig.suptitle("Accuracy & F1 — Half 1", fontsize=16, fontweight="bold")
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    for idx, name in enumerate(results):
        ax = axes[idx]
        h = results[name]["history"]
        epochs = range(1, len(h["train_acc"]) + 1)
        f1_key = "macro_f1" if "macro_f1" in h else "val_macro_f1"
        ax.plot(
            epochs, h["train_acc"], "o-", color=colors[idx], label="Train Acc", lw=2
        )
        if "test_acc" in h:
            ax.plot(
                epochs,
                h["test_acc"],
                "s--",
                color=colors[idx],
                alpha=0.6,
                label="Val Acc",
            )
        ax.plot(epochs, h[f1_key], "d:", color="#9b59b6", label="Macro F1")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score (%)")
        ax.set_title(name.upper(), fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)
    plt.tight_layout()
    path = Path(out) / "training_curves_accuracy.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_model_comparison(results, out):
    models = list(results.keys())
    f1_key = "best_f1" if "best_f1" in results[models[0]] else "best_macro_f1"
    best_f1s = [results[m].get(f1_key, 0) for m in models]
    times = [
        results[m].get("total_time_seconds", results[m].get("total_time_sec", 0)) / 3600
        for m in models
    ]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"][: len(models)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Model Comparison — Half 1", fontsize=16, fontweight="bold")

    bars = axes[0].bar([m.upper() for m in models], best_f1s, color=colors)
    axes[0].set_title("Best Macro F1", fontweight="bold")
    axes[0].set_ylabel("F1 Score (%)")
    for bar, val in zip(bars, best_f1s):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%",
            ha="center",
            fontweight="bold",
        )

    bars = axes[1].bar([m.upper() for m in models], times, color=colors)
    axes[1].set_title("Training Time", fontweight="bold")
    axes[1].set_ylabel("Hours")
    for bar, val in zip(bars, times):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.1f}h",
            ha="center",
            fontweight="bold",
        )

    plt.tight_layout()
    path = Path(out) / "model_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results", default="results", help="Directory with *_summary.json"
    )
    parser.add_argument("--output", default="plots")
    args = parser.parse_args()

    Path(args.output).mkdir(exist_ok=True)
    results = load_results(args.results)

    if not results:
        print("No results found!")
        return

    print(f"Loaded {len(results)} models: {', '.join(results.keys())}")
    print()

    plot_training_curves(results, args.output)
    plot_accuracy_f1(results, args.output)
    plot_model_comparison(results, args.output)

    print()
    print(f"All plots saved to: {args.output}/")


if __name__ == "__main__":
    main()
