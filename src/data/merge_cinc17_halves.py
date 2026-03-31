"""
============================================================================
 MERGE RESULTS — Combine Half 1 and Half 2 CINC 2017 training results
============================================================================
 Run this locally AFTER downloading checkpoints from both Kaggle notebooks.
 
 Usage:
   python merge_cinc17_halves.py --half1_dir ./half1_results --half2_dir ./half2_results
============================================================================
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def load_summary(directory, half, model_name):
    path = os.path.join(directory, f"cinc17_half{half}_{model_name.lower()}_summary.json")
    if not os.path.exists(path):
        print(f"  ⚠ Not found: {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--half1_dir", required=True, help="Directory with Half 1 results")
    parser.add_argument("--half2_dir", required=True, help="Directory with Half 2 results")
    parser.add_argument("--output_dir", default="results/cinc17_merged", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    models = ["Mamba", "Transformer", "LSTM", "SSM"]

    combined = {}
    for model_name in models:
        s1 = load_summary(args.half1_dir, 1, model_name)
        s2 = load_summary(args.half2_dir, 2, model_name)

        if s1 is None or s2 is None:
            print(f"  Skipping {model_name} — missing data")
            continue

        # Average best F1 across both halves
        avg_f1 = (s1["best_macro_f1"] + s2["best_macro_f1"]) / 2
        total_time = s1["total_time_sec"] + s2["total_time_sec"]

        combined[model_name] = {
            "avg_macro_f1": avg_f1,
            "half1_f1": s1["best_macro_f1"],
            "half2_f1": s2["best_macro_f1"],
            "total_training_time_sec": total_time,
            "half1_epochs": s1["epochs_completed"],
            "half2_epochs": s2["epochs_completed"],
        }

        print(f"\n{model_name}:")
        print(f"  Half 1 F1: {s1['best_macro_f1']:.1f}% ({s1['epochs_completed']} epochs)")
        print(f"  Half 2 F1: {s2['best_macro_f1']:.1f}% ({s2['epochs_completed']} epochs)")
        print(f"  Average F1: {avg_f1:.1f}%")
        print(f"  Total time: {total_time:.0f}s")

    # Save combined JSON
    with open(os.path.join(args.output_dir, "cinc17_combined_results.json"), "w") as f:
        json.dump(combined, f, indent=2)

    # Plot comparison
    if combined:
        names = list(combined.keys())
        avg_f1s = [combined[n]["avg_macro_f1"] for n in names]
        h1_f1s = [combined[n]["half1_f1"] for n in names]
        h2_f1s = [combined[n]["half2_f1"] for n in names]

        x = np.arange(len(names))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width, h1_f1s, width, label="Half 1", color="steelblue")
        ax.bar(x, h2_f1s, width, label="Half 2", color="indianred")
        ax.bar(x + width, avg_f1s, width, label="Average", color="forestgreen")

        ax.set_ylabel("Macro-F1 (%)")
        ax.set_title("PhysioNet CINC 2017 — Model Comparison (Split Training)")
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 100)

        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, "cinc17_combined_comparison.png"), dpi=200)
        plt.close(fig)
        print(f"\nPlot saved to {args.output_dir}/cinc17_combined_comparison.png")

    print(f"\nAll results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
