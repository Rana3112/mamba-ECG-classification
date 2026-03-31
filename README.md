# Mamba-Inspired State Space Models for Efficient Multi-Class Cardiac Arrhythmia Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper: ICANN 2026](https://img.shields.io/badge/paper-ICANN%202026-green.svg)](paper/)
[![Dataset: PhysioNet CinC 2017](https://img.shields.io/badge/dataset-PhysioNet%20CinC%202017-orange.svg)](https://physionet.org/content/challenge-2017/)

Official code repository for the paper accepted at **ICANN 2026** (International Conference on Artificial Neural Networks):

> **"Mamba-Inspired State Space Models for Efficient Multi-Class Cardiac Arrhythmia Classification: A Comparative Study on Edge-Deployable ECG Monitoring"**

---

## Abstract

Real-time cardiac arrhythmia detection on edge devices demands architectures that balance classification accuracy with computational efficiency. This paper presents a comparative evaluation of four sequence modeling architectures — **Mamba** (Selective State Space Model), **Transformer**, **LSTM**, and a **baseline SSM** — for multi-class ECG arrhythmia classification on the PhysioNet/CinC 2017 Challenge dataset. All models are trained on identical data splits with consistent preprocessing and evaluated on a held-out test set of 1,706 ECG recordings across four classes: Normal (N), Other (O), Atrial Fibrillation (A), and Noise (~). Mamba achieves the highest Macro F1 score of **42.91%**, outperforming the Transformer (24.24%), SSM (27.61%), and LSTM (18.80%). Critically, Mamba operates with **O(N) linear complexity**, compared to the Transformer's O(N²) quadratic attention, making it uniquely suited for real-time inference on resource-constrained hardware.

---

## Dataset: PhysioNet/CinC 2017

| Property | Value |
|----------|-------|
| Total recordings | 8,528 single-lead ECG |
| Sampling rate | 300 Hz (downsampled to 150 Hz) |
| Max sequence length | 4,500 time steps |
| Train / Test split | 6,822 / 1,706 (80/20 stratified) |
| Random seed | 42 |

### Class Distribution

```
Normal (N)          ████████████████████████████████████████  5,076 (59.5%)
Other (O)           ████████████████████                      2,311 (27.1%)
Atrial Fibrib (A)   ███████                                    758 ( 8.9%)
Noise (~)           ███                                        383 ( 4.5%)
```

The severe class imbalance — particularly the 4.5% representation of Noise signals — poses a significant challenge for all architectures.

### Preprocessing Pipeline

```
Raw ECG (300 Hz)
    │
    ▼
Downsample by 2× → 150 Hz
    │
    ▼
Truncate to max 4,500 time steps
    │
    ▼
80/20 stratified split (seed=42)
    │
    ▼
No additional normalization — raw signal values used
```

---

## Architectures Compared

### 1. Mamba (Selective State Space Model)

- **2 SelectiveSSMLayer blocks**, d_model=64, state_dim=16
- Input-dependent selection: parameters A, B, C, and Δ are functions of the input
- Causal 1D depthwise convolution (kernel size=3) for local pattern capture
- SiLU activation with input-dependent gates
- Length-aware pooling over sequence

### 2. Transformer

- **2-layer TransformerEncoder**, 4 attention heads, d_model=64
- Sinusoidal positional encoding (max_len=20,000)
- Mean pooling over sequence
- O(N²) self-attention across all time steps

### 3. LSTM

- **2-layer unidirectional LSTM**, hidden_size=64
- Final hidden state as sequence representation
- O(N) complexity but sequential processing prevents GPU parallelization

### 4. SSM (Baseline S4-style)

- Fixed-coefficient State Space Model with learnable A, B, C matrices
- HiPPO initialization for A matrix
- Same wrapper and pooling as Mamba
- Serves as controlled comparison to isolate Mamba's selective mechanism

---

## Training Configuration

All models trained with **identical hyperparameters** for fair comparison:

| Parameter | Value |
|-----------|-------|
| Epochs | 6 (completed) |
| Batch size | 16 (effective, via 4×4 gradient accumulation) |
| Optimizer | AdamW |
| Learning rate | 1e-3 with cosine annealing |
| Loss function | Cross-entropy (unweighted) |
| Mixed precision | Disabled (T4 GPU compatibility) |
| Device | NVIDIA Tesla T4 (16GB) |
| Checkpoint | Best Macro F1 saved per model |

---

## Results

### Validation Performance (Best Macro F1)

| Model | Best Macro F1 | Final Train Acc | Final Val Loss | Training Time |
|-------|:-------------:|:---------------:|:--------------:|:-------------:|
| **Mamba** | **42.91%** | 55.26% | 1.085 | 9.1 hrs |
| SSM | 27.61% | 52.42% | 1.241 | 5.6 hrs |
| Transformer | 24.24% | 58.25% | 1.206 | 0.25 hrs |
| LSTM | 18.80% | 60.19% | 1.270 | 0.30 hrs |

### F1 Score Comparison (Validation)

```
Macro F1 (%)
50 ┤
   │
45 ┤  ████ Mamba (42.91%)
   │
40 ┤  ████
   │
35 ┤  ████
   │
30 ┤  ████        ████ SSM (27.61%)
   │
25 ┤  ████        ████        ████ Transformer (24.24%)
   │
20 ┤  ████        ████        ████        ████ LSTM (18.80%)
   │
15 ┤  ████        ████        ████        ████
   │
10 ┤  ████        ████        ████        ████
   │
 5 ┤  ████        ████        ████        ████
   │
 0 ┼──┴───────────┴───────────┴───────────┴────
       Mamba       SSM      Transformer     LSTM
```

**Mamba achieves 1.55× higher Macro F1** than the next best model (SSM) and **1.77× higher** than the Transformer.

### Training Dynamics Summary

| Metric | Mamba | SSM | Transformer | LSTM |
|--------|-------|-----|-------------|------|
| Loss reduction | 1.29 → 1.07 (steepest) | Gradual | 1.29 → 1.20 (minimal) | Flat |
| F1 progression | 18.8% → 42.9% (peak epoch 5) | 18.8% → 27.6% | Flat ~24% | Flat ~18.8% |
| Overfitting signal | Yes (epoch 5→6 drop) | No | No | No |

### Inference Complexity Comparison

| Model | Per-step complexity | Memory (states) | Parallelizable | Edge-suitable |
|-------|:-------------------:|:---------------:|:--------------:|:-------------:|
| **Mamba** | O(1) | O(d×N) | No (sequential) | **Yes** |
| Transformer | O(N) | O(N²) | Yes | No |
| LSTM | O(1) | O(h) | No (sequential) | Partial |
| SSM | O(1) | O(d×N) | No (sequential) | Yes |

### Model Size (Edge Deployment)

| Model | Approx. Parameters | Estimated Size | Fits on MCU? |
|-------|:------------------:|:--------------:|:------------:|
| Mamba | ~50K | ~200 KB | Yes (256KB+ RAM) |
| Transformer | ~45K | ~180 KB | Yes (256KB+ RAM) |
| LSTM | ~35K | ~140 KB | Yes (256KB+ RAM) |
| SSM | ~48K | ~192 KB | Yes (256KB+ RAM) |

All models fit comfortably on microcontrollers with 256KB+ RAM (e.g., ARM Cortex-M4/M7, ESP32).

---

## Why Mamba Outperforms

### The Selective State Space Mechanism

The core differentiator between Mamba and the baseline SSM is the **input-dependent selection mechanism**:

- **Δ (discretization step), B, and C matrices** are computed as functions of the input at each time step
- This allows the model to **selectively propagate or suppress information** based on ECG morphology
- A sharp QRS complex → large Δ → model "remembers" it
- A flat baseline segment → small Δ → model "forgets" it

The baseline SSM uses **fixed matrices** that cannot distinguish between clinically relevant and irrelevant segments. This explains the **15.3 percentage point gap** between Mamba (42.9%) and SSM (27.6%).

### Why Transformers Struggle

1. **Positional encoding gap:** Sinusoidal encodings designed for discrete tokens, not continuous waveforms
2. **Attention dilution:** With N=4,500 time steps, self-attention spreads focus across all positions. Clinically relevant segments (QRS complex spans ~15 time steps) are a tiny fraction
3. **No causal structure:** Transformer sees all time steps simultaneously; ECG benefits from causal, stateful processing

### Why LSTM Fails

- **Vanishing gradients over 4,500 steps:** Information from early in the recording is lost
- **Unidirectional processing:** Cannot use future context to disambiguate
- **Fixed hidden state size:** 64 dimensions insufficient for full ECG morphology variety

---

## Repository Structure

```
├── paper/                          # Paper and manuscript
│   ├── latex/                      # LaTeX source files for Overleaf
│   │   ├── cinc17_experiments.tex  # Main LaTeX manuscript
│   │   └── *.png                   # Paper figures (fig1-fig8)
│   ├── figures/
│   │   └── architecture/           # Model architecture diagrams
│   ├── research_document.md        # Full paper in Markdown
│   └── Mamba_Inspired_State_Space_Models.pdf  # Final PDF
├── src/
│   ├── models/                     # Model training & analysis
│   │   ├── ablation_mamba.py       # Mamba ablation studies
│   │   ├── generate_paper_plots.py # Plot generation for paper
│   │   ├── inference_plot.py       # Inference visualization
│   │   ├── model_size_analysis.py  # Model size comparison
│   │   ├── realtime_ecg_monitor.py # Real-time ECG monitoring
│   │   └── visualize_results.py    # Results visualization
│   └── data/                       # Data processing
│       ├── download_models.py      # Dataset download script
│       ├── find_kaggle_path.py     # Kaggle dataset path finder
│       ├── merge_cinc17_halves.py  # Dataset merge utility
│       ├── kaggle_cinc17_half1.py  # Half 1 data processing
│       ├── kaggle_cinc17_half2.py  # Half 2 data processing
│       └── kaggle_clean_notebook.py # Data cleaning notebook
├── notebooks/
│   └── ecgMonitorInterface.ipynb   # ECG monitoring interface
├── results/
│   ├── training_plots/             # Training curves (PNG + PDF)
│   ├── inference_plots/            # Inference analysis plots
│   └── model_analysis/             # Model size & efficiency
├── data/                           # Dataset (not tracked in git)
│   └── raw/                        # Raw PhysioNet/CinC 2017 data
├── models/                         # Trained checkpoints (not tracked)
│   └── checkpoints/                # Model weights
├── configs/                        # Configuration files
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Rana3112/mamba-ECG-classification.git
cd mamba-ECG-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Data Setup

```bash
# Download the dataset
python src/data/download_models.py

# Or place training2017.zip in data/ and extract manually
unzip data/training2017.zip -d data/raw/
```

### Data Preprocessing

```bash
# Process dataset halves
python src/data/kaggle_cinc17_half1.py
python src/data/kaggle_cinc17_half2.py

# Merge dataset halves
python src/data/merge_cinc17_halves.py
```

### Analysis & Visualization

```bash
# Generate paper plots (fig1-fig8)
python src/models/generate_paper_plots.py

# Run inference analysis
python src/models/inference_plot.py

# Model size comparison
python src/models/model_size_analysis.py

# Visualize all results
python src/models/visualize_results.py
```

### Real-Time Monitoring

```bash
# Launch ECG monitoring interface
python src/models/realtime_ecg_monitor.py

# Or use the Jupyter notebook
jupyter notebook notebooks/ecgMonitorInterface.ipynb
```

---

## Figures Available

All figures referenced in the paper (fig1–fig8) are available in both **PDF (vector)** and **PNG (300 DPI)** formats:

| Figure | Description | Location |
|--------|-------------|----------|
| Fig. 1 | Loss curves per model | `results/training_plots/fig1_loss_curves.*` |
| Fig. 2 | F1 score curves per model | `results/training_plots/fig2_f1_curves.*` |
| Fig. 3 | Model comparison bar chart | `results/training_plots/fig3_model_comparison.*` |
| Fig. 4 | Training accuracy curves | `results/training_plots/fig4_train_accuracy.*` |
| Fig. 5 | Summary table | `results/training_plots/fig5_summary_table.*` |
| Fig. 6 | Class distribution | `results/training_plots/fig6_class_distribution.*` |
| Fig. 7 | Loss per model | `results/training_plots/fig7_loss_per_model.*` |
| Fig. 8 | Efficiency scatter (F1 vs. time) | `results/training_plots/fig8_efficiency_scatter.*` |

Additional inference plots (confusion matrices, ROC curves, precision-recall, per-class F1) are in `results/inference_plots/`.

---

## Limitations

- **Limited epochs:** All models trained for only 6 epochs due to Kaggle GPU time constraints (30 hrs/week)
- **No hyperparameter tuning:** Identical hyperparameters used for all models
- **Class imbalance:** No focal loss, class weighting, or oversampling applied
- **Single-lead ECG:** CinC 2017 uses lead I only; multi-lead ECG may favor architectures with more parameters

---

## Future Work

1. Longer training with early stopping and class-weighted loss
2. Multi-lead ECG input to test scalability
3. Quantization and pruning for further edge optimization
4. Comparison with hybrid Mamba-Transformer architectures

---

## Citing This Work

If you use this code or reference our work, please cite:

```bibtex
@article{mamba-ecg-2026,
  title={Mamba-Inspired State Space Models for Efficient Multi-Class Cardiac Arrhythmia Classification: A Comparative Study on Edge-Deployable ECG Monitoring},
  author={Utkarsh Rana},
  journal={International Conference on Artificial Neural Networks (ICANN)},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **PhysioNet/CinC 2017 Challenge** for the dataset
- **Gu & Dao (2023)** — Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- **Clifford et al. (2017)** — AF Classification from a Short Single Lead ECG Recording
- All contributors and reviewers
