# Mamba-Inspired State Space Models for Efficient Multi-Class Cardiac Arrhythmia Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-ICANN%202026-green.svg)](paper/)

Official code repository for the ICANN 2026 paper: **"Mamba-Inspired State Space Models for Efficient Multi-Class Cardiac Arrhythmia Classification: A Comparative Study on Edge-Deployable ECG Monitoring"**

## Overview

This repository contains the complete implementation for comparing four sequence modeling architectures — **Mamba (Selective State Space Model)**, **Transformer**, **LSTM**, and **baseline SSM** — for multi-class ECG arrhythmia classification on the PhysioNet/CinC 2017 Challenge dataset.

### Key Findings

- **Mamba achieves the highest Macro F1 score of 42.91%**, outperforming Transformer (24.24%), SSM (27.61%), and LSTM (18.80%)
- Mamba operates with **O(N) linear complexity** vs Transformer's O(N²) quadratic attention
- All models trained on identical data splits with consistent preprocessing
- Evaluated on a held-out test set of 1,706 ECG recordings across 4 classes: Normal (N), Other (O), Atrial Fibrillation (A), and Noise (~)

## Repository Structure

```
├── paper/                          # Paper and manuscript
│   ├── latex/                      # LaTeX source files for Overleaf
│   │   ├── cinc17_experiments.tex  # Main LaTeX manuscript
│   │   └── *.png                   # Paper figures
│   ├── figures/                    # High-resolution figures
│   │   ├── architecture/           # Model architecture diagrams
│   │   └── pipeline/               # Data pipeline flowcharts
│   ├── research_document.md        # Full paper in Markdown
│   └── Mamba_Inspired_State_Space_Models.pdf  # Final PDF
├── src/                            # Source code
│   ├── models/                     # Model definitions and training
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
├── notebooks/                      # Jupyter notebooks
│   └── ecgMonitorInterface.ipynb   # ECG monitoring interface
├── results/                        # Experimental results
│   ├── training_plots/             # Training curves and metrics
│   ├── inference_plots/            # Inference analysis plots
│   └── model_analysis/             # Model size and efficiency analysis
├── data/                           # Dataset (not tracked in git)
│   ├── raw/                        # Raw PhysioNet/CinC 2017 data
│   └── training2017.zip            # Original dataset archive
├── models/                         # Trained model checkpoints (not tracked)
│   └── models_export/              # Exported model weights (.pth)
├── configs/                        # Configuration files
├── .gitignore                      # Git ignore rules
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Dataset

This project uses the **PhysioNet/Computing in Cardiology Challenge 2017** dataset:
- 8,528 single-lead ECG recordings sampled at 300 Hz
- Four classes: Normal (59.5%), Other (27.1%), Atrial Fibrillation (8.9%), Noise (4.5%)
- Download from: [PhysioNet CinC 2017](https://physionet.org/content/challenge-2017/) or [Kaggle](https://www.kaggle.com/datasets/kinguistics/heartbeat-sounds)

### Setup Data

```bash
# Download the dataset
python src/data/download_models.py

# Or place training2017.zip in data/ and extract manually
unzip data/training2017.zip -d data/raw/
```

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/mamba-ecg-classification.git
cd mamba-ecg-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training Models

```bash
# Run data preprocessing
python src/data/kaggle_cinc17_half1.py
python src/data/kaggle_cinc17_half2.py

# Merge dataset halves
python src/data/merge_cinc17_halves.py
```

### Analysis and Visualization

```bash
# Generate paper plots
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

## Results Summary

| Model | Macro F1 | Weighted F1 | Accuracy | Complexity |
|-------|----------|-------------|----------|------------|
| Mamba | 42.91%   | -           | -        | O(N)       |
| Transformer | 24.24% | -        | -        | O(N²)      |
| SSM   | 27.61%   | -           | -        | O(N)       |
| LSTM  | 18.80%   | -           | -        | O(N)       |

*Validation metrics from training. Full test-set analysis in `results/inference_plots/inference_analysis.md`*

## Citing This Work

If you use this code or reference our work, please cite:

```bibtex
@article{mamba-ecg-2026,
  title={Mamba-Inspired State Space Models for Efficient Multi-Class Cardiac Arrhythmia Classification: A Comparative Study on Edge-Deployable ECG Monitoring},
  author={YOUR NAME},
  journal={International Conference on Artificial Neural Networks (ICANN)},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PhysioNet/CinC 2017 Challenge for the dataset
- Mamba: Selective State Space Models (Gu & Dao, 2023)
- All contributors and reviewers
