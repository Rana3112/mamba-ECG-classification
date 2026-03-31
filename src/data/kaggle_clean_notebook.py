"""
Clean Kaggle Notebook — Load Trained Models & Run Everything
Upload these as Kaggle Datasets:
  1. physionet (training2017 folder with REFERENCE.csv + .mat files)
  2. model_export (all .pth and _summary.json files)
Then set MODEL_DIR and DATA_DIR below to match your dataset paths.
"""

import os, json, math, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    f1_score,
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ============================================================
# 1. CONFIG — Change these paths to match your Kaggle datasets
# ============================================================
DATA_DIR = "/kaggle/input/datasets/nebula3112/physionet3112/training2017/"
MODEL_DIR = "/kaggle/input/datasets/nebula3112/model-export/models_export/"
OUTPUT_DIR = "/kaggle/working/"
DOWNSAMPLE = 2
MAX_LEN = 4500
CLASSES = ["N", "O", "A", "~"]
CLASS_NAMES = {"N": "Normal", "O": "Other", "A": "AFib", "~": "Noise"}
CLASS_NAMES_LIST = ["Normal", "Other", "AFib", "Noise"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Check what files exist
print(f"\nModel files in {MODEL_DIR}:")
for f in sorted(os.listdir(MODEL_DIR)):
    size = os.path.getsize(os.path.join(MODEL_DIR, f)) / (1024 * 1024)
    print(f"  {f} ({size:.1f} MB)")


# ============================================================
# 2. MODEL ARCHITECTURES (must match training exactly)
# ============================================================


class SelectiveSSMLayer(nn.Module):
    def __init__(self, d_model=64, state_dim=16):
        super().__init__()
        self.d_model, self.N = d_model, state_dim
        self.A_log = nn.Parameter(
            torch.log(
                torch.arange(1, state_dim + 1).float().unsqueeze(0).repeat(d_model, 1)
            )
        )
        self.D = nn.Parameter(torch.ones(d_model))
        self.B_proj = nn.Linear(d_model, state_dim, bias=False)
        self.C_proj = nn.Linear(d_model, state_dim, bias=False)
        self.delta_proj = nn.Linear(d_model, d_model, bias=True)
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.out_proj = nn.Linear(d_model, d_model)
        self.conv1d = nn.Conv1d(d_model, d_model, 3, padding=1, groups=d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B_sz, L, D = x.shape
        A = -torch.exp(self.A_log)
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_in.transpose(1, 2)).transpose(1, 2)
        x_conv = F.silu(x_conv)
        delta = F.softplus(self.delta_proj(x_conv))
        B_sel, C_sel = self.B_proj(x_conv), self.C_proj(x_conv)
        h = torch.zeros(B_sz, D, self.N, device=x.device)
        outputs = []
        for t in range(L):
            dt = delta[:, t, :]
            A_bar = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))
            b_t = dt.unsqueeze(-1) * B_sel[:, t, :].unsqueeze(1)
            h = A_bar * h + b_t * x_conv[:, t, :].unsqueeze(-1)
            outputs.append((h * C_sel[:, t, :].unsqueeze(1)).sum(-1))
        y = torch.stack(outputs, dim=1) + x_conv * self.D.unsqueeze(0).unsqueeze(0)
        return self.out_proj(y * F.silu(z))


class MambaClassifier(nn.Module):
    def __init__(
        self, input_dim=1, d_model=64, state_dim=16, num_layers=2, num_classes=4
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList(
            [SelectiveSSMLayer(d_model, state_dim) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.classifier = nn.Linear(d_model, num_classes)


class S4Layer(nn.Module):
    def __init__(self, d_model=64, state_dim=16):
        super().__init__()
        self.d_model, self.N = d_model, state_dim
        self.A = nn.Parameter(-torch.exp(torch.randn(d_model, state_dim)))
        self.B = nn.Parameter(torch.randn(d_model, state_dim))
        self.C = nn.Parameter(torch.randn(d_model, state_dim))
        self.D = nn.Parameter(torch.ones(d_model))
        self.delta = nn.Parameter(torch.rand(d_model) * 0.1 + 0.001)

    def forward(self, x):
        B_sz, L, D = x.shape
        delta = torch.sigmoid(self.delta).view(1, 1, self.d_model, 1)
        A_bar = torch.exp(delta * self.A.unsqueeze(0).unsqueeze(0))
        B_bar = delta * self.B.unsqueeze(0).unsqueeze(0)
        h = torch.zeros(B_sz, D, self.N, device=x.device)
        outputs = []
        for t in range(L):
            x_t = x[:, t, :].unsqueeze(-1)
            h = A_bar.squeeze(0).squeeze(0) * h + B_bar.squeeze(0).squeeze(0) * x_t
            outputs.append((h * self.C.unsqueeze(0)).sum(-1))
        return torch.stack(outputs, dim=1) + x * self.D.unsqueeze(0).unsqueeze(0)


class SSMClassifier(nn.Module):
    def __init__(self, d_model=64, state_dim=16, num_layers=2, num_classes=4):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.layers = nn.ModuleList(
            [S4Layer(d_model, state_dim) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.classifier = nn.Linear(d_model, num_classes)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        pe[:, 0::2] = torch.sin(
            position
            * torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
        )
        pe[:, 1::2] = torch.cos(
            position
            * torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
        )
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=1, d_model=64, num_classes=4, num_layers=2, nhead=4):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers
        )
        self.classifier = nn.Linear(d_model, num_classes)


class LSTMClassifier(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=False
        )
        self.classifier = nn.Linear(hidden_size, num_classes)


# ============================================================
# 3. WRAPPERS (must match training exactly)
# ============================================================


class WrapperBase(nn.Module):
    def __init__(self, backbone, d_model, num_classes):
        super().__init__()
        self.backbone = backbone
        if hasattr(self.backbone, "classifier"):
            self.backbone.classifier = nn.Identity()
        self.head = nn.Linear(d_model, num_classes)

    def pool(self, x, lengths):
        mask = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None]
        mask = mask.unsqueeze(-1).float()
        return (x * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)


class MambaWrapper(WrapperBase):
    def forward(self, x, lengths):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x = self.backbone.input_proj(x)
        for layer, norm in zip(self.backbone.layers, self.backbone.norms):
            x = norm(x + layer(x))
        return self.head(self.pool(x, lengths))


class SSMWrapper(WrapperBase):
    def forward(self, x, lengths):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x = self.backbone.input_proj(x)
        for layer, norm in zip(self.backbone.layers, self.backbone.norms):
            x = norm(x + layer(x))
        return self.head(self.pool(x, lengths))


class TransformerWrapper(WrapperBase):
    def forward(self, x, lengths):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        embedded = self.backbone.embedding(x)
        mask = (
            torch.arange(embedded.size(1), device=x.device)[None, :] >= lengths[:, None]
        )
        encoded = self.backbone.pos_encoder(embedded)
        encoded = self.backbone.transformer_encoder(encoded, src_key_padding_mask=mask)
        return self.head(self.pool(encoded, lengths))


class LSTMWrapper(WrapperBase):
    def forward(self, x, lengths):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.backbone.lstm(packed)
        return self.head(hidden[-1])


# ============================================================
# 4. LOAD MODELS
# ============================================================


def load_model(wrapper_cls, backbone_cls, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        first_key = list(ckpt.keys())[0] if ckpt else None
        if first_key and isinstance(ckpt[first_key], torch.Tensor):
            state = ckpt
        else:
            print(f"  Unknown format, keys: {list(ckpt.keys())[:5]}")
            return None

    backbone = backbone_cls()
    model = wrapper_cls(backbone, d_model=64, num_classes=4)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


MODELS = {}
MODEL_CONFIG = {
    "Mamba": (MambaWrapper, MambaClassifier, "cinc17_half1_mamba_best.pth"),
    "Transformer": (
        TransformerWrapper,
        TransformerClassifier,
        "cinc17_half1_transformer_best.pth",
    ),
    "LSTM": (LSTMWrapper, LSTMClassifier, "cinc17_half1_lstm_best.pth"),
    "SSM": (SSMWrapper, SSMClassifier, "cinc17_half1_ssm_best.pth"),
}

print("\nLoading models...")
for name, (wcls, bcls, fname) in MODEL_CONFIG.items():
    path = os.path.join(MODEL_DIR, fname)
    if os.path.exists(path):
        m = load_model(wcls, bcls, path, DEVICE)
        if m is not None:
            MODELS[name] = m
            print(f"  {name} loaded")
        else:
            print(f"  {name} FAILED")
    else:
        print(f"  {name} file not found: {fname}")

print(f"\n{len(MODELS)}/4 models loaded")


# ============================================================
# 5. LOAD DATA
# ============================================================

ref_path = os.path.join(DATA_DIR, "REFERENCE.csv")
with open(ref_path) as f:
    df_lines = [l.strip().split(",") for l in f if len(l.strip().split(",")) == 2]
records = [r for r, _ in df_lines]
labels = [l for _, l in df_lines]

train_recs, test_recs, train_labels, test_labels = train_test_split(
    records, labels, test_size=0.2, random_state=42, stratify=labels
)
print(f"Test set: {len(test_recs)} recordings")


# ============================================================
# 6. HELPER FUNCTIONS
# ============================================================


def load_ecg(record_id):
    mat = sio.loadmat(os.path.join(DATA_DIR, f"{record_id}.mat"))
    sig = mat["val"].flatten().astype(np.float32)
    if DOWNSAMPLE > 1:
        sig = sig[::DOWNSAMPLE]
    if len(sig) > MAX_LEN:
        sig = sig[:MAX_LEN]
    return sig


def predict_single(signal, model, device):
    sig_t = torch.from_numpy(signal).unsqueeze(0).to(device)
    length = torch.tensor([len(signal)]).to(device)
    with torch.no_grad():
        logits = model(sig_t, length)
        probs = torch.softmax(logits.float(), dim=1)
        probs = torch.nan_to_num(probs, nan=0.25)
    return probs[0].cpu().numpy()


def predict_all(signal):
    results = {}
    for name, model in MODELS.items():
        probs = predict_single(signal, model, DEVICE)
        pred_id = int(np.argmax(probs))
        results[name] = {
            "class": CLASSES[pred_id],
            "class_name": CLASS_NAMES_LIST[pred_id],
            "confidence": float(probs[pred_id]),
            "probs": {c: float(probs[i]) for i, c in enumerate(CLASSES)},
        }
    return results


# ============================================================
# 7. INFERENCE ON TEST SET
# ============================================================

print("\n" + "=" * 50)
print("FULL TEST SET INFERENCE")
print("=" * 50)


def run_full_inference():
    all_preds = {name: [] for name in MODELS}
    all_labels = []
    all_probs = {name: [] for name in MODELS}
    t0 = time.time()

    for i, (rec_id, true_label) in enumerate(zip(test_recs, test_labels)):
        signal = load_ecg(rec_id)
        true_id = CLASSES.index(true_label)
        all_labels.append(true_id)

        for name, model in MODELS.items():
            probs = predict_single(signal, model, DEVICE)
            all_preds[name].append(int(np.argmax(probs)))
            all_probs[name].append(probs)

        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{len(test_recs)}")

    elapsed = time.time() - t0
    all_labels = np.array(all_labels)

    for name in MODELS:
        preds = np.array(all_preds[name])
        probs = np.array(all_probs[name])
        macro_f1 = f1_score(all_labels, preds, average="macro") * 100
        acc = np.mean(preds == all_labels) * 100
        print(f"\n{name}: Macro F1={macro_f1:.2f}% | Accuracy={acc:.2f}%")
        print(
            classification_report(
                all_labels,
                preds,
                target_names=CLASS_NAMES_LIST,
                digits=2,
                zero_division=0,
            )
        )

    return all_preds, all_labels, all_probs, elapsed


all_preds, all_labels, all_probs, elapsed = run_full_inference()
print(f"\nTotal inference time: {elapsed:.1f}s")


# ============================================================
# 8. PLOTS
# ============================================================

PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# --- 8a. Confusion Matrices ---
n = len(MODELS)
fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
if n == 1:
    axes = [axes]
fig.suptitle("Confusion Matrices — Half 1 Test Set", fontsize=16, fontweight="bold")
for ax, (name, model) in zip(axes, MODELS.items()):
    cm = confusion_matrix(all_labels, all_preds[name])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES_LIST,
        yticklabels=CLASS_NAMES_LIST,
        ax=ax,
    )
    ax.set_title(name, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/confusion_matrices.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved: confusion_matrices.png")

# --- 8b. Per-Class F1 ---
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle("Per-Class F1 — Half 1 Test Set", fontsize=16, fontweight="bold")
colors = {
    "Mamba": "#E74C3C",
    "Transformer": "#3498DB",
    "LSTM": "#2ECC71",
    "SSM": "#F39C12",
}
x = np.arange(4)
w = 0.8 / len(MODELS)
for i, name in enumerate(MODELS):
    f1s = f1_score(all_labels, all_preds[name], average=None, labels=range(4)) * 100
    offset = (i - len(MODELS) / 2 + 0.5) * w
    bars = ax.bar(
        x + offset,
        f1s,
        w,
        label=name,
        color=colors.get(name, "#999"),
        edgecolor="black",
        linewidth=0.5,
    )
    for bar, val in zip(bars, f1s):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.5,
                f"{val:.1f}",
                ha="center",
                fontsize=7,
            )
ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES_LIST)
ax.set_ylabel("F1 (%)")
ax.legend()
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/per_class_f1.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved: per_class_f1.png")

# --- 8c. Overall Metrics ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Overall Metrics — Half 1 Test Set", fontsize=16, fontweight="bold")
model_names = list(MODELS.keys())
bar_colors = [colors.get(n, "#999") for n in model_names]
for idx, (metric, func) in enumerate(
    [
        (
            "Macro F1",
            lambda n: f1_score(all_labels, all_preds[n], average="macro") * 100,
        ),
        (
            "Weighted F1",
            lambda n: f1_score(all_labels, all_preds[n], average="weighted") * 100,
        ),
        ("Accuracy", lambda n: np.mean(np.array(all_preds[n]) == all_labels) * 100),
    ]
):
    vals = [func(n) for n in model_names]
    bars = axes[idx].bar(
        model_names, vals, color=bar_colors, edgecolor="black", linewidth=0.5
    )
    axes[idx].set_title(metric, fontweight="bold")
    axes[idx].set_ylim(0, max(vals) * 1.2)
    for bar, val in zip(bars, vals):
        axes[idx].text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.5,
            f"{val:.1f}%",
            ha="center",
            fontweight="bold",
            fontsize=9,
        )
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/overall_metrics.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved: overall_metrics.png")


# ============================================================
# 9. SINGLE SAMPLE COMPARISON
# ============================================================

print("\n" + "=" * 50)
print("SINGLE SAMPLE COMPARISON")
print("=" * 50)

idx = random.randint(0, len(test_recs) - 1)
rec_id, true_label = test_recs[idx], test_labels[idx]
signal = load_ecg(rec_id)
predictions = predict_all(signal)

print(f"Record: {rec_id} | True: {CLASS_NAMES[true_label]} ({true_label})")
for name, pred in predictions.items():
    correct = "✓" if pred["class"] == true_label else "✗"
    print(f"  {name:<12}: {pred['class']} ({pred['confidence']:.1%}) {correct}")

# Plot
fig = plt.figure(figsize=(16, 10))
ax_main = plt.subplot2grid((6, 4), (0, 0), colspan=4, rowspan=3)
ax_main.plot(signal, color="#1a1a1a", linewidth=0.8)
ax_main.set_title(
    f"ECG Record {rec_id} — True: {CLASS_NAMES[true_label]} ({true_label})",
    fontsize=13,
    fontweight="bold",
)
ax_main.set_xlabel("Time Steps (150 Hz)")
ax_main.set_ylabel("Amplitude")
ax_main.grid(True, alpha=0.3)

for idx, (name, pred) in enumerate(predictions.items()):
    ax = plt.subplot2grid((6, 4), (3, idx), rowspan=3)
    probs = pred["probs"]
    cls_colors = [colors.get(CLASS_NAMES_LIST[i], "#999") for i in range(4)]
    vals = [probs[c] for c in CLASSES]
    bars = ax.bar(
        CLASSES,
        vals,
        color=["#2ecc71", "#3498db", "#e74c3c", "#f39c12"],
        edgecolor="#333",
        linewidth=0.5,
    )
    for bar, cls in zip(bars, CLASSES):
        if cls == pred["class"]:
            bar.set_edgecolor("black")
            bar.set_linewidth(2.5)
    correct = "✓" if pred["class"] == true_label else "✗"
    c = "#059669" if pred["class"] == true_label else "#dc2626"
    ax.set_title(f"{name}: {pred['class']} {correct}", color=c, fontweight="bold")
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, vals):
        if val > 0.02:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.02,
                f"{val:.0%}",
                ha="center",
                fontsize=9,
            )

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/sample_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved: sample_comparison.png")


# ============================================================
# 10. MODEL SIZE ANALYSIS
# ============================================================

print("\n" + "=" * 50)
print("MODEL SIZE ANALYSIS")
print("=" * 50)

print(f"\n{'Model':<15} {'Params':>10} {'Float32 KB':>12} {'File MB':>10}")
print("-" * 50)
for name, model in MODELS.items():
    params = sum(p.numel() for p in model.parameters())
    kb = params * 4 / 1024
    fname = MODEL_CONFIG[name][2]
    fpath = os.path.join(MODEL_DIR, fname)
    mb = os.path.getsize(fpath) / (1024 * 1024) if os.path.exists(fpath) else 0
    print(f"{name:<15} {params:>10,} {kb:>12.1f} {mb:>10.2f}")

print("\nDone. All plots saved to:", PLOT_DIR)
