"""
Inference + Visualization for ALL 4 Models (Half 1)
Runs on Kaggle T4 GPU.
Upload all .pth files to /kaggle/working/ as a dataset.
"""

import os, json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
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

# ===== Config =====
DATA_DIR = "/kaggle/input/datasets/nebula3112/physionet/training2017/"
MODEL_DIR = "/kaggle/working/"
PLOT_DIR = "/kaggle/working/plots/"
DOWNSAMPLE = 2
MAX_LEN = 4500
CLASSES = ["N", "O", "A", "~"]
CLASS_NAMES = ["Normal", "Other", "AFib", "Noise"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(PLOT_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ===== Dataset =====
class ECGDataset(Dataset):
    def __init__(self, records, labels, data_dir):
        self.records, self.labels, self.data_dir = records, labels, data_dir

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        try:
            mat = sio.loadmat(os.path.join(self.data_dir, f"{self.records[idx]}.mat"))
            sig = mat["val"].flatten().astype(np.float32)
        except Exception:
            sig = np.zeros(1, dtype=np.float32)
        if DOWNSAMPLE > 1:
            sig = sig[::DOWNSAMPLE]
        if len(sig) > MAX_LEN:
            sig = sig[:MAX_LEN]
        return (
            torch.from_numpy(sig),
            torch.tensor(CLASSES.index(self.labels[idx]), dtype=torch.long),
            torch.tensor(len(sig)),
        )


def collate_pad(batch):
    sigs, labels, lens = zip(*batch)
    return (
        pad_sequence(sigs, batch_first=True, padding_value=0.0),
        torch.stack(labels),
        torch.stack(lens),
    )


# ===== Models (from kaggle_cinc17_half1.py) =====
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

    def forward(self, x):
        x = self.input_proj(x)
        for layer, norm in zip(self.layers, self.norms):
            x = norm(x + layer(x))
        return self.classifier(x.mean(dim=1))


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

    def forward(self, x):
        x = self.input_proj(x)
        for layer, norm in zip(self.layers, self.norms):
            x = norm(x + layer(x))
        return self.classifier(x.mean(dim=1))


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

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return self.classifier(x.mean(dim=1))


class LSTMClassifier(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=False
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        return self.classifier(out[:, -1, :])


# ===== Wrappers =====
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


# ===== Load Data =====
ref_path = os.path.join(DATA_DIR, "REFERENCE.csv")
with open(ref_path) as f:
    df_lines = [l.strip().split(",") for l in f if len(l.strip().split(",")) == 2]
records = [r for r, _ in df_lines]
labels = [l for _, l in df_lines]

train_recs, test_recs, train_labels, test_labels = train_test_split(
    records, labels, test_size=0.2, random_state=42, stratify=labels
)
print(f"Test samples: {len(test_recs)}")

test_dataset = ECGDataset(test_recs, test_labels, DATA_DIR)
test_loader = DataLoader(
    test_dataset, batch_size=16, shuffle=False, collate_fn=collate_pad
)


# ===== Load Models =====
def load_model(wrapper_cls, backbone_cls, path, device, **kwargs):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    backbone = backbone_cls(**kwargs)
    model = wrapper_cls(backbone, d_model=64, num_classes=4)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


MODELS = {}
print("\nLoading models...")

# Mamba
path = os.path.join(MODEL_DIR, "cinc17_half1_mamba_latest.pth")
if os.path.exists(path):
    MODELS["Mamba"] = load_model(MambaWrapper, MambaClassifier, path, DEVICE)
    print(f"  Mamba loaded")

# Transformer
path = os.path.join(MODEL_DIR, "cinc17_half1_transformer_latest.pth")
if os.path.exists(path):
    MODELS["Transformer"] = load_model(
        TransformerWrapper, TransformerClassifier, path, DEVICE
    )
    print(f"  Transformer loaded")

# LSTM
path = os.path.join(MODEL_DIR, "cinc17_half1_lstm_latest.pth")
if os.path.exists(path):
    MODELS["LSTM"] = load_model(LSTMWrapper, LSTMClassifier, path, DEVICE)
    print(f"  LSTM loaded")

# SSM
path = os.path.join(MODEL_DIR, "cinc17_half1_ssm_latest.pth")
if os.path.exists(path):
    MODELS["SSM"] = load_model(SSMWrapper, SSMClassifier, path, DEVICE)
    print(f"  SSM loaded")

print(f"\nTotal models loaded: {len(MODELS)}")


# ===== Inference =====
def run_inference(model, loader, device):
    all_preds, all_labels, all_probs = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for signals, labs, lengths in loader:
            signals, lengths = signals.to(device), lengths.to(device)
            logits = model(signals, lengths)
            probs = torch.softmax(logits.float(), dim=1)
            probs = torch.nan_to_num(probs, nan=0.25, posinf=1.0, neginf=0.0)
            all_preds.extend(probs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labs.numpy())
            all_probs.extend(probs.cpu().numpy())
    elapsed = time.time() - t0
    return np.array(all_preds), np.array(all_labels), np.array(all_probs), elapsed


results = {}
for name, model in MODELS.items():
    print(f"\nRunning {name} inference...")
    preds, labs, probs, elapsed = run_inference(model, test_loader, DEVICE)
    results[name] = {"preds": preds, "labels": labs, "probs": probs, "time": elapsed}
    macro_f1 = f1_score(labs, preds, average="macro") * 100
    print(f"  Macro F1: {macro_f1:.2f}% | Time: {elapsed:.1f}s")

    print(f"\n{name} — Classification Report")
    print(classification_report(labs, preds, target_names=CLASS_NAMES, digits=2))


# ===== Plot 1: Confusion Matrices =====
n = len(results)
fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
if n == 1:
    axes = [axes]
fig.suptitle("Confusion Matrices — Half 1 Test Set", fontsize=16, fontweight="bold")
for ax, (name, r) in zip(axes, results.items()):
    cm = confusion_matrix(r["labels"], r["preds"])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_title(name, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(
    os.path.join(PLOT_DIR, "confusion_matrices.png"), dpi=300, bbox_inches="tight"
)
plt.close()
print(f"Saved: confusion_matrices.png")


# ===== Plot 2: Per-Class F1 =====
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle("Per-Class F1 — Half 1 Test Set", fontsize=16, fontweight="bold")
colors = {
    "Mamba": "#E74C3C",
    "Transformer": "#3498DB",
    "LSTM": "#2ECC71",
    "SSM": "#F39C12",
}
x = np.arange(4)
w = 0.8 / len(results)
for i, (name, r) in enumerate(results.items()):
    f1s = f1_score(r["labels"], r["preds"], average=None, labels=range(4)) * 100
    offset = (i - len(results) / 2 + 0.5) * w
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
                bar.get_height() + 0.5,
                f"{val:.1f}",
                ha="center",
                fontsize=7,
            )
ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES)
ax.set_ylabel("F1 Score (%)")
ax.legend()
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "per_class_f1.png"), dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: per_class_f1.png")


# ===== Plot 3: ROC Curves =====
fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
if n == 1:
    axes = [axes]
fig.suptitle("ROC Curves — Half 1 Test Set", fontsize=16, fontweight="bold")
cls_colors = ["#2ECC71", "#E74C3C", "#3498DB", "#F39C12"]
for ax, (name, r) in zip(axes, results.items()):
    labels_oh = np.eye(4)[r["labels"]]
    for i, (cls, color) in enumerate(zip(CLASS_NAMES, cls_colors)):
        fpr, tpr, _ = roc_curve(labels_oh[:, i], r["probs"][:, i])
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls} (AUC={auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_title(name, fontweight="bold")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "roc_curves.png"), dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: roc_curves.png")


# ===== Plot 4: Precision-Recall =====
fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
if n == 1:
    axes = [axes]
fig.suptitle("Precision-Recall — Half 1 Test Set", fontsize=16, fontweight="bold")
for ax, (name, r) in zip(axes, results.items()):
    labels_oh = np.eye(4)[r["labels"]]
    for i, (cls, color) in enumerate(zip(CLASS_NAMES, cls_colors)):
        p, rec, _ = precision_recall_curve(labels_oh[:, i], r["probs"][:, i])
        ap = average_precision_score(labels_oh[:, i], r["probs"][:, i])
        ax.plot(rec, p, color=color, lw=2, label=f"{cls} (AP={ap:.2f})")
    ax.set_title(name, fontweight="bold")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
plt.tight_layout()
plt.savefig(
    os.path.join(PLOT_DIR, "precision_recall.png"), dpi=300, bbox_inches="tight"
)
plt.close()
print(f"Saved: precision_recall.png")


# ===== Plot 5: Overall Metrics =====
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Overall Metrics — Half 1 Test Set", fontsize=16, fontweight="bold")
model_names = list(results.keys())
bar_colors = [colors.get(n, "#999") for n in model_names]

for idx, (metric, key) in enumerate(
    [
        (
            "Macro F1",
            lambda r: f1_score(r["labels"], r["preds"], average="macro") * 100,
        ),
        (
            "Weighted F1",
            lambda r: f1_score(r["labels"], r["preds"], average="weighted") * 100,
        ),
        ("Accuracy", lambda r: np.mean(r["preds"] == r["labels"]) * 100),
    ]
):
    vals = [key(results[n]) for n in model_names]
    bars = axes[idx].bar(
        model_names, vals, color=bar_colors, edgecolor="black", linewidth=0.5
    )
    axes[idx].set_title(metric, fontweight="bold")
    axes[idx].set_ylabel("Score (%)")
    axes[idx].set_ylim(0, max(vals) * 1.2)
    for bar, val in zip(bars, vals):
        axes[idx].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%",
            ha="center",
            fontweight="bold",
            fontsize=9,
        )
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "overall_metrics.png"), dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: overall_metrics.png")


# ===== Plot 6: Inference Speed =====
fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle("Inference Speed — Half 1 Test Set", fontsize=16, fontweight="bold")
times = [results[n]["time"] for n in model_names]
bars = ax.bar(model_names, times, color=bar_colors, edgecolor="black", linewidth=0.5)
ax.set_ylabel("Time (seconds)")
for bar, val in zip(bars, times):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.3,
        f"{val:.1f}s",
        ha="center",
        fontweight="bold",
    )
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "inference_speed.png"), dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: inference_speed.png")


# ===== Plot 7: Summary Table =====
fig, ax = plt.subplots(figsize=(12, 2 + len(model_names) * 0.5))
ax.axis("off")
table_data = [["Model", "Macro F1", "Weighted F1", "Accuracy", "Inference (s)"]]
for name in model_names:
    r = results[name]
    table_data.append(
        [
            name,
            f"{f1_score(r['labels'], r['preds'], average='macro') * 100:.2f}%",
            f"{f1_score(r['labels'], r['preds'], average='weighted') * 100:.2f}%",
            f"{np.mean(r['preds'] == r['labels']) * 100:.2f}%",
            f"{r['time']:.1f}",
        ]
    )
table = ax.table(cellText=table_data, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.8)
for j in range(5):
    table[0, j].set_facecolor("#2C3E50")
    table[0, j].set_text_props(color="white", fontweight="bold")
ax.set_title("Inference Results Summary", fontsize=14, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "summary_table.png"), dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: summary_table.png")

print(f"\nAll plots saved to: {PLOT_DIR}")
