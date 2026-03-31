"""
Real-Time ECG Monitor with 4 Trained Models
Compares inference predictions vs true labels on real CinC 2017 test data.
Run on Kaggle T4 — upload .pth files to /kaggle/working/
"""

import os, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import scipy.io as sio
from sklearn.model_selection import train_test_split
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import clear_output, display

# ===== Config =====
DATA_DIR = "/kaggle/input/datasets/nebula3112/physionet/training2017/"
MODEL_DIR = "/kaggle/working/"
PLOT_DIR = "/kaggle/working/monitor/"
DOWNSAMPLE = 2
MAX_LEN = 4500
CLASSES = ["N", "O", "A", "~"]
CLASS_NAMES = {"N": "Normal", "O": "Other", "A": "AFib", "~": "Noise"}
CLASS_COLORS = {"N": "#2ecc71", "O": "#3498db", "A": "#e74c3c", "~": "#f39c12"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(PLOT_DIR, exist_ok=True)

print(f"Device: {DEVICE}")


# ===== Models =====
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


# ===== Load Models =====
def load_model(wrapper_cls, backbone_cls, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    backbone = backbone_cls()
    model = wrapper_cls(backbone, d_model=64, num_classes=4)

    # Handle different checkpoint formats
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        # Check if it's a raw state dict
        first_key = list(ckpt.keys())[0] if ckpt else None
        if first_key and isinstance(ckpt[first_key], torch.Tensor):
            state = ckpt
        else:
            print(f"  Unknown format, keys: {list(ckpt.keys())[:5]}")
            return None

    model.load_state_dict(state)
    model.to(device).eval()
    return model


MODELS = {}
for name, wcls, bcls, fname in [
    ("Mamba", MambaWrapper, MambaClassifier, "cinc17_half1_mamba_best.pth"),
    (
        "Transformer",
        TransformerWrapper,
        TransformerClassifier,
        "cinc17_half1_transformer_best.pth",
    ),
    ("LSTM", LSTMWrapper, LSTMClassifier, "cinc17_half1_lstm_best.pth"),
    ("SSM", SSMWrapper, SSMClassifier, "cinc17_half1_ssm_best.pth"),
]:
    path = os.path.join(MODEL_DIR, fname)
    if os.path.exists(path):
        m = load_model(wcls, bcls, path, DEVICE)
        if m is not None:
            MODELS[name] = m
            print(f"  {name} loaded")
        else:
            print(f"  {name} FAILED to load")

print(f"\n{len(MODELS)} models loaded")


# ===== Load Test Data =====
ref_path = os.path.join(DATA_DIR, "REFERENCE.csv")
with open(ref_path) as f:
    df_lines = [l.strip().split(",") for l in f if len(l.strip().split(",")) == 2]
records = [r for r, _ in df_lines]
labels = [l for _, l in df_lines]

train_recs, test_recs, train_labels, test_labels = train_test_split(
    records, labels, test_size=0.2, random_state=42, stratify=labels
)
print(f"Test set: {len(test_recs)} recordings")


# ===== Load ECG Signal =====
def load_ecg(record_id):
    mat = sio.loadmat(os.path.join(DATA_DIR, f"{record_id}.mat"))
    sig = mat["val"].flatten().astype(np.float32)
    if DOWNSAMPLE > 1:
        sig = sig[::DOWNSAMPLE]
    if len(sig) > MAX_LEN:
        sig = sig[:MAX_LEN]
    return sig


# ===== Predict with All Models =====
def predict_all(signal):
    """Run inference on a single signal with all models."""
    sig_t = torch.from_numpy(signal).unsqueeze(0).to(DEVICE)
    length = torch.tensor([len(signal)]).to(DEVICE)
    results = {}
    with torch.no_grad():
        for name, model in MODELS.items():
            logits = model(sig_t, length)
            probs = torch.softmax(logits.float(), dim=1)
            probs = torch.nan_to_num(probs, nan=0.25)
            pred_id = int(probs.argmax(dim=1).item())
            results[name] = {
                "class": CLASSES[pred_id],
                "class_name": CLASS_NAMES[CLASSES[pred_id]],
                "confidence": float(probs[0, pred_id]),
                "probs": {c: float(probs[0, i]) for i, c in enumerate(CLASSES)},
            }
    return results


# ===== Real-Time Monitor Demo =====
def realtime_ecg_monitor(
    record_index=0, seconds_on_screen=4.0, step_sec=0.03, duration_sec=None
):
    """
    Simulates real-time ECG monitoring by scrolling through a test recording.
    Shows ground truth vs all 4 model predictions.
    """
    rec_id = test_recs[record_index]
    true_label = test_labels[record_index]

    signal = load_ecg(rec_id)
    fs = 150  # after downsampling

    if duration_sec is None:
        duration_sec = len(signal) / fs

    screen_len = int(seconds_on_screen * fs)
    step_len = max(1, int(step_sec * fs))
    buf = np.zeros(screen_len, dtype=np.float32)

    # Run full prediction once
    predictions = predict_all(signal)

    fig, axes = plt.subplots(
        5, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1, 1, 1, 1]}
    )
    fig.patch.set_facecolor("black")
    fig.suptitle(
        f"ECG Monitor — Record {rec_id} | True Label: {CLASS_NAMES[true_label]} ({true_label})",
        color="white",
        fontsize=13,
        fontweight="bold",
    )

    total_steps = min(len(signal) - step_len, int(duration_sec * fs))

    for t in range(0, total_steps, step_len):
        new = signal[t : t + step_len]
        buf = np.roll(buf, -step_len)
        buf[-step_len:] = new

        # Main ECG plot
        axes[0].clear()
        axes[0].set_facecolor("black")
        axes[0].plot(buf, color="#00ff88", linewidth=1.2)
        axes[0].set_xlim(0, screen_len)
        y_min, y_max = signal.min(), signal.max()
        axes[0].set_ylim(y_min * 1.2, y_max * 1.2)
        axes[0].set_title(
            f"ECG Signal — {rec_id} ({true_label})", color="white", fontsize=11
        )
        axes[0].tick_params(colors="white")
        for spine in axes[0].spines.values():
            spine.set_color("#333")

        # Probability bars for each model
        for idx, (model_name, pred) in enumerate(predictions.items()):
            ax = axes[idx + 1]
            ax.clear()
            ax.set_facecolor("#111")
            probs = pred["probs"]

            colors = [CLASS_COLORS[c] for c in CLASSES]
            bar_vals = [probs[c] for c in CLASSES]

            bars = ax.barh(
                CLASSES, bar_vals, color=colors, edgecolor="#333", linewidth=0.5
            )

            # Mark predicted class
            for bar, cls in zip(bars, CLASSES):
                if cls == pred["class"]:
                    bar.set_edgecolor("white")
                    bar.set_linewidth(2)

            correct = "✓" if pred["class"] == true_label else "✗"
            color = "#00ff88" if pred["class"] == true_label else "#ff4444"

            ax.set_title(
                f"{model_name}: {pred['class']} ({pred['confidence']:.0%}) {correct}",
                color=color,
                fontsize=10,
                fontweight="bold",
            )
            ax.set_xlim(0, 1)
            ax.tick_params(colors="white", labelsize=9)
            for spine in ax.spines.values():
                spine.set_color("#333")

            # Add percentage labels
            for bar, val in zip(bars, bar_vals):
                if val > 0.02:
                    ax.text(
                        val + 0.02,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.0%}",
                        va="center",
                        color="white",
                        fontsize=8,
                    )

        plt.tight_layout()
        clear_output(wait=True)
        display(fig)
        time.sleep(0.02)

    plt.close(fig)

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"Record: {rec_id} | True Label: {CLASS_NAMES[true_label]} ({true_label})")
    print(f"{'=' * 50}")
    for model_name, pred in predictions.items():
        correct = "✓" if pred["class"] == true_label else "✗"
        print(
            f"  {model_name:<12}: {pred['class']} ({pred['confidence']:.1%}) {correct}"
        )
    print()

    return predictions


# ===== Static Comparison Plot =====
def plot_comparison(record_index=0):
    """Static plot comparing all 4 models on a single ECG recording."""
    rec_id = test_recs[record_index]
    true_label = test_labels[record_index]
    signal = load_ecg(rec_id)
    predictions = predict_all(signal)

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#fafafa")

    # Main ECG
    ax_main = plt.subplot2grid((6, 4), (0, 0), colspan=4, rowspan=3)
    ax_main.plot(signal, color="#1a1a1a", linewidth=0.8)
    ax_main.set_title(
        f"ECG Record {rec_id} — True Label: {CLASS_NAMES[true_label]} ({true_label})",
        fontsize=13,
        fontweight="bold",
    )
    ax_main.set_xlabel("Time Steps (150 Hz)")
    ax_main.set_ylabel("Amplitude")
    ax_main.grid(True, alpha=0.3)

    # Model predictions
    model_names = list(predictions.keys())
    for idx, model_name in enumerate(model_names):
        ax = plt.subplot2grid((6, 4), (3, idx), rowspan=3)
        pred = predictions[model_name]
        probs = pred["probs"]
        colors = [CLASS_COLORS[c] for c in CLASSES]
        bar_vals = [probs[c] for c in CLASSES]

        bars = ax.bar(CLASSES, bar_vals, color=colors, edgecolor="#333", linewidth=0.5)

        # Highlight predicted
        for bar, cls in zip(bars, CLASSES):
            if cls == pred["class"]:
                bar.set_edgecolor("black")
                bar.set_linewidth(2.5)

        correct = "✓" if pred["class"] == true_label else "✗"
        color = "#059669" if pred["class"] == true_label else "#dc2626"
        ax.set_title(
            f"{model_name}: {pred['class']} {correct}", color=color, fontweight="bold"
        )
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)

        for bar, val in zip(bars, bar_vals):
            if val > 0.02:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 0.02,
                    f"{val:.0%}",
                    ha="center",
                    fontsize=9,
                    fontweight="bold",
                )

    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOT_DIR, f"comparison_{rec_id}.png"), dpi=300, bbox_inches="tight"
    )
    plt.show()
    print(f"Saved: {PLOT_DIR}comparison_{rec_id}.png")


# ===== Run =====
# Pick a random test sample and run the monitor
import random

random_idx = random.randint(0, len(test_recs) - 1)
print(
    f"\nRunning monitor on test sample #{random_idx}: {test_recs[random_idx]} (True: {test_labels[random_idx]})\n"
)

# Static comparison first
plot_comparison(random_idx)

# Then real-time scrolling monitor
realtime_ecg_monitor(record_index=random_idx, seconds_on_screen=4.0, step_sec=0.03)
