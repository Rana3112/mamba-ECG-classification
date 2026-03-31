"""
Ablation Study for Mamba Architecture
Tests the contribution of each component to final performance.

Ablations:
1. Baseline Mamba (2 layers, state_dim=16, with conv) — already trained
2. Single layer (1 layer, state_dim=16, with conv)
3. Deep (4 layers, state_dim=16, with conv)
4. Small state (2 layers, state_dim=8, with conv)
5. Large state (2 layers, state_dim=32, with conv)
6. No convolution (2 layers, state_dim=16, without conv)

Run on Kaggle T4 GPU. Each variant takes ~9 hours.
Run one variant per week, or run all in sequence (~54 hours).
"""

import os, json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from collections import Counter

# ===== Config =====
DATA_DIR = "/kaggle/input/datasets/nebula3112/physionet/training2017/"
OUTPUT_DIR = "/kaggle/working/"
DOWNSAMPLE = 2
MAX_LEN = 4500
BATCH_SIZE = 16
GRAD_ACCUM = 4
EPOCHS = 7
LR = 1e-3
WEIGHT_DECAY = 1e-4
CLASSES = ["N", "O", "A", "~"]

# ===== Ablation Configs =====
ABLATIONS = {
    "baseline": {"num_layers": 2, "state_dim": 16, "use_conv": True, "d_model": 64},
    "1layer": {"num_layers": 1, "state_dim": 16, "use_conv": True, "d_model": 64},
    "4layer": {"num_layers": 4, "state_dim": 16, "use_conv": True, "d_model": 64},
    "state8": {"num_layers": 2, "state_dim": 8, "use_conv": True, "d_model": 64},
    "state32": {"num_layers": 2, "state_dim": 32, "use_conv": True, "d_model": 64},
    "no_conv": {"num_layers": 2, "state_dim": 16, "use_conv": False, "d_model": 64},
}

# Pick which ablation to run (change this to run different variants)
ABLATION_NAME = (
    "1layer"  # CHANGE THIS: baseline, 1layer, 4layer, state8, state32, no_conv
)

CFG = ABLATIONS[ABLATION_NAME]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Ablation: {ABLATION_NAME}")
print(f"Config: {CFG}")


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


# ===== Model =====
class SelectiveSSMLayer(nn.Module):
    def __init__(self, d_model=64, state_dim=16, use_conv=True):
        super().__init__()
        self.d_model, self.N, self.use_conv = d_model, state_dim, use_conv
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
        if use_conv:
            self.conv1d = nn.Conv1d(d_model, d_model, 3, padding=1, groups=d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B_sz, L, D = x.shape
        A = -torch.exp(self.A_log)
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)

        if self.use_conv:
            x_conv = self.conv1d(x_in.transpose(1, 2)).transpose(1, 2)
            x_conv = F.silu(x_conv)
        else:
            x_conv = F.silu(x_in)

        delta = F.softplus(self.delta_proj(x_conv))
        B_sel = self.B_proj(x_conv)
        C_sel = self.C_proj(x_conv)
        h = torch.zeros(B_sz, D, self.N, device=x.device)
        outputs = []
        for t in range(L):
            dt = delta[:, t, :]
            A_bar = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))
            b_t = dt.unsqueeze(-1) * B_sel[:, t, :].unsqueeze(1)
            h = A_bar * h + b_t * x_conv[:, t, :].unsqueeze(-1)
            outputs.append((h * C_sel[:, t, :].unsqueeze(1)).sum(-1))
        y = torch.stack(outputs, dim=1)
        if self.use_conv:
            y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)
        y = y * F.silu(z)
        return self.out_proj(y)


class MambaClassifier(nn.Module):
    def __init__(
        self,
        input_dim=1,
        d_model=64,
        state_dim=16,
        num_layers=2,
        num_classes=4,
        use_conv=True,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList(
            [SelectiveSSMLayer(d_model, state_dim, use_conv) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        for layer, norm in zip(self.layers, self.norms):
            x = norm(x + layer(x))
        return self.classifier(x.mean(dim=1))


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


# ===== Load Data =====
ref_path = os.path.join(DATA_DIR, "REFERENCE.csv")
with open(ref_path) as f:
    df_lines = [l.strip().split(",") for l in f if len(l.strip().split(",")) == 2]
records = [r for r, _ in df_lines]
labels = [l for _, l in df_lines]

train_recs, test_recs, train_labels, test_labels = train_test_split(
    records, labels, test_size=0.2, random_state=42, stratify=labels
)
print(f"Train: {len(train_recs)}, Test: {len(test_recs)}")

# Class-weighted sampler for imbalance
label_counts = Counter(train_labels)
weights = [1.0 / label_counts[l] for l in train_labels]
sampler = WeightedRandomSampler(weights, len(weights))

train_dataset = ECGDataset(train_recs, train_labels, DATA_DIR)
test_dataset = ECGDataset(test_recs, test_labels, DATA_DIR)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    collate_fn=collate_pad,
    num_workers=0,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_pad,
    num_workers=0,
)


# ===== Training Functions =====
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)
    for batch_idx, (signals, labs, lengths) in enumerate(loader, 1):
        signals, labs, lengths = signals.to(device), labs.to(device), lengths.to(device)
        logits = model(signals, lengths)
        loss = criterion(logits, labs) / GRAD_ACCUM
        loss.backward()
        if batch_idx % GRAD_ACCUM == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item() * GRAD_ACCUM * labs.size(0)
        correct += (logits.argmax(1) == labs).sum().item()
        total += labs.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for signals, labs, lengths in loader:
        signals, labs, lengths = signals.to(device), labs.to(device), lengths.to(device)
        logits = model(signals, lengths)
        total_loss += criterion(logits, labs).item() * labs.size(0)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labs.cpu().numpy())
    acc = 100.0 * np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = 100.0 * f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return (
        total_loss / len(all_labels),
        acc,
        f1,
        np.array(all_preds),
        np.array(all_labels),
    )


# ===== Run Ablation =====
backbone = MambaClassifier(
    d_model=CFG["d_model"],
    state_dim=CFG["state_dim"],
    num_layers=CFG["num_layers"],
    use_conv=CFG["use_conv"],
)
model = MambaWrapper(backbone, d_model=CFG["d_model"], num_classes=4).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()

history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_macro_f1": [],
    "val_acc": [],
}
best_f1 = 0.0
start_time = time.time()

for epoch in range(EPOCHS):
    t0 = time.time()
    train_loss, train_acc = train_one_epoch(
        model, train_loader, optimizer, criterion, DEVICE
    )
    val_loss, val_acc, val_f1, _, _ = evaluate(model, test_loader, criterion, DEVICE)
    scheduler.step()
    elapsed = time.time() - t0

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["val_macro_f1"].append(val_f1)

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_score": best_f1,
                "history": history,
            },
            os.path.join(OUTPUT_DIR, f"ablation_{ABLATION_NAME}_best.pth"),
        )

    print(
        f"  Epoch {epoch + 1}/{EPOCHS} | Loss={train_loss:.4f} | Acc={train_acc:.1f}% | Val F1={val_f1:.1f}% | Time={elapsed:.0f}s"
    )

total_time = time.time() - start_time

# Final evaluation
val_loss, val_acc, val_f1, preds, labels_arr = evaluate(
    model, test_loader, criterion, DEVICE
)
print(f"\nFinal Classification Report:")
print(
    classification_report(
        labels_arr, preds, target_names=["Normal", "Other", "AFib", "Noise"], digits=2
    )
)

# Save summary
summary = {
    "ablation": ABLATION_NAME,
    "config": CFG,
    "params": total_params,
    "epochs_completed": EPOCHS,
    "best_macro_f1": best_f1,
    "final_macro_f1": val_f1,
    "final_acc": val_acc,
    "total_time_sec": total_time,
    "history": history,
}
with open(os.path.join(OUTPUT_DIR, f"ablation_{ABLATION_NAME}_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nAblation '{ABLATION_NAME}' completed in {total_time / 3600:.1f}h")
print(f"Best F1: {best_f1:.2f}% | Final F1: {val_f1:.2f}%")
print(f"Results saved to: {OUTPUT_DIR}ablation_{ABLATION_NAME}_summary.json")
