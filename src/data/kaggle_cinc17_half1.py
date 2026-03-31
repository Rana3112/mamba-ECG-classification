"""
============================================================================
 KAGGLE NOTEBOOK — PhysioNet CINC 2017 Training (HALF 1 of 2)
============================================================================
 Target GPU : NVIDIA P100 (16 GB VRAM)
 Dataset    : PhysioNet/CinC 2017 Challenge — FIRST HALF (records 0-4263)
 Models     : Mamba, Transformer, LSTM, SSM (S4)

 INSTRUCTIONS FOR KAGGLE:
 1. Add the "PhysioNet CinC 2017 Challenge" dataset to your notebook.
    It should be available at: /kaggle/input/training2017/training2017/
 2. Copy-paste this entire script into a Kaggle notebook cell.
 3. Set accelerator to GPU (P100).
 4. Run. Checkpoints are saved to /kaggle/working/ after every epoch.
 5. Download the checkpoint files from /kaggle/working/ when done.

 RESUME TRAINING:
 If the notebook times out, re-upload the checkpoint files to a Kaggle
 dataset, mount them, and set RESUME_DIR below to that path. The script
 will automatically resume from the last saved epoch.
============================================================================
"""

import math
import os
import json
import random
import time
from contextlib import nullcontext

import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Sampler

# ==========================================================================
# CONFIGURATION — EDIT THESE
# ==========================================================================
DATA_DIR = (
    "/kaggle/input/datasets/nebula3112/physionet/training2017/"  # Kaggle dataset path
)
OUTPUT_DIR = "/kaggle/working/"  # Save checkpoints here
RESUME_DIR = "/kaggle/working/"  # Resume from checkpoints saved in /kaggle/working/

HALF = 1  # This script trains on HALF 1 (first 50% of records)
EPOCHS = 7  # 4 models on T4 (~25-30 hrs)
BATCH_SIZE = 4  # Batch size (P100 can handle this)
ACCUM_STEPS = 4  # Gradient accumulation steps (effective batch = 16)
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.1
DOWNSAMPLE_FACTOR = 2  # Downsample 300Hz -> 150Hz to reduce sequence length
CROP_LENGTH = 4500  # Max sequence length after downsampling (~30s at 150Hz)
D_MODEL = 64
STATE_DIM = 16
NUM_CLASSES = 4
MODELS_TO_TRAIN = ["Mamba", "Transformer", "LSTM", "SSM"]

CINC_LABEL_MAP = {"N": 0, "A": 1, "O": 2, "~": 3}
CINC_LABEL_NAMES = ["N", "A", "O", "Noise"]

# ==========================================================================
# MODEL DEFINITIONS (self-contained, no external imports needed)
# ==========================================================================


# --- Mamba (Selective SSM / S6) ---
class SelectiveSSMLayer(nn.Module):
    def __init__(self, d_model=64, state_dim=16):
        super().__init__()
        self.d_model = d_model
        self.N = state_dim
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
        B_sel = self.B_proj(x_conv)
        C_sel = self.C_proj(x_conv)
        h = torch.zeros(B_sz, D, self.N, device=x.device)
        outputs = []
        for t in range(L):
            dt = delta[:, t, :]
            A_bar = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))
            b_t = dt.unsqueeze(-1) * B_sel[:, t, :].unsqueeze(1)
            x_t = x_conv[:, t, :]
            h = A_bar * h + b_t * x_t.unsqueeze(-1)
            c_t = C_sel[:, t, :].unsqueeze(1)
            y_t = (h * c_t).sum(-1)
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)
        y = y * F.silu(z)
        return self.out_proj(y)


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


# --- Transformer ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=1, d_model=64, num_classes=4, num_layers=2, nhead=4):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return self.classifier(x.mean(dim=1))


# --- LSTM ---
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


# --- SSM (S4) ---
class S4Layer(nn.Module):
    def __init__(self, d_model=64, state_dim=16):
        super().__init__()
        self.d_model = d_model
        self.N = state_dim
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
            y_t = (h * self.C.unsqueeze(0)).sum(-1)
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)
        return y + x * self.D.unsqueeze(0).unsqueeze(0)


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


# ==========================================================================
# WRAPPERS (masked pooling for variable-length CINC sequences)
# ==========================================================================


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
        pooled = (x * mask).sum(dim=1)
        counts = mask.sum(dim=1)
        return pooled / (counts + 1e-6)


class MambaWrapper(WrapperBase):
    def forward(self, x, lengths):
        x = self.backbone.input_proj(x)
        for layer, norm in zip(self.backbone.layers, self.backbone.norms):
            x = norm(x + layer(x))
        return self.head(self.pool(x, lengths))


class SSMWrapper(WrapperBase):
    def forward(self, x, lengths):
        x = self.backbone.input_proj(x)
        for layer, norm in zip(self.backbone.layers, self.backbone.norms):
            x = norm(x + layer(x))
        return self.head(self.pool(x, lengths))


class TransformerWrapper(WrapperBase):
    def forward(self, x, lengths):
        embedded = self.backbone.embedding(x)
        mask = (
            torch.arange(embedded.size(1), device=x.device)[None, :] >= lengths[:, None]
        )
        encoded = self.backbone.pos_encoder(embedded)
        encoded = self.backbone.transformer_encoder(encoded, src_key_padding_mask=mask)
        return self.head(self.pool(encoded, lengths))


class LSTMWrapper(WrapperBase):
    def forward(self, x, lengths):
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.backbone.lstm(packed)
        return self.head(hidden[-1])


def build_model_suite():
    models = {}
    models["Mamba"] = MambaWrapper(
        MambaClassifier(1, D_MODEL, STATE_DIM, 2, NUM_CLASSES), D_MODEL, NUM_CLASSES
    )
    models["SSM"] = SSMWrapper(
        SSMClassifier(D_MODEL, STATE_DIM, 2, NUM_CLASSES), D_MODEL, NUM_CLASSES
    )
    models["LSTM"] = LSTMWrapper(
        LSTMClassifier(1, D_MODEL, 2, NUM_CLASSES), D_MODEL, NUM_CLASSES
    )
    models["Transformer"] = TransformerWrapper(
        TransformerClassifier(1, D_MODEL, NUM_CLASSES), D_MODEL, NUM_CLASSES
    )
    return models


# ==========================================================================
# DATASET & DATA LOADER
# ==========================================================================


def read_record_length(root_dir, record_name):
    header_path = os.path.join(root_dir, record_name + ".hea")
    try:
        with open(header_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip().split()
        if len(first_line) >= 4:
            return int(first_line[3])
    except OSError:
        pass
    return None


def crop_signal(signal, target_length, mode="random"):
    if target_length is None or signal.shape[0] <= target_length:
        return signal
    if mode == "random":
        start = random.randint(0, signal.shape[0] - target_length)
    elif mode == "center":
        start = (signal.shape[0] - target_length) // 2
    else:
        start = 0
    return signal[start : start + target_length]


class Cinc17Dataset(Dataset):
    def __init__(
        self,
        records,
        labels,
        root_dir,
        downsample_factor=1,
        target_length=None,
        crop_mode="center",
    ):
        self.records = list(records)
        self.labels = list(labels)
        self.root_dir = root_dir
        self.downsample_factor = max(1, int(downsample_factor))
        self.target_length = target_length
        self.crop_mode = crop_mode
        self.raw_lengths = [
            read_record_length(root_dir, r) or target_length or 0 for r in self.records
        ]
        self.processed_lengths = [self._proc_len(l) for l in self.raw_lengths]

    def _proc_len(self, raw_len):
        l = max(1, math.ceil(raw_len / self.downsample_factor))
        if self.target_length is not None:
            l = min(l, self.target_length)
        return l

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        mat_path = os.path.join(self.root_dir, rec + ".mat")
        try:
            mat = scipy.io.loadmat(mat_path)
            sig = mat["val"][0].astype(np.float32)
            sig = sig[:: self.downsample_factor]
            sig = crop_signal(sig, self.target_length, self.crop_mode)
            sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-6)
            sig = torch.tensor(sig, dtype=torch.float32).unsqueeze(-1)
            label = torch.tensor(
                CINC_LABEL_MAP.get(self.labels[idx], 3), dtype=torch.long
            )
            return sig, label
        except Exception as e:
            print(f"Error loading {rec}: {e}")
            return torch.zeros(1000, 1), torch.tensor(3, dtype=torch.long)


class LengthBucketSampler(Sampler):
    def __init__(self, lengths, batch_size, shuffle=True, bucket_mult=25):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bucket_size = max(batch_size, batch_size * bucket_mult)

    def __iter__(self):
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            random.shuffle(indices)
        batches = []
        for s in range(0, len(indices), self.bucket_size):
            bucket = indices[s : s + self.bucket_size]
            bucket.sort(key=lambda i: self.lengths[i], reverse=True)
            for bs in range(0, len(bucket), self.batch_size):
                batch = bucket[bs : bs + self.batch_size]
                if batch:
                    batches.append(batch)
        if self.shuffle:
            random.shuffle(batches)
        for b in batches:
            yield b

    def __len__(self):
        return math.ceil(len(self.lengths) / self.batch_size)


def collate_pad(batch):
    signals, labels = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in signals], dtype=torch.long)
    padded = torch.nn.utils.rnn.pad_sequence(signals, batch_first=True)
    return padded, torch.stack(labels), lengths


def get_dataloaders(records, labels, root_dir):
    """Split the given records into train/test and return loaders."""
    X_train, X_test, y_train, y_test = train_test_split(
        records, labels, test_size=0.2, random_state=42, stratify=labels
    )
    train_ds = Cinc17Dataset(
        X_train,
        y_train,
        root_dir,
        downsample_factor=DOWNSAMPLE_FACTOR,
        target_length=CROP_LENGTH,
        crop_mode="random",
    )
    test_ds = Cinc17Dataset(
        X_test,
        y_test,
        root_dir,
        downsample_factor=DOWNSAMPLE_FACTOR,
        target_length=CROP_LENGTH,
        crop_mode="center",
    )
    common = {
        "collate_fn": collate_pad,
        "num_workers": 2,
        "pin_memory": True,
        "persistent_workers": True,
    }
    train_loader = DataLoader(
        train_ds,
        batch_sampler=LengthBucketSampler(
            train_ds.processed_lengths, BATCH_SIZE, shuffle=True
        ),
        **common,
    )
    test_loader = DataLoader(
        test_ds,
        batch_sampler=LengthBucketSampler(
            test_ds.processed_lengths, BATCH_SIZE, shuffle=False
        ),
        **common,
    )
    return train_loader, test_loader, train_ds


# ==========================================================================
# TRAINING LOOP WITH EPOCH CHECKPOINTING
# ==========================================================================


def compute_class_weights(dataset, device):
    counts = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    for lab in dataset.labels:
        counts[CINC_LABEL_MAP.get(lab, NUM_CLASSES - 1)] += 1
    weights = counts.sum() / torch.clamp(counts, min=1.0)
    weights = weights / weights.mean()
    return weights.to(device)


def checkpoint_path(model_name, epoch):
    return os.path.join(
        OUTPUT_DIR, f"cinc17_half{HALF}_{model_name.lower()}_epoch{epoch}.pth"
    )


def latest_checkpoint_path(model_name):
    return os.path.join(
        OUTPUT_DIR, f"cinc17_half{HALF}_{model_name.lower()}_latest.pth"
    )


def save_checkpoint(
    model, optimizer, scheduler, scaler, epoch, best_score, history, model_name
):
    """Save everything needed to resume training."""
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "best_score": best_score,
        "history": history,
    }
    # Save epoch-specific checkpoint
    path = checkpoint_path(model_name, epoch)
    torch.save(ckpt, path)
    print(f"  💾 Checkpoint saved: {path}")

    # Also save as 'latest' for easy resume
    latest = latest_checkpoint_path(model_name)
    torch.save(ckpt, latest)


def load_checkpoint(model, optimizer, scheduler, scaler, model_name):
    """Try to load the latest checkpoint. Returns start_epoch, best_score, history."""
    # Check resume dir first, then output dir
    for search_dir in [RESUME_DIR, OUTPUT_DIR] if RESUME_DIR else [OUTPUT_DIR]:
        path = os.path.join(
            search_dir, f"cinc17_half{HALF}_{model_name.lower()}_latest.pth"
        )
        if os.path.exists(path):
            print(f"  🔄 Resuming from checkpoint: {path}")
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if scaler and ckpt["scaler_state_dict"]:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            return ckpt["epoch"] + 1, ckpt["best_score"], ckpt["history"]
    return (
        0,
        float("-inf"),
        {"train_loss": [], "train_acc": [], "val_loss": [], "val_macro_f1": []},
    )


def autocast_ctx(device, enabled):
    if enabled and device.type == "cuda":
        return torch.amp.autocast("cuda")
    return nullcontext()


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, use_amp):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (signals, labels, lengths) in enumerate(loader, 1):
        signals = signals.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        with autocast_ctx(device, use_amp):
            logits = model(signals, lengths)
            loss = criterion(logits, labels) / ACCUM_STEPS

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if batch_idx % ACCUM_STEPS == 0:
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * ACCUM_STEPS
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Print progress every 50 batches
        if batch_idx % 50 == 0:
            print(
                f"    Batch {batch_idx}/{len(loader)} | Loss: {total_loss / batch_idx:.4f}"
            )

    # Flush remaining gradients
    if total > 0 and batch_idx % ACCUM_STEPS != 0:
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / max(len(loader), 1), 100.0 * correct / max(total, 1)


def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for signals, labels, lengths in loader:
            signals = signals.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            with autocast_ctx(device, use_amp):
                logits = model(signals, lengths)
                loss = criterion(logits, labels)

            total_loss += loss.item()
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    label_ids = list(range(NUM_CLASSES))
    report = classification_report(
        all_labels,
        all_preds,
        labels=label_ids,
        target_names=CINC_LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    acc = accuracy_score(all_labels, all_preds) * 100.0
    macro_f1 = (
        f1_score(all_labels, all_preds, labels=label_ids, average="macro") * 100.0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=label_ids).tolist()

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "loss": total_loss / max(len(loader), 1),
        "per_class_f1": {
            name: report[name]["f1-score"] * 100 for name in CINC_LABEL_NAMES
        },
        "confusion_matrix": cm,
    }


def train_model(name, model, train_loader, test_loader, train_ds, device):
    print(f"\n{'=' * 60}")
    print(f"  TRAINING: {name} (Half {HALF})")
    print(f"{'=' * 60}")

    model = model.to(device)
    use_amp = False  # Disable AMP for P100 compatibility (sm_60)
    scaler = None
    class_weights = compute_class_weights(train_ds, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Try to resume
    start_epoch, best_score, history = load_checkpoint(
        model, optimizer, scheduler, scaler, name
    )

    if start_epoch >= EPOCHS:
        print(f"  ✅ {name} already completed {EPOCHS} epochs. Skipping.")
        return history

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()

    try:
        for epoch in range(start_epoch, EPOCHS):
            epoch_start = time.time()

            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scaler, use_amp
            )
            val_metrics = evaluate(model, test_loader, criterion, device, use_amp)
            scheduler.step()

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_metrics["loss"])
            history["val_macro_f1"].append(val_metrics["macro_f1"])

            epoch_time = time.time() - epoch_start
            print(
                f"  {name} | Epoch {epoch + 1}/{EPOCHS} | "
                f"Train Loss={train_loss:.4f} | Train Acc={train_acc:.1f}% | "
                f"Val Acc={val_metrics['accuracy']:.1f}% | "
                f"Val F1={val_metrics['macro_f1']:.1f}% | "
                f"Time={epoch_time:.0f}s"
            )
            for cls_name, f1_val in val_metrics["per_class_f1"].items():
                print(f"    F1({cls_name}): {f1_val:.1f}%")

            if val_metrics["macro_f1"] > best_score:
                best_score = val_metrics["macro_f1"]
                best_path = os.path.join(
                    OUTPUT_DIR, f"cinc17_half{HALF}_{name.lower()}_best.pth"
                )
                torch.save(model.state_dict(), best_path)
                print(f"  ⭐ New best F1={best_score:.1f}% — saved to {best_path}")

            # Save checkpoint EVERY epoch
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, best_score, history, name
            )

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  ❌ {name} OOM at epoch {epoch + 1}!")
            if device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            raise

    total_time = time.time() - start_time
    print(f"  {name} completed in {total_time:.0f}s")

    # Save final summary
    summary = {
        "model": name,
        "half": HALF,
        "epochs_completed": len(history["train_loss"]),
        "best_macro_f1": best_score,
        "total_time_sec": total_time,
        "history": history,
    }
    with open(
        os.path.join(OUTPUT_DIR, f"cinc17_half{HALF}_{name.lower()}_summary.json"), "w"
    ) as f:
        json.dump(summary, f, indent=2)

    return history


# ==========================================================================
# MAIN
# ==========================================================================


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        # Test CUDA compatibility for P100 (sm_60)
        try:
            x = torch.randn(2, 4, device=device, requires_grad=True)
            w = torch.randn(4, 8, device=device)
            y = x @ w
            y.sum().backward()
            print("CUDA operations OK")
        except RuntimeError as e:
            print(f"CUDA test failed: {e}")
            print(
                "Falling back to CPU. Try: !pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121"
            )
            device = torch.device("cpu")

    # Load reference file
    ref_path = os.path.join(DATA_DIR, "REFERENCE.csv")
    if not os.path.exists(ref_path):
        # Some Kaggle datasets have different structures
        for alt in [
            "/kaggle/input/datasets/nebula3112/physionet/REFERENCE.csv",
            "/kaggle/input/datasets/nebula3112/physionet/training2017/REFERENCE.csv",
        ]:
            if os.path.exists(alt):
                ref_path = alt
                break

    df = pd.read_csv(ref_path, header=None, names=["record", "label"])
    print(f"Total records in dataset: {len(df)}")

    # Split into halves
    df_sorted = df.sort_values("record").reset_index(drop=True)
    mid = len(df_sorted) // 2

    if HALF == 1:
        df_half = df_sorted.iloc[:mid].reset_index(drop=True)
    else:
        df_half = df_sorted.iloc[mid:].reset_index(drop=True)

    print(f"Half {HALF}: {len(df_half)} records")
    print(f"Label distribution:\n{df_half['label'].value_counts().to_string()}")

    # Determine root dir (where .mat files are)
    root_dir = os.path.dirname(ref_path)
    # Check if .mat files are in root_dir
    sample_rec = df_half["record"].iloc[0]
    if not os.path.exists(os.path.join(root_dir, sample_rec + ".mat")):
        # Try parent
        if os.path.exists(os.path.join(os.path.dirname(root_dir), sample_rec + ".mat")):
            root_dir = os.path.dirname(root_dir)

    records = df_half["record"].values
    labels = df_half["label"].values

    train_loader, test_loader, train_ds = get_dataloaders(records, labels, root_dir)
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # Build and train each model
    model_suite = build_model_suite()

    for name in MODELS_TO_TRAIN:
        if name in model_suite:
            train_model(
                name, model_suite[name], train_loader, test_loader, train_ds, device
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            print(f"Unknown model: {name}")

    print("\n" + "=" * 60)
    print("  ALL TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Checkpoints saved in: {OUTPUT_DIR}")
    print("Download these files before the notebook session expires!")


if __name__ == "__main__":
    main()
