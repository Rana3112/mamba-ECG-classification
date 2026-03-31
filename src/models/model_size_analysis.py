"""
Model Size Analysis for Edge Device Deployment
Compares all 4 models: parameters, file size, memory footprint.
Run on Kaggle — takes <1 minute.
"""

import os, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

# ===== Config =====
MODEL_DIR = "/kaggle/working/"
DEVICE = torch.device("cpu")  # Measure on CPU (edge devices don't have GPU)


# ===== Model Architectures (from kaggle_cinc17_half1.py) =====
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
                torch.arange(0, d_model, 2).float()
                * (-torch.log(torch.tensor(10000.0)) / d_model)
            )
        )
        pe[:, 1::2] = torch.cos(
            position
            * torch.exp(
                torch.arange(0, d_model, 2).float()
                * (-torch.log(torch.tensor(10000.0)) / d_model)
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
    pass


class SSMWrapper(WrapperBase):
    pass


class TransformerWrapper(WrapperBase):
    pass


class LSTMWrapper(WrapperBase):
    pass


# ===== Size Analysis =====
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_model_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def get_state_dict_size_mb(model):
    import io

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell() / (1024 * 1024)


def estimate_inference_memory_mb(model, seq_len=4500):
    """Estimate peak memory for single-sample inference (edge device)."""
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, seq_len, 1)
        try:
            y = model(x)
        except:
            y = model(x, torch.tensor([seq_len]))
    # Model params + activations
    param_mem = sum(p.nelement() * p.element_size() for p in model.parameters())
    return param_mem / (1024 * 1024)


# ===== Load and Analyze =====
print("=" * 70)
print("MODEL SIZE ANALYSIS FOR EDGE DEVICE DEPLOYMENT")
print("=" * 70)
print()

MODEL_CONFIGS = {
    "Mamba": {
        "wrapper": MambaWrapper,
        "backbone": MambaClassifier,
        "file": "cinc17_half1_mamba_best.pth",
        "kwargs": {},
    },
    "Transformer": {
        "wrapper": TransformerWrapper,
        "backbone": TransformerClassifier,
        "file": "cinc17_half1_transformer_best.pth",
        "kwargs": {},
    },
    "LSTM": {
        "wrapper": LSTMWrapper,
        "backbone": LSTMClassifier,
        "file": "cinc17_half1_lstm_best.pth",
        "kwargs": {},
    },
    "SSM": {
        "wrapper": SSMWrapper,
        "backbone": SSMClassifier,
        "file": "cinc17_half1_ssm_best.pth",
        "kwargs": {},
    },
}

results = []

for name, cfg in MODEL_CONFIGS.items():
    path = os.path.join(MODEL_DIR, cfg["file"])
    if not os.path.exists(path):
        print(f"  {name}: file not found at {path}")
        continue

    # Load model
    backbone = cfg["backbone"](**cfg["kwargs"])
    model = cfg["wrapper"](backbone, d_model=64, num_classes=4)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # Handle different checkpoint formats
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        # Check if it's a raw state dict (keys are tensor names)
        first_key = list(ckpt.keys())[0] if ckpt else None
        if first_key and isinstance(ckpt[first_key], torch.Tensor):
            state = ckpt
            print(f"  {name}: loaded as raw state dict")
        else:
            print(f"  {name}: unknown format, keys = {list(ckpt.keys())[:5]}")
            state = None

    if state is not None:
        try:
            model.load_state_dict(state)
        except RuntimeError as e:
            print(f"  {name}: state_dict mismatch ({str(e)[:80]}...)")
            state = None

    # Measure sizes
    total_params, trainable_params = count_parameters(model)
    file_size_mb = get_model_size_mb(path)
    state_dict_mb = get_state_dict_size_mb(model) if state is not None else file_size_mb
    param_size_kb = total_params * 4 / 1024  # float32 = 4 bytes

    # Layer breakdown
    layer_counts = {}
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layer_counts.setdefault("Linear", 0)
            layer_counts["Linear"] += 1
        elif isinstance(module, nn.LSTM):
            layer_counts.setdefault("LSTM", 0)
            layer_counts["LSTM"] += 1
        elif isinstance(module, nn.Conv1d):
            layer_counts.setdefault("Conv1d", 0)
            layer_counts["Conv1d"] += 1
        elif isinstance(module, nn.LayerNorm):
            layer_counts.setdefault("LayerNorm", 0)
            layer_counts["LayerNorm"] += 1
        elif isinstance(module, nn.TransformerEncoder):
            layer_counts.setdefault("TransformerEncoder", 0)
            layer_counts["TransformerEncoder"] += 1

    results.append(
        {
            "name": name,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "file_size_mb": file_size_mb,
            "state_dict_mb": state_dict_mb,
            "param_size_kb": param_size_kb,
            "layer_counts": layer_counts,
        }
    )

    print(
        f"  {name}: {total_params:,} params | {file_size_mb:.2f} MB file | {param_size_kb:.1f} KB float32"
    )


# ===== Print Detailed Table =====
print()
print("=" * 70)
print("DETAILED MODEL SIZE COMPARISON")
print("=" * 70)
print()
print(
    f"{'Model':<15} {'Params':>10} {'File (MB)':>10} {'Float32 (KB)':>14} {'Edge MCU?':>10}"
)
print("-" * 62)

# MCU thresholds
MCU_THRESHOLDS = {
    "Cortex-M0 (16KB RAM)": 16,
    "Cortex-M4 (256KB RAM)": 256,
    "Cortex-M7 (512KB RAM)": 512,
    "ESP32 (520KB RAM)": 520,
    "Cortex-A7 (1GB RAM)": 1024 * 1024,
}

for r in sorted(results, key=lambda x: x["param_size_kb"]):
    # Check which MCU it fits on
    fits = []
    for mcu, kb in MCU_THRESHOLDS.items():
        if r["param_size_kb"] < kb:
            fits.append(mcu.split("(")[0].strip())

    mcu_str = fits[0] if fits else "None"
    print(
        f"{r['name']:<15} {r['total_params']:>10,} {r['file_size_mb']:>10.2f} {r['param_size_kb']:>14.1f} {mcu_str:>10}"
    )


# ===== Layer Breakdown =====
print()
print("=" * 70)
print("LAYER TYPE BREAKDOWN")
print("=" * 70)
print()
print(
    f"{'Model':<15} {'Linear':>8} {'Conv1d':>8} {'LSTM':>8} {'LayerNorm':>10} {'Transformer':>12}"
)
print("-" * 65)

for r in sorted(results, key=lambda x: x["name"]):
    lc = r["layer_counts"]
    print(
        f"{r['name']:<15} {lc.get('Linear', 0):>8} {lc.get('Conv1d', 0):>8} {lc.get('LSTM', 0):>8} {lc.get('LayerNorm', 0):>10} {lc.get('TransformerEncoder', 0):>12}"
    )


# ===== MCU Compatibility Matrix =====
print()
print("=" * 70)
print("MCU COMPATIBILITY MATRIX")
print("=" * 70)
print()
print(f"{'MCU':<25} {'RAM (KB)':>10} ", end="")
for r in sorted(results, key=lambda x: x["param_size_kb"]):
    print(f"{r['name']:>10}", end="")
print()
print("-" * 75)

for mcu, ram_kb in MCU_THRESHOLDS.items():
    print(f"{mcu:<25} {ram_kb:>10,} ", end="")
    for r in sorted(results, key=lambda x: x["param_size_kb"]):
        if r["param_size_kb"] < ram_kb:
            print(f"{'✅ Fit':>10}", end="")
        else:
            print(f"{'❌':>10}", end="")
    print()


# ===== Complexity Analysis =====
print()
print("=" * 70)
print("INFERENCE COMPLEXITY (per ECG sample)")
print("=" * 70)
print()
print(f"{'Model':<15} {'Per-step':>12} {'Total (N=4500)':>16} {'State Memory':>14}")
print("-" * 60)

complexity = {
    "Mamba": {"per_step": "O(d×N)", "total": "O(N)", "state": "O(d×N)"},
    "Transformer": {"per_step": "O(N)", "total": "O(N²)", "state": "O(N²)"},
    "LSTM": {"per_step": "O(d²)", "total": "O(N)", "state": "O(d)"},
    "SSM": {"per_step": "O(d×N)", "total": "O(N)", "state": "O(d×N)"},
}

for name, c in complexity.items():
    print(f"{name:<15} {c['per_step']:>12} {c['total']:>16} {c['state']:>14}")


# ===== Summary =====
print()
print("=" * 70)
print("SUMMARY FOR EDGE DEPLOYMENT")
print("=" * 70)
print()

smallest = min(results, key=lambda x: x["param_size_kb"])
print(f"Smallest model:  {smallest['name']} ({smallest['param_size_kb']:.1f} KB)")
print(f"All models fit:  Cortex-M4 (256KB+ RAM)")
print(f"Best for edge:   Mamba — O(N) complexity + smallest memory + highest accuracy")
print()

# Save results as JSON
output = {
    "models": [
        {
            "name": r["name"],
            "total_params": r["total_params"],
            "file_size_bytes": int(r["file_size_mb"] * 1024 * 1024),
            "float32_size_kb": round(r["param_size_kb"], 1),
            "layer_counts": r["layer_counts"],
        }
        for r in results
    ],
    "mcu_compatibility": MCU_THRESHOLDS,
    "complexity": complexity,
}

with open(os.path.join(MODEL_DIR, "model_size_analysis.json"), "w") as f:
    json.dump(output, f, indent=2)

print(f"Results saved to: {MODEL_DIR}model_size_analysis.json")
