# exploration stage :,O

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import json
import os

# Global constants
MAX_EPOCHS = 300        # hard cap
PATIENCE   = 30         # epochs allowed with no improvement
MIN_DELTA  = 1e-4       # smallest improvement to consider as an 'improvement'
# Variables to keep track
best_val_loss = float("inf")
best_state = None
epochs_no_improve = 0
history = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using hw device:", device)

# STEP 1: BUILD TRAINING DATA
# We read CSV and get the shape of X
# (N, C, T) = (num_windows, num_channels, window_len_samples)

CSV_PATH = "eeg_windows_4.csv"
df = pd.read_csv(CSV_PATH)

# expected columns from CSV:
# window_idx, ui_state, is_trimmed, is_bad, sample_idx, ch0..chN-1, testfreq_e, testfreq_hz
all_cols = list(df.columns)
tf_e_i = all_cols.index("testfreq_e") # training frequency enum
sample_i = all_cols.index("sample_idx")
ch_cols = all_cols[sample_i + 1 : tf_e_i]
n_ch = len(ch_cols)

print("rows:", len(df))
print("n_ch:", n_ch)
print("first 10 cols:", all_cols[:10])
print("last 10 cols:", all_cols[-10:])

# Keep only trimmed rows 
df = df[df["is_trimmed"] == 1].copy()

# Drop "bad" rows (artifactual windows)
df = df[df["is_bad"] != 1].copy()

print("after filters rows:", len(df))

grouped = df.groupby("window_idx", sort=True)
windows = []
labels = []

for wid, g in grouped:
    g = g.sort_values("sample_idx")
    tf_hz = int(g["testfreq_hz"].iloc[0])
    if tf_hz < 0:
        continue  # ignore TestFreq_None windows for now

    x = g[ch_cols].to_numpy(dtype=np.float32)  # (T, C)
    if x.ndim != 2 or x.shape[1] != n_ch:
        continue

    windows.append(x.T)  # -> (C, T)
    labels.append(tf_hz)

X = np.stack(windows, axis=0)  # (N, C, T)
y = np.array(labels)

print("X shape (N,C,T):", X.shape)
print("y shape (labels):", y.shape)
print("unique labels tf_hz:", sorted(set(y.tolist())))
assert X.shape[0] == y.shape[0], f"X has N={X.shape[0]} windows but y has {y.shape[0]} labels"

# STEP 2: BASIC DATA NORMALIZATION
eps = 1e-6
mu = X.mean(axis=2, keepdims=True)          # (N, C, 1)
sd = X.std(axis=2, keepdims=True)           # (N, C, 1)
Xn = (X - mu) / (sd + eps)
# quick sanity checks
print("X mean (should be ~0):", float(Xn.mean()))
print("X std  (should be ~1):", float(Xn.std()))
print("Any NaNs?", np.isnan(Xn).any())
print("Any infs?", np.isinf(Xn).any())
X = Xn

# STEP 3: BUILD TENSORS (DEFINING EEGNET CNN)
# PyTorch Conv2d wants (B, 1, C, T) so we need to reshape
# i.e. each training batch gets:
# B is the batch size (how many windows processed together) e.g. 1 window only would have B = 1
# in_channels is 1 because EEG data itself is 1D (whereas RGB would be 3D wtv)
# C is the spatial dimension (height) -> how many channels there are (8)
# T is the temporal dimension (width) -> time samples (number of samples per window)
N, C, T = X.shape
print("Step3: starting with X shape:", X.shape, " y shape:", y.shape)

# ---- 1) Convert Hz labels -> class indices 0..K-1 ----
# PyTorch classification expects labels like [0,1,2,3,4] not [8,9,10,11,12]
unique_hz = sorted(set(y.tolist()))
K = len(unique_hz)
hz_to_class = {hz: i for i, hz in enumerate(unique_hz)}
class_to_hz = {i: hz for hz, i in hz_to_class.items()}

y_cls = np.array([hz_to_class[int(h)] for h in y], dtype=np.int64)  # int64 for PyTorch
print("unique_hz:", unique_hz)
print("K classes:", K)
print("y_cls unique:", sorted(set(y_cls.tolist())))

# ---- 2) Reshape X for Conv2d: (N,C,T) -> (N,1,C,T) ----
X_4d = X[:, None, :, :]  # insert dimension at axis=1
print("X_4d shape:", X_4d.shape)
assert X_4d.shape == (N, 1, C, T)

# ---- 3) Make PyTorch tensors ----
# X tensor must be float32. y tensor for classification must be int64 (Long).
X_t = torch.from_numpy(X_4d).float()
y_t = torch.from_numpy(y_cls).long()

# ---- 4) Build a Dataset + DataLoaders ----
dataset = TensorDataset(X_t, y_t)
# 80/20 split 
n_val = max(1, int(0.2 * len(dataset)))
n_train = len(dataset) - n_val
train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
batch_size = 16  # small since dataset is currently very small (will need to be tuned fs)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

print(f"train windows: {len(train_ds)}, val windows: {len(val_ds)}")

# STEP 4: EEGNET (CNN) MODEL DEFINITION
# A layer is a block in the NN
# A kernel is a set of learned weights inside a CONVOLUTIONAL layer specifically
# EEGNet has a small number of layers, but each convolutional layer contains many kernels
# - for each layer, the number of kernels = the number of out_channels 
#   (each kernel maps to one feature)
# CURRENT ARCHITECTURE
#   - 4 layer CNN Input: (B, 1, C, T) Output logits: (B, K)
#   - (logits are the unnormalized class scores before any softmax)
#   - (we use logits bcuz the cross entropy loss expects that, not probabilities)
# 1) TEMPORAL convolution block: slides along time (no channel mixing) to learn temporal ssvep patterns
#   - single convolutional layer + BatchNorm
#   - number of kernels = F1
#   - output shape (B, F1, C, T)
# 2) SPATIAL convolution block: slides along space (mixing channels) to learn spatial EEG patterns (electrode combinations)
#    - single convolutional layer + BatchNorm + AvgPool + Dropout
#    - depthwise = ONE spatial filter for every temporal filter (each temporal feature gets its own spatial projection)
#    - output shape (B, F1*D, C, T)
#    - where D is the number of ftrs you get from a single spatial kernel (i.e. num of spatial combinations you learn per temporal filter)
# 3) SEPARABLE convolution layer
#    - 2 convolutional layers (depthwise & pointwise) + BatchNorm + AvgPool + Dropout
#    - each spatial map from 2) gets its own temporal refinement
#    - number of feature maps stays the same (F1*D)
#    - output shape (B, F1*D, C, T') <- time downsampled due to pooling
#    - so final out channels = F1*D = F2 = number of learned features passed to classifier
# 4) Flatten
# 5) Classifier 
#    - maps to k classes (B, k)
class EEGNet(nn.Module):
    def __init__(self, n_ch: int, n_time: int, n_classes: int,
                 F1: int = 8,      # number of temporal filters
                 D: int = 2,       # depth multiplier for spatial filters
                 F2: int = 16,     # number of pointwise filters after separable conv
                 kernel_length: int = 64,
                 dropout: float = 0.25):
        super().__init__()
        self.n_ch = n_ch
        self.n_time = n_time
        self.n_classes = n_classes

        # ------------------------------
        # temporal conv
        # Conv2d over time only:
        # input  (B, 1, C, T)
        # output (B, F1, C, T)
        # ------------------------------
        self.conv_temporal = nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, kernel_length),
            padding=(0, kernel_length // 2),
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1)

        # ------------------------------
        # Depthwise spatial conv (mix across channels)
        # fixed kernel size (C, 1), vary D
        # output shape becomes (B, F1*D, 1, T)
        # ------------------------------
        self.conv_spatial = nn.Conv2d(
            in_channels=F1,
            out_channels=F1 * D,
            kernel_size=(n_ch, 1),
            groups=F1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.act = nn.ELU()
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))  # downsample time by 4
        self.drop1 = nn.Dropout(dropout)

        # ------------------------------
        # separable conv (depthwise temporal + pointwise)
        # Depthwise temporal conv: groups = F1*D
        # Pointwise conv: 1x1 mixes feature maps
        # ------------------------------
        self.sep_depth = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F1 * D,
            kernel_size=(1, 16),
            padding=(0, 16 // 2),
            groups=F1 * D,
            bias=False
        )
        self.sep_point = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F2,
            kernel_size=(1, 1),
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))  # downsample time again
        self.drop2 = nn.Dropout(dropout)

        # ------------------------------
        # Classifier: we need to know the flattened feature size.
        # We compute it by doing a dummy forward pass with zeros.
        # ------------------------------
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_ch, n_time)
            feat = self._forward_features(dummy)
            feat_dim = feat.shape[1]  # (B, feat_dim)
        self.classifier = nn.Linear(feat_dim, n_classes)

    def _forward_features(self, x):
        # x: (B, 1, C, T)
        x = self.conv_temporal(x)
        x = self.bn1(x)

        x = self.conv_spatial(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.sep_depth(x)
        x = self.sep_point(x)
        x = self.bn3(x)
        x = self.act(x)
        x = self.pool2(x)
        x = self.drop2(x)

        # flatten everything but batch
        x = x.flatten(start_dim=1)  # (B, features)
        return x

    def forward(self, x):
        # return logits (no softmax here; CrossEntropyLoss expects raw logits)
        feats = self._forward_features(x)
        logits = self.classifier(feats)
        return logits
    
# Instantiate model
model = EEGNet(n_ch=C, n_time=T, n_classes=K).to(device)
print(model)

# Loss + optimizer
# CrossEntropyLoss expects:
#   logits (current 'scores'): (B, K)
#   labels: (B,) with values 0..K-1
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# STEP 5: TRAINING LOOP FUNCTION
def run_epoch(model, loader, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total = 0

    # no_grad in eval to speed up + reduce memory
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for xb, yb in loader:
            xb = xb.to(device)  # (B,1,C,T)
            yb = yb.to(device)  # (B,)

            if train:
                optimizer.zero_grad()

            logits = model(xb)               # (B,K)
            loss = criterion(logits, yb)     # scalar

            if train:
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item()) * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += int((preds == yb).sum().item())
            total += xb.size(0)

    avg_loss = total_loss / max(1, total)
    acc = total_correct / max(1, total)
    return avg_loss, acc

def run_training_to_convergence(model, train_loader, val_loader,
                                max_epochs=MAX_EPOCHS, patience=PATIENCE, min_delta=MIN_DELTA):
    """
    Trains until val loss stops improving.
    Returns: (best_state_dict, history_list)
    """
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = []

    for ep in range(1, max_epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, train=True)
        va_loss, va_acc = run_epoch(model, val_loader,   train=False)

        history.append({
            "epoch": ep,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": va_loss,
            "val_acc": va_acc,
        })

        improved = (best_val_loss - va_loss) > min_delta

        if improved:
            best_val_loss = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            star = " *best*"
        else:
            epochs_no_improve += 1
            star = ""

        print(
            f"Epoch {ep:03d} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.3f} | "
            f"no_improve={epochs_no_improve}/{patience}{star}"
        )

        if epochs_no_improve >= patience:
            print(f"Early stop at epoch {ep} (best val loss {best_val_loss:.4f}).")
            break

    # Restore best weights into the *training* model
    if best_state is not None:
        model.load_state_dict(best_state)
        print("Restored best model weights (lowest val loss).")
    else:
        print("WARNING: best_state never set (unexpected).")

    return best_state, history


best_state, history = run_training_to_convergence(
    model, train_loader, val_loader
)

model.eval()
with torch.no_grad():
    xb0, yb0 = next(iter(val_loader))
    xb0 = xb0.to(device)
    out = model(xb0)
    print("logits shape:", tuple(out.shape), " (should be (B,K))")


# STEP 6: ONNX EXPORT FOR C++
onnx_path = "eegnet.onnx"
model_export = EEGNet(n_ch=C, n_time=T, n_classes=K)
model_export.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
model_export.eval()

dummy_input = torch.zeros(1, 1, C, T, dtype=torch.float32)

torch.onnx.export(
    model_export,
    dummy_input,
    onnx_path,
    input_names=["x"],
    output_names=["logits"],
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}},
)
print("Exported ONNX ->", os.path.abspath(onnx_path))

# STEP 7: JSON META EXPORT FOR C++
meta = {
    "model_type": "EEGNet",
    "input_shape": [1, C, T],
    "expects_4d": True,
    "classes_hz": unique_hz,
    "norm": {
        "type": "zscore_per_window_per_channel",
        "eps": 1e-6
    }
}

with open("eegnet_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print("Wrote meta ->", os.path.abspath("eegnet_meta.json"))