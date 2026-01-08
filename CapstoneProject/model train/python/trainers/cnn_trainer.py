# trainers/cnn_trainer.py

from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

import utils.utils as utils


# EEGNET (CNN) MODEL DEFINITION
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


# TRAINING LOOP FUNCTION
def run_epoch(model, loader, train: bool, *, device, optimizer, criterion):
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
                                *, device, optimizer, criterion,
                                max_epochs, patience, min_delta):
    """
    Trains until val loss stops improving.
    Returns: (best_state_dict, history_list)
    prioritizes val loss to give an idea of MODEL CONFIDENCE.
    """
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = []

    for ep in range(1, max_epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, train=True,
                                    device=device, optimizer=optimizer, criterion=criterion)
        va_loss, va_acc = run_epoch(model, val_loader,   train=False,
                                    device=device, optimizer=optimizer, criterion=criterion)

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


def train_cnn(
    *,
    X: np.ndarray,             # (N,C,T) float32 (already normalized in train_ssvep.py)
    y_cls: np.ndarray,         # (N,) int64 class indices 0..K-1
    n_ch: int,
    n_time: int,
    n_classes: int,
    seed: int,
    val_ratio: float,
    batch_size: int,
    learning_rate: float,
    max_epochs: int,
    patience: int,
    min_delta: float,
    device: torch.device,
) -> Tuple[EEGNet, Dict[str, torch.Tensor], List[Dict[str, float]]]:
    """
    Returns:
      model (with best weights loaded),
      best_state (cpu tensors),
      history (list of dicts)
    """
    # BUILD TENSORS (DEFINING EEGNET CNN)
    # PyTorch Conv2d wants (B, 1, C, T) so we need to reshape
    # i.e. each training batch gets:
    # B is the batch size (how many windows processed together) e.g. 1 window only would have B = 1
    # in_channels is 1 because EEG data itself is 1D (whereas RGB would be 3D wtv)
    # C is the spatial dimension (height) -> how many channels there are (8)
    # T is the temporal dimension (width) -> time samples (number of samples per window)
    N, C, T = X.shape
    assert C == n_ch
    assert T == n_time
    assert X.shape[0] == y_cls.shape[0], f"X has N={X.shape[0]} windows but y has {y_cls.shape[0]} labels"

    # ---- Reshape X for Conv2d: (N,C,T) -> (N,1,C,T) ----
    X_4d = X[:, None, :, :]  # insert dimension at axis=1
    print("X_4d shape:", X_4d.shape)
    assert X_4d.shape == (N, 1, C, T)

    # ---- Make PyTorch tensors ----
    # X tensor must be float32. y tensor for classification must be int64 (Long).
    X_t = torch.from_numpy(X_4d).float()
    y_t = torch.from_numpy(y_cls.astype(np.int64)).long()

    # ---- Build a Dataset + DataLoaders ----
    dataset = TensorDataset(X_t, y_t)
    # 80/20 split
    n_val = max(1, int(val_ratio * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    # small since dataset is currently very small (will need to be tuned fs)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"train windows: {len(train_ds)}, val windows: {len(val_ds)}")

    # Instantiate model
    model = EEGNet(n_ch=C, n_time=T, n_classes=n_classes).to(device)
    print(model)

    # Loss + optimizer
    # CrossEntropyLoss expects:
    #   logits (current 'scores'): (B, K)
    #   labels: (B,) with values 0..K-1
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_state, history = run_training_to_convergence(
        model, train_loader, val_loader,
        device=device, optimizer=optimizer, criterion=criterion,
        max_epochs=max_epochs, patience=patience, min_delta=min_delta,
    )

    model.eval()
    with torch.no_grad():
        xb0, yb0 = next(iter(val_loader))
        xb0 = xb0.to(device)
        out = model(xb0)
        print("logits shape:", tuple(out.shape), " (should be (B,K))")

    return model, best_state, history


def train_cnn_on_split(
    *,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray,   y_val: np.ndarray,
    n_ch: int, n_time: int,
    seed: int,
    batch_size: int,
    learning_rate: float,
    max_epochs: int,
    patience: int,
    min_delta: float,
    device: torch.device,
) -> tuple[float, float]:
    """
    Trains CNN on a specific split and returns (val_bal_acc, val_acc).
    This avoids random_split so we can do proper CV.
    """
    # Make training more repeatable across runs
    torch.manual_seed(seed)
    np.random.seed(seed)


    # reshape: (N,C,T) -> (N,1,C,T)
    Xtr = torch.from_numpy(X_train[:, None, :, :]).float()
    ytr = torch.from_numpy(y_train.astype(np.int64)).long()
    Xva = torch.from_numpy(X_val[:, None, :, :]).float()
    yva = torch.from_numpy(y_val.astype(np.int64)).long()

    train_ds = TensorDataset(Xtr, ytr)
    val_ds   = TensorDataset(Xva, yva)

    # deterministic shuffling
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model = EEGNet(n_ch=n_ch, n_time=n_time, n_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    run_training_to_convergence(
        model, train_loader, val_loader,
        device=device, optimizer=optimizer, criterion=criterion,
        max_epochs=max_epochs, patience=patience, min_delta=min_delta,
    )

    # Evaluate on val
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            p = torch.argmax(logits, dim=1).cpu().numpy()
            preds.append(p)
            trues.append(yb.numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)

    val_acc = float((y_pred == y_true).mean())
    val_bal = utils.balanced_accuracy(y_true, y_pred)
    return val_bal, val_acc


def export_cnn_onnx(
    *,
    model: EEGNet,          # trained model (weights already loaded)
    n_ch: int,
    n_time: int,
    out_path: Path,
) -> Path:
    """
    ONNX export helper (called by train_ssvep.py)
    NOTE: requires `pip install onnx`
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Export from CPU model for fewer surprises
    model_export = EEGNet(n_ch=n_ch, n_time=n_time, n_classes=model.n_classes)
    model_export.load_state_dict({k: v.detach().cpu() for k, v in model.state_dict().items()})
    model_export.eval()

    dummy_input = torch.zeros(1, 1, n_ch, n_time, dtype=torch.float32)

    torch.onnx.export(
        model_export,
        dummy_input,
        str(out_path),
        input_names=["x"],
        output_names=["logits"],
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}},
    )

    print("Exported ONNX ->", str(out_path.resolve()))
    return out_path