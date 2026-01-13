# trainers/cnn_trainer.py

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, BatchSampler

import utils.utils as utils

# CNN Hyperparameters & Configs DataClass
@dataclass(frozen=True)
class CNNTrainConfig:
    # Optimization & Convergence Stability
    max_epochs: int = 400
    batch_size: int = 12             # how many training windows the CNN sees at once before updating its weights (1 optimizer step) -> keep batches small for overlapping EEG windows
    learning_rate: float = 1e-3      # magnitude of gradient descent steps. smaller batches require smaller LR. (1e-3 is adam optimizer default)
    seed: int = 0

    # Generalization Control
    patience: int = 25               # Number of successive iterations we'll continue for when seeing no improvement [larger = less premature stopping but nore overfit risk]
    min_delta: float = 1e-3          # numerical change in loss func necessary to consider real improvement 
    
    # TODO: should choose batch_size based on number of training windows if arg is not given and scale LR accordingly
    # TODO: tune hyperparams... [patience, learning_rate, batch_size, kernel_length, F1, D, F2] ??
    # TODO: option on UI to tune (longer) or use defaults (faster)?

    # Model Capacity & Sizing
    F1: int = 8                      # [temporal frequency detectors] number of output channels from the first layer (inputs to 2nd layer), aka: number of different temporal kernels ('weight matrices') generated
    kernel_length: int = 125         # [125 samples at 250Hz -> 500ms EEG temporal summaries] length of temporal kernel [essentially FIR filters applied to the per-channel temporal streams] 
    
    D: int = 2                       # [spatial variants per frequency] number of output channels from the 2nd layer (inputs to 3rd layer), aka: number of different spatial kernels ('weight matrices') generated, e.g: more weights on occipital from ssvep
    pooling_factor: int = 5          # [downsamples by 5 after spatial block (250Hz/5 = new sampling rate of 50Hz)] for optimal tradeoff between preserving temporal frequency detail, regularization/stability/reduce overfitting, & speed/"cost"
    # TODO: shift up if using high freq

    F2: int = 16                     # [compressed ftr set for classifier] number of output channels from the 3rd layer (inputs to final layer), representing spatiotemporal mixed info blocks (for the 500ms segments)
    pooling_factor_final: int = 10    # [downsamples again by 10 now (50Hz/10 = new sampling rate of 5Hz)]
    
    dropout: float = 0.5             # [regularization trick] "turns off" a fraction of activations to reduce overfitting (not rely too much on any single pathway), greater = greater overfitting resistance
    

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
#    - each spatiotemporal map from 2) gets its own temporal refinement (again)
#    - 1) depthwise temporal conv
#         - number of feature maps stays the same (F1*D), just refined
#         - (B,F1*D,1,T1) -> (B,F1*D,1,T2)
#    - 2) pointwise (1x1) conv (COMPRESSION)
#         - AT EACH TIME INDEX in T2: apply another learned map from F1*D to F2 output channels 
#         - so final out channels = F2 = number of learned features passed to classifier
# 4) Flatten
# 5) Classifier
#    - maps to k classes (B, k)
class EEGNet(nn.Module):
    def __init__(self, n_ch: int, n_time: int, n_classes: int,
                 F1: int = 8,
                 D: int = 2,
                 F2: int = 16,
                 kernel_length: int = 125,
                 pooling_factor: int = 5,
                 pooling_factor_final: int = 10,
                 dropout: float = 0.5):
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
        self.bn2 = nn.BatchNorm2d(F1 * D)              # batch normalization
        self.act = nn.ELU()                            # exponential linear unit (ELU) non-linearity 
        self.pool1 = nn.AvgPool2d(kernel_size=(1, pooling_factor))
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
        self.pool2 = nn.AvgPool2d(kernel_size=(1, pooling_factor_final))  # downsample time again
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

class ListBatchSampler(BatchSampler):
    """
    PyTorch BatchSampler that yields precomputed index lists.
    """
    def __init__(self, batches: list[list[int]]):
        self._batches = batches

    def __iter__(self):
        for b in self._batches:
            yield b

    def __len__(self):
        return len(self._batches)

    
def make_trial_batches(
    trial_ids: np.ndarray,
    *,
    max_trials_per_batch: int = 1,
    max_windows_per_batch: int = 16,
    seed: int = 0,
) -> list[list[int]]:
    """
    Builds batches as lists of indices.
    - Groups by trial_id (no mixing across trials unless max_trials_per_batch > 1)
    - Caps batch size so huge trials get split into multiple optimizer steps
    """
    trial_ids = np.asarray(trial_ids).astype(np.int64)
    uniq = np.unique(trial_ids)

    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)

    # map: trial -> indices (shuffled inside trial)
    trial_to_idx = {}
    for t in uniq:
        idx = np.where(trial_ids == t)[0].astype(np.int64)
        rng.shuffle(idx)
        trial_to_idx[int(t)] = idx.tolist()

    batches: list[list[int]] = []
    for i in range(0, len(uniq), max_trials_per_batch):
        trials = uniq[i : i + max_trials_per_batch]

        # collect windows from these trials
        pool: list[int] = []
        for t in trials:
            pool.extend(trial_to_idx[int(t)])

        # chunk pool into capped batches
        for j in range(0, len(pool), max_windows_per_batch):
            batches.append(pool[j : j + max_windows_per_batch])

    return batches


# TODO: Apply z score normalization to avoid magnitude related skews
def apply_z_score_normalization(X: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    # X: (N, C, T)
    mu = X.mean(axis=2, keepdims=True)
    sd = X.std(axis=2, keepdims=True)
    return (X - mu) / (sd + eps)

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

def train_final_cnn_and_export(
    *,
    X_pair: np.ndarray,
    y_pair: np.ndarray,
    trial_ids_pair: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    n_ch: int,
    n_time: int,
    out_onnx_path: Path,
) -> utils.FinalTrainResults:
    """
    FINAL TRAINING + EXPORT (shared trainer API)

    What we do here:
    1) Run CV to report FINAL metrics (loss/acc on held-out folds)
       - This gives an honest estimate of generalization for the chosen pair.
    2) Train ONE final model on ALL data (max data = best deployable model)
    3) Export that final model to ONNX

    """

    cfg = CNNTrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) CV METRICS (for reporting)
    # We'll average final_train_loss/acc and final_val_loss/acc across folds for FinalTrainResults.
    tr_losses: list[float] = []
    tr_accs:   list[float] = []
    va_losses: list[float] = []
    va_accs:   list[float] = []

    if folds and len(folds) >= 2:
        for fi, (tr_idx, va_idx) in enumerate(folds):
            fold_seed = int(cfg.seed + 1000 * fi)

            va_bal, va_acc, tr_loss, tr_acc, va_loss = train_cnn_on_split(
                X_train=X_pair[tr_idx], y_train=y_pair[tr_idx],
                X_val=X_pair[va_idx],   y_val=y_pair[va_idx],
                trial_ids_train=trial_ids_pair[tr_idx],
                trial_ids_val=trial_ids_pair[va_idx],
                n_ch=n_ch, n_time=n_time,
                seed=fold_seed,
                batch_size=min(cfg.batch_size, int(len(tr_idx))) if len(tr_idx) > 0 else cfg.batch_size,
                learning_rate=cfg.learning_rate,
                max_epochs=cfg.max_epochs,      # final training can use full schedule
                patience=cfg.patience,
                min_delta=cfg.min_delta,
                device=device,
            )

            tr_losses.append(float(tr_loss))
            tr_accs.append(float(tr_acc))
            va_losses.append(float(va_loss))
            va_accs.append(float(va_acc))

    # 2) TRAIN FINAL MODEL ON ALL DATA
    # We train on ALL X_pair/y_pair so the exported model sees maximum calibration windows.
    # We'll create a "val_loader" that is just the train set again, so early-stopping doesn't crash.

    # Normalize inside trainer (per-window z-score)
    X_all = apply_z_score_normalization(X_pair)

    # reshape: (N,C,T) -> (N,1,C,T)
    X_all_t = torch.from_numpy(X_all[:, None, :, :]).float()
    y_all_t = torch.from_numpy(y_pair.astype(np.int64)).long()

    ds_all = TensorDataset(X_all_t, y_all_t)

    g = torch.Generator().manual_seed(cfg.seed)
    # final data loader creation -> prioritize trial based batching
    if trial_ids_pair is not None:
        batches = make_trial_batches(
            trial_ids_pair,
            max_trials_per_batch=1, # don't mix trials (just shuffles)
            max_windows_per_batch=cfg.batch_size, # still keep steps small
            seed=cfg.seed,
        )
        train_loader = DataLoader(
            ds_all,
            batch_sampler=ListBatchSampler(batches),
        )
    else:
        g = torch.Generator().manual_seed(cfg.seed)
        train_loader = DataLoader(
            ds_all,
            batch_size=min(cfg.batch_size, len(ds_all)) if len(ds_all) > 0 else cfg.batch_size,
            shuffle=True,
            generator=g
        )

    # Build model using cfg hyperparams
    model = EEGNet(
        n_ch=n_ch, n_time=n_time, n_classes=2,
        F1=cfg.F1, D=cfg.D, F2=cfg.F2, 
        kernel_length=cfg.kernel_length, dropout=cfg.dropout,
        pooling_factor=cfg.pooling_factor, pooling_factor_final=cfg.pooling_factor_final,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # Train to convergence
    for ep in range(cfg.max_epochs):
        tr_loss, tr_acc = run_epoch(model, train_loader, train=True, device=device, optimizer=optimizer, criterion=criterion)
        print(f"[FINAL] Epoch {ep+1:03d}/{cfg.max_epochs} train_loss={tr_loss:.4f} train_acc={tr_acc:.3f}")

    # 3) EXPORT
    # export helper expects trained model weights
    export_ok = True
    try:
        export_cnn_onnx(
            model=model,
            n_ch=n_ch,
            n_time=n_time,
            out_path=Path(out_onnx_path),
        )
    except Exception as e:
        print("[CNN] ONNX export failed:", e)
        export_ok = False

    # FINAL RESULT OBJECT
    # If folds weren't provided, we still trained/exported, but metrics will be zeros.
    def _mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    return utils.FinalTrainResults(
        train_ok=True,
        onnx_export_ok=bool(export_ok),
        final_train_loss=_mean(tr_losses),
        final_train_acc=_mean(tr_accs),
        final_val_loss=_mean(va_losses),
        final_val_acc=_mean(va_accs),
    )

def train_cnn_on_split(
    *,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray,   y_val: np.ndarray,
    trial_ids_train: np.ndarray | None = None,
    trial_ids_val: np.ndarray | None = None,
    n_ch: int, n_time: int,
    seed: int,
    batch_size: int,
    learning_rate: float,
    max_epochs: int,
    patience: int,
    min_delta: float,
    device: torch.device,
) -> tuple[float, float, float, float, float]:
    """
    Trains CNN on a specific split and returns:
     (val_bal_acc, val_acc, train_loss, train_acc, val_loss)
    This avoids random_split so we can do proper CV.
    """
    # Make training more repeatable across runs
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # normalize by z-score to avoid large magnitude skews
    X_train = apply_z_score_normalization(X_train)
    X_val   = apply_z_score_normalization(X_val)

    # reshape: (N,C,T) -> (N,1,C,T) for Conv2d:
    # C is the spatial dimension (height) -> how many channels there are (8)
    # T is the temporal dimension (width) -> time samples (number of samples per window)
    Xtr = torch.from_numpy(X_train[:, None, :, :]).float()
    ytr = torch.from_numpy(y_train.astype(np.int64)).long()
    Xva = torch.from_numpy(X_val[:, None, :, :]).float()
    yva = torch.from_numpy(y_val.astype(np.int64)).long()

    train_ds = TensorDataset(Xtr, ytr)
    val_ds   = TensorDataset(Xva, yva)

    # deterministic shuffling for the train loader
    g = torch.Generator().manual_seed(seed)
    if trial_ids_train is not None:
        # DESIRED: Trial-batched training: one optimizer step per trial (or per small group of trials)
        batches = make_trial_batches(
            trial_ids_train,
            max_trials_per_batch=1,   # 1 trial per step
            seed=seed,
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=ListBatchSampler(batches),
        )
    else:
        # fallback: window batching (less good for strongly overlapping ssvep windows...)
        g = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            train_ds,
            batch_size=min(batch_size, len(train_ds)) if len(train_ds) > 0 else batch_size,
            shuffle=True,
            generator=g
        )

    # window-batching
    val_loader = DataLoader(
        val_ds,
        batch_size=min(batch_size, len(val_ds)) if len(val_ds) > 0 else batch_size,
        shuffle=False
    )

    # Instantiate model with configs
    cfg = CNNTrainConfig()
    model = EEGNet(
        n_ch=n_ch, n_time=n_time, n_classes=2,
        F1=cfg.F1, D=cfg.D, F2=cfg.F2, pooling_factor=cfg.pooling_factor, pooling_factor_final=cfg.pooling_factor_final,
        kernel_length=cfg.kernel_length, dropout=cfg.dropout
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    run_training_to_convergence(
        model, train_loader, val_loader,
        device=device, optimizer=optimizer, criterion=criterion,
        max_epochs=max_epochs, patience=patience, min_delta=min_delta,
    )

    # FINAL METRICS
    # train and val loss/acc
    train_loss, train_acc = run_epoch(
        model, train_loader, train=False,
        device=device, optimizer=optimizer, criterion=criterion
    )
    val_loss, val_acc = run_epoch(
        model, val_loader, train=False,
        device=device, optimizer=optimizer, criterion=criterion
    )
    # Balanced accuracy on val
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
    return (
        float(val_bal),
        float(val_acc),
        float(train_loss),
        float(train_acc),
        float(val_loss),
    )


def score_pair_cv_cnn(
    *,
    X_pair: np.ndarray,
    y_pair: np.ndarray,
    trial_ids_pair: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    n_ch: int,
    n_time: int,
    freq_a_hz: int,
    freq_b_hz: int,
) -> utils.ModelMetrics:
    """
    Cross-val scoring for ONE (already-binary) pair.
    Folds are built in train_ssvep.py (shared across archs).
    Returns ModelMetrics (shared contract).
    """
    cfg = CNNTrainConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bals: list[float] = []
    accs: list[float] = []

    for fi, (tr_idx, va_idx) in enumerate(folds):
        # per-fold seed so folds are repeatable but not identical
        fold_seed = int(cfg.seed + 1000 * fi)

        val_bal, val_acc, _, _, _ = train_cnn_on_split(
            X_train=X_pair[tr_idx], y_train=y_pair[tr_idx],
            X_val=X_pair[va_idx],   y_val=y_pair[va_idx],
            trial_ids_train=trial_ids_pair[tr_idx],
            trial_ids_val=trial_ids_pair[va_idx],
            n_ch=n_ch, n_time=n_time,
            seed=fold_seed,
            batch_size=min(cfg.batch_size, int(len(tr_idx))) if len(tr_idx) > 0 else cfg.batch_size,
            learning_rate=cfg.learning_rate,
            max_epochs=cfg.max_epochs,
            patience=cfg.patience,
            min_delta=cfg.min_delta,
            device=device,
        )
        bals.append(float(val_bal))
        accs.append(float(val_acc))

    if not bals:
        return utils.ModelMetrics(
            freq_a_hz=int(freq_a_hz),
            freq_b_hz=int(freq_b_hz),
            cv_ok=False,
            avg_fold_balanced_accuracy=0.0,
            std_fold_balanced_accuracy=0.0,
            avg_fold_accuracy=0.0,
            std_fold_accuracy=0.0,
        )

    return utils.ModelMetrics(
        freq_a_hz=int(freq_a_hz),
        freq_b_hz=int(freq_b_hz),
        cv_ok=True,
        avg_fold_balanced_accuracy=float(np.mean(bals)),
        std_fold_balanced_accuracy=float(np.std(bals)),
        avg_fold_accuracy=float(np.mean(accs)),
        std_fold_accuracy=float(np.std(accs)),
    )


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
    model_export = model.to("cpu")
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