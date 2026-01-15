# trainers/cnn_trainer.py

from __future__ import annotations
from dataclasses import dataclass, replace, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, BatchSampler
from typing import Iterator, List
from itertools import product

import utils.utils as utils

# TODO: option on UI to tune (longer) or use defaults (faster)?
# TODO: add 3rd neutral class -> 3 class classifiers!!!
# TODO: decide pooling args & hyperparam tuning range based on frequencies were looking at. shift up if using high freq to preserve more info (at least 60Hz by Nyquist)

# ===================== DEBUG HELPERS =====================
def _log(logger: utils.DebugLogger | None, msg: str = "") -> None:
    if logger is None:
        print(msg)
    else:
        logger.log(msg)

def eval_pred_stats(model, loader, *, device) -> dict:
    model.eval()
    y_true_all = []
    y_pred_all = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred_all.append(pred)
            y_true_all.append(yb.numpy())

    y_true = np.concatenate(y_true_all) if y_true_all else np.array([], dtype=np.int64)
    y_pred = np.concatenate(y_pred_all) if y_pred_all else np.array([], dtype=np.int64)

    # binary confusion
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    pred0 = int((y_pred == 0).sum())
    pred1 = int((y_pred == 1).sum())
    n = int(len(y_true))

    return {"n": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "pred0": pred0, "pred1": pred1}


# CNN Hyperparameters & Configs DataClass
# Capitals indicate const variables that should never be mutated in this architecture.
@dataclass(frozen=True)
class CNNTrainConfig:
    # Optimization & Convergence Stability
    MAX_EPOCHS: int = 400            # number of epochs for final model training
    MAX_EPOCHS_CV: int = 300         # number of epochs while running all the pairwise models for comparison
    MAX_EPOCHS_HTUNING: int = 200    # number of epochs while running all the candidate grids for hparam tuning
    batch_size: int = 12             # how many (mostly indep, non-overlapping due to batching strategy) training windows the CNN sees at once before updating its weights (1 optimizer step) -> keep batches small for overlapping EEG windows
    learning_rate: float = 1e-3      # magnitude of gradient descent steps. smaller batches require smaller LR. (1e-3 is adam optimizer default)
    seed: int = 0

    # Generalization Control
    patience: int = 25               # Number of successive iterations we'll continue for when seeing no improvement [larger = less premature stopping but nore overfit risk]
    min_delta: float = 1e-3          # numerical change in loss func necessary to consider real improvement 

    # Model Capacity & Sizing
    F1: int = 8                      # [temporal frequency detectors] number of output channels from the first layer (inputs to 2nd layer), aka: number of different temporal kernels ('weight matrices') generated
    kernel_length: int = 125         # [125 samples at 250Hz -> 500ms EEG temporal summaries] length of temporal kernel [essentially FIR filters applied to the per-channel temporal streams] 
    
    D: int = 2                       # [spatial variants per frequency] number of output channels from the 2nd layer (inputs to 3rd layer), aka: number of different spatial kernels ('weight matrices') generated, e.g: more weights on occipital from ssvep
    pooling_factor: int = 5          # [downsamples by 5 after spatial block (250Hz/5 = new sampling rate of 50Hz)] for optimal tradeoff between preserving temporal frequency detail, regularization/stability/reduce overfitting, & speed/"cost"

    pooling_factor_final: int = 10   # [downsamples again by 10 now (50Hz/10 = new sampling rate of 5Hz)]
    
    DROPOUT: float = 0.5             # [regularization trick] "turns off" a fraction of activations to reduce overfitting (not rely too much on any single pathway), greater = greater overfitting resistance

# Hyperparameter tuning space [if hparam tuning on, for final model training only] 
HPARAM_SPACE = {
    "kernel_length": [63, 125, 187],  # 250 ms, 500 ms, or 750 ms temporal segments as layer 2 inputs
    "pooling_factor": [2, 4, 5],      # layer 2 temporal downsampling 
    "pooling_factor_final": [8, 10],  # final layer temporal downsampling
    "F1": [8, 16],                    # number of output (125-length) temporal combinations from layer 1
    "learning_rate": [1e-3, 5e-4],
    "batch_size": [8, 12, 16],
    # total combos = 3 x 3 x 2 x 2 x 2 x 3 = 72
}

def _derived_F2(cfg: CNNTrainConfig) -> int:
    # F2 is fixed in this arch by F1 and D
    # [compressed ftr set for classifier] number of output channels from the 3rd layer (inputs to final layer), representing spatiotemporal mixed info blocks (for the 500ms segments)
    return int(cfg.F1 * cfg.D)

# Iterator to generate candidate configs
def iter_hparam_candidates(
    base_cfg: CNNTrainConfig,
    space: dict[str, list],
) -> Iterator[CNNTrainConfig]:
    keys = list(space.keys())
    for vals in product(*(space[k] for k in keys)): # itertools.product is the equivalent of nested loops (grid search across all key values in space) written compactly
        patch = dict(zip(keys, vals)) 
        yield replace(base_cfg, **patch) # patch updates base (default) cfg with only values that are different in current space key values from product loop

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
            kernel_size=(1, 15),
            padding=(0, 15 // 2),
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
    - NOTE: max_windows_per_batch will always == batch size as long as trials are long enough compared to window length (which they are)
    This is for comparing hparam configs + freq pairs.
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

def make_stratified_trial_batches_new(
    trial_ids: np.ndarray,
    y: np.ndarray,
    *,
    max_windows_per_batch: int = 16,
    seed: int = 0,
    logger: utils.DebugLogger | None = None,
) -> list[list[int]]:
    """
    Build batches ensuring BOTH classes present and roughly balanced.
    Accepts slight imbalance in final batches to avoid dropping data.
    """
    rng = np.random.default_rng(seed)
    trial_ids = np.asarray(trial_ids).astype(np.int64)
    y = np.asarray(y).astype(np.int64)
    
    # Group trials by their majority class
    trials_by_class: dict[int, list[int]] = {0: [], 1: []}
    trial_to_indices: dict[int, np.ndarray] = {}
    
    for tid in np.unique(trial_ids):
        mask = trial_ids == tid
        indices = np.where(mask)[0]
        majority_class = int(np.round(y[mask].mean()))
        trials_by_class[majority_class].append(int(tid))
        trial_to_indices[int(tid)] = indices
    
    # Validate we have trials for both classes
    if len(trials_by_class[0]) == 0 or len(trials_by_class[1]) == 0:
        raise ValueError(
            f"Cannot create stratified batches: "
            f"class0 has {len(trials_by_class[0])} trials, "
            f"class1 has {len(trials_by_class[1])} trials. "
            f"Need at least 1 trial per class."
        )
    
    # Shuffle trials within each class
    for class_trials in trials_by_class.values():
        rng.shuffle(class_trials)
    
    # Collect ALL windows from each class
    windows_class0: list[int] = []
    windows_class1: list[int] = []
    
    for tid in trials_by_class[0]:
        indices = trial_to_indices[tid].copy()
        rng.shuffle(indices)
        windows_class0.extend(indices.tolist())
    
    for tid in trials_by_class[1]:
        indices = trial_to_indices[tid].copy()
        rng.shuffle(indices)
        windows_class1.extend(indices.tolist())
    
    # Shuffle the pools
    rng.shuffle(windows_class0)
    rng.shuffle(windows_class1)
    
    # Build batches with both classes
    batches: list[list[int]] = []
    windows_per_class = max_windows_per_batch // 2
    
    ptr0 = 0
    ptr1 = 0
    
    while ptr0 < len(windows_class0) or ptr1 < len(windows_class1):
        batch: list[int] = []
        
        # Take from class 0 (up to windows_per_class or whatever's left)
        remaining0 = len(windows_class0) - ptr0
        remaining1 = len(windows_class1) - ptr1
        
        # Decide how many to take from each class
        if remaining0 >= windows_per_class and remaining1 >= windows_per_class:
            # Normal case: take balanced amounts
            n_take0 = windows_per_class
            n_take1 = windows_per_class
        elif remaining0 > 0 and remaining1 > 0:
            # Leftover case: take what's available but keep it reasonable
            # Don't allow more than 2:1 ratio
            n_take0 = min(remaining0, windows_per_class)
            n_take1 = min(remaining1, windows_per_class)
            
            # Enforce minimum representation (at least 2 from minority class if possible)
            min_per_class = 2
            if n_take0 >= min_per_class and n_take1 >= min_per_class:
                # Both have enough - keep them balanced
                n_take = min(n_take0, n_take1)
                n_take0 = n_take
                n_take1 = n_take
        else:
            # One class exhausted - stop making batches
            break
        
        # Add windows to batch
        batch.extend(windows_class0[ptr0:ptr0 + n_take0])
        batch.extend(windows_class1[ptr1:ptr1 + n_take1])
        
        ptr0 += n_take0
        ptr1 += n_take1
        
        # Verify both classes present
        if len(batch) > 0:
            batch_has_0 = any(y[idx] == 0 for idx in batch)
            batch_has_1 = any(y[idx] == 1 for idx in batch)
            
            if batch_has_0 and batch_has_1:
                rng.shuffle(batch)
                batches.append(batch)
    
    if len(batches) == 0:
        raise ValueError("Could not create any batches with both classes present")
    
    # Log what we kept
    total_used = sum(len(b) for b in batches)
    total_available = len(windows_class0) + len(windows_class1)
    dropped = total_available - total_used
    
    if dropped > 0:
        _log(logger, f"[BATCH INFO] Used {total_used}/{total_available} windows ({100*total_used/total_available:.1f}%), dropped {dropped} to maintain class balance")
    
    return batches


def make_stratified_trial_batches(
    trial_ids: np.ndarray,
    y: np.ndarray,
    *,
    max_windows_per_batch: int = 16,
    seed: int = 0,
) -> list[list[int]]:
    """
    Build batches ensuring EVERY batch has samples from BOTH classes.
    Uses cycling to handle unequal trial counts.
    """
    rng = np.random.default_rng(seed)
    trial_ids = np.asarray(trial_ids).astype(np.int64)
    y = np.asarray(y).astype(np.int64)
    
    # Group trials by their majority class
    trials_by_class: dict[int, list[int]] = {0: [], 1: []}
    trial_to_indices: dict[int, np.ndarray] = {}
    
    for tid in np.unique(trial_ids):
        mask = trial_ids == tid
        indices = np.where(mask)[0]
        majority_class = int(np.round(y[mask].mean()))
        trials_by_class[majority_class].append(int(tid))
        trial_to_indices[int(tid)] = indices
    
    # Validate we have trials for both classes
    if len(trials_by_class[0]) == 0 or len(trials_by_class[1]) == 0:
        raise ValueError(
            f"Cannot create stratified batches: "
            f"class0 has {len(trials_by_class[0])} trials, "
            f"class1 has {len(trials_by_class[1])} trials. "
            f"Need at least 1 trial per class."
        )
    
    # Shuffle trials within each class
    for class_trials in trials_by_class.values():
        rng.shuffle(class_trials)
    
    # ===== NEW APPROACH: Collect ALL windows, then batch them =====
    # This ensures we use all data and can guarantee balanced batches
    
    all_windows_by_class: dict[int, list[int]] = {0: [], 1: []}
    
    for cls in [0, 1]:
        for tid in trials_by_class[cls]:
            indices = trial_to_indices[tid].copy()
            rng.shuffle(indices)
            all_windows_by_class[cls].extend(indices.tolist())
    
    # Now create batches by alternating between classes
    batches: list[list[int]] = []
    
    ptr0 = 0  # Pointer for class 0 windows
    ptr1 = 0  # Pointer for class 1 windows
    
    windows_per_class = max_windows_per_batch // 2
    
    while ptr0 < len(all_windows_by_class[0]) or ptr1 < len(all_windows_by_class[1]):
        batch_windows: list[int] = []
        
        # Take from class 0
        end0 = min(ptr0 + windows_per_class, len(all_windows_by_class[0]))
        if ptr0 < len(all_windows_by_class[0]):
            batch_windows.extend(all_windows_by_class[0][ptr0:end0])
            ptr0 = end0
        
        # Take from class 1
        end1 = min(ptr1 + windows_per_class, len(all_windows_by_class[1]))
        if ptr1 < len(all_windows_by_class[1]):
            batch_windows.extend(all_windows_by_class[1][ptr1:end1])
            ptr1 = end1
        
        # Only add batch if it has windows from BOTH classes
        has_class0 = any(y[idx] == 0 for idx in batch_windows)
        has_class1 = any(y[idx] == 1 for idx in batch_windows)
        
        if has_class0 and has_class1:
            rng.shuffle(batch_windows)
            batches.append(batch_windows)
        elif len(batch_windows) > 0:
            # Last batch might be incomplete - merge with previous if possible
            if batches:
                batches[-1].extend(batch_windows)
                rng.shuffle(batches[-1])
    
    return batches

def make_stratified_trial_batches_old(
    trial_ids: np.ndarray,
    y: np.ndarray,
    *,
    max_windows_per_batch: int = 16,
    seed: int = 0,
) -> list[list[int]]:
    """
    Build batches ensuring each has samples from BOTH classes.
    Each batch contains windows from exactly 1 trial per class.
    
    This is optimal for heavily overlapping windows because:
    - Maximizes trial diversity per batch
    - Reduces within-batch correlation
    - Forces model to generalize across trials
    
    Args:
        trial_ids: Trial ID for each window
        y: Binary labels (0 or 1) for each window
        max_windows_per_batch: Cap on batch size
        seed: Random seed for reproducibility
    
    Returns:
        List of index lists (one per batch)
    """
    rng = np.random.default_rng(seed)
    trial_ids = np.asarray(trial_ids).astype(np.int64)
    y = np.asarray(y).astype(np.int64)
    
    # Group trials by their majority class
    trials_by_class: dict[int, list[int]] = {0: [], 1: []}
    trial_to_indices: dict[int, np.ndarray] = {}
    
    for tid in np.unique(trial_ids):
        mask = trial_ids == tid
        indices = np.where(mask)[0]
        
        # Assign trial to class based on majority vote
        majority_class = int(np.round(y[mask].mean()))  # 0 or 1
        trials_by_class[majority_class].append(int(tid))
        trial_to_indices[int(tid)] = indices
    
    # Shuffle trials within each class for randomness
    for class_trials in trials_by_class.values():
        rng.shuffle(class_trials)
    
    # Build batches by interleaving trials from each class
    batches: list[list[int]] = []
    max_trials = max(len(trials_by_class[0]), len(trials_by_class[1]))
    
    for i in range(max_trials):
        batch_windows: list[int] = []
        
        # Take windows from one trial per class (if available)
        for cls in [0, 1]:
            if i < len(trials_by_class[cls]):
                tid = trials_by_class[cls][i]
                indices = trial_to_indices[tid].copy()
                rng.shuffle(indices)  # Shuffle within trial
                
                # Take up to half the batch size from this trial
                n_take = min(len(indices), max_windows_per_batch // 2)
                batch_windows.extend(indices[:n_take].tolist())
        
        if len(batch_windows) > 0:
            rng.shuffle(batch_windows)  # Mix the two classes
            
            # Cap at max batch size
            batch_windows = batch_windows[:max_windows_per_batch]
            batches.append(batch_windows)
    
    return batches

def make_trial_holdout_split(
    trial_ids: np.ndarray,
    *,
    holdout_frac: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (train_idx, holdout_idx) by splitting UNIQUE trial IDs.
    Prevents overlap leakage from heavily overlapping windows.
    This is for FINAL MODEL EVALUATION ON SIMPLE HOLDOUT.
    """
    trial_ids = np.asarray(trial_ids).astype(np.int64)
    uniq_trials = np.unique(trial_ids)

    rng = np.random.default_rng(int(seed))
    rng.shuffle(uniq_trials)

    n_holdout_trials = max(1, int(np.ceil(len(uniq_trials) * float(holdout_frac))))
    holdout_trials = set(uniq_trials[:n_holdout_trials].tolist())

    holdout_mask = np.isin(trial_ids, list(holdout_trials))
    holdout_idx = np.where(holdout_mask)[0].astype(np.int64)
    train_idx = np.where(~holdout_mask)[0].astype(np.int64)

    # fall back to no-holdout
    if len(train_idx) == 0 or len(holdout_idx) == 0:
        return np.arange(len(trial_ids), dtype=np.int64), np.array([], dtype=np.int64)

    return train_idx, holdout_idx

# Apply z score normalization to avoid magnitude related skews
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
                                max_epochs, patience, min_delta,
                                logger: utils.DebugLogger | None = None):
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
        _optim_loss, _optim_acc = run_epoch(model, train_loader, train=True,
                                   device=device, optimizer=optimizer, criterion=criterion)

        # report metrics in eval mode (no dropout) for consistent reporting
        tr_loss, tr_acc = run_epoch(model, train_loader, train=False,
                                    device=device, optimizer=optimizer, criterion=criterion)
        va_loss, va_acc = run_epoch(model, val_loader,   train=False,
                                    device=device, optimizer=optimizer, criterion=criterion)

        stats = eval_pred_stats(model, val_loader, device=device)
        _log(logger,
            f"           val_pred_dist: pred0={stats['pred0']} pred1={stats['pred1']} | "
            f"conf: tn={stats['tn']} fp={stats['fp']} fn={stats['fn']} tp={stats['tp']}"
        )

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

        _log(logger,
            f"Epoch {ep:03d} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.3f} | "
            f"no_improve={epochs_no_improve}/{patience}{star}"
        )

        if epochs_no_improve >= patience:
            _log(logger,f"Early stop at epoch {ep} (best val loss {best_val_loss:.4f}).")
            break

    # Restore best weights into the *training* model
    if best_state is not None:
        model.load_state_dict(best_state)
        _log(logger,"Restored best model weights (lowest val loss).")
    else:
        _log(logger,"WARNING: best_state never set (unexpected).")

    return best_state, history

def train_single_final_model_with_holdout_and_export(
    *,
    X_pair: np.ndarray,
    y_pair: np.ndarray,
    trial_ids_pair: np.ndarray,
    n_ch: int,
    n_time: int,
    cfg: CNNTrainConfig,
    out_onnx_path: Path,
    device: torch.device,
    holdout_frac: float = 0.1,
    logger: utils.DebugLogger | None = None
) -> tuple[bool, bool, float, float, float]:
    """
    Trains ONE final model using a small trial-wise holdout for early stopping, then exports ONNX.

    Returns:
      (train_ok, export_ok, holdout_loss, holdout_acc, holdout_bal_acc)
    """
    # Build trial-wise holdout split
    tr_idx, ho_idx = make_trial_holdout_split(
        trial_ids_pair, holdout_frac=float(holdout_frac), seed=int(cfg.seed)
    )

    if len(ho_idx) == 0:
        # No holdout possible; train on all data without early stop and export.
        # (But in practice with multiple trials, ho_idx should not be empty.)
        ho_loss = 0.0
        ho_acc = 0.0
        ho_bal = 0.0
        train_ok = True
        export_ok = True

        # Train loader on all data
        X_all = apply_z_score_normalization(X_pair)
        X_all_t = torch.from_numpy(X_all[:, None, :, :]).float()
        y_all_t = torch.from_numpy(y_pair.astype(np.int64)).long()
        ds_all = TensorDataset(X_all_t, y_all_t)

        batches = make_stratified_trial_batches_new(
            trial_ids=trial_ids_pair,
            y=y_pair,
            max_windows_per_batch=int(cfg.batch_size),
            seed=int(cfg.seed),
        )
        train_loader = DataLoader(ds_all, batch_sampler=ListBatchSampler(batches))

        model = EEGNet(
            n_ch=n_ch, n_time=n_time, n_classes=2,
            F1=int(cfg.F1), D=int(cfg.D), F2=int(_derived_F2(cfg)),
            kernel_length=int(cfg.kernel_length),
            pooling_factor=int(cfg.pooling_factor),
            pooling_factor_final=int(cfg.pooling_factor_final),
            dropout=float(cfg.DROPOUT),
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.learning_rate))

        for ep in range(int(cfg.MAX_EPOCHS)):
            tr_loss, tr_acc = run_epoch(model, train_loader, train=True,
                                        device=device, optimizer=optimizer, criterion=criterion)
            _log(logger,f"[FINAL-NO-HOLDOUT] Epoch {ep+1:03d}/{int(cfg.MAX_EPOCHS)} "
                  f"train_loss={tr_loss:.4f} train_acc={tr_acc:.3f}")

        try:
            export_cnn_onnx(model=model, n_ch=n_ch, n_time=n_time, out_path=Path(out_onnx_path))
        except Exception as e:
            _log(logger, f"[CNN] ONNX export failed: {e}")
            export_ok = False

        return bool(train_ok), bool(export_ok), float(ho_loss), float(ho_acc), float(ho_bal)

    # Use existing trainer on the split (this includes early stopping)
    val_bal, val_acc, train_loss, train_acc, val_loss, model = train_cnn_on_split(
        X_train=X_pair[tr_idx], y_train=y_pair[tr_idx],
        X_val=X_pair[ho_idx],   y_val=y_pair[ho_idx],
        trial_ids_train=trial_ids_pair[tr_idx],
        trial_ids_val=trial_ids_pair[ho_idx],
        n_ch=n_ch, n_time=n_time,
        cfg=cfg,
        max_epochs=int(cfg.MAX_EPOCHS),
        device=device,
        return_model=True,
    )

    export_ok = True
    try:
        export_cnn_onnx(model=model, n_ch=n_ch, n_time=n_time, out_path=Path(out_onnx_path))
    except Exception as e:
        _log(logger, f"[CNN] ONNX export failed: {e}")
        export_ok = False

    return True, bool(export_ok), float(val_loss), float(val_acc), float(val_bal)

def train_final_cnn_and_export(
    *,
    X_pair: np.ndarray,
    y_pair: np.ndarray,
    trial_ids_pair: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    n_ch: int,
    n_time: int,
    out_onnx_path: Path,
    hparam_tuning: str,
    logger: utils.DebugLogger | None = None,
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
    cv_bals:       list[float] = []
    cv_accs:       list[float] = []
    cv_val_losses: list[float] = []

    # 2) Run grid search for hyperparameter tuning if ON
    # we will update this config object to get best 
    if(hparam_tuning == "ON"):
        best_cfg = cfg
        best_score = -1.0
        
        # grid search
        for cand in iter_hparam_candidates(cfg, HPARAM_SPACE):
            score = score_hparam_cfg_cv_cnn(
                cfg=cand,
                X_pair=X_pair,
                y_pair=y_pair,
                trial_ids_pair=trial_ids_pair,
                folds=folds,
                n_ch=n_ch,
                n_time=n_time,
                device=device,
                max_epochs=int(cfg.MAX_EPOCHS_HTUNING),
                logger=logger,
            )

            _log(logger,
                "[HTUNE] "
                f"score={score:.4f} "
                f"F1={cand.F1} D={cand.D} "
                f"k={cand.kernel_length} "
                f"p1={cand.pooling_factor} p2={cand.pooling_factor_final} "
                f"bs={cand.batch_size} lr={cand.learning_rate}"
            )

            if score > best_score:
                best_score = score
                best_cfg = cand

        cfg = best_cfg
        _log(logger, f"[HTUNE] Best cfg: {asdict(cfg)}")
        _log(logger, f"[HTUNE] Best mean CV balanced acc: {float(best_score):.4f}")

    # 3) Always compute CV metrics for the cfg we will actually deploy
    if folds and len(folds) >= 2:
        for fi, (tr_idx, va_idx) in enumerate(folds):
            # per-fold seed for reproducibility
            fold_cfg = replace(cfg, seed=int(cfg.seed + 1000 * fi))

            va_bal, va_acc, _, _, va_loss = train_cnn_on_split(
                X_train=X_pair[tr_idx], y_train=y_pair[tr_idx],
                X_val=X_pair[va_idx],   y_val=y_pair[va_idx],
                trial_ids_train=trial_ids_pair[tr_idx],
                trial_ids_val=trial_ids_pair[va_idx],
                n_ch=n_ch, n_time=n_time,
                cfg=fold_cfg,
                max_epochs=cfg.MAX_EPOCHS,      # final training can use full schedule
                device=device,
                logger=logger,
            )

            cv_bals.append(float(va_bal))
            cv_accs.append(float(va_acc))
            cv_val_losses.append(float(va_loss))

    # 4) BUILD FINAL MODEL FROM BEST CFGS AND TRAIN with simple holdout, THEN EXPORT
    # We train on ALL X_pair/y_pair except small 10% holdout for early stopping, so the exported model sees maximum calibration windows but doesn't overfit.
    train_ok, export_ok, ho_loss, ho_acc, ho_bal = train_single_final_model_with_holdout_and_export(
        X_pair=X_pair,
        y_pair=y_pair,
        trial_ids_pair=trial_ids_pair,
        n_ch=n_ch,
        n_time=n_time,
        cfg=cfg,
        out_onnx_path=Path(out_onnx_path),
        device=device,
        holdout_frac=0.1,
    )

    # 4) Return FINAL RESULT OBJECT
    # If folds weren't provided, we still trained/exported, but metrics will be zeros.
    cv_ok = bool(len(cv_bals) > 0)
    return utils.FinalTrainResults(
        train_ok=bool(train_ok),
        onnx_export_ok=bool(export_ok),

        cv_ok=cv_ok,
        cv_mean_bal_acc=utils.mean(cv_bals),
        cv_std_bal_acc=utils.std(cv_bals),
        cv_mean_acc=utils.mean(cv_accs),
        cv_std_acc=utils.std(cv_accs),
        cv_mean_val_loss=utils.mean(cv_val_losses),
        cv_std_val_loss=utils.std(cv_val_losses),

        final_holdout_ok=True,
        final_holdout_loss=float(ho_loss),
        final_holdout_acc=float(ho_acc),
        final_holdout_bal_acc=float(ho_bal),

        cfg=asdict(cfg),
    )

def train_cnn_on_split(
    *,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray,   y_val: np.ndarray,
    trial_ids_train: np.ndarray | None = None,
    trial_ids_val: np.ndarray | None = None,
    n_ch: int, n_time: int,
    cfg: CNNTrainConfig,
    max_epochs: int,
    device: torch.device,
    return_model: bool = False, # need model when we do onnx export at the end (once only)
    logger: utils.DebugLogger | None = None,
) -> tuple[float, float, float, float, float] | tuple[float, float, float, float, float, EEGNet]:
    """
    Trains CNN on a specific split and returns:
     (val_bal_acc, val_acc, train_loss, train_acc, val_loss)
    This avoids random_split so we can do proper CV.
    - Uses cfg for ALL hyperparams (F1/kernel/pooling/batch/lr/patience/etc).
    """
    # Make training more repeatable across runs
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))

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
    if trial_ids_train is not None:
        batches = make_stratified_trial_batches_new(
            trial_ids=trial_ids_train,
            y=y_train,
            max_windows_per_batch=int(cfg.batch_size),
            seed=int(cfg.seed),
            logger=logger,
        )
        train_loader = DataLoader(train_ds, batch_sampler=ListBatchSampler(batches))
        # In train_cnn_on_split(), after creating batches:
        _log(logger,f"\n[BATCH VALIDATION]")
        for i, batch_idx in enumerate(batches):
            batch_labels = y_train[batch_idx]
            n0 = int((batch_labels == 0).sum())
            n1 = int((batch_labels == 1).sum())

            if n0 == 0 or n1 == 0:
                raise RuntimeError(f"Batch {i} is unbalanced: class0={n0}, class1={n1}")

            _log(logger,f"  Batch {i}: size={len(batch_idx)}, class0={n0}, class1={n1}")

        _log(logger,f"All {len(batches)} batches are balanced! ✓\n")
    else:
        # fallback: window batching (less good for strongly overlapping ssvep windows...)
        g = torch.Generator().manual_seed(int(cfg.seed))
        train_loader = DataLoader(
            train_ds,
            batch_size=min(int(cfg.batch_size), len(train_ds)) if len(train_ds) > 0 else int(cfg.batch_size),
            shuffle=True,
            generator=g
        )

    # val loader uses window batching
    val_loader = DataLoader(
        val_ds,
        batch_size=min(int(cfg.batch_size), len(val_ds)) if len(val_ds) > 0 else int(cfg.batch_size),
        shuffle=False,
    )

    _log(logger, f"[SPLIT] train windows={len(train_ds)} c0={(y_train==0).sum()} c1={(y_train==1).sum()}")
    _log(logger, f"[SPLIT] val   windows={len(val_ds)}   c0={(y_val==0).sum()} c1={(y_val==1).sum()}")

    # build model using cfg
    F2 = _derived_F2(cfg)
    model = EEGNet(
        n_ch=n_ch, n_time=n_time, n_classes=2,
        F1=int(cfg.F1),
        D=int(cfg.D),
        F2=int(F2),
        kernel_length=int(cfg.kernel_length),
        pooling_factor=int(cfg.pooling_factor),
        pooling_factor_final=int(cfg.pooling_factor_final),
        dropout=float(cfg.DROPOUT),
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.learning_rate))

    run_training_to_convergence(
        model,
        train_loader,
        val_loader,
        device=device,
        optimizer=optimizer,
        criterion=criterion,
        max_epochs=int(max_epochs), # depends on htuning vs pairwise comps vs final model
        patience=int(cfg.patience),
        min_delta=float(cfg.min_delta),
        logger=logger,
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

    if return_model:
        return (
            float(val_bal),
            float(val_acc),
            float(train_loss),
            float(train_acc),
            float(val_loss),
            model,
        )

    return (
        float(val_bal),
        float(val_acc),
        float(train_loss),
        float(train_acc),
        float(val_loss),
    )

def score_hparam_cfg_cv_cnn(
    *,
    cfg: CNNTrainConfig,
    X_pair: np.ndarray,
    y_pair: np.ndarray,
    trial_ids_pair: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    n_ch: int,
    n_time: int,
    device: torch.device,
    max_epochs: int,
    logger: utils.DebugLogger | None = None,
) -> float:
    """
    Returns mean CV balanced accuracy for this hparam cfg.
    Uses per-fold seed offsets for repeatability.
    """
    bals: list[float] = []
    for fi, (tr_idx, va_idx) in enumerate(folds):
        fold_cfg = replace(cfg, seed=int(cfg.seed + 1000 * fi))
        val_bal, _, _, _, _ = train_cnn_on_split(
            X_train=X_pair[tr_idx], y_train=y_pair[tr_idx],
            X_val=X_pair[va_idx],   y_val=y_pair[va_idx],
            trial_ids_train=trial_ids_pair[tr_idx],
            trial_ids_val=trial_ids_pair[va_idx],
            n_ch=n_ch, n_time=n_time,
            cfg=fold_cfg,
            max_epochs=int(max_epochs),
            device=device,
            logger=logger,
        )
        bals.append(float(val_bal))
    return float(np.mean(bals)) if bals else 0.0

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
    logger: utils.DebugLogger | None = None
) -> utils.ModelMetrics:
    """
    Cross-val scoring for ONE (already-binary) pair.
    Folds are built in train_ssvep.py (shared across archs).
    Returns ModelMetrics (shared contract).
    """
    cfg = CNNTrainConfig() # use default configs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bals: list[float] = []
    accs: list[float] = []

    for fi, (tr_idx, va_idx) in enumerate(folds):
        # per-fold seed so folds are repeatable but not identical
        fold_cfg = replace(cfg, seed=int(cfg.seed + 1000 * fi))

        val_bal, val_acc, _, _, _ = train_cnn_on_split(
            X_train=X_pair[tr_idx], y_train=y_pair[tr_idx],
            X_val=X_pair[va_idx],   y_val=y_pair[va_idx],
            trial_ids_train=trial_ids_pair[tr_idx],
            trial_ids_val=trial_ids_pair[va_idx],
            n_ch=n_ch, n_time=n_time,
            cfg=fold_cfg,
            max_epochs=int(cfg.MAX_EPOCHS_CV),
            device=device,
            logger=logger,
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