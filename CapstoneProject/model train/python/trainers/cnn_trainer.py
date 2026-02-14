# trainers/cnn_trainer.py

from __future__ import annotations
from dataclasses import dataclass, replace, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, BatchSampler
from typing import Iterator, List, Any, Set
from itertools import product
from collections import deque, defaultdict

import utils.utils as utils

# TODO: option on UI to tune (longer) or use defaults (faster)?
# TODO: plotting training ?? if time 
# TODO: Display hparam plot on ui

# ===================== DEBUG HELPERS =====================
def _log(logger, *parts) -> None:
    msg = " ".join(str(p) for p in parts)
    if logger is None:
        print(msg)
    else:
        logger.log(msg)

def eval_pred_stats(model, loader, *, device) -> dict[str, Any]:
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

    # 3-class confusion matrix (rows = true, cols = pred)
    n = int(len(y_true))
    conf = np.zeros((3,3), dtype=np.int64)
    for t,p in zip(y_true.tolist(), y_pred.tolist()): # iterate over each combo of true,pred using zip
        if 0 <= t <= 2 and 0 <= p <= 2: # guard allowed classes
            conf[t][p] += 1 # count

    pred0 = int((y_pred == 0).sum())
    pred1 = int((y_pred == 1).sum())
    pred2 = int((y_pred == 2).sum())

    return {"n": n, "pred0": pred0, "pred1": pred1, "pred2": pred2, "conf": conf.tolist()}

# CNN Hyperparameters & Configs DataClass
# Capitals indicate const variables that should never be mutated in this architecture.
@dataclass(frozen=True)
class CNNTrainConfig:
    # Optimization & Convergence Stability
    MAX_EPOCHS: int = 400            # number of epochs for final model training
    MAX_EPOCHS_CV: int = 300         # number of epochs while running all the pairwise models for comparison
    MAX_EPOCHS_HTUNING: int = 250    # number of epochs while running all the candidate grids for hparam tuning
    batch_size: int = 18             # how many (mostly indep, non-overlapping due to batching strategy) training windows the CNN sees at once before updating its weights (1 optimizer step) -> keep batches small for overlapping EEG windows
    learning_rate: float = 1e-3      # magnitude of gradient descent steps. smaller batches require smaller LR. (1e-3 is adam optimizer default)
    weight_decay: float = 1e-5       # regularization constant in Adam that penalizes large model weights to reduce overfitting
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
    
    dropout: float = 0.5             # [regularization trick] "turns off" a fraction of activations to reduce overfitting (not rely too much on any single pathway), greater = greater overfitting resistance

    val_batch_mode: str = "FULL"     # how validation windows are batched for early-stopping stability ("FULL" = all at once, "FIXED" = constant-size batches)
    val_batch_size: int = 27         # used only if FIXED (kept larger than training batches for stable loss estimates)

    norm_mode: str = "BatchNorm"     # normalization type ("GroupNorm" = batch-size independent, stable for small/variable batches; "BatchNorm" = slightly better with large, consistent batches but unstable for small ones)

# Hyperparameter tuning space [if hparam tuning on, for final model training only] 
HPARAM_SPACE = {
    "kernel_length": [63, 125, 187],  # 250 ms, 500 ms, or 750 ms temporal segments as layer 2 inputs
    "pooling_factor": [3, 4, 5],      # layer 2 temporal downsampling 
    "pooling_factor_final": [8, 10],  # final layer temporal downsampling
    "F1": [8, 16],                    # number of output temporal combinations from layer 1
    "learning_rate": [1e-3, 5e-4],
    "batch_size": [12, 15, 18],        # should be multiples of 3 for class balance per batch
    "dropout": [0.3, 0.5],
    "norm_mode": ["GroupNorm", "BatchNorm"],
    "patience": [20, 25],
    # total combos = 3 x 3 x 2 x 2 x 2 x 3 x 2 x 2 x 2 x 2 = 1152
}

# more minimal hparam tuning with params that empirically had the greatest impact
HPARAM_SPACE_MINIMAL = {
    "kernel_length": [125, 187],  # 250 ms, 500 ms, or 750 ms temporal segments as layer 2 inputs
    "pooling_factor": [4, 5],      # layer 2 temporal downsampling 
    "pooling_factor_final": [8, 10],  # final layer temporal downsampling
    "F1": [8, 16],                    # number of output temporal combinations from layer 1
    "dropout": [0.3, 0.5],
    "batch_size": [15, 18],  
    # total combos = 2 x 2 x 2 x 2 x 2 x 2 = 64
}

def _derived_F2(cfg: CNNTrainConfig) -> int:
    # F2 is fixed in this arch by F1 and D
    # [compressed ftr set for classifier] number of output channels from the 3rd layer (inputs to final layer), representing spatiotemporal mixed info blocks (for the 500ms segments)
    return int(cfg.F1 * cfg.D)

def iter_hparam_candidates(
    base_cfg: CNNTrainConfig,
    space: dict[str,list],
) -> Iterator[CNNTrainConfig]:
    """
    YIELDS CNNTrainConfig objects for every combination in the hparam grid (space) dict[str,list]
    """
    for patch in (
        dict(zip(space.keys(), values)) # (1) zip assigns each prod (combination) we obtained in step 2 to the initial keys in order, so like [a,c] gets mapped back to relevant keys ("lr",a), ("kernel",b); then it becomes dict {"lr":a, "kernel":b} format
        for values in product(*space.values()) # (2) takes values of dict, so the hparam lists here, [a,b..] and [c,d..] then makes products [a,c], [a,d], [b,c], [b,d] so every possible combination...
    ):
        yield replace(base_cfg, **patch) # (3) replace creates a copy of base_cfg with only the specified fields in patch changed and yield turns a fxn into an iterator that produces ('yields') values one at a time instead of all at once

def pick_gn_groups(num_channels: int, preferred: int = 8) -> int:
    """
    Pick a GroupNorm group count that divides num_channels.
    Tries preferred -> 4 -> 2 -> 1.
    """
    for g in (preferred, 4, 2, 1):
        if num_channels % g == 0:
            return g
    return 1

# EEGNET (CNN) MODEL DEFINITION
# A layer is a block in the NN
# A kernel is a set of learned weights inside a CONVOLUTIONAL layer specifically
# EEGNet has a small number of layers, but each convolutional layer contains many kernels
# - for each layer, the number of kernels = the number of out_channels
#   (each kernel maps to one feature)
# CURRENT ARCHITECTURE
#   - 4 layer CNN Input: (B, 1, C, T) Output logits: (B, K)
#   - (logits are the unnormalized class scores before any softmax)
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
                 F1: int,
                 D: int,
                 F2: int,
                 kernel_length: int,
                 pooling_factor: int,
                 pooling_factor_final: int,
                 dropout: float,
                 norm_mode: str):
        super().__init__() # init nn.Module parent class, then added stuff
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
        if norm_mode == "GroupNorm":
            self.bn1 = nn.GroupNorm(pick_gn_groups(F1), F1)
        else:
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
        if norm_mode == "GroupNorm":
            self.bn2 = nn.GroupNorm(pick_gn_groups(F1 * D), F1 * D)
        else:
            self.bn2 = nn.BatchNorm2d(F1 * D)          # batch normalization
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
        if norm_mode == "GroupNorm":
            self.bn3 = nn.GroupNorm(pick_gn_groups(F2), F2)
        else:
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

def get_trials_per_class(
    trial_ids: np.ndarray,
    y: np.ndarray,
) -> tuple[dict[int, list[int]], dict[int, list[int]]]: 
    uniq_trials = np.unique(trial_ids)
    trials_by_class: dict[int, list[int]] = {0: [], 1: [], 2: []}
    trial_to_indices: dict[int, np.ndarray] = {}
    for tid in uniq_trials:
        mask = (trial_ids == tid)
        idx = np.where(mask)[0].astype(np.int64)
        if idx.size == 0:
            continue
        curr_trial_class = y[idx][0] # label purity guaranteed per trial
        trials_by_class[curr_trial_class].append(int(tid))
        trial_to_indices[int(tid)] = idx
    return trials_by_class, trial_to_indices

def make_stratified_trial_batches(
    trial_ids: np.ndarray,
    y: np.ndarray,
    *,
    max_windows_per_batch: int,
    hz_a: int,
    hz_b: int,
    seed: int = 0,
    logger: utils.DebugLogger | None = None,
) -> tuple[list[list[int]], list[utils.TrainIssue]]:
    """
    Balanced, low-correlation batching for overlapping SSVEP windows.
    Note the main anti-leakage mechanism is the groups implemented in train_ssvep.py.

    Strategy:
    - Within each trial pair, take up to max_windows_per_batch//3 per class.
    - Every returned batch contains BOTH classes + REST.
    
    This is optimal for heavily overlapping windows because:
    - Maximizes trial diversity per batch
    - Reduces within-batch correlation
    - Forces model to generalize across trials
    
    Args:
        trial_ids: Trial ID for each window
        y: labels (0,1,2) for each window
        max_windows_per_batch: Cap on batch size
        seed: Random seed for reproducibility
    
    Returns:
        List of index lists (one per batch)
    """
    trial_queues: dict[int, deque[int]] = {} # working window indices for a given trial
    batch_issues: list[utils.TrainIssue] = []
    rng = np.random.default_rng(seed)
    trial_ids = np.asarray(trial_ids).astype(np.int64)
    y = np.asarray(y).astype(np.int64)

    if max_windows_per_batch < 3:
        utils.abort(
            "CNN_BATCH",
            "max_windows_per_batch must be >= 3",
            {"max_windows_per_batch": int(max_windows_per_batch)},
        )
    if max_windows_per_batch % 3 != 0:
        utils.abort(
            "CNN_BATCH",
            "max_windows_per_batch must be divisible by 3 for even 3-class batching",
            {"max_windows_per_batch": int(max_windows_per_batch)},
        )
    
    windows_per_class = int(max_windows_per_batch // 3)
    trials_by_class, trial_to_indices = get_trials_per_class(trial_ids=trial_ids,y=y)
    uniq_trials = np.unique(trial_ids)

    # small batch: 2/3 of full batch (still balanced)
    small_batch_size = int((2 * max_windows_per_batch) // 3)
    windows_per_class_small_batch = small_batch_size // 3
    # guard against too small 
    if(windows_per_class_small_batch == 0):
        small_batch_size = 3
        windows_per_class_small_batch = 1

    # check minimum number of windows per batch
    N0 = int(np.sum(y==0))
    N1 = int(np.sum(y==1))
    N2 = int(np.sum(y==2))
    min_windows_needed_per_class = windows_per_class*utils.MIN_NUM_BATCHES_PER_CLASS
    for cls, count in [(0, N0), (1, N1), (2, N2)]:
        if count < min_windows_needed_per_class:
            batch_issues.append(utils.issue(
                "CNN_BATCH",
                f"Class {cls} has insufficient windows to create MIN_NUM_BATCHES_PER_CLASS batches",
                {
                    "class": cls,
                    "n_windows": count,
                    "min_batches_required": utils.MIN_NUM_BATCHES_PER_CLASS,
                    "windows_per_batch": max_windows_per_batch,
                    "min_windows_needed_per_class": min_windows_needed_per_class,
                },
                data_insufficiency={
                    "frequency_hz": -1 if cls == 2 else (hz_a if cls == 0 else hz_b),
                    "metric": "windows",
                    "required": min_windows_needed_per_class,
                    "actual": count,
                }
            ))

    # Must have at least two trials per class
    for c in [0,1,2]:
        if len(trials_by_class[c]) < utils.MIN_TRIALS_PER_CLASS_FOR_BATCHING:
            utils.abort(
                "CNN_BATCH",
                "Cannot create stratified batches: Not enough trials per class",
                {
                    "n_trials_class0": int(len(trials_by_class[0])),
                    "n_trials_class1": int(len(trials_by_class[1])),
                    "n_trials_class2": int(len(trials_by_class[2])),
                    "uniq_trials_total": int(len(uniq_trials)),
                    "max_windows_per_batch": int(max_windows_per_batch),
                },
                data_insufficiency={
                        "frequency_hz": -1 if cls == 2 else (hz_a if cls == 0 else hz_b),
                        "metric": "trials",
                        "required": utils.MIN_TRIALS_PER_CLASS_FOR_BATCHING,
                        "actual": len(trials_by_class[c]),
                }
            )
    
    # Per-trial work queues of unused window indices
    for trial in uniq_trials:
        rng.shuffle(trial_to_indices[int(trial)]) # shuffle indices for given trial
        trial_queues[int(trial)] = deque(trial_to_indices[int(trial)].tolist())
        # now every trial_queues[trial] has a deque containing global indices where that trial is located, in random order

    # Shuffle trial order within each class (deterministic given seed)
    rng.shuffle(trials_by_class[0])
    rng.shuffle(trials_by_class[1])
    rng.shuffle(trials_by_class[2])

    # for each trial, split into (1/3) batches
    small_batch_count = 0
    batches: list[list[int]] = []
    minibatches_per_class: dict[int, list[int]] = {0:[], 1:[], 2:[]}
    # use a set for exhausted trials because we do not want to allow duplicate values (appending the same trial everytime we try & exhaust it...) 
    exhausted_trials_per_class: dict[int, Set[int]] = {0: set(), 1: set(), 2: set()}
    example_ctrs: dict[int, int] = {0:0, 1:0, 2:0}
    trial_ctrs: dict[int, int] = {0:0, 1:0, 2:0}

    # Safety Check 1: Max outer iters to prevent inf looping
    MAX_OUTER_ITERS = 10000
    outer_iteration = 0

    while True:
        outer_iteration += 1
        if outer_iteration > MAX_OUTER_ITERS:
            if logger:
                logger.log(f"[CNN_BATCH] WARNING: Hit max iterations ({MAX_OUTER_ITERS}), stopping batch creation")
            break
        # (1) select a random trial for each class by going through trials_by_class (alr random shuffled)
        # (2) pop from that trial's deque if its not empty & append example to batch
        # (3) if queue is empty, try another trial
        # (4) if all trial queues are empty (exhausted), break
        for cidx in (0,1,2):
            # Safety Check 2: Max inner iters per class
            MAX_INNER_ITERS = 1000
            inner_iter = 0

            while(example_ctrs[cidx] < windows_per_class):
                inner_iter += 1
                if inner_iter > MAX_INNER_ITERS:
                    if logger:
                        logger.log(f"[CNN_BATCH] WARNING: Inner loop for class {cidx} hit max iterations")
                    break
                # if we've exhausted all trials -> exit, this is the most we can get for this minibatch
                if len(exhausted_trials_per_class[cidx]) >= len(trials_by_class[cidx]):
                    break
                
                # iterate through trials_by_class 1 by 1
                trial_idx = trial_ctrs[cidx]
                trial_ctrs[cidx] = trial_ctrs[cidx]+1 # for next time
                
                if(trial_idx >= len(trials_by_class[cidx])):
                    # cycle (we've gone through all trials once)
                    trial_idx = 0
                    trial_ctrs[cidx] = 0
                # trials_by_class will give [1,4,5] etc trial nums for that class in random order
                trial = trials_by_class[cidx][trial_idx]
                if trial in exhausted_trials_per_class[cidx]:
                    continue  # don't waste iterations on known-empty trials

                # so now we've selected a trial for our class
                # need to pop from this trial's work queue and bring it into batch
                if len(trial_queues[trial]) > 0:
                    winidx = trial_queues[int(trial)].pop()
                    minibatches_per_class[cidx].append(winidx)
                    example_ctrs[cidx] = example_ctrs[cidx]+1
                else: 
                    # this trial's work queue is empty, thus we've exhausted it, try cycling
                    exhausted_trials_per_class[cidx].add(trial)
                    continue
        
        # we made a batch, let's check it and build it
        notEnough = False
        if (len(minibatches_per_class[0]) == windows_per_class and
            len(minibatches_per_class[1]) == windows_per_class and
            len(minibatches_per_class[2]) == windows_per_class):
            batch = minibatches_per_class[0] + minibatches_per_class[1] + minibatches_per_class[2]
            rng.shuffle(batch)
            batches.append(batch)
        else:
            # try making small batches or stop
            if small_batch_count >= utils.MAX_SMALL_BATCHES:
                notEnough = True
            for c in (0,1,2):
                if len(minibatches_per_class[c]) < windows_per_class_small_batch:
                    notEnough = True
            if notEnough:
                break
            else:
                # ensure balance before building (truncate to limiting size)
                k = min(len(minibatches_per_class[0]), len(minibatches_per_class[1]), len(minibatches_per_class[2]))
                minibatches_per_class[0] = minibatches_per_class[0][:k]
                minibatches_per_class[1] = minibatches_per_class[1][:k]
                minibatches_per_class[2] = minibatches_per_class[2][:k]
                batch = minibatches_per_class[0] + minibatches_per_class[1] + minibatches_per_class[2]
                rng.shuffle(batch)
                batches.append(batch)
                small_batch_count = small_batch_count+1

        if notEnough:
            break # exit

        # reset flags for next batch
        example_ctrs[0] = 0
        example_ctrs[1] = 0
        example_ctrs[2] = 0
        minibatches_per_class = {0: [], 1: [], 2: []}

    # Check if we produced the expected number of batches
    # we have N0, N1, N2 total numbers of wins per class across all trials
    # we can maximally put B/3 windows per class per full batch where B = max_batch_size
    # we can additionally have 4 small batches with w_s = (2B/3)/3 windows per class
    
    # number of batches limited by smallest contributing class
    full_batches_possible = min(N0,N1,N2)//windows_per_class

    #leftovers after full batches
    R0 = N0 - full_batches_possible*windows_per_class
    R1 = N1 - full_batches_possible*windows_per_class
    R2 = N2 - full_batches_possible*windows_per_class
    # number of batches per small batch limited by smallest contributing class that remains
    min_num_small_batches = min(R0,R1,R2)//windows_per_class_small_batch
    n_small_possible = min(utils.MAX_SMALL_BATCHES, min_num_small_batches)

    max_num_batches = full_batches_possible + n_small_possible # include possible small batches
    if len(batches) < 0.9*full_batches_possible or len(batches) > max_num_batches:
        batch_issues.append(utils.issue(
            "CNN_BATCH",
            "Fewer OR too many batches produced than expected (possible early-stop or window reuse issue)",
            {
                "expected complete batches possible": int(full_batches_possible),
                "expected maximum batches possible": int(max_num_batches),
                "actual_batches": int(len(batches)),
                "N0": int(N0), "N1": int(N1), "N2": int(N2),
            },
        ))
    if len(batches) == 0:
         utils.abort(
            "CNN_BATCH",
            "Could not create any balanced batches with both classes present",
            {
                "n_batches": int(len(batches)),
                "n_trials_c0": int(len(trials_by_class[0])),
                "n_trials_c1": int(len(trials_by_class[1])),
                "n_trials_c2": int(len(trials_by_class[2])),
                "max_windows_per_batch": int(max_windows_per_batch),
            },
        )

    return batches, batch_issues

def make_pooled_batches(
    trial_ids: np.ndarray,
    y: np.ndarray,
    *,
    max_windows_per_batch: int,
    hz_a: int,
    hz_b: int,
    seed: int = 0,
    logger: utils.DebugLogger | None = None,
) -> tuple[list[list[int]], list[utils.TrainIssue]]:
    """
    Pool-based batching with EXACT batch size behavior as current implementation.
    
    Batch sizes:
    - Full batches: max_windows_per_batch (e.g., 18)
    - Small batches: Balanced to minimum available class
    - Minimum size: 3 (1 per class)
    """
    print("py ENTERED POOLED BATCHING \n")
    batch_issues: list[utils.TrainIssue] = []
    rng = np.random.default_rng(seed)
    trial_ids = np.asarray(trial_ids).astype(np.int64)
    y = np.asarray(y).astype(np.int64)

    # Validation
    if max_windows_per_batch < 3:
        utils.abort("CNN_BATCH", "max_windows_per_batch must be >= 3",
                   {"max_windows_per_batch": int(max_windows_per_batch)})
    if max_windows_per_batch % 3 != 0:
        utils.abort("CNN_BATCH", "max_windows_per_batch must be divisible by 3",
                   {"max_windows_per_batch": int(max_windows_per_batch)})
    
    windows_per_class = int(max_windows_per_batch // 3)
    
    # Small batch minimum
    small_batch_size = int((2 * max_windows_per_batch) // 3)
    windows_per_class_small = small_batch_size // 3
    if windows_per_class_small == 0:
        windows_per_class_small = 1
    
    N0 = int((y == 0).sum())
    N1 = int((y == 1).sum())
    N2 = int((y == 2).sum())
    
    # 1) Create shuffled pools per class
    pools = {0: [], 1: [], 2: []}
    
    for cls in [0, 1, 2]:
        # Group by trial first (for diversity)
        trial_groups = {}
        for idx in np.where(y == cls)[0]:
            tid = int(trial_ids[idx])
            if tid not in trial_groups:
                trial_groups[tid] = []
            trial_groups[tid].append(int(idx))
        
        # Shuffle within each trial
        for tid in trial_groups:
            rng.shuffle(trial_groups[tid])
        
        # Shuffle trial order
        trial_list = list(trial_groups.keys())
        rng.shuffle(trial_list)
        
        # Build pool by concatenating shuffled trials
        for tid in trial_list:
            pools[cls].extend(trial_groups[tid])
    
    # 2) Validation: min trials check
    unique_trials_per_class = {
        0: len(set(trial_ids[y == 0])),
        1: len(set(trial_ids[y == 1])),
        2: len(set(trial_ids[y == 2])),
    }
    
    for cls in [0, 1, 2]:
        if unique_trials_per_class[cls] < utils.MIN_TRIALS_PER_CLASS_FOR_BATCHING:
            utils.abort(
                "CNN_BATCH",
                "Not enough trials per class",
                {
                    "class": cls,
                    "n_trials": unique_trials_per_class[cls],
                    "required": utils.MIN_TRIALS_PER_CLASS_FOR_BATCHING,
                },
                data_insufficiency={
                    "frequency_hz": -1 if cls == 2 else (hz_a if cls == 0 else hz_b),
                    "metric": "trials",
                    "required": utils.MIN_TRIALS_PER_CLASS_FOR_BATCHING,
                    "actual": unique_trials_per_class[cls],
                }
            )
    
    # Check min windows
    min_windows_needed = windows_per_class * utils.MIN_NUM_BATCHES_PER_CLASS
    for cls, count in [(0, N0), (1, N1), (2, N2)]:
        if count < min_windows_needed:
            batch_issues.append(utils.issue(
                "CNN_BATCH",
                f"Class {cls} has insufficient windows",
                {
                    "class": cls,
                    "n_windows": count,
                    "required": min_windows_needed,
                },
                data_insufficiency={
                    "frequency_hz": -1 if cls == 2 else (hz_a if cls == 0 else hz_b),
                    "metric": "windows",
                    "required": min_windows_needed,
                    "actual": count,
                }
            ))
    
    # 3) Fill batches
    batches = []
    pos = {0: 0, 1: 0, 2: 0}
    small_count = 0
    
    while True:
        # Check available windows in each pool
        avail = {
            0: len(pools[0]) - pos[0],
            1: len(pools[1]) - pos[1],
            2: len(pools[2]) - pos[2],
        }
        
        # Try to form a FULL batch (windows_per_class from each class)
        if all(avail[c] >= windows_per_class for c in [0, 1, 2]):
            batch = []
            for cls in [0, 1, 2]:
                batch.extend(pools[cls][pos[cls]:pos[cls] + windows_per_class])
                pos[cls] += windows_per_class
            rng.shuffle(batch)
            batches.append(batch)
            continue  # Try next batch
        
        # Can't form full batch. Try SMALL batch.
        # accept if >= windows_per_class_small for all classes
        if (small_count < utils.MAX_SMALL_BATCHES and
            all(avail[c] >= windows_per_class_small for c in [0, 1, 2])):
            
            # truncate to minimum available
            # we accept variable sizes like 15, 16, etc...
            k = min(avail[0], avail[1], avail[2])
            
            batch = []
            for cls in [0, 1, 2]:
                batch.extend(pools[cls][pos[cls]:pos[cls] + k])
                pos[cls] += k
            
            rng.shuffle(batch)
            batches.append(batch)
            small_count += 1
            continue
        
        # Can't form any more batches
        if logger:
            logger.log(f"[CNN_BATCH] Stopping: avail c0={avail[0]} c1={avail[1]} c2={avail[2]}, "
                      f"need full={windows_per_class} or small>={windows_per_class_small}")
        break
    
    # 4) final Validation/debug
    if len(batches) == 0:
        utils.abort(
            "CNN_BATCH",
            "Could not create any batches",
            {
                "n_trials_c0": unique_trials_per_class[0],
                "n_trials_c1": unique_trials_per_class[1],
                "n_trials_c2": unique_trials_per_class[2],
            },
        )
    
    # Expected batch count check (same as current)
    full_batches_possible = min(N0, N1, N2) // windows_per_class
    R0 = N0 - full_batches_possible * windows_per_class
    R1 = N1 - full_batches_possible * windows_per_class
    R2 = N2 - full_batches_possible * windows_per_class
    small_possible = min(R0, R1, R2) // windows_per_class_small
    max_total = full_batches_possible + min(utils.MAX_SMALL_BATCHES, small_possible)
    
    if len(batches) < 0.9 * full_batches_possible or len(batches) > max_total:
        batch_issues.append(utils.issue(
            "CNN_BATCH",
            "Batch count outside expected range",
            {
                "expected_min": int(0.9 * full_batches_possible),
                "expected_max": int(max_total),
                "actual": len(batches),
            },
        ))
    
    if logger:
        # Report batch size statistics (for verification)
        batch_sizes = [len(b) for b in batches]
        logger.log(f"[CNN_BATCH] Created {len(batches)} batches: "
                  f"sizes min={min(batch_sizes)} max={max(batch_sizes)} mean={np.mean(batch_sizes):.1f}")
        
        # Report class balance in first few batches
        for i, batch in enumerate(batches[:3]):
            c0 = sum(1 for idx in batch if y[idx] == 0)
            c1 = sum(1 for idx in batch if y[idx] == 1)
            c2 = sum(1 for idx in batch if y[idx] == 2)
            logger.log(f"[CNN_BATCH] Batch {i}: size={len(batch)} c0={c0} c1={c1} c2={c2}")
    
    return batches, batch_issues

def debug_all_batches(loader, *, logger):
    """
    Reports any imbalanced batches and returns issues.
    """
    tag="BATCH"
    _log(logger, f"[{tag}] Inspecting ALL batches...")
    
    batch_issues = []
    
    batch_stats = []
    for batch_i, (xb, yb_batch) in enumerate(loader):
        c0 = int((yb_batch == 0).sum())
        c1 = int((yb_batch == 1).sum())
        c2 = int((yb_batch == 2).sum())
        total = len(yb_batch)
        
        batch_stats.append({
            "batch": batch_i,
            "size": total,
            "c0": c0,
            "c1": c1,
            "c2": c2,
        })
        
        # Check batch quality
        if total > 0:
            frac_c0 = c0 / total
            frac_c1 = c1 / total
            frac_c2 = c2 / total
            
            # CHECK: MIN_FRAC_PER_CLASS_IN_BATCH for c0
            if frac_c0 < utils.MIN_FRAC_PER_CLASS_IN_BATCH:
                batch_issues.append(utils.issue(
                    "CNN_BATCH",
                    f"Batch {batch_i}: class 0 fraction below minimum",
                    {"batch": batch_i, "fraction": float(frac_c0), "min": utils.MIN_FRAC_PER_CLASS_IN_BATCH},
                    data_insufficiency={
                        "frequency_hz": None,
                        "metric": "windows",
                        "required": int(total * utils.MIN_FRAC_PER_CLASS_IN_BATCH),
                        "actual": c0,
                    }
                ))
                _log(logger,f"[{tag}] Batch {batch_i}: c0_frac={frac_c0:.2f} < {utils.MIN_FRAC_PER_CLASS_IN_BATCH}")
            
            # CHECK: MIN_FRAC_PER_CLASS_IN_BATCH for c1
            if frac_c1 < utils.MIN_FRAC_PER_CLASS_IN_BATCH:
                batch_issues.append(utils.issue(
                    "CNN_BATCH",
                    f"Batch {batch_i}: class 1 fraction below minimum",
                    {"batch": batch_i, "fraction": float(frac_c1), "min": utils.MIN_FRAC_PER_CLASS_IN_BATCH},
                    data_insufficiency={
                        "frequency_hz": None,
                        "metric": "windows",
                        "required": int(total * utils.MIN_FRAC_PER_CLASS_IN_BATCH),
                        "actual": c1,
                    }
                ))
                _log(logger,f"[{tag}] Batch {batch_i}: c1_frac={frac_c1:.2f} < {utils.MIN_FRAC_PER_CLASS_IN_BATCH}")
            
            # CHECK: MAX_REST_FRAC_IN_BATCH
            if frac_c2 > utils.MAX_REST_FRAC_IN_BATCH:
                batch_issues.append(utils.issue(
                    "CNN_BATCH",
                    f"Batch {batch_i}: REST fraction exceeds maximum",
                    {"batch": batch_i, "fraction": float(frac_c2), "max": utils.MAX_REST_FRAC_IN_BATCH},
                    data_insufficiency={
                        "frequency_hz": -1,
                        "metric": "windows",
                        "required": int(total * utils.MAX_REST_FRAC_IN_BATCH),
                        "actual": c2,
                    }
                ))
                _log(logger,f"[{tag}] Batch {batch_i}: rest_frac={frac_c2:.2f} > {utils.MAX_REST_FRAC_IN_BATCH}")
            
            # CHECK: MAX_CLASS_IMBALANCE_IN_BATCH
            if c0 + c1 > 0:
                imbalance = abs(c0 - c1) / (c0 + c1)
                if imbalance > utils.MAX_CLASS_IMBALANCE_IN_BATCH:
                    batch_issues.append(utils.issue(
                        "CNN_BATCH",
                        f"Batch {batch_i}: class imbalance exceeds maximum",
                        {"batch": batch_i, "imbalance": float(imbalance), "max": utils.MAX_CLASS_IMBALANCE_IN_BATCH},
                        data_insufficiency={
                            "frequency_hz": None,
                            "metric": "windows",
                            "required": max(c0, c1),
                            "actual": min(c0, c1),
                        }
                    ))
                    _log(logger,f"[{tag}] Batch {batch_i}: imbalance={imbalance:.2f} > {utils.MAX_CLASS_IMBALANCE_IN_BATCH}")
    
    # Summary statistics
    total_batches = len(batch_stats)
    if total_batches == 0:
        _log(logger,f"[{tag}] No batches to inspect")
        return batch_issues
    
    sizes = [b["size"] for b in batch_stats]
    c0_counts = [b["c0"] for b in batch_stats]
    c1_counts = [b["c1"] for b in batch_stats]
    c2_counts = [b["c2"] for b in batch_stats]
    
    _log(logger,f"[{tag}] Total batches: {total_batches}")
    _log(logger,f"[{tag}] Batch sizes: min={min(sizes)} max={max(sizes)} mean={np.mean(sizes):.1f}")
    _log(logger,f"[{tag}] Class 0: min={min(c0_counts)} max={max(c0_counts)} mean={np.mean(c0_counts):.1f}")
    _log(logger,f"[{tag}] Class 1: min={min(c1_counts)} max={max(c1_counts)} mean={np.mean(c1_counts):.1f}")
    _log(logger,f"[{tag}] Class 2: min={min(c2_counts)} max={max(c2_counts)} mean={np.mean(c2_counts):.1f}")
    
    return batch_issues

def make_trial_holdout_split(
    trial_ids: np.ndarray,
    y: np.ndarray,
    *,
    holdout_frac: float = 0.15,
    seed: int = 0,
    logger: utils.DebugLogger | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stratified trial-wise holdout split.
    - This is for FINAL MODEL EVALUATION ON SIMPLE HOLDOUT.
    - Ensure both classes appear in holdout.
    - Returns (train_idx, holdout_idx)
    """
    trial_ids = np.asarray(trial_ids).astype(np.int64)
    y = np.asarray(y).astype(np.int64)
    # Build (trial_id, label) -> window indices mapping
    trial_label_to_indices = defaultdict(list)
    for idx in range(len(trial_ids)):
        key = (int(trial_ids[idx]), int(y[idx]))
        trial_label_to_indices[key].append(idx)
    
    # Count windows per class
    n_c0 = int((y == 0).sum())
    n_c1 = int((y == 1).sum())
    n_c2 = int((y == 2).sum())
    
    trial_groups_per_class = {0: [], 1: [], 2: []}
    for (tid, label), indices in trial_label_to_indices.items():
        trial_groups_per_class[label].append({
            'tid': tid,
            'label': label,
            'indices': indices,
            'size': len(indices)
        })
    
    for cls in [0, 1, 2]:
        if len(trial_groups_per_class[cls]) < utils.MIN_TRIALS_PER_CLASS_FOR_HOLDOUT:
            if logger:
                logger.log(f"[HOLDOUT] Class {cls} has only {len(trial_groups_per_class[cls])} trial-groups, need {utils.MIN_TRIALS_PER_CLASS_FOR_HOLDOUT}")
            return np.arange(len(trial_ids), dtype=np.int64), np.array([], dtype=np.int64)
    
    # ADAPTIVE TARGETS: Ensure minimum viable batch creation
    MIN_WINDOWS_PER_CLASS = 18  # Need for batch_size=18 (6 per class × 3 batches)

    # Compute required fraction for each class
    required_fracs = []
    for cls, n_cls in enumerate([n_c0, n_c1, n_c2]):
        if n_cls > 0:
            req_frac = MIN_WINDOWS_PER_CLASS / n_cls
            required_fracs.append(req_frac)
            if logger:
                logger.log(f"[HOLDOUT] Class {cls}: N={n_cls}, need={MIN_WINDOWS_PER_CLASS}, req_frac={req_frac:.3f}")

    # Use maximum required fraction (most constraining class)
    if required_fracs:
        adaptive_frac = max(required_fracs)
        adaptive_frac = min(adaptive_frac, 0.5)  # Cap at 50%

        if adaptive_frac > holdout_frac:
            if logger:
                logger.log(f"[HOLDOUT] Adjusting: {holdout_frac:.2f} → {adaptive_frac:.2f}")
            holdout_frac = adaptive_frac

    # Now compute targets with adaptive fraction
    target_c0 = max(MIN_WINDOWS_PER_CLASS, int(np.ceil(n_c0 * holdout_frac)))
    target_c1 = max(MIN_WINDOWS_PER_CLASS, int(np.ceil(n_c1 * holdout_frac)))
    target_c2 = max(MIN_WINDOWS_PER_CLASS, int(np.ceil(n_c2 * holdout_frac)))

    if logger:
        logger.log(f"[HOLDOUT] Targets: c0={target_c0}/{n_c0}, c1={target_c1}/{n_c1}, c2={target_c2}/{n_c2}")

    rng = np.random.default_rng(seed)
    
    # Shuffle trial-groups per class
    for cls in [0, 1, 2]:
        rng.shuffle(trial_groups_per_class[cls])
    
    # use round-robin selection to prevent large trials from dominating holdout split
    holdout_indices = []
    selected_counts = {0: 0, 1: 0, 2: 0}
    targets = {0: target_c0, 1: target_c1, 2: target_c2}
    # Track position in each class's trial list
    positions = {0: 0, 1: 0, 2: 0}
    
    # Round-robin: cycle through classes, take one trial from each
    max_iterations = 1000  # Safety limit
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        made_progress = False
        
        for cls in [0, 1, 2]:
            # Skip if we've hit target for this class
            if selected_counts[cls] >= targets[cls]:
                continue
            
            # Skip if we've exhausted trials for this class
            if positions[cls] >= len(trial_groups_per_class[cls]):
                continue
            
            # Get next trial for this class
            trial_group = trial_groups_per_class[cls][positions[cls]]
            positions[cls] += 1
            
            # Add to holdout
            holdout_indices.extend(trial_group['indices'])
            selected_counts[cls] += trial_group['size']
            made_progress = True
        
        # Stop if no class made progress (all exhausted or hit targets)
        if not made_progress:
            break
    
    # Create masks
    holdout_indices = np.array(sorted(holdout_indices), dtype=np.int64)
    all_indices = np.arange(len(trial_ids), dtype=np.int64)
    holdout_mask = np.zeros(len(trial_ids), dtype=bool)
    holdout_mask[holdout_indices] = True
    
    train_idx = all_indices[~holdout_mask]
    holdout_idx = all_indices[holdout_mask]
    
    # Validate both splits have all classes
    train_c0 = int((y[train_idx] == 0).sum())
    train_c1 = int((y[train_idx] == 1).sum())
    train_c2 = int((y[train_idx] == 2).sum())
    
    hold_c0 = int((y[holdout_idx] == 0).sum())
    hold_c1 = int((y[holdout_idx] == 1).sum())
    hold_c2 = int((y[holdout_idx] == 2).sum())
    
    if logger:
        logger.log(f"[HOLDOUT] train: N={len(train_idx)} c0={train_c0} c1={train_c1} c2={train_c2}")
        logger.log(f"[HOLDOUT] hold:  N={len(holdout_idx)} c0={hold_c0} c1={hold_c1} c2={hold_c2}")
    
    # Check for missing classes
    if train_c0 == 0 or train_c1 == 0 or train_c2 == 0:
        if logger:
            logger.log("[HOLDOUT] Train missing a class, falling back to no-holdout")
        return np.arange(len(trial_ids), dtype=np.int64), np.array([], dtype=np.int64)
    
    if hold_c0 == 0 or hold_c1 == 0 or hold_c2 == 0:
        if logger:
            logger.log("[HOLDOUT] Holdout missing a class, falling back to no-holdout")
        return np.arange(len(trial_ids), dtype=np.int64), np.array([], dtype=np.int64)
    
    return train_idx, holdout_idx

def make_trial_holdout_split(
    trial_ids: np.ndarray,
    y: np.ndarray,
    *,
    holdout_frac: float = 0.15,
    seed: int = 0,
    logger: utils.DebugLogger | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stratified TRIAL-wise holdout split (3-class).
    Intended behavior:
      - Holdout is ~holdout_frac of TOTAL windows (not per-class), chosen by selecting WHOLE trials.
      - Stratified by trial label (one label per trial).
      - Uses round-robin across classes to avoid one class/trial dominating.
      - Never splits a trial across train/holdout.
      - Guarantees each class appears in holdout + train (if feasible), else falls back to no-holdout.
    Returns:
      (train_idx, holdout_idx)  (both are arrays of window indices)
    """
    trial_ids = np.asarray(trial_ids).astype(np.int64)
    y = np.asarray(y).astype(np.int64)

    n_total = int(len(trial_ids))
    if n_total == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    # ----------------------------
    # 1) Build trial -> indices mapping (DO NOT key by (trial,label))
    # ----------------------------
    trial_to_indices: dict[int, list[int]] = defaultdict(list)
    for i, tid in enumerate(trial_ids.tolist()):
        trial_to_indices[int(tid)].append(int(i))

    uniq_trials = sorted(trial_to_indices.keys())

    # ----------------------------
    # 2) Infer ONE label per trial; if not label-pure, we cannot safely do trial-wise stratification
    # ----------------------------
    trial_to_label: dict[int, int] = {}
    non_pure_trials: list[int] = []

    for tid in uniq_trials:
        idxs = np.asarray(trial_to_indices[tid], dtype=np.int64)
        labels = np.unique(y[idxs])
        if labels.size != 1:
            non_pure_trials.append(tid)
            continue
        trial_to_label[tid] = int(labels[0])

    if non_pure_trials:
        # This is a real bug upstream (trial id assignment / label purity invariant).
        # Safer to fall back than silently create nonsense splits.
        if logger:
            logger.log(
                f"[HOLDOUT] ERROR: Found {len(non_pure_trials)} non-label-pure trials "
                f"(examples: {non_pure_trials[:5]}). Falling back to no-holdout."
            )
        return np.arange(n_total, dtype=np.int64), np.array([], dtype=np.int64)

    # ----------------------------
    # 3) Group trials by class
    # ----------------------------
    trials_per_class: dict[int, list[int]] = {0: [], 1: [], 2: []}
    for tid, lab in trial_to_label.items():
        if lab in trials_per_class:
            trials_per_class[lab].append(tid)

    # Need at least MIN_TRIALS_PER_CLASS_FOR_HOLDOUT trials in each class to have any chance
    for cls in [0, 1, 2]:
        if len(trials_per_class[cls]) < utils.MIN_TRIALS_PER_CLASS_FOR_HOLDOUT:
            if logger:
                logger.log(
                    f"[HOLDOUT] Class {cls} has only {len(trials_per_class[cls])} trials, "
                    f"need {utils.MIN_TRIALS_PER_CLASS_FOR_HOLDOUT}. Returning no-holdout."
                )
            return np.arange(n_total, dtype=np.int64), np.array([], dtype=np.int64)

    # ----------------------------
    # 4) Compute target holdout size as a FRACTION OF TOTAL windows
    #    (because whole-trial selection will overshoot a bit, we treat this as a soft target)
    # ----------------------------
    holdout_frac = float(np.clip(holdout_frac, 0.0, 0.5))
    target_holdout_total = int(np.ceil(n_total * holdout_frac))

    if logger:
        n_c0 = int((y == 0).sum())
        n_c1 = int((y == 1).sum())
        n_c2 = int((y == 2).sum())
        logger.log(f"[HOLDOUT] Total windows={n_total} (c0={n_c0} c1={n_c1} c2={n_c2})")
        logger.log(f"[HOLDOUT] Target holdout ~= {target_holdout_total} windows ({holdout_frac:.2f} of total)")

    # ----------------------------
    # 5) Shuffle trials within each class; round-robin pick whole trials until we hit target_total
    #    Also: ensure at least 1 trial from each class is picked (if possible) so holdout has all classes.
    # ----------------------------
    rng = np.random.default_rng(int(seed))
    for cls in [0, 1, 2]:
        rng.shuffle(trials_per_class[cls])

    positions = {0: 0, 1: 0, 2: 0}
    holdout_trials: set[int] = set()
    holdout_counts = {0: 0, 1: 0, 2: 0}
    holdout_total = 0

    # Phase A: guarantee presence (one trial per class)
    for cls in [0, 1, 2]:
        tid = trials_per_class[cls][0]
        positions[cls] = 1
        holdout_trials.add(tid)
        sz = len(trial_to_indices[tid])
        holdout_counts[cls] += sz
        holdout_total += sz

    # Phase B: round-robin fill until soft target reached
    max_iters = 100000
    it = 0
    while holdout_total < target_holdout_total and it < max_iters:
        it += 1
        made_progress = False

        for cls in [0, 1, 2]:
            if holdout_total >= target_holdout_total:
                break

            if positions[cls] >= len(trials_per_class[cls]):
                continue

            tid = trials_per_class[cls][positions[cls]]
            positions[cls] += 1
            if tid in holdout_trials:
                continue

            holdout_trials.add(tid)
            sz = len(trial_to_indices[tid])
            holdout_counts[cls] += sz
            holdout_total += sz
            made_progress = True

        if not made_progress:
            break

    # ----------------------------
    # 6) Convert selected trials to indices
    # ----------------------------
    holdout_indices_list: list[int] = []
    for tid in holdout_trials:
        holdout_indices_list.extend(trial_to_indices[tid])

    holdout_idx = np.array(sorted(holdout_indices_list), dtype=np.int64)
    holdout_mask = np.zeros(n_total, dtype=bool)
    holdout_mask[holdout_idx] = True
    train_idx = np.arange(n_total, dtype=np.int64)[~holdout_mask]

    # ----------------------------
    # 7) Validate both splits contain all classes
    # ----------------------------
    def _counts(idxs: np.ndarray) -> tuple[int, int, int]:
        return (
            int((y[idxs] == 0).sum()),
            int((y[idxs] == 1).sum()),
            int((y[idxs] == 2).sum()),
        )

    tr_c0, tr_c1, tr_c2 = _counts(train_idx)
    ho_c0, ho_c1, ho_c2 = _counts(holdout_idx)

    if logger:
        logger.log(
            f"[HOLDOUT] Selected trials: {len(holdout_trials)} | "
            f"hold windows={len(holdout_idx)} (c0={ho_c0} c1={ho_c1} c2={ho_c2}) | "
            f"train windows={len(train_idx)} (c0={tr_c0} c1={tr_c1} c2={tr_c2})"
        )

    # If any class missing in either split, fallback to no-holdout
    if (tr_c0 == 0 or tr_c1 == 0 or tr_c2 == 0) or (ho_c0 == 0 or ho_c1 == 0 or ho_c2 == 0):
        if logger:
            logger.log("[HOLDOUT] Missing class in train/holdout after trial selection. Returning no-holdout.")
        return np.arange(n_total, dtype=np.int64), np.array([], dtype=np.int64)

    return train_idx, holdout_idx

def make_group_holdout_split(
    group_ids: np.ndarray,
    y: np.ndarray,
    *,
    holdout_frac: float = 0.15,
    seed: int = 0,
    logger=None,
    min_groups_per_class: int = 1,   # require at least this many groups in holdout per class (if possible)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Leakage-safe holdout split where the leakage unit is group_id.

    - Splits by group_id (ALL windows in a group stay together).
    - Targets ~holdout_frac of TOTAL windows (not per-class targets).
    - Uses round-robin across classes to avoid one class dominating.
    - Tries to avoid large overshoot by skipping a group if it would overshoot badly.

    Returns: (train_idx, holdout_idx)
    """

    group_ids = np.asarray(group_ids).astype(np.int64)
    y = np.asarray(y).astype(np.int64)
    n = int(len(y))
    if n == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    # Map group_id -> window indices
    gid_to_idx: dict[int, np.ndarray] = {}
    for gid in np.unique(group_ids):
        gid_to_idx[int(gid)] = np.where(group_ids == gid)[0].astype(np.int64)

    # Determine each group's (assumed) label and size
    # If a group is not label-pure, we abort to avoid silent leakage/bugs.
    groups_by_class: dict[int, list[dict]] = {0: [], 1: [], 2: []}
    for gid, idxs in gid_to_idx.items():
        labels = y[idxs]
        uniq = np.unique(labels)
        cls = int(uniq[0])
        groups_by_class[cls].append({"gid": int(gid), "idxs": idxs, "size": int(len(idxs))})

    # If any class has zero groups, can't do stratified holdout safely
    for cls in (0, 1, 2):
        if len(groups_by_class[cls]) == 0:
            if logger:
                logger.log(f"[HOLDOUT] Class {cls} has 0 groups; falling back to no-holdout")
            return np.arange(n, dtype=np.int64), np.array([], dtype=np.int64)

    rng = np.random.default_rng(seed)
    for cls in (0, 1, 2):
        rng.shuffle(groups_by_class[cls])

    total_target = int(np.ceil(n * float(holdout_frac)))
    total_target = max(1, total_target)

    if logger:
        c0 = int((y == 0).sum()); c1 = int((y == 1).sum()); c2 = int((y == 2).sum())
        logger.log(f"[HOLDOUT] Total windows={n} (c0={c0} c1={c1} c2={c2})")
        logger.log(f"[HOLDOUT] Target holdout ~= {total_target} windows ({holdout_frac:.2f} of total)")

    # Round-robin selection over classes, choosing groups (not trials)
    pos = {0: 0, 1: 0, 2: 0}
    holdout_gids: list[int] = []
    holdout_indices: list[int] = []
    holdout_count = {0: 0, 1: 0, 2: 0}
    holdout_total = 0

    # Helper: check if we can still add at least one group from each class
    def has_remaining(cls: int) -> bool:
        return pos[cls] < len(groups_by_class[cls])

    # Phase 1: ensure each class appears in holdout (if possible)
    for cls in (0, 1, 2):
        if not has_remaining(cls):
            if logger:
                logger.log(f"[HOLDOUT] No remaining groups for class {cls}; falling back to no-holdout")
            return np.arange(n, dtype=np.int64), np.array([], dtype=np.int64)

        g = groups_by_class[cls][pos[cls]]
        pos[cls] += 1
        holdout_gids.append(g["gid"])
        holdout_indices.extend(g["idxs"].tolist())
        holdout_count[cls] += g["size"]
        holdout_total += g["size"]

    # Phase 2: Size-aware round-robin (LFD strategy)
    # Sort groups by size descending within each class
    for cls in (0, 1, 2):
        groups_by_class[cls].sort(key=lambda g: -g["size"])

    max_iters = 10000
    iters = 0

    while holdout_total < total_target and iters < max_iters:
        iters += 1
        made_progress = False

        for cls in (0, 1, 2):
            if holdout_total >= total_target:
                break
            if not has_remaining(cls):
                continue

            g = groups_by_class[cls][pos[cls]]
            gsize = g["size"]

            # IMPROVED: Only skip if we'd MASSIVELY overshoot AND we have alternatives
            new_total = holdout_total + gsize
            overshoot = new_total - total_target

            # More lenient overshoot tolerance for small groups
            if gsize <= 3:
                # Always take small groups (helps class 2 REST)
                pass
            elif overshoot > max(10, total_target * 0.2):
                # Only skip if huge overshoot AND we can try other classes
                remaining_classes = sum(1 for c in (0,1,2) if has_remaining(c))
                if remaining_classes > 1:
                    pos[cls] += 1
                    continue

            # Take group
            pos[cls] += 1
            holdout_gids.append(g["gid"])
            holdout_indices.extend(g["idxs"].tolist())
            holdout_count[cls] += gsize
            holdout_total = new_total
            made_progress = True

        if not made_progress:
            # Force progress by taking smallest available
            candidates = []
            for cls in (0, 1, 2):
                if has_remaining(cls):
                    candidates.append((groups_by_class[cls][pos[cls]]["size"], cls))
            if not candidates:
                break
            _, best_cls = min(candidates, key=lambda t: t[0])
            g = groups_by_class[best_cls][pos[best_cls]]
            pos[best_cls] += 1
            holdout_gids.append(g["gid"])
            holdout_indices.extend(g["idxs"].tolist())
            holdout_count[best_cls] += g["size"]
            holdout_total += g["size"]

    holdout_idx = np.array(sorted(set(holdout_indices)), dtype=np.int64)
    hold_mask = np.zeros(n, dtype=bool)
    hold_mask[holdout_idx] = True
    train_idx = np.arange(n, dtype=np.int64)[~hold_mask]

    # Final sanity: all classes present in both splits
    def class_counts(idxs: np.ndarray) -> tuple[int, int, int]:
        return (int((y[idxs] == 0).sum()), int((y[idxs] == 1).sum()), int((y[idxs] == 2).sum()))

    trc0, trc1, trc2 = class_counts(train_idx)
    hoc0, hoc1, hoc2 = class_counts(holdout_idx)

    if logger:
        logger.log(f"[HOLDOUT] Selected groups: {len(set(holdout_gids))} | hold windows={len(holdout_idx)} "
                   f"(c0={hoc0} c1={hoc1} c2={hoc2}) | train windows={len(train_idx)} "
                   f"(c0={trc0} c1={trc1} c2={trc2})")

    # If holdout lost a class (can happen with extreme imbalance + huge groups), fail safe
    if hoc0 == 0 or hoc1 == 0 or hoc2 == 0:
        if logger:
            logger.log("[HOLDOUT] Holdout missing a class after group selection; falling back to no-holdout")
        return np.arange(n, dtype=np.int64), np.array([], dtype=np.int64)

    if trc0 == 0 or trc1 == 0 or trc2 == 0:
        if logger:
            logger.log("[HOLDOUT] Train missing a class after group selection; falling back to no-holdout")
        return np.arange(n, dtype=np.int64), np.array([], dtype=np.int64)

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
                                max_epochs, patience, min_delta, debug_heavy = True,
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
        conf = stats["conf"]
        
        if debug_heavy:
            _log(
                logger,
                f"val_pred_count: pred0={stats['pred0']} pred1={stats['pred1']} pred2={stats['pred2']} | "
                f"conf rows=true: "
                f"t0={conf[0]} t1={conf[1]} t2={conf[2]}"
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

        if debug_heavy:
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
    else:
        _log(logger,"WARNING: best_state never set (unexpected).")

    return best_state, history

def train_single_final_model_with_holdout_and_export(
    *,
    X_pair: np.ndarray,
    y_pair: np.ndarray,
    trial_ids_pair: np.ndarray,
    group_pair: np.ndarray,
    n_ch: int,
    n_time: int,
    cfg: CNNTrainConfig,
    out_onnx_path: Path,
    device: torch.device,
    holdout_frac: float = 0.15,
    hz_a: int,
    hz_b: int,
    zscore_norm: str,
    logger: utils.DebugLogger | None = None
) -> tuple[bool, bool, float, float, float, float, list[utils.TrainIssue]]:
    """
    Trains ONE final model using a small trial-wise holdout for early stopping, then exports ONNX.

    Returns:
      (train_ok, export_ok, holdout_loss, holdout_acc, holdout_bal_acc)
    """
    final_issues: list[utils.TrainIssue] = []
    train_ok = False
    export_ok = False
    # Build trial-wise holdout split
    tr_idx, ho_idx = make_group_holdout_split(
        group_pair, y_pair, holdout_frac=float(holdout_frac), seed=int(cfg.seed), logger=logger,
    )
    _log(logger, f"[HOLDOUT] train windows={len(tr_idx)} c0={(y_pair[tr_idx]==0).sum()} c1={(y_pair[tr_idx]==1).sum()} c2={(y_pair[tr_idx]==2).sum()}")
    _log(logger, f"[HOLDOUT] hold  windows={len(ho_idx)} c0={(y_pair[ho_idx]==0).sum()} c1={(y_pair[ho_idx]==1).sum()} c2={(y_pair[ho_idx]==2).sum()}")

    if len(ho_idx) == 0:
        final_issues.append(utils.issue("HOLDOUT", "failed to make holdout; attempting to train on all data without early stopping"))
        # No holdout possible; train on all data without early stop and export.
        # (But in practice with multiple trials, ho_idx should not be empty.)
        ho_loss = 0.0
        ho_acc = 0.0
        ho_bal = 0.0
        # Train loader on all data
        if zscore_norm == "ON":
            X_all = apply_z_score_normalization(X_pair)
        else:
            X_all = X_pair
        X_all_t = torch.from_numpy(X_all[:, None, :, :]).float()
        y_all_t = torch.from_numpy(y_pair.astype(np.int64)).long()
        ds_all = TensorDataset(X_all_t, y_all_t)
        try:
            batches, batch_issues = make_pooled_batches(
                trial_ids=trial_ids_pair,
                y=y_pair,
                max_windows_per_batch=int(cfg.batch_size),
                seed=int(cfg.seed),
                hz_a=hz_a,
                hz_b=hz_b,
                logger=logger,
            )
            final_issues.extend(batch_issues)
            train_loader = DataLoader(ds_all, batch_sampler=ListBatchSampler(batches))

            model = EEGNet(
                n_ch=n_ch, n_time=n_time, n_classes=3,
                F1=int(cfg.F1), D=int(cfg.D), F2=int(_derived_F2(cfg)),
                kernel_length=int(cfg.kernel_length),
                pooling_factor=int(cfg.pooling_factor),
                pooling_factor_final=int(cfg.pooling_factor_final),
                dropout=float(cfg.dropout), norm_mode=cfg.norm_mode
            ).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))
            train_ok = True
        except Exception as e:
            _log(logger, f"[CNN] Final holdout training failed: {e}")
            return False, False, float(ho_loss), float(ho_acc), float(ho_bal), 0.0, final_issues
        for ep in range(int(cfg.MAX_EPOCHS)):
            tr_loss, tr_acc = run_epoch(model, train_loader, train=True,
                                        device=device, optimizer=optimizer, criterion=criterion)
        try:
            export_cnn_onnx(model=model, n_ch=n_ch, n_time=n_time, out_path=Path(out_onnx_path))
            export_ok = True
        except Exception as e:
            _log(logger, f"[CNN] ONNX export failed: {e}")
        return bool(train_ok), bool(export_ok), float(ho_loss), float(ho_acc), float(ho_bal), float(tr_acc), final_issues

    # REGULAR FLOW: Use existing trainer on the train/holdout split (this includes early stopping)
    try:
        val_bal, val_acc, _, train_acc, val_loss,  training_issues, model = train_cnn_on_split(
            X_train=X_pair[tr_idx], y_train=y_pair[tr_idx],
            X_val=X_pair[ho_idx],   y_val=y_pair[ho_idx],
            trial_ids_train=trial_ids_pair[tr_idx],
            trial_ids_val=trial_ids_pair[ho_idx],
            n_ch=n_ch, n_time=n_time,
            cfg=cfg,
            max_epochs=int(cfg.MAX_EPOCHS),
            device=device,
            hz_a=hz_a, hz_b=hz_b,
            zscorearg=zscore_norm,
            return_model=True,
        )
        final_issues.extend(training_issues)
        train_ok = True
    except Exception as e:
        _log(logger, f"[CNN] Final training with holdout failed: {e}")
        train_ok = False
    
    # only export if train ok
    if not train_ok:
        return False, False, float("inf"), 0.0, 0.0, 0.0, final_issues
    try:
        export_cnn_onnx(model=model, n_ch=n_ch, n_time=n_time, out_path=Path(out_onnx_path))
        export_ok = True
    except Exception as e:
        _log(logger, f"[CNN] ONNX export failed: {e}")
        export_ok = False

    return bool(train_ok), bool(export_ok), float(val_loss), float(val_acc), float(val_bal), float(train_acc), final_issues

def train_final_cnn_and_export(
    *,
    X_pair: np.ndarray,
    y_pair: np.ndarray,
    trial_ids_pair: np.ndarray,
    group_pair: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    n_ch: int,
    n_time: int,
    out_onnx_path: Path,
    hparam_tuning: str,
    zscore_norm: str,
    hz_a: int,
    hz_b: int,
    logger: utils.DebugLogger | None = None,
) -> tuple[utils.FinalTrainResults, list[utils.TrainIssue]]:
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
    issues: list[utils.TrainIssue] = []

    # 1) CV METRICS (for reporting)
    # We'll average final_train_loss/acc and final_val_loss/acc across folds for FinalTrainResults.
    cv_bals:       list[float] = []
    cv_accs:       list[float] = []
    cv_val_losses: list[float] = []

    # 2) Run grid search for hyperparameter tuning if ON
    # we will update this config object to get best 
    if(hparam_tuning != "OFF"):
        best_cfg = cfg
        best_score = -1.0
        if(hparam_tuning == "FULL"):
            search_space = HPARAM_SPACE
        elif(hparam_tuning == "QUICK"):
            search_space = HPARAM_SPACE_MINIMAL
        else:
            utils.abort("[TRAIN]", "hparam arg not recognized")
        
        # grid search -> iter releases new candidate each time it gets called
        for cand in iter_hparam_candidates(cfg, search_space):
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
                hz_a=hz_a, hz_b=hz_b,
                logger=logger,
                zscorearg=zscore_norm,
            )

            _log(logger,
                "[HTUNE] "
                f"score={score:.4f} "
                f"F1={cand.F1} D={cand.D} "
                f"k={cand.kernel_length} "
                f"p1={cand.pooling_factor} p2={cand.pooling_factor_final} "
                f"bs={cand.batch_size} lr={cand.learning_rate}"
                f"norm_mode={cand.norm_mode} dropout={cand.dropout}"
                f"patience={cand.patience} val_batch_mode={cand.val_batch_mode}"
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

            va_bal, va_acc, _, _, va_loss, training_issues = train_cnn_on_split(
                X_train=X_pair[tr_idx], y_train=y_pair[tr_idx],
                X_val=X_pair[va_idx],   y_val=y_pair[va_idx],
                trial_ids_train=trial_ids_pair[tr_idx],
                trial_ids_val=trial_ids_pair[va_idx],
                n_ch=n_ch, n_time=n_time,
                cfg=fold_cfg,
                max_epochs=cfg.MAX_EPOCHS,      # final training can use full schedule
                device=device,
                hz_a=hz_a, hz_b=hz_b,
                zscorearg=zscore_norm,
                logger=logger,
            )
            issues.extend(training_issues)
            cv_bals.append(float(va_bal))
            cv_accs.append(float(va_acc))
            cv_val_losses.append(float(va_loss))

    # 4) BUILD FINAL MODEL FROM BEST CFGS AND TRAIN with simple holdout, THEN EXPORT
    # We train on ALL X_pair/y_pair except small 10% holdout for early stopping, so the exported model sees maximum calibration windows but doesn't overfit.
    train_ok, export_ok, ho_loss, ho_acc, ho_bal, tr_acc, training_issues_2 = train_single_final_model_with_holdout_and_export(
        X_pair=X_pair,
        y_pair=y_pair,
        trial_ids_pair=trial_ids_pair,
        group_pair=group_pair,
        n_ch=n_ch,
        n_time=n_time,
        cfg=cfg,
        out_onnx_path=Path(out_onnx_path),
        device=device,
        logger=logger,
        holdout_frac=0.15,
        hz_a=hz_a,
        hz_b=hz_b,
        zscore_norm=zscore_norm,
    )
    issues.extend(training_issues_2)

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

        final_train_acc=float(tr_acc),

        cfg=asdict(cfg),
    ), issues

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
    zscorearg: str,
    hz_a: int,
    hz_b: int,
    return_model: bool = False, # need model when we do onnx export at the end (once only)
    logger: utils.DebugLogger | None = None,
) -> tuple[float, float, float, float, float, list[utils.TrainIssue]] | tuple[float, float, float, float, float, list[utils.TrainIssue], EEGNet]:
    """
    Trains CNN on a specific tr/val split and returns:
     (val_bal_acc, val_acc, train_loss, train_acc, val_loss)
    This avoids random_split so we can do proper CV.
    - Uses cfg for ALL hyperparams (F1/kernel/pooling/batch/lr/patience/etc).
    """
    issue_list: list[utils.TrainIssue] = []
    # Make training more repeatable across runs
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    if torch.cuda.is_available():
       torch.cuda.manual_seed_all(int(cfg.seed))
       torch.backends.cudnn.deterministic = True
       torch.backends.cudnn.benchmark = False

    # normalize by z-score to avoid large magnitude skews
    if zscorearg == "ON":
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
        batches, batch_issues = make_pooled_batches(
            trial_ids=trial_ids_train,
            y=y_train,
            max_windows_per_batch=int(cfg.batch_size),
            seed=int(cfg.seed),
            hz_a=hz_a,
            hz_b=hz_b,
            logger=logger,
        )
        issue_list.extend(batch_issues)
        train_loader = DataLoader(train_ds, batch_sampler=ListBatchSampler(batches))
        #batch_debug_issues = debug_all_batches(train_loader, logger=logger)
        #issue_list.extend(batch_debug_issues)
    else:
        # fallback: window batching (less good for strongly overlapping ssvep windows...)
        g = torch.Generator().manual_seed(int(cfg.seed))
        train_loader = DataLoader(
            train_ds,
            batch_size=min(int(cfg.batch_size), len(train_ds)) if len(train_ds) > 0 else int(cfg.batch_size),
            shuffle=True,
            generator=g
        )
        issue_list.append(utils.issue("CNN_BATCH","Batching fell back to window rather than grouped (degraded behavior for highly overlapping windows)"))

    if logger:
        logger.log(f"[BATCH_DEBUG] Inspecting first 3 batches...")
        for batch_i, (xb, yb) in enumerate(train_loader):
            if batch_i >= 3:
                break
            c0 = int((yb == 0).sum())
            c1 = int((yb == 1).sum())
            c2 = int((yb == 2).sum())
            logger.log(f"[BATCH_DEBUG] Batch {batch_i}: size={len(yb)} | c0={c0} c1={c1} c2={c2}")

    # val loader uses random window batching
    if len(val_ds) == 0:
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    else:
        if getattr(cfg, "val_batch_mode", "FULL") == "FULL":
            # single batch = cleanest val loss/acc for early stopping
            val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)
        else:
            # constant, non-tiny batches
            bs = min(int(getattr(cfg, "val_batch_size", 27)), len(val_ds))
            val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, drop_last=False)


    _log(logger, f"[SPLIT] train windows={len(train_ds)} c0={(y_train==0).sum()} c1={(y_train==1).sum()} c2={(y_train==2).sum()}")
    _log(logger, f"[SPLIT] val   windows={len(val_ds)}   c0={(y_val==0).sum()} c1={(y_val==1).sum()} c2={(y_val==2).sum()}")

    # build model using cfg
    F2 = _derived_F2(cfg)
    model = EEGNet(
        n_ch=n_ch, n_time=n_time, n_classes=3,
        F1=int(cfg.F1),
        D=int(cfg.D),
        F2=int(F2),
        kernel_length=int(cfg.kernel_length),
        pooling_factor=int(cfg.pooling_factor),
        pooling_factor_final=int(cfg.pooling_factor_final),
        dropout=float(cfg.dropout), norm_mode=cfg.norm_mode
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))

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
        debug_heavy=False,
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
    val_bal = utils.balanced_accuracy_multiclass(y_true, y_pred)

    if return_model:
        return (
            float(val_bal),
            float(val_acc),
            float(train_loss),
            float(train_acc),
            float(val_loss),
            issue_list,
            model,
        )

    return (
        float(val_bal),
        float(val_acc),
        float(train_loss),
        float(train_acc),
        float(val_loss),
        issue_list
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
    hz_a: int,
    hz_b: int,
    logger: utils.DebugLogger | None = None,
    zscorearg: str,
) -> float:
    """
    Returns mean CV balanced accuracy for this hparam cfg.
    Uses per-fold seed offsets for repeatability.
    """
    bals: list[float] = []
    for fi, (tr_idx, va_idx) in enumerate(folds):
        fold_cfg = replace(cfg, seed=int(cfg.seed + 1000 * fi))
        val_bal, _, _, _, _, _ = train_cnn_on_split(
            X_train=X_pair[tr_idx], y_train=y_pair[tr_idx],
            X_val=X_pair[va_idx],   y_val=y_pair[va_idx],
            trial_ids_train=trial_ids_pair[tr_idx],
            trial_ids_val=trial_ids_pair[va_idx],
            n_ch=n_ch, n_time=n_time,
            cfg=fold_cfg,
            max_epochs=int(max_epochs),
            device=device,
            hz_a=hz_a, hz_b=hz_b,
            zscorearg=zscorearg,
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
    zscorearg: str,
    logger: utils.DebugLogger | None = None
) -> tuple[utils.ModelMetrics, list[utils.TrainIssue]]:
    """
    Cross-val scoring for ONE (already-binary) pair.
    Folds are built in train_ssvep.py (shared across archs).
    Returns ModelMetrics (shared contract).
    """
    pair_issues: list[utils.TrainIssue] = []
    cfg = CNNTrainConfig() # use default configs
    # if high frequency, reduce pooling factors to preserve greater freq range that supports Nyquist
    if freq_a_hz >= 25.0 or freq_b_hz >= 25.0:
        cfg = replace(cfg, pooling_factor=4) # 250 Hz / 4 = 62.5 Hz preservation

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bals: list[float] = []
    accs: list[float] = []

    for fi, (tr_idx, va_idx) in enumerate(folds):
        # per-fold seed so folds are repeatable but not identical
        fold_cfg = replace(cfg, seed=int(cfg.seed + 1000 * fi))

        val_bal, val_acc, _, _, _, training_issues = train_cnn_on_split(
            X_train=X_pair[tr_idx], y_train=y_pair[tr_idx],
            X_val=X_pair[va_idx],   y_val=y_pair[va_idx],
            trial_ids_train=trial_ids_pair[tr_idx],
            trial_ids_val=trial_ids_pair[va_idx],
            n_ch=n_ch, n_time=n_time,
            cfg=fold_cfg,
            max_epochs=int(cfg.MAX_EPOCHS_CV),
            device=device,
            hz_a=freq_a_hz,
            hz_b=freq_b_hz,
            zscorearg=zscorearg,
            logger=logger,
        )
        pair_issues.extend(training_issues)
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
        ), pair_issues

    return utils.ModelMetrics(
        freq_a_hz=int(freq_a_hz),
        freq_b_hz=int(freq_b_hz),
        cv_ok=True,
        avg_fold_balanced_accuracy=float(np.mean(bals)),
        std_fold_balanced_accuracy=float(np.std(bals)),
        avg_fold_accuracy=float(np.mean(accs)),
        std_fold_accuracy=float(np.std(accs)),
    ), pair_issues


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