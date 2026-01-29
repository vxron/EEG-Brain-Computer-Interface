# Collection of commonly used stats utils for features, scoring, etc...
from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
from datetime import datetime

# ==================== TEXT FILE LOGGER ==============================
class DebugLogger:
    """
    Simple line-oriented debug logger that writes to a file.
    """
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        header = (
            f"==== DEBUG LOG START ====\n"
            f"created: {datetime.now().isoformat()}\n"
            f"path: {self.path.resolve()}\n"
            f"=========================\n\n"
        )
        self.path.write_text(header)

    def log(self, msg: str = ""):
        with self.path.open("a", encoding="utf-8") as f:
            f.write(msg.rstrip() + "\n")

# =============== TRAINING DATA MINIMUM REQS & VALIDATOR FUNCTIONS ====================
HOP_SAMPLES = 80                       # matches C++
MIN_TOTAL_WINDOWS = 80                 # after filters (trimmed + not bad)
MIN_FREQS_FOR_PAIR_SEARCH = 2          # need at least 2 freqs with enough windows
MIN_WINDOWS_PER_FREQ_FOR_SHORTLIST = 40 
# For pair-level training sanity
MIN_PAIR_WINDOWS_PER_CLASS = 40        # for a stable pair evaluation/training
MIN_PAIR_TRIALS_PER_CLASS = 2 
# For CV fold building
MIN_GROUPS_PER_CLASS_FOR_CV = 3        # absolute minimum 
MIN_FOLDS_MIN = 2                      # hard minimum; k_eff will degrade from here
MIN_WINDOWS_PER_CLASS_VAL = 10          # 5 validation samples per class
MIN_WINDOWS_PER_CLASS_TRAIN = 20       # 10 training samples per class
# For holdout training
MIN_TRIALS_PER_CLASS_FOR_HOLDOUT = 1   # if fewer, skip holdout (fall back to no-holdout)
# For batches
MIN_TRIALS_PER_CLASS_FOR_BATCHING = 1
MIN_NUM_BATCHES_PER_CLASS = 10
MIN_FRAC_PER_CLASS_IN_BATCH = 0.20
MAX_REST_FRAC_IN_BATCH = 0.55
MAX_CLASS_IMBALANCE_IN_BATCH = 0.30    # |c0-c1|/(c0+c1)
MAX_SMALL_BATCHES = 2 # balance trying to use maximum data with consistent batch sizing

# =================== SHARED DATACLASSES ============================
@dataclass(frozen=True)
class DatasetInfo:
    ch_cols: list[str]
    n_ch: int
    n_time: int
    classes_hz: list[int]

@dataclass(frozen=True)
class GeneralTrainingConfigs:
    number_cross_val_folds: int = 5  # cross val is used for assessing pairwise combinations & selecting bset frequency-pair
    test_split_fraction: float = 0.10   # [should be rlly small test fold, maybe 10%] regular test/train is used for final model training (deployment)

# Custom metrics that select_best_pair expects trainers to output for their pairwise models
@dataclass(frozen=True)
class ModelMetrics:
    freq_a_hz: int
    freq_b_hz: int
    cv_ok: bool                            # report any failures if not ok
    avg_fold_balanced_accuracy: float      # currently the deciding factor between paired models (protects against class imbalance)
    std_fold_balanced_accuracy: float
    avg_fold_accuracy: float
    std_fold_accuracy: float

@dataclass(frozen=True)
class FinalTrainResults:
    train_ok: bool = False
    onnx_export_ok: bool = False
    
    # Cross-validation summary
    cv_ok: bool = False
    cv_mean_bal_acc: float = 0.0
    cv_std_bal_acc: float = 0.0
    cv_mean_acc: float = 0.0
    cv_std_acc: float = 0.0
    cv_mean_val_loss: float = 0.0 # cross-entropy loss is objective that optimizer is minimizing
    cv_std_val_loss: float = 0.0

    # Final exported model early-stop holdout metrics (not used for selection across freq pairs)
    final_holdout_ok: bool = False
    final_holdout_loss: float = float("inf") # loss on holdout set
    final_holdout_acc: float = 0.0
    final_holdout_bal_acc: float = 0.0
    cfg: dict[str, Any] | None = None # FOR CNN ARCH (svm may not need cfg)
    final_train_loss: float | None = None
    final_train_acc: float | None = None

# ==================== SHARED ERROR HANDLING ============================
# One shared error payload
@dataclass(frozen=True)
class TrainIssue:
    stage: str
    message: str
    details: dict[str,Any] | None = None
    data_insufficiency: dict[str, Any] | None = None  # receives NONE when not ui-facing issue
    # frequency: which freq was insufficient
    # metric: trials | windows | groups
    # required
    # actual

# for NON-FATAL problems (e.g. skip a pair, etc)
def issue(stage: str, message: str, details: dict[str, Any] | None = None, data_insufficiency: dict[str,Any] | None = None) -> TrainIssue:
    return TrainIssue(stage=stage, message=message, details=details, data_insufficiency=data_insufficiency)

def issues_to_json(issues: list[TrainIssue]) -> list[dict[str, Any]]: # each issue becomes a dict
    return [asdict(i) for i in issues]

class TrainAbort(RuntimeError): # TrainAbort class inherits from RuntimeError
    def __init__(self, issue: TrainIssue):
        super().__init__(f"[{issue.stage}] {issue.message}") # calls parent class constructor RuntimeError.__init__ and changes message format to issue
        self.issue = issue # assign issue to self so we can access it later e.g. "e.issue.stage"

# for FATAL error conditions
def abort(stage: str, message: str, details: dict[str,Any] | None = None, data_insufficiency: dict[str,Any] | None = None):
    raise TrainAbort(issue(stage,message,details,data_insufficiency)) # calls a new trainabort object

# =================== METRIC CALCULATIONS ============================

def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Balanced accuracy = mean(recall per class).
    Safer than raw accuracy when class counts differ.
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)

    accs = []
    for cls in (0, 1):
        m = (y_true == cls)
        if m.sum() == 0:
            continue
        accs.append(float((y_pred[m] == cls).mean()))
    return float(np.mean(accs)) if accs else 0.0

def balanced_accuracy_multiclass(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    recalls = []
    for c in range(n_classes):
        mask = (y_true == c)
        denom = int(mask.sum())
        if denom == 0:
            return 0.0  # if a class is missing, treat as invalid fold
        recalls.append(float((y_pred[mask] == c).mean()))
    return float(np.mean(recalls))

# ======================= BASIC STATS ===================================
def mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0

def std(xs: list[float]) -> float:
    return float(np.std(xs)) if xs else 0.0
