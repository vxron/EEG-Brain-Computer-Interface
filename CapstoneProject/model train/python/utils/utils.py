# Collection of commonly used stats utils for features, scoring, etc...

import pandas as pd
import numpy as np
from dataclasses import dataclass
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
    avg_fold_balanced_accuracy: float      # currently the deciding factor
    std_fold_balanced_accuracy: float
    avg_fold_accuracy: float
    std_fold_accuracy: float
    # TODO: investigate metrics used in literature for this purpose

@dataclass(frozen=True)
class FinalTrainResults:
    train_ok: bool
    onnx_export_ok: bool
    
    # Cross-validation summary
    cv_ok: bool
    cv_mean_bal_acc: float
    cv_std_bal_acc: float
    cv_mean_acc: float
    cv_std_acc: float
    cv_mean_val_loss: float
    cv_std_val_loss: float

    # Final exported model early-stop holdout metrics (not used for selection across freq pairs)
    final_holdout_ok: bool
    final_holdout_loss: float
    final_holdout_acc: float
    final_holdout_bal_acc: float
    cfg: dict[str, Any] | None = None # FOR CNN ARCH (svm may not need cfg)
    # Optional extras (won’t be required at init)
    final_train_loss: float | None = None
    final_train_acc: float | None = None
    final_val_loss: float | None = None
    final_val_acc: float | None = None

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

# ======================= BASIC STATS ===================================
def mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0

def std(xs: list[float]) -> float:
    return float(np.std(xs)) if xs else 0.0
