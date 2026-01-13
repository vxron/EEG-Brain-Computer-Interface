# Collection of commonly used stats utils for features, scoring, etc...

import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path

# =================== SHARED DATACLASSES ============================
@dataclass(frozen=True)
class DatasetInfo:
    ch_cols: list[str]
    n_ch: int
    n_time: int
    classes_hz: list[int]

@dataclass(frozen=True)
class GeneralTrainingConfigs:
    number_cross_val_folds: int = 6  # cross val is used for assessing pairwise combinations & selecting bset frequency-pair
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
    final_train_loss: float
    final_train_acc: float
    final_val_loss: float
    final_val_acc: float

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