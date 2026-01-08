# Collection of commonly used stats utils for features, scoring, etc...

# HADEEL Ftr extraction utils should be put here (as well as any reusable scoring you use, or just use whats alr here if needed too!)

import pandas as pd
import numpy as np

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