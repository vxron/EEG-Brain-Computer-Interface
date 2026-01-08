#!/usr/bin/env python3
"""
train_ssvep.py

Script for training an SSVEP model from calibration data.
This script is called automatically by the C++ backend after a calibration
session. It MUST match the CLI interface expected by C++. 
(see training manager thread in CapstoneProject.cpp for details)

- Loads windowed EEG from one or many eeg_windows.csv files.
- Applies per-window, per-channel z-score normalization.
- Selects BEST LEFT/RIGHT frequency pair using cross-validated scoring on model arch.
    --arch SVM : linear SVM
    --arch CNN : compact CNN (EEGNet) in PyTorch
- Trains final model on that binary pair.
- Exports ONNX model to <model_dir>/ssvep_model.onnx
- Writes train_result.json to <model_dir>/train_result.json

Expected args:
    --data <path>      directory containing calibration data
    --model <path>     directory where ONNX + meta.json should be written
    --subject <str>    subject ID
    --session <str>    session ID
    --arch <CNN|SVM>
    --calibsetting <all_sessions|most_recent_only>
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from dataclasses import dataclass
from typing import Any

import utils.utils as utils

# Trainers live in trainers/
from trainers.cnn_trainer import train_cnn, export_cnn_onnx, train_cnn_on_split
#rom trainers.svm_trainer import train_svm, export_svm_onnx # TODO HADEEL

# ------------------------------
# PATH ROOTS (repo-anchored)
# ------------------------------
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
DATA_ROOT = REPO_ROOT / "data"
MODELS_ROOT = REPO_ROOT / "models"


# ------------------------------
# CLI ARG PARSER
# ------------------------------
def get_args():
    parser = argparse.ArgumentParser(
        description="Train SSVEP SVM model from calibration data."
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to directory containing calibration data (eeg_windows.csv).",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Output directory where ONNX + metadata will be saved.",
    )

    parser.add_argument(
        "--subject",
        type=str,
        required=True,
        help="Subject ID for this training run.",
    )

    parser.add_argument(
        "--session",
        type=str,
        required=True,
        help="Session ID for this training run.",
    )

    parser.add_argument(
        "--arch",
        type=str,
        required=True,
        choices=["CNN", "SVM"],
        help="Choice of ML model (SVM or CNN).",
    )

    parser.add_argument(
        "--calibsetting",
        type=str,
        required=True,
        choices=["all_sessions", "most_recent_only"],
        help="Choice of calib data setting (all sessions or most recent).",
    )

    # Shared pipeline knobs (regardless of model)
    parser.add_argument("--val_ratio", type=float, default=0.2) # 20% val, 80% train used for FINAL MODEL TRAINING
    parser.add_argument("--seed", type=int, default=0)

    # Pair selection knobs (regardless of model)
    parser.add_argument("--pick_top_k_freqs", type=int, default=6,
                        help="Shortlist top-K freqs before pairwise search (cap compute).")
    parser.add_argument("--cv_folds", type=int, default=5,
                   help="Cross-validation folds used for best-pair scoring.") # used for SCORING CANDIDATE PAIRS
    parser.add_argument("--min_windows_per_class", type=int, default=4,
                   help="Minimum number of windows per class required to consider a frequency.")

    # CNN specific knobs
    parser.add_argument("--max_epochs", type=int, default=300)
    
    parser.add_argument(
        "--patience", 
        type=int, 
        default=30,
        help="Number of successive iterations we'll continue for when seeing no improvement.",
    )

    parser.add_argument(
        "--min_delta", 
        type=float, 
        default=1e-4,
        help="numerical change in loss func necessary to consider real improvement",
    ) 
    
    # TODO: should choose batch_size based on number of training windows if arg is not given
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=12,
        help = "how many training windows the CNN sees at once before updating its weights (1 optimizer step)",
    ) 
    # greater batch sizes produce a more stable, accurate gradient (more of training set being used at once) and faster per-epoch compute times on GPUs, at the cost of less frequent weight updates (slower learning dynamics)
    # greater batch sizes (toward whole train set) are also at a greater risk of overfitting because gradient is more stable (tends towards sharp minima)...
    #  whereas small batch w diff gradients per step adds intrinsic regularization/stochastic noise (flatter minima, small input noise doesn't matter)
    # for small eeg datasets -> small batch sizes preferred

    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=1e-3,
        help="magnitude of gradient descent steps. smaller batches require smaller LR.",
    ) 

    # SVM specific knobs HERE (if any)

    return parser.parse_args()

# -----------------------------
# Shared Between All Trainers (CNN, SVM, ETC...) 
# Data Loading + Preprocessing
# -----------------------------
@dataclass
class DatasetInfo:
    ch_cols: list[str]
    n_ch: int
    n_time: int
    classes_hz: list[int]

# Group rows by session folder name in addition to window_idx so we avoid collisions
@dataclass(frozen=True)
class CsvSource:
    src_id: str          # session folder name
    path: Path

@dataclass
class NormalizerSpec:
    """
    per-window zscores
    """
    type: str = "zscore_per_window_per_channel"
    eps: float = 1e-6

    def apply(self, X: np.ndarray) -> np.ndarray:
        # X: (N, C, T)
        mu = X.mean(axis=2, keepdims=True)
        sd = X.std(axis=2, keepdims=True)
        return (X - mu) / (sd + self.eps)

# Handle "most recent" vs "all sessions" calib data settings
def list_window_csvs(data_session_dir: Path, calibsetting: str) -> list[CsvSource]:
    """
    data_session_dir: <root>/data/<subject>/<session>
    Returns sources with stable src_id to prevent window_idx collisions across sessions.
    """
    data_session_dir = Path(data_session_dir)

    if calibsetting == "most_recent_only":
        p = data_session_dir / "eeg_windows.csv"
        return [CsvSource(src_id=data_session_dir.name, path=p)]

    if calibsetting == "all_sessions":
        subject_dir = data_session_dir.parent  # <root>/data/<subject>
        sources: list[CsvSource] = []

        for sess_dir in subject_dir.iterdir():
            if not sess_dir.is_dir():
                continue
            if "__IN_PROGRESS" in sess_dir.name:
                continue

            p = sess_dir / "eeg_windows.csv"
            if p.exists():
                sources.append(CsvSource(src_id=sess_dir.name, path=p))

        # stable ordering
        sources.sort(key=lambda s: s.src_id)
        return sources

    raise ValueError(f"Unknown calibsetting: {calibsetting}")


# BUILD TRAINING DATA
# We read CSV(s) and get the shape of X
# (N, C, T) = (num_windows, num_channels, window_len_samples)
def load_windows_csv(sources: list[CsvSource]) -> tuple[np.ndarray, np.ndarray, DatasetInfo]:
    """
    Reads window-level CSV(s) and returns:
      X: (N, C, T)
      y_hz: (N,)
      info: DatasetInfo

    Expected columns:
      window_idx, is_trimmed, is_bad, sample_idx, eeg1..eeg8, testfreq_e, testfreq_hz
    """
    frames: list[pd.DataFrame] = []

    required = {"window_idx", "is_trimmed", "is_bad", "sample_idx", "testfreq_hz", "testfreq_e"}

    for src in sources:
        if not src.path.exists():
            continue

        df = pd.read_csv(src.path)

        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{src.path} missing columns: {sorted(missing)}")

        # filters (keep only trimmed and not-bad)
        df = df[df["is_trimmed"] == 1].copy()
        df = df[df["is_bad"] != 1].copy()

        # tag source to prevent window_idx collisions across sessions
        df["_src"] = src.src_id

        frames.append(df)

    if not frames:
        raise FileNotFoundError("No usable eeg_windows.csv found for requested setting.")

    df = pd.concat(frames, ignore_index=True)

    # Detect channel columns as the ones between sample_idx and testfreq_e
    cols = list(df.columns)
    sample_i = cols.index("sample_idx")
    tf_e_i = cols.index("testfreq_e")
    ch_cols = cols[sample_i + 1 : tf_e_i]
    n_ch = len(ch_cols)
    if n_ch <= 0:
        raise ValueError("No EEG channel columns detected between sample_idx and testfreq_e.")

    windows: list[np.ndarray] = []
    labels_hz: list[int] = []

    # group by (_src, window_idx) to avoid collisions
    grouped = df.groupby(["_src", "window_idx"], sort=True)

    for (_src, wid), g in grouped:
        g = g.sort_values("sample_idx")

        tf_hz = int(g["testfreq_hz"].iloc[0])
        if tf_hz < 0:
            continue

        x_tc = g[ch_cols].to_numpy(dtype=np.float32)  # (T, C)
        if x_tc.ndim != 2 or x_tc.shape[1] != n_ch:
            continue

        x_ct = x_tc.T  # (C, T)

        # Allow small T (window length) differences across sessions:
        # We standardize by cropping everyone to the smallest observed T.
        if not windows:
            target_T = x_ct.shape[1]
        else:
            target_T = windows[0].shape[1]

        if x_ct.shape[1] > target_T:
            x_ct = standardize_T_crop(x_ct, target_T)

        # If we ever encounter a shorter window, we retroactively crop all prior windows.
        if x_ct.shape[1] < target_T:
            target_T = x_ct.shape[1]
            windows = [standardize_T_crop(w, target_T) for w in windows]

        windows.append(x_ct)
        labels_hz.append(tf_hz)

    if not windows:
        raise RuntimeError("No valid windows found after grouping/filters.")

    X = np.stack(windows, axis=0).astype(np.float32)  # (N, C, T)
    y_hz = np.array(labels_hz, dtype=np.int64)

    info = DatasetInfo(
        ch_cols=ch_cols,
        n_ch=n_ch,
        n_time=int(X.shape[2]),
        classes_hz=sorted(set(y_hz.tolist())),
    )
    return X, y_hz, info


# -----------------------------
# Best Pair Selection Logic
# -----------------------------

def standardize_T_crop(x_ct: np.ndarray, target_T: int) -> np.ndarray:
    """
    x_ct: (C, T)
    Crop windows to target_T deterministically.
    THIS IS FOR WHEN WE FIND WINDOWS AREN'T EXACTLY SAME LENGTH & WE DONT WANT TO FAIL.

    We crop from the END so alignment is stable if windows were built sequentially.
    """
    C, T = x_ct.shape
    if T == target_T:
        return x_ct
    if T < target_T:
        raise ValueError("standardize_T_crop only supports cropping (T >= target_T).")
    return x_ct[:, -target_T:]


def make_binary_pair_dataset(X: np.ndarray, y_hz: np.ndarray, hz_a: int, hz_b: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Filters to only windows in {hz_a, hz_b} and returns:
      Xp: (Npair, C, T)
      yb: (Npair,) where hz_a -> 0, hz_b -> 1
    """
    mask = (y_hz == hz_a) | (y_hz == hz_b)
    Xp = X[mask]
    yp = y_hz[mask]
    yb = np.where(yp == hz_a, 0, 1).astype(np.int64)
    return Xp, yb

def make_cv_folds_binary(yb: np.ndarray, k: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Returns list of (train_idx, val_idx) for k-fold CV.
    Ensures each fold gets some samples from each class when possible.
    """
    rng = np.random.default_rng(seed)
    idx0 = np.where(yb == 0)[0]
    idx1 = np.where(yb == 1)[0]
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    folds0 = np.array_split(idx0, k)
    folds1 = np.array_split(idx1, k)

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(k):
        val_idx = np.concatenate([folds0[i], folds1[i]]) if (len(folds0[i]) + len(folds1[i])) else np.array([], dtype=np.int64)
        train_parts = [folds0[j] for j in range(k) if j != i] + [folds1[j] for j in range(k) if j != i]
        train_idx = np.concatenate(train_parts) if train_parts else np.array([], dtype=np.int64)

        if val_idx.size == 0 or train_idx.size == 0:
            continue

        folds.append((train_idx, val_idx))

    return folds

def shortlist_freqs(y_hz: np.ndarray, *, min_windows_per_class: int, pick_top_k: int) -> list[int]:
    """
    Keep only frequencies with enough windows, then take the top-K by count.
    """
    vals, counts = np.unique(y_hz, return_counts=True)
    keep = [(int(v), int(c)) for v, c in zip(vals, counts) if int(c) >= int(min_windows_per_class)]
    if len(keep) < 2:
        return []

    # Sort by count desc (more data = more stable), then by freq asc
    keep.sort(key=lambda t: (-t[1], t[0]))
    freqs = [hz for hz, _ in keep]
    return freqs[: min(pick_top_k, len(freqs))]

def score_pair_cnn_cv(
    *,
    X: np.ndarray, y_hz: np.ndarray,
    hz_a: int, hz_b: int,
    n_ch: int, n_time: int,
    cv_folds: int,
    seed: int,
    batch_size: int,
    learning_rate: float,
    max_epochs: int,
    patience: int,
    min_delta: float,
    device: torch.device,
) -> dict[str, Any]:
    """
    Returns dict with mean/std balanced accuracy across folds.
    """
    Xp, yb = make_binary_pair_dataset(X, y_hz, hz_a, hz_b)

    # Guard: need enough per class
    c0 = int((yb == 0).sum())
    c1 = int((yb == 1).sum())
    if c0 < 2 or c1 < 2:
        return {"ok": False, "reason": "not_enough_data", "c0": c0, "c1": c1}

    # Auto-reduce k so CV still works on small datasets.
    k = int(min(cv_folds, c0, c1))
    if k < 2:
        return {"ok": False, "reason": "k_too_small", "c0": c0, "c1": c1, "k": k}

    folds = make_cv_folds_binary(yb, k, seed)
    if len(folds) < max(2, k - 1):
        return {"ok": False, "reason": "fold_build_failed", "c0": c0, "c1": c1, "k": k}

    # Use shorter training during selection (faster but same arch)
    max_epochs_sel = min(max_epochs, 80)
    patience_sel   = min(patience, 10)

    bals = []
    accs = []
    for fi, (tr_idx, va_idx) in enumerate(folds):
        val_bal, val_acc = train_cnn_on_split(
            X_train=Xp[tr_idx], y_train=yb[tr_idx],
            X_val=Xp[va_idx],   y_val=yb[va_idx],
            n_ch=n_ch, n_time=n_time,
            seed=seed + 1000 * fi,
            batch_size=min(batch_size, int(len(tr_idx))),
            learning_rate=learning_rate,
            max_epochs=max_epochs_sel,
            patience=patience_sel,
            min_delta=min_delta,
            device=device,
        )
        bals.append(val_bal)
        accs.append(val_acc)

    return {
        "ok": True,
        "hz_a": hz_a, "hz_b": hz_b,
        "n_pair": int(len(yb)),
        "c0": c0, "c1": c1,
        "mean_bal_acc": float(np.mean(bals)),
        "std_bal_acc": float(np.std(bals)),
        "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs)),
        "folds_used": int(len(folds)),
        "k_requested": int(cv_folds),
        "k_used": int(k),
    }

def select_best_pair(
    *,
    X: np.ndarray, y_hz: np.ndarray,
    info: DatasetInfo,
    args,
    device: torch.device,
) -> tuple[int, int, dict[str, Any]]:
    """
    Returns: (best_left_hz, best_right_hz, debug_info)
    Scoring metric: mean CV balanced accuracy.
    """
    cand_freqs = shortlist_freqs(
        y_hz,
        min_windows_per_class=args.min_windows_per_class,
        pick_top_k=args.pick_top_k_freqs,
    )
    if len(cand_freqs) < 2:
        raise RuntimeError("Not enough usable frequencies after min_windows_per_class filtering.")

    pairs = [(cand_freqs[i], cand_freqs[j]) for i in range(len(cand_freqs)) for j in range(i + 1, len(cand_freqs))]

    best = None
    best_score = -1.0
    all_scores = []

    print(f"[PY] Pair search candidates: freqs={cand_freqs} -> {len(pairs)} pairs, arch={args.arch}")

    for hz_a, hz_b in pairs:
        if args.arch == "CNN":
            res = score_pair_cnn_cv(
                X=X, y_hz=y_hz,
                hz_a=hz_a, hz_b=hz_b,
                n_ch=info.n_ch, n_time=info.n_time,
                cv_folds=args.cv_folds,
                seed=args.seed,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                max_epochs=args.max_epochs,
                patience=args.patience,
                min_delta=args.min_delta,
                device=device,
            )
        else:
            res = {"ok": False, "reason": "svm_not_implemented"}
            print("hadeel todo for svm")

        all_scores.append(res)

        if not res.get("ok", False):
            print(f"[PY] pair ({hz_a},{hz_b}) skipped: {res.get('reason')}")
            continue

        score = float(res["mean_bal_acc"])
        print(f"[PY] pair ({hz_a},{hz_b}) mean_bal_acc={score:.3f} (+/-{res['std_bal_acc']:.3f}) n={res['n_pair']}")

        if score > best_score:
            best_score = score
            best = res

    if best is None:
        raise RuntimeError("Failed to select any valid pair (all candidates lacked data).")

    a = int(best["hz_a"])
    b = int(best["hz_b"])

    # Make left/right deterministic
    # Convention: left = lower Hz, right = higher Hz
    left_hz, right_hz = (a, b) if a < b else (b, a)

    debug = {
        "candidate_freqs": cand_freqs,
        "pair_scores": all_scores,
        "best_score_mean_bal_acc": best_score,
    }
    return left_hz, right_hz, debug


# -----------------------------
# JSON Contract & ONNX Export
# -----------------------------
def write_train_result_json(model_dir: Path, *, ok: bool, arch: str, calibsetting: str,
                            subject: str, session: str,
                            left_hz: int, right_hz: int,
                            left_e: int, right_e: int,
                            extra: dict[str, Any] | None = None) -> Path:
    payload: dict[str, Any] = {
        "ok": bool(ok),
        "arch": arch,
        "calibsetting": calibsetting,
        "subject_id": subject,
        "session_id": session,
        "best_freq_left_hz": int(left_hz),
        "best_freq_right_hz": int(right_hz),
        "best_freq_left_e": int(left_e),
        "best_freq_right_e": int(right_e),
    }
    if extra:
        payload["extra"] = extra

    out_path = Path(model_dir) / "train_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[PY] wrote train_result -> {out_path.resolve()}")
    return out_path

def hz_to_enum_mapping() -> dict[int, int]:
    """
    MUST keep this mapping consistent with C++ TestFreq_E.
    """
    return {
        8: 1,
        9: 2,
        10: 3,
        11: 4,
        12: 5,
        20: 6,
        25: 7,
        30: 8,
        35: 9,
    }


# -----------------------------
# Orchestrator (MAIN)
# -----------------------------
def main():
    args = get_args()

    # 0: RESOLVE PATHS FROM ARGS
    out_dir_arg = Path(args.model)
    out_dir = out_dir_arg if out_dir_arg.is_absolute() else (MODELS_ROOT / out_dir_arg)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[PY] model output dir:", out_dir.resolve())

    data_arg = Path(args.data)
    data_session_dir = data_arg if data_arg.is_absolute() else (DATA_ROOT / data_arg)
    data_session_dir = data_session_dir.resolve()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[PY] device:", device)

    # 1) Resolve sources based on calibsetting
    sources = list_window_csvs(data_session_dir, args.calibsetting)
    print(f"[PY] loading {len(sources)} eeg_windows.csv sources")

    # 2) Load data
    X, y_hz, info = load_windows_csv(sources)
    print("[PY] Loaded X:", X.shape, "y_hz:", y_hz.shape, "classes_hz:", info.classes_hz)

    # 3) Normalize (safe: per-window zscore)
    norm = NormalizerSpec()
    X = norm.apply(X).astype(np.float32)

    # 4) Select best pair using CV scoring with same arch
    best_left_hz, best_right_hz, debug = select_best_pair(X=X, y_hz=y_hz, info=info, args=args, device=device)
    print(f"[PY] BEST PAIR: {best_left_hz}Hz vs {best_right_hz}Hz")

    # 5) Train final model on winning pair
    Xp, yb = make_binary_pair_dataset(X, y_hz, best_left_hz, best_right_hz)

    if args.arch == "CNN":
        model, best_state, history = train_cnn(
            X=Xp,
            y_cls=yb,
            n_ch=info.n_ch,
            n_time=info.n_time,
            n_classes=2,
            seed=args.seed,
            val_ratio=args.val_ratio,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_epochs=args.max_epochs,
            patience=args.patience,
            min_delta=args.min_delta,
            device=device,
        )

        export_cnn_onnx(
            model=model,
            n_ch=info.n_ch,
            n_time=info.n_time,
            out_path=(out_dir / "ssvep_model.onnx"),
        )

    elif args.arch == "SVM":
        raise NotImplementedError("Wire SVM final training + export here.")

    # 6) Write train_result.json (contract consumed by C++)
    hz2e = hz_to_enum_mapping()
    if best_left_hz not in hz2e or best_right_hz not in hz2e:
        raise ValueError(f"Best pair ({best_left_hz},{best_right_hz}) not in hz_to_enum_mapping()")

    write_train_result_json(
        out_dir,
        ok=True,
        arch=args.arch,
        calibsetting=args.calibsetting,
        subject=args.subject,
        session=args.session,
        left_hz=best_left_hz,
        right_hz=best_right_hz,
        left_e=hz2e[best_left_hz],
        right_e=hz2e[best_right_hz],
        extra={
            "n_windows_total": int(X.shape[0]),
            "n_windows_pair": int(Xp.shape[0]),
            "pair_selection_metric": "mean_cv_balanced_accuracy",
            "cv_folds_requested": int(args.cv_folds),
            "candidate_freqs": debug.get("candidate_freqs"),
            "best_score_mean_bal_acc": debug.get("best_score_mean_bal_acc"),
            "pair_scores": debug.get("pair_scores"),
        },
    )

    return 0

# ENTRY POINT
if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"[PY] FATAL TRAINING ERROR: {e}", file=sys.stderr)
        sys.exit(1)
