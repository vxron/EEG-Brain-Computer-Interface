#!/usr/bin/env python3
"""
train_ssvep.py

Script for training an SSVEP model from calibration data.
This script is called automatically by the C++ backend after a calibration
session. It MUST match the CLI interface expected by C++. 
(see training manager thread in CapstoneProject.cpp for details)

- Loads windowed EEG from one or many eeg_windows.csv files.
- Selects BEST LEFT/RIGHT frequency pair using cross-validated scoring on model arch.
    --arch SVM : linear SVM
    --arch CNN : compact CNN (EEGNet) in PyTorch
- Trains final model on that binary pair.
- Exports ONNX model to <model_dir>/ssvep_model.onnx
- Writes train_result.json to <model_dir>/train_result.json

Expected args:
    --data <path>      directory containing calibration data
    --model <path>     directory where ONNX + meta.json should be written
    --arch <CNN|SVM>
    --calibsetting <all_sessions|most_recent_only>
    --tunehparams <on|off>
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Any
from dataclasses import dataclass, asdict

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold

import utils.utils as utils

# Trainers live in trainers/
from trainers.cnn_trainer import score_pair_cv_cnn, train_final_cnn_and_export
# from trainers.svm_trainer # TODO HADEEL

# ^ THOSE ARE THE "API" FUNCS WE USE from our trainers. 
# Here are more details on the signature they must take.
# 1) score_pair_cv_<arch>
#    - inputs: REQUIRED FOR ALL* 
#        - X_pair
#        - y_pair
#        - folds: premade cross-val folds for cross-val training (shared between archs)
#        - trial_ids_pair: (for trial-level batching if desired, otherwise unused)
#        - n_ch: spatial dimension 
#        - n_time: time dimension
#        - freq_a_hz
#        - freq_b_hz
#        - (...) any addtn args as required per specific model implementation
#    - return: ModelMetrics (REQUIRED FOR ALL* see utils dataclass)
# 2) train_final_<arch>_and_export
#    - inputs: REQUIRED FOR ALL* 
#        - X_pair
#        - y_pair
#        - folds: premade cross-val folds for FINAL cross-val training (for reporting purposes)
#        - trial_ids_pair: (for trial-level batching if desired, otherwise unused)
#        - n_ch: spatial dimension 
#        - n_time: time dimension
#        - out_onnx_path: Path
#        - hparam_tuning: whether or not we want to run hparam tuning
#        - (...) any addtn args as required per specific model implementation
#    - return: FinalTrainResults (REQUIRED FOR ALL* see utils dataclass)  

def validate_loaded_dataset(
    *,
    X: np.ndarray,
    y_hz: np.ndarray,
    trial_ids: np.ndarray,
    window_ids: np.ndarray,
) -> None:
    # Basic shape checks (aborts if failure)
    if X.ndim !=3:
        utils.abort("DATA", "X must be (N,C,T)", {"shape": list(X.shape)})
    N = int(X.shape[0])
    if N <= 0:
        utils.abort("DATA", "No windows loaded", "N=0")
    if len(y_hz) != N or len(trial_ids) != N or len(window_ids) != N:
        utils.abort("DATA", "length mismatch", f"N={N} y_hz={len(y_hz)} trial_ids={len(trial_ids)} window_ids={len(window_ids)}")
    if N < utils.MIN_TOTAL_WINDOWS:
        utils.abort("DATA", "Too few usable windows after loading", f"N={N} < MIN_TOTAL_WINDOWS={utils.MIN_TOTAL_WINDOWS}")
    # Needs at least 2 usable frequencies
    vals, counts = np.unique(y_hz, return_counts = True)
    n_freqs = int(len(vals))
    if n_freqs < utils.MIN_FREQS_FOR_PAIR_SEARCH:
        utils.abort("DATA", "Not enough frequencies in training data", f"Need at least {utils.MIN_FREQS_FOR_PAIR_SEARCH} frequencies to train a binary SSVEP model. Found {n_freqs}: {vals.tolist()}")

def validate_pair_dataset(
    *,
    hz_a: int,
    hz_b: int,
    Xp: np.ndarray,
    yb: np.ndarray,
    tp: np.ndarray,
    k_req: int, 
) -> tuple[bool, utils.TrainIssue | None]:
    """
    Returns (ok, reason)
    If not ok, caller can skip this pair and try next
    """
    N = int(len(yb))
    c0 = int((yb == 0).sum())
    c1 = int((yb == 1).sum())

    if N == 0:
        return False, utils.issue(
            "PAIR_SEARCH",
            "pair forms empty dataset",
            {"hz_a": hz_a, "hz_b": hz_b},
        )

    if c0 < utils.MIN_PAIR_WINDOWS_PER_CLASS or c1 < utils.MIN_PAIR_WINDOWS_PER_CLASS:
        return False, utils.issue(
            "PAIR_SEARCH",
            "Insufficient windows per class",
            {"hz_a": hz_a, "hz_b": hz_b, "c0": c0, "c1": c1, "min": utils.MIN_PAIR_WINDOWS_PER_CLASS}
        )
    
    uniq_trials = np.unique(tp)
    trials_by_class = {0: set(), 1: set()}
    for tid in uniq_trials:
        m = (tp == tid)
        lab = 1 if float(yb[m].mean()) >= 0.5 else 0 # majority vote
        trials_by_class[lab].add(int(tid))
    
    n_t0 = len(trials_by_class[0])
    n_t1 = len(trials_by_class[1])
    if n_t0 < utils.MIN_PAIR_TRIALS_PER_CLASS or n_t1 < utils.MIN_PAIR_TRIALS_PER_CLASS:
        return False, utils.issue(
            "PAIR_SEARCH",
            "Insufficient trials per class for pair",
            {"hz_a": hz_a, "hz_b": hz_b, "t0": n_t0, "t1": n_t1, "min": int(utils.MIN_PAIR_TRIALS_PER_CLASS)},
        )
    
    # demand at least k = 2 for cross fold
    if int(k_req) < utils.MIN_FOLDS_MIN:
        return False, utils.issue(
            "PAIR_SEARCH",
            "Requested fold count too small",
            {"k_req": int(k_req), "MIN_FOLDS_MIN": int(utils.MIN_FOLDS_MIN)},
        )
    
    return True, None

# ------------------------------
# DEBUG
# ------------------------------
def _summarize_blocked_folds(
    *,
    yb: np.ndarray,
    trial_ids: np.ndarray,
    groups: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    block_size_windows: int,
    hop_samples: int,
    n_time: int,
    logger: utils.DebugLogger,
    tag: str = "CV",
) -> None:
    yb = np.asarray(yb).astype(np.int64)
    trial_ids = np.asarray(trial_ids).astype(np.int64)
    groups = np.asarray(groups).astype(np.int64)

    N = int(len(yb))
    uniq_groups = np.unique(groups)

    logger.log(f"[{tag}] ===== Blocked-fold summary =====")
    logger.log(f"[{tag}] N_windows={N}")
    logger.log(f"[{tag}] n_time(W)={int(n_time)} samples")
    logger.log(f"[{tag}] hop_samples(H)={int(hop_samples)}")
    logger.log(f"[{tag}] block_size_windows=ceil(W/H)={int(block_size_windows)}")
    logger.log(f"[{tag}] n_groups={len(uniq_groups)}")
    logger.log(f"[{tag}] k_used={len(folds)}")
    logger.log()

    # group purity checks + build group->label and group->trial maps
    bad_label_groups = 0
    bad_trial_groups = 0
    group_to_label: dict[int, int] = {}
    group_to_trial: dict[int, int] = {}

    for g in uniq_groups:
        idx_g = np.where(groups == g)[0]
        labs = np.unique(yb[idx_g])
        trs = np.unique(trial_ids[idx_g])
        if len(labs) != 1:
            bad_label_groups += 1
        if len(trs) != 1:
            bad_trial_groups += 1
        # even if not pure, choose first for logging
        group_to_label[int(g)] = int(yb[idx_g[0]])
        group_to_trial[int(g)] = int(trial_ids[idx_g[0]])

    if bad_label_groups or bad_trial_groups:
        logger.log(
            f"[{tag}] WARNING: group purity failed | "
            f"bad_label_groups={bad_label_groups}, "
            f"bad_trial_groups={bad_trial_groups}"
        )
    else:
        logger.log(f"[{tag}] group purity OK ✓")

    logger.log()

    # overall class balance (windows)
    c0 = int((yb == 0).sum())
    c1 = int((yb == 1).sum())
    logger.log(f"[{tag}] overall class counts: c0={c0}, c1={c1}")
    logger.log()

    # per-fold stats
    for fi, (tr_idx, va_idx) in enumerate(folds):
        tr_idx = np.asarray(tr_idx, dtype=np.int64)
        va_idx = np.asarray(va_idx, dtype=np.int64)

        # window counts
        tr0 = int((yb[tr_idx] == 0).sum())
        tr1 = int((yb[tr_idx] == 1).sum())
        va0 = int((yb[va_idx] == 0).sum())
        va1 = int((yb[va_idx] == 1).sum())

        # leakage check: any group present in both train and val?
        tr_gset = set(groups[tr_idx].tolist())
        va_gset = set(groups[va_idx].tolist())
        inter = tr_gset.intersection(va_gset)

        logger.log(
            f"[{tag}] Fold {fi:02d}: "
            f"train N={len(tr_idx)} (c0={tr0}, c1={tr1}) | "
            f"val N={len(va_idx)} (c0={va0}, c1={va1}) | "
            f"leak_groups={len(inter)}"
        )

        if inter:
            logger.log(
                f"[{tag}]   ERROR: leakage detected! example groups={list(inter)[:5]}"
            )

        # group counts (by label), computed correctly PER FOLD
        tr_groups = np.array(sorted(tr_gset), dtype=np.int64)
        va_groups = np.array(sorted(va_gset), dtype=np.int64)
        tr_g0 = int(sum(group_to_label[int(g)] == 0 for g in tr_groups))
        tr_g1 = int(sum(group_to_label[int(g)] == 1 for g in tr_groups))
        va_g0 = int(sum(group_to_label[int(g)] == 0 for g in va_groups))
        va_g1 = int(sum(group_to_label[int(g)] == 1 for g in va_groups))

        logger.log(
            f"[{tag}]   groups: "
            f"train G={len(tr_groups)} (g0={tr_g0}, g1={tr_g1}) | "
            f"val G={len(va_groups)} (g0={va_g0}, g1={va_g1})"
        )
    logger.log(f"[{tag}] ===== End summary =====\n")

# ------------------------------
# PATH ROOTS (repo-anchored)
# ------------------------------
def find_repo_root(start: Path | None = None) -> Path:
    """
    Walk upward from `start` (default: this file) until we find a repo-root marker.
    """
    start = (start or Path(__file__)).resolve()
    markers = [
        ".git",            # directory
        "README.md",       # file
    ]
    for p in [start] + list(start.parents):
        for m in markers:
            if (p / m).exists():
                # If marker is .git, this still works (exists() checks dir)
                return p
    raise RuntimeError(f"Could not find repo root walking upward from {start}")

REPO_ROOT = find_repo_root()
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
        help="Passes user/session_dir.",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Output directory where ONNX + metadata will be saved.",
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

    parser.add_argument(
        "--tunehparams",
        type=str,
        required=True,
        choices=["ON", "OFF"],
        help="Choice of using default values for hyperparams (faster) or running explicit tuning algorithms (slower)."
    )
    
    return parser.parse_args()


# -----------------------------
# Shared Between All Trainers (CNN, SVM, ETC...) 
# Data Loading + Preprocessing
# -----------------------------
# Group rows by session folder name in addition to window_idx so we avoid collisions
@dataclass(frozen=True)
class CsvSource:
    src_id: str          # session folder name
    path: Path

def list_window_csvs(data_session_dir: Path, calibsetting: str) -> list[CsvSource]:
    """
    Handles "most recent" vs "all sessions" calib data settings
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

        sources.sort(key=lambda s: s.src_id)
        return sources

    utils.abort("INIT", "Unknown calibsetting", {"calibsetting": calibsetting})

def add_trial_ids_per_window(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds an integer trial_id per window based on contiguous runs of testfreq_hz within each _src csv.
    Trial increments when testfreq_hz changes (in WINDOW order).
    This is essential so that we can batch per trial instead of per window in CNN arch (to optimize learning steps with richer info).
    """
    # one row per window
    win = (
        df.groupby(["_src", "window_idx"], sort=False)
          .agg(testfreq_hz=("testfreq_hz", "first"))
          .reset_index()
    )
    # sort windows by time within each session using WINDOW_IDX
    win = win.sort_values(["_src", "window_idx"], kind="stable")

    # trial increments on frequency changes within each session
    def _assign_trials(g: pd.DataFrame) -> pd.DataFrame:
        freq = g["testfreq_hz"].to_numpy()
        # new trial when freq changes vs previous window
        new_trial = np.empty(len(freq), dtype=np.int64)
        new_trial[0] = 0
        for i in range(1, len(freq)):
            new_trial[i] = new_trial[i - 1] + (freq[i] != freq[i - 1])
        g = g.copy()
        g["trial_local"] = new_trial
        return g

    # Make trial ids globally unique ints across sessions
    win = win.groupby("_src", group_keys=False, sort=False).apply(_assign_trials)
    trial_key = list(zip(win["_src"].astype(str), win["trial_local"].astype(int)))
    win["trial_id"] = pd.factorize(trial_key)[0].astype(np.int64)

    # Merge trial_id back onto per-sample rows
    df = df.merge(win[["_src", "window_idx", "trial_id"]], on=["_src", "window_idx"], how="left")
    if df["trial_id"].isna().any():
        utils.abort("LOAD", "TRIAL ID merge fail", "trial_id merge failed for some rows")
    df["trial_id"] = df["trial_id"].astype(np.int64)
    return df

def log_window_idx_gaps(
    df: pd.DataFrame,
    *,
    logger: utils.DebugLogger | None = None,
    tag: str = "GAPCHECK",
) -> None:
    """
    - Checks for missing window_idx after filters (is_trimmed==1, is_bad!=1).
    - Needed for when we split into CV folds at window block level (overlapping windows being "forced" into same block)
    - Works per _src because window_idx restarts per session csv.
    - this is really PURELY FOR DEBUG
    """
    if "_src" not in df.columns:
        utils.abort("LOAD", "Missing src col", "df must contain _src column before gap checking")
    # one row per window
    win = (
        df.groupby(["_src", "window_idx"], sort=False)
          .size()
          .reset_index(name="n_rows")
          .sort_values(["_src", "window_idx"], kind="stable")
          .reset_index(drop=True)
    )
    # compute diffs within each _src
    win["jump"] = win.groupby("_src")["window_idx"].diff()
    gaps = win[win["jump"] > 1].copy()
    msg = (
        f"[{tag}] windows_kept={len(win)} | "
        f"gap_events(jump>1)={len(gaps)} | "
    )
    if logger is None:
        print(msg)
    else:
        logger.log(msg)

def load_windows_csv(sources: list[CsvSource]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, utils.DatasetInfo, list[utils.TrainIssue]]:
    """
    BUILD TRAINING DATA
    Reads window-level CSV(s) and returns:
      X: (N, C, T) = (num_windows, num_channels, window_len_samples)
      y_hz: (N,)
      trial_ids_np: single array with the trial ids for each window so that we can shuffle by trial if desired
      window_ids_np: window_idx (per-session idx, used to make overlap-safe window blocks for CV fold creation)
      info: DatasetInfo

    Expected columns:
      window_idx, is_trimmed, is_bad, sample_idx, eeg1..eeg8, testfreq_e, testfreq_hz
    """
    frames: list[pd.DataFrame] = []
    load_issues: list[utils.TrainIssue] = []
    required = {"window_idx", "is_trimmed", "is_bad", "sample_idx", "testfreq_hz", "testfreq_e"}

    for src in sources:
        if not src.path.exists():
            continue

        df = pd.read_csv(src.path)

        missing = required - set(df.columns)
        if missing:
            utils.abort(
                "LOAD",
                "CSV missing required columns",
                {"path": str(src.path), "missing": sorted(missing)},
            )

        # filters (keep only trimmed and not-bad)
        df = df[df["is_trimmed"] == 1].copy()
        df = df[df["is_bad"] != 1].copy()
        # tag source to prevent window_idx collisions across sessions
        df["_src"] = src.src_id

        # LOGGING (purely debug)
        log_window_idx_gaps(df, logger=None, tag=f"GAPCHECK {src.src_id}")
        # per-file read success + per-frequency window counts 
        n_rows = int(len(df))
        if n_rows == 0:
            load_issues.append(utils.issue(
                "LOAD",
                "eeg_windows.csv has 0 usable rows",
                {"session": src.src_id, "path": str(src.path)},
            ))
            continue
        else:
            # count distinct windows per frequency (NOT rows)
            per_win = (
                df.groupby(["window_idx"], sort=False)
                  .agg(testfreq_hz=("testfreq_hz", "first"))
                  .reset_index()
            )
            freq_counts = per_win["testfreq_hz"].value_counts().sort_index()
            freq_str = ", ".join([f"{int(hz)}Hz:{int(cnt)}" for hz, cnt in freq_counts.items()])
            n_windows = int(len(per_win))
            print(f"[PY] read OK: {src.src_id} -> {src.path} | windows={n_windows} | {freq_str}")

        frames.append(df)

    if not frames:
        utils.abort(
            "LOAD",
            "No usable eeg_windows.csv found for requested setting.",
            {"n_sources": len(sources)},
        )

    df = pd.concat(frames, ignore_index=True)

    # assign trial ids before grouping windows
    df = add_trial_ids_per_window(df)

    # Detect channel columns as the ones between sample_idx and testfreq_e
    # TODO: detect by name instead of between these columns, cuz it's too brittle if we change column arch
    cols = list(df.columns)
    sample_i = cols.index("sample_idx")
    tf_e_i = cols.index("testfreq_e")
    ch_cols = cols[sample_i + 1 : tf_e_i]
    n_ch = len(ch_cols)
    if n_ch <= 0:
        utils.abort("LOAD", "No EEG channel columns detected between sample_idx and testfreq_e.")

    windows: list[np.ndarray] = []
    labels_hz: list[int] = []
    trial_ids: list[int] = []
    window_ids: list[int] = []

    # group by (_src, window_idx) to avoid collisions
    grouped = df.groupby(["_src", "window_idx"], sort=True)
    target_T: int | None = None

    for (_src, wid), g in grouped:
        g = g.sort_values("sample_idx")

        tf_hz = int(g["testfreq_hz"].iloc[0])
        if tf_hz < 0:
            continue

        trial_id = int(g["trial_id"].iloc[0])

        x_tc = g[ch_cols].to_numpy(dtype=np.float32)  # (T, C)
        if x_tc.ndim != 2 or x_tc.shape[1] != n_ch:
            continue

        x_ct = x_tc.T  # (C, T)

        if not windows:
            target_T = x_ct.shape[1]
        else:
            target_T = windows[0].shape[1] # regular window length

        if int(x_ct.shape[1]) != int(target_T):
            utils.abort(
                "LOAD",
                "Window length mismatch",
                {"_src": str(_src), "window_idx": int(wid), "T": int(x_ct.shape[1]), "target_T": int(target_T)},
            )

        windows.append(x_ct)
        labels_hz.append(tf_hz)
        trial_ids.append(trial_id)
        window_ids.append(int(wid))

    if not windows:
        utils.abort("LOAD", "No valid windows found after grouping/filters")

    X = np.stack(windows, axis=0).astype(np.float32)  # (N, C, T)
    y_hz = np.array(labels_hz, dtype=np.int64)
    trial_ids_np = np.array(trial_ids, dtype=np.int64)
    window_ids_np = np.array(window_ids, dtype=np.int64)

    info = utils.DatasetInfo(
        ch_cols=ch_cols,
        n_ch=n_ch,
        n_time=int(X.shape[2]),
        classes_hz=sorted(set(y_hz.tolist())),
    )
    return X, y_hz, trial_ids_np, window_ids_np, info, load_issues

# -----------------------------
# CV Fold building
# -----------------------------

def _count_groups_per_class(
    *,
    yb: np.ndarray,
    groups: np.ndarray,
) -> tuple[int, int, np.ndarray, np.ndarray]:
    """
    Returns:
      n_groups_c0, n_groups_c1,
      uniq_groups (sorted),
      group_label aligned with uniq_groups (0/1)
    Assumes each group is label-pure (we still verify elsewhere).
    """
    yb = np.asarray(yb).astype(np.int64)
    groups = np.asarray(groups).astype(np.int64)

    uniq_groups = np.unique(groups)
    group_label = np.empty(len(uniq_groups), dtype=np.int64)

    for gi, g in enumerate(uniq_groups):
        idx_g = np.where(groups == g)[0]
        if idx_g.size == 0:
            utils.abort("FOLDS", "Empty group encountered (should be impossible)")
        group_label[gi] = int(yb[idx_g[0]])

    n_groups_c0 = int((group_label == 0).sum())
    n_groups_c1 = int((group_label == 1).sum())
    return n_groups_c0, n_groups_c1, uniq_groups, group_label

def _validate_folds_strict(
    *,
    yb: np.ndarray,
    groups: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[bool, list[utils.TrainIssue]]:
    """
    Strict validation:
      - at least 2 folds
      - no group leakage between train and val
      - each fold has both classes in val AND train
    Returns (ok, reasons)
    """
    yb = np.asarray(yb).astype(np.int64)
    groups = np.asarray(groups).astype(np.int64)
    issues: list[utils.TrainIssue] = []

    if len(folds) < 2:
        issues.append(utils.issue("FOLDS", "len(folds) < 2", {"n_folds": int(len(folds))}))
        return False, issues

    for fi, (tr_idx, va_idx) in enumerate(folds):
        tr_idx = np.asarray(tr_idx, dtype=np.int64)
        va_idx = np.asarray(va_idx, dtype=np.int64)

        # group leakage check
        tr_g = set(groups[tr_idx].tolist())
        va_g = set(groups[va_idx].tolist())
        inter = tr_g.intersection(va_g)
        if inter:
            issues.append(utils.issue(
                "FOLDS",
                "Group leakage between train and val",
                {"fold": int(fi), "n_leak_groups": int(len(inter))},
            ))
            continue

        # class presence check (val)
        va0 = int((yb[va_idx] == 0).sum())
        va1 = int((yb[va_idx] == 1).sum())
        if va0 == 0 or va1 == 0:
            issues.append(utils.issue(
                "FOLDS",
                "Validation split has only one class",
                {"fold": int(fi), "va0": va0, "va1": va1},
            ))

        # class presence check (train)
        tr0 = int((yb[tr_idx] == 0).sum())
        tr1 = int((yb[tr_idx] == 1).sum())
        if tr0 == 0 or tr1 == 0:
            issues.append(utils.issue(
                "FOLDS",
                "Training split has only one class",
                {"fold": int(fi), "tr0": tr0, "tr1": tr1},
            ))

        # size sanity
        if tr_idx.size == 0 or va_idx.size == 0:
            issues.append(utils.issue(
                "FOLDS",
                "Fold has empty train or val split",
                {"fold": int(fi), "train_size": int(tr_idx.size), "val_size": int(va_idx.size)},
            ))

    ok = (len(issues) == 0)
    return ok, issues

def make_cv_folds_binary_by_blocked_windows(
    *,
    yb: np.ndarray,
    trial_ids: np.ndarray,
    window_ids: np.ndarray,
    k: int,
    seed: int,
    n_time: int,
    hop_samples: int,
    fs_hz: float = 250.0,
    debug_logger: utils.DebugLogger | None = None,
    debug_tag: str = "CV",
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[utils.TrainIssue]]:
    """
    Stratified CV on *blocked window groups*:
      - windows close in time (overlapping) are grouped into the same "block" group
        block_size_windows ~= ceil(W/H) where W=n_time and H=hop_samples
      - we then do StratifiedGroupKFold so each fold has both classes, but blocks
        never get split across train/val.
    This is desired compromise:
      random-ish windows in each fold, but overlap leakage is prevented because
      overlapping neighbors live in the same group.
    
    Steps:
      1) Build blocked groups (overlap-safe) within trials.
      2) Compute k upper bound based on groups-per-class.
      3) For k_try from k_eff down to 2:
            try a fixed seed list (deterministic),
            build StratifiedGroupKFold splits,
            validate strictly.
         Return first valid folds.
      4) If nothing valid, return [].

    Guarantees (if returns non-empty):
      - leak_groups=0 in every fold
      - each fold val has both classes
      - each fold train has both classes
    """
    issues: list[utils.TrainIssue] = []
    split_issues: list[utils.TrainIssue] = []
    yb = np.asarray(yb).astype(np.int64)
    trial_ids = np.asarray(trial_ids).astype(np.int64)
    window_ids = np.asarray(window_ids).astype(np.int64)
    if yb.shape[0] != window_ids.shape[0]:
        utils.abort("FOLDS", "yb and window_ids must have same length",
                    {"len_yb": int(yb.shape[0]), "len_window_ids": int(window_ids.shape[0])})
    if yb.shape[0] != trial_ids.shape[0]:
        utils.abort("FOLDS", "yb and trial_ids must have same length",
                    {"len_yb": int(yb.shape[0]), "len_trial_ids": int(trial_ids.shape[0])})

    # 1) Build block groups
    block_size_windows = _compute_block_size_windows(
        n_time=int(n_time),
        hop_samples=int(hop_samples),
    )
    groups = make_block_groups_within_trials_by_window_idx(
        trial_ids=trial_ids,
        yb=yb,
        window_ids=window_ids,
        block_size_windows=int(block_size_windows),
    )

    # 2) Compute k bound from groups-per-class
    n_groups_c0, n_groups_c1, uniq_groups, group_label = _count_groups_per_class(
        yb=yb, groups=groups
    )
    k_req = int(k)
    k_eff = int(min(k_req, n_groups_c0, n_groups_c1))

    if debug_logger is not None:
        debug_logger.log(f"[{debug_tag}] ===== Fold builder (blocked groups) =====")
        debug_logger.log(f"[{debug_tag}] N_windows={len(yb)}")
        debug_logger.log(f"[{debug_tag}] n_time={int(n_time)} hop_samples={int(hop_samples)} fs_hz={float(fs_hz)}")
        debug_logger.log(f"[{debug_tag}] block_size_windows=ceil(W/H)={block_size_windows}")
        debug_logger.log(f"[{debug_tag}] n_groups_total={len(uniq_groups)}")
        debug_logger.log(f"[{debug_tag}] n_groups_c0={n_groups_c0}, n_groups_c1={n_groups_c1}")
        debug_logger.log(f"[{debug_tag}] k requested={k_req} -> k_eff (group-limited)={k_eff}")
        debug_logger.log()

    if k_eff < 2:
        issues.append(utils.issue(
            "FOLDS",
            "Not enough groups per class to build >=2 folds",
            {"k_req": k_req, "k_eff": k_eff, "n_groups_c0": n_groups_c0, "n_groups_c1": n_groups_c1},
        ))
        return [], issues

    # Deterministic seed schedule:
    # - include the provided seed first
    # - then a fixed list of additional seeds
    seed_base = int(seed)
    seed_list = [seed_base, 0, 1, 2, 3, 5, 7, 11, 13, 17]
    # remove duplicates while preserving order
    seen = set()
    seed_list = [s for s in seed_list if not (s in seen or seen.add(s))]

    idx = np.arange(len(yb), dtype=np.int64)

    # 3) Retry + degrade loop
    for k_try in range(k_eff, 1, -1):
        if debug_logger is not None:
            debug_logger.log(f"[{debug_tag}] Trying k_try={k_try} with seeds={seed_list}")

        for s in seed_list:
            sgkf = StratifiedGroupKFold(n_splits=int(k_try), shuffle=True, random_state=int(s))
            folds: list[tuple[np.ndarray, np.ndarray]] = []

            try:
                for tr_idx, va_idx in sgkf.split(idx, yb, groups):
                    folds.append((tr_idx.astype(np.int64), va_idx.astype(np.int64)))
            except Exception as e:
                split_issues.append(utils.issue(
                    "FOLDS",
                    "Exception during StratifiedGroupKFold split",
                    {"seed": int(s), "k_try": int(k_try), "err": str(e)},
                ))
                continue

            ok, reasons = _validate_folds_strict(yb=yb, groups=groups, folds=folds)

            if debug_logger is not None:
                debug_logger.log(f"[{debug_tag}]   seed={s}: built {len(folds)} folds -> ok={ok}")
                if not ok:
                    for r in reasons[:6]:
                        debug_logger.log(f"[{debug_tag}]     issue: {r.stage} | {r.message}")
                    if len(reasons) > 6:
                        debug_logger.log(f"[{debug_tag}]     ... ({len(reasons)-6} more)")

            if ok:
                # 4) Summarize the successful folds in detail
                if debug_logger is not None:
                    _summarize_blocked_folds(
                        yb=yb,
                        trial_ids=trial_ids,
                        groups=groups,
                        folds=folds,
                        block_size_windows=int(block_size_windows),
                        hop_samples=int(hop_samples),
                        n_time=int(n_time),
                        logger=debug_logger,
                        tag=debug_tag,
                    )
                    debug_logger.log(f"[{debug_tag}] DONE: using k={k_try}, seed={s}")
                    debug_logger.log(f"[{debug_tag}] ========================================\n")
                return folds, []
            
            for it in reasons:
                split_issues.append(it)

        if debug_logger is not None:
            debug_logger.log(f"[{debug_tag}] No valid folds for k_try={k_try}. Degrading k...\n")

    issues.append(utils.issue(
        "FOLDS",
        "Could not produce valid folds for any k>=2",
        {"k_eff": int(k_eff), "k_req": int(k_req)},
    ))

    # Keep only a limited number to avoid giant JSON
    issues.extend(split_issues[:25])
    return [], issues


def _compute_block_size_windows(
    *,
    n_time: int,
    hop_samples: int,
) -> int:
    """
    Block size (in windows) ~ ceil(W/H) where:
      W = window_len_samples = n_time
      H = hop_samples
    This groups windows that overlap in raw EEG samples into the same block.
    """
    if hop_samples <= 0:
        utils.abort("FOLDS", "hop_samples must be > 0", {"hop_samples": int(hop_samples)})
    if n_time <= 0:
        utils.abort("FOLDS", "n_time must be > 0", {"n_time": int(n_time)})
    # fs_hz isn't mathematically required for W/H, but we keep it in signature
    # because you'll likely want to log/validate it.
    block = int(np.ceil(float(n_time) / float(hop_samples)))
    return max(1, block)

def make_block_groups_within_trials_by_window_idx(
    *,
    trial_ids: np.ndarray,
    yb: np.ndarray,
    window_ids: np.ndarray,
    block_size_windows: int,
) -> np.ndarray:
    """
    Goal: 
    - assign each window to a "group ID" so StratifiedGroupKFold can keep overlapping windows together.
    - WHY?? PREVENT LEAKAGE.
    
    Group key = (trial_id, label, block_bin), where:
        block_bin = floor(window_idx / block_size_windows) i.e. overlapping chunk index
        label = test frequency for that block
        trial_id = monotonic counter value for all groups
    Example: tuple (12, 0, 3) means trial 12, class 0, block #3: treated as one inseparable group during stratification.

      - this assumes the incoming arrays are already in time order (which they are from load_windows_csv)
      - groups never cross trial/label boundaries
      - within each (trial_id, label, block_bin) segment, consecutive windows are chunked into blocks
        of size block_size_windows
      - this supports cases when some window_idx are missing due to artifacts
    """
    trial_ids = np.asarray(trial_ids).astype(np.int64)
    yb = np.asarray(yb).astype(np.int64)
    window_ids = np.asarray(window_ids).astype(np.int64)

    if not (len(trial_ids) == len(yb) == len(window_ids)):
        utils.abort(
            "FOLDS",
            "trial_ids, yb, window_ids must have same length",
            {"len_trial_ids": int(len(trial_ids)), "len_yb": int(len(yb)), "len_window_ids": int(len(window_ids))},
        )

    # this bins each window into a chunk index, where chunks are size block_size_windows
    # and where block_size_windows = ceil (W/H), answering how many consecutive windows are overlapping
    block_bin = (window_ids // int(block_size_windows)).astype(np.int64) 
    # group nearby-in-time windows within the trial into overlap-safe chunks
    # build a composit key for every window
    key = list(zip(trial_ids.tolist(), yb.tolist(), block_bin.tolist()))
    groups = pd.factorize(key)[0].astype(np.int64)
    return groups

# -----------------------------
# Best Pair Selection Logic
# -----------------------------
def make_binary_pair_dataset(X: np.ndarray, y_hz: np.ndarray, trial_ids: np.ndarray, window_ids: np.ndarray, hz_a: int, hz_b: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Filters to only windows in {hz_a, hz_b} and returns:
      Xp: (Npair, C, T)
      yb: (Npair,) where hz_a -> 0, hz_b -> 1
    """
    mask = (y_hz == hz_a) | (y_hz == hz_b)
    Xp = X[mask]
    yp = y_hz[mask]
    tp = trial_ids[mask]
    wp = window_ids[mask]
    yb = np.where(yp == hz_a, 0, 1).astype(np.int64)
    return Xp, yb, tp, wp

def shortlist_freqs(y_hz: np.ndarray, pick_top_k: int) -> list[int]:
    """
    Keep only frequencies with enough windows, then take the top-K by count.
    """
    k_folds = 5 # REQUIRE AT LEAST K_FOLDS WINDOWS PER CLASS (bare minimum) and 20 for reasonability
    min_windows_per_class = max(utils.MIN_WINDOWS_PER_FREQ_FOR_SHORTLIST, int(k_folds * 2))
    vals, counts = np.unique(y_hz, return_counts=True)
    keep = [(int(v), int(c)) for v, c in zip(vals, counts) if int(c) >= int(min_windows_per_class)]
    if len(keep) < 2:
        return []

    # Sort by count desc (more data = more stable), then by freq asc
    keep.sort(key=lambda t: (-t[1], t[0]))
    freqs = [hz for hz, _ in keep]
    return freqs[: min(pick_top_k, len(freqs))]

def select_best_pair(
    *,
    X: np.ndarray, y_hz: np.ndarray, trial_ids: np.ndarray,
    info: utils.DatasetInfo,
    gen_cfg: utils.GeneralTrainingConfigs,
    window_ids: np.ndarray,
    debug_logger: utils.DebugLogger | None = None,
    args,
) -> tuple[int, int, dict[str, Any]]:
    """
    Returns: (best_left_hz, best_right_hz, debug_info)
    Scoring metric: mean CV balanced accuracy.
    """
    pair_issues: list[utils.TrainIssue] = [] # error catching
    cand_freqs = shortlist_freqs(
        y_hz,
        pick_top_k=6,
    )
    pairs = [(cand_freqs[i], cand_freqs[j]) for i in range(len(cand_freqs)) for j in range(i + 1, len(cand_freqs))]

    best_metrics = None
    best_score = -1.0
    all_metrics = []
    print(f"[PY] Pair search candidates: freqs={cand_freqs} -> {len(pairs)} pairs, arch={args.arch}")

    for hz_a, hz_b in pairs:

        # ====== 1) build binary dataset for this candidate pair =====
        Xp, yb, tp, wp = make_binary_pair_dataset(X, y_hz, trial_ids, window_ids, hz_a, hz_b)

        ok, issue = validate_pair_dataset(
            hz_a=hz_a,
            hz_b=hz_b,
            Xp=Xp,
            yb=yb,
            tp=tp,
            k_req=int(gen_cfg.number_cross_val_folds),
        )
        if not ok:
            if issue is not None:
                pair_issues.append(utils.issue(
                "PAIR_SEARCH",
                issue.message,
                {
                    "hz_a": int(hz_a),
                    "hz_b": int(hz_b),
                    "n_pair": int(len(yb)),
                    "inner": utils.issue_to_dict(issue) if hasattr(utils, "issue_to_dict") else asdict(issue),
                },
            ))
            else:
                pair_issues.append(utils.issue(
                    "PAIR_SEARCH", "Pair dataset invalid for unknown reason",
                    {"hz_a": int(hz_a), "hz_b": int(hz_b), "n_pair": int(len(yb))}
                ))
            continue

        k = int(gen_cfg.number_cross_val_folds)
        folds, fold_issues = make_cv_folds_binary_by_blocked_windows(
            yb=yb,
            trial_ids=tp,
            window_ids=wp,
            k=k,
            seed=0,
            n_time=info.n_time,    # window length in samples
            hop_samples=utils.HOP_SAMPLES,
            fs_hz=250.0,
            debug_logger=debug_logger,
            debug_tag=f"PAIR {hz_a}vs{hz_b}",
        )
        if len(folds) < 2:
            pair_issues.append(utils.issue(
                "PAIR_SEARCH",
                "Fold build failed for pair",
                {"hz_a": int(hz_a), "hz_b": int(hz_b), "n_fold_issues": int(len(fold_issues))},
            ))
            pair_issues.extend(fold_issues)
            continue

        # ===== 2) score the pair using trainer API =====
        if args.arch == "CNN":
            metrics, score_issues = score_pair_cv_cnn(
                X_pair=Xp,
                y_pair=yb,
                trial_ids_pair=tp,
                folds=folds,
                n_ch=info.n_ch,
                n_time=info.n_time,
                freq_a_hz=hz_a,
                freq_b_hz=hz_b,
                logger=debug_logger,
            )
            pair_issues.extend(score_issues)
        elif args.arch == "SVM":
            metrics = utils.ModelMetrics(0, -99, -99)
            print("hadeel todo for svm")
        else:
            utils.abort("PAIR_SEARCH", "Unknown model train architecture", {"arch": str(args.arch)})

        all_metrics.append(metrics)

        if not metrics.cv_ok:
            print(f"[PY] pair ({hz_a},{hz_b}) skipped due to cv error")
            pair_issues.append(utils.issue(
                "PAIR_SEARCH",
                "CV failed for pair",
                {"hz_a": hz_a, "hz_b": hz_b}
            ))
            continue

        score = metrics.avg_fold_balanced_accuracy # current scoring metric we're using
        std_score = metrics.std_fold_balanced_accuracy
        print(f"[PY] pair ({hz_a},{hz_b}) mean_bal_acc={score:.3f} (+/-{std_score:.3f})")

        if score > best_score:
            best_score = score
            best_metrics = metrics

    if best_metrics is None:
        utils.abort(
            "PAIR_SEARCH",
            "Failed to select any valid pair (all candidates skipped).",
            {"cand_freqs": cand_freqs, "n_pairs": int(len(pairs)), "skip_count": int(len(pair_issues))},
        )

    a = best_metrics.freq_a_hz
    b = best_metrics.freq_b_hz

    # Make left/right deterministic
    # Convention: left = lower Hz, right = higher Hz
    left_hz, right_hz = (a, b) if a < b else (b, a)

    debug = {
        "candidate_freqs": cand_freqs,
        "pair_scores": [asdict(m) for m in all_metrics], # dictionary format for JSON serialization
        "best_score_mean_bal_acc": best_score,
        "pair_skip_reasons": utils.issues_to_json(pair_issues),
    }
    return left_hz, right_hz, debug


# -----------------------------
# JSON Contract & ONNX Export
# -----------------------------
def write_train_result_json(model_dir: Path, *, train_ok: bool, onnx_ok: bool, cv_ok: bool, final_holdout_ok: bool, arch: str, calibsetting: str,
                            left_hz: int, right_hz: int,
                            left_e: int, right_e: int,
                            final_holdout_bal_acc: float = 0.0, final_train_acc: float = 0.0,
                            issues: list[utils.TrainIssue] | None = None,
                            extra: dict[str, Any] | None = None) -> Path:
    payload = {
        "train_ok": bool(train_ok),
        "onnx_ok": bool(onnx_ok),
        "cv_ok": bool(cv_ok),
        "final_holdout_ok": bool(final_holdout_ok),
        "arch": arch,
        "calibsetting": calibsetting,
        "best_freq_left_hz": int(left_hz),
        "best_freq_right_hz": int(right_hz),
        "best_freq_left_e": int(left_e),
        "best_freq_right_e": int(right_e),
        "final_holdout_acc": float(final_holdout_bal_acc),
        "final_train_acc": float(final_train_acc),
        "issues": utils.issues_to_json(issues or [])
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
        13: 6,
        14: 7,
        15: 8,
        16: 9,
        17: 10,
        18: 11,
        20: 12,
        25: 13,
        30: 14,
        35: 15,
        -1: 99
    }


# -----------------------------
# Orchestrator (MAIN)
# -----------------------------
def main():
    args = get_args()

    issues: list[utils.TrainIssue] = [] # should contain utils.TrainIssue

    # 0) INITS (default final results)
    best_left_hz = -1
    best_right_hz = -1
    final = utils.FinalTrainResults()
    X = None
    Xp = None
    debug: dict[str, Any] = {}

    # 1) RESOLVE PATHS FROM ARGS
    out_dir_arg = Path(args.model)
    out_dir = out_dir_arg if out_dir_arg.is_absolute() else (MODELS_ROOT / out_dir_arg)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[PY] model output dir:", out_dir.resolve())

    # init centralized debug logger
    debug_log = utils.DebugLogger(out_dir / "train_ssvep_debug.txt")
    print(f"[PY] CV debug log -> {debug_log.path.resolve()}")

    data_arg = Path(args.data)
    data_session_dir = data_arg if data_arg.is_absolute() else (DATA_ROOT / data_arg)
    data_session_dir = data_session_dir.resolve()
    hparam_arg = args.tunehparams
    
    gen_cfg = utils.GeneralTrainingConfigs()

    try:
        # 2) Resolve sources based on calibsetting
        sources = list_window_csvs(data_session_dir, args.calibsetting)
        print(f"[PY] loading {len(sources)} eeg_windows.csv sources")

        # 3) Load data
        X, y_hz, trial_ids, window_ids, info, load_issues = load_windows_csv(sources)
        issues.extend(load_issues)
        print("[PY] Loaded X:", X.shape, "y_hz:", y_hz.shape, "classes_hz:", info.classes_hz)
        validate_loaded_dataset(
            X=X,
            y_hz=y_hz,
            trial_ids=trial_ids,
            window_ids=window_ids,
        )

        # 4) Select best pair using CV scoring with same arch
        best_left_hz, best_right_hz, debug = select_best_pair(X=X, y_hz=y_hz, trial_ids=trial_ids, window_ids=window_ids, info=info, gen_cfg=gen_cfg, debug_logger=debug_log, args=args)
        print(f"[PY] BEST PAIR: {best_left_hz}Hz vs {best_right_hz}Hz")

        # 5) Train final model on winning pair
        Xp, yb, tp, wp = make_binary_pair_dataset(X, y_hz, trial_ids, window_ids, best_left_hz, best_right_hz)

        # 6) Validate final folds
        ok, reason = validate_pair_dataset(
            hz_a=best_left_hz,
            hz_b=best_right_hz,
            Xp=Xp,
            yb=yb,
            tp=tp,
            k_req=int(gen_cfg.number_cross_val_folds),
        )
        if not ok:
            utils.abort(
                "FINAL",
                "Winning pair failed sanity checks",
                {"hz_a": int(best_left_hz), "hz_b": int(best_right_hz), "inner": utils.issue_to_dict(reason) if reason else None},
            )

        # Build folds for the winning pair (used for final CV reporting)
        c0 = int((yb == 0).sum())
        c1 = int((yb == 1).sum())
        k_final = int(gen_cfg.number_cross_val_folds)
        folds_final: list[tuple[np.ndarray, np.ndarray]] = []
        fold_issues: list[utils.TrainIssue] = []
        if c0 >= 2 and c1 >= 2:
            folds_final, fold_issues = make_cv_folds_binary_by_blocked_windows(
                yb=yb,
                trial_ids=tp,
                window_ids=wp,
                k=k_final,
                seed=0,
                n_time=info.n_time,
                hop_samples=utils.HOP_SAMPLES,
                fs_hz=250.0,
                debug_logger=debug_log,
                debug_tag=f"FINAL {best_left_hz}vs{best_right_hz}",
            )
        if len(folds_final) < 2:
            fold_issues.append(utils.issue(
                "FINAL",
                "Final folds not usable; CV metrics may be zero",
                {"k_final": int(gen_cfg.number_cross_val_folds), "n_folds": int(len(folds_final)), "n_fold_issues": int(len(fold_issues))},
            ))
        
        issues.extend(fold_issues)

        if args.arch == "CNN":
            final, cnn_issues = train_final_cnn_and_export(
                X_pair=Xp,
                y_pair=yb,
                trial_ids_pair=tp,
                folds=folds_final,
                n_ch=info.n_ch,
                n_time=info.n_time,
                out_onnx_path=(out_dir / "ssvep_model.onnx"),
                hparam_tuning=hparam_arg,
                logger=debug_log,
            )
            issues.extend(cnn_issues)

        elif args.arch == "SVM":
            raise NotImplementedError("Wire SVM final training + export here.")

        # 7) Write train_result.json (contract consumed by C++)
        hz2e = hz_to_enum_mapping()
        if best_left_hz not in hz2e or best_right_hz not in hz2e:
            utils.abort(
                "FINALIZE",
                "Best pair not in hz_to_enum_mapping()",
                {"best_left_hz": int(best_left_hz), "best_right_hz": int(best_right_hz)},
            )

        write_train_result_json(
            out_dir,
            train_ok=final.train_ok,
            onnx_ok=final.onnx_export_ok,
            cv_ok=final.cv_ok,
            final_holdout_ok=final.final_holdout_ok,
            arch=args.arch,
            calibsetting=args.calibsetting,
            left_hz=best_left_hz,
            right_hz=best_right_hz,
            left_e=hz2e[best_left_hz],
            right_e=hz2e[best_right_hz],
            final_holdout_bal_acc=final.final_holdout_bal_acc,
            final_train_acc=final.final_train_acc,
            issues=issues,
            extra={
                "n_windows_total": int(X.shape[0]) if X is not None else 0,
                "n_windows_pair": int(Xp.shape[0]) if Xp is not None else 0,
                "pair_selection_metric": "mean_cv_balanced_accuracy",
                "candidate_freqs": debug.get("candidate_freqs"),
                "best_score_mean_bal_acc": debug.get("best_score_mean_bal_acc"),
                "pair_scores": debug.get("pair_scores"),
                "pair_skip_reasons": debug.get("pair_skip_reasons"),
            },
        )
        return 0
    
    # Known failures for UI reporting
    except utils.TrainAbort as ta:
        issues.append(ta.issue)

    # Unexpected crashes
    except Exception as e:
        issues.append(utils.issue("FATAL", "UNHANDLED_EXCEPTION", str(e)))

    # unified failure write
    hz2e = hz_to_enum_mapping()
    left_e=hz2e.get(best_left_hz, -1),
    right_e=hz2e.get(best_right_hz, -1),

    write_train_result_json(
        out_dir,
        train_ok=False,
        onnx_ok=False,
        cv_ok=False,
        final_holdout_ok=False,
        arch=args.arch,
        calibsetting=args.calibsetting,
        left_hz=best_left_hz,
        right_hz=best_right_hz,
        left_e=left_e,
        right_e=right_e,
        issues=issues,
        extra={
            "fatal_stage": issues[-1].stage if issues else "UNKNOWN",
            "n_windows_total": int(X.shape[0]) if X is not None else 0,
        },
    )
    return 1


# ENTRY POINT
if __name__ == "__main__":
    sys.exit(main())

