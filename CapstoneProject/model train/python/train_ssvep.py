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
    --zscorenormalization <on|off>
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
from collections import defaultdict
from sklearn.model_selection import StratifiedGroupKFold
import time

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
    y_hz = y_hz[y_hz != -1] # rest doesn't count
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
    c2 = int((yb == 2).sum())

    if N == 0:
        return False, utils.issue(
            "PAIR_SEARCH",
            "pair forms empty dataset",
            {"hz_a": hz_a, "hz_b": hz_b},
        )

    if c0 < utils.MIN_PAIR_WINDOWS_PER_CLASS:
        return False, utils.issue(
            "PAIR_SEARCH",
            "Insufficient windows in class 0 (Note, given order of checks, other freq classes may be missing data as well)",
            {"hz_a": hz_a, "hz_b": hz_b, "c0": c0, "c1": c1, "c2": c2, "min": utils.MIN_PAIR_WINDOWS_PER_CLASS},
            data_insufficiency={
                    "frequency_hz": hz_a,
                    "metric": "windows",
                    "required": utils.MIN_PAIR_WINDOWS_PER_CLASS,
                    "actual": c0,
            }
        )
    
    if c1 < utils.MIN_PAIR_WINDOWS_PER_CLASS:
        return False, utils.issue(
            "PAIR_SEARCH",
            "Insufficient windows in class 1 (Note, given order of checks, rest class may also be missing wins)",
            {"hz_a": hz_a, "hz_b": hz_b, "c0": c0, "c1": c1, "c2": c2, "min": utils.MIN_PAIR_WINDOWS_PER_CLASS},
            data_insufficiency={
                    "frequency_hz": hz_b,
                    "metric": "windows",
                    "required": utils.MIN_PAIR_WINDOWS_PER_CLASS,
                    "actual": c1,
            }
        )
    
    if c2 < utils.MIN_PAIR_WINDOWS_PER_CLASS:
        return False, utils.issue(
            "PAIR_SEARCH",
            "Insufficient windows in rest class (Note this means freqs a & b have enough wins)",
            {"hz_a": hz_a, "hz_b": hz_b, "c0": c0, "c1": c1, "c2": c2, "min": utils.MIN_PAIR_WINDOWS_PER_CLASS},
            data_insufficiency={
                    "frequency_hz": -1,
                    "metric": "windows",
                    "required": utils.MIN_PAIR_WINDOWS_PER_CLASS,
                    "actual": c2,
            }
        )
    
    uniq_trials = np.unique(tp)
    trials_by_class = {0: set(), 1: set(), 2: set()}
    for tid in uniq_trials:
        desidx = np.where(tp==tid)[0] # global idx for this trial id
        classnum = yb[desidx][0] # protocol guarantees trial purity
        trials_by_class[classnum].add(int(tid))
    
    n_t0 = len(trials_by_class[0])
    n_t1 = len(trials_by_class[1])
    n_t2 = len(trials_by_class[2])
    if n_t0 < utils.MIN_PAIR_TRIALS_PER_CLASS:
        return False, utils.issue(
            "PAIR_SEARCH",
            "Insufficient trials in freq class 0",
            {"hz_a": hz_a, "hz_b": hz_b, "t0": n_t0, "t1": n_t1, "t2": n_t2, "min": int(utils.MIN_PAIR_TRIALS_PER_CLASS)},
            data_insufficiency={
                        "frequency_hz": hz_a,
                        "metric": "trials",
                        "required": utils.MIN_PAIR_TRIALS_PER_CLASS,
                        "actual": n_t0,
            }
        )
    if n_t1 < utils.MIN_PAIR_TRIALS_PER_CLASS:
        return False, utils.issue(
            "PAIR_SEARCH",
            "Insufficient trials in freq class 1",
            {"hz_a": hz_a, "hz_b": hz_b, "t0": n_t0, "t1": n_t1, "t2": n_t2, "min": int(utils.MIN_PAIR_TRIALS_PER_CLASS)},
            data_insufficiency={
                        "frequency_hz": hz_b,
                        "metric": "trials",
                        "required": utils.MIN_PAIR_TRIALS_PER_CLASS,
                        "actual": n_t1,
            }
        )
    if n_t2 < utils.MIN_PAIR_TRIALS_PER_CLASS:
        return False, utils.issue(
            "PAIR_SEARCH",
            "Insufficient trials in rest class",
            {"hz_a": hz_a, "hz_b": hz_b, "t0": n_t0, "t1": n_t1, "t2": n_t2, "min": int(utils.MIN_PAIR_TRIALS_PER_CLASS)},
            data_insufficiency={
                        "frequency_hz": -1,
                        "metric": "trials",
                        "required": utils.MIN_PAIR_TRIALS_PER_CLASS,
                        "actual": n_t2,
            }
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
def summarize_blocked_folds(
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
    c2 = int((yb == 2).sum())
    logger.log(f"[{tag}] overall class counts: c0={c0}, c1={c1}, c2={c2}")
    logger.log()

    # per-fold stats
    for fi, (tr_idx, va_idx) in enumerate(folds):
        tr_idx = np.asarray(tr_idx, dtype=np.int64)
        va_idx = np.asarray(va_idx, dtype=np.int64)

        # window counts
        tr0 = int((yb[tr_idx] == 0).sum())
        tr1 = int((yb[tr_idx] == 1).sum())
        tr2 = int((yb[tr_idx] == 2).sum())
        va0 = int((yb[va_idx] == 0).sum())
        va1 = int((yb[va_idx] == 1).sum())
        va2 = int((yb[va_idx] == 2).sum())

        # leakage check: any group present in both train and val?
        tr_gset = set(groups[tr_idx].tolist())
        va_gset = set(groups[va_idx].tolist())
        inter = tr_gset.intersection(va_gset)

        logger.log(
            f"[{tag}] Fold {fi:02d}: "
            f"train N={len(tr_idx)} (c0={tr0}, c1={tr1}, c2={tr2}) | "
            f"val N={len(va_idx)} (c0={va0}, c1={va1}, c2={va2}) | "
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
        tr_g2 = int(sum(group_to_label[int(g)] == 2 for g in tr_groups))
        va_g0 = int(sum(group_to_label[int(g)] == 0 for g in va_groups))
        va_g1 = int(sum(group_to_label[int(g)] == 1 for g in va_groups))
        va_g2 = int(sum(group_to_label[int(g)] == 2 for g in va_groups))

        logger.log(
            f"[{tag}]   groups: "
            f"train G={len(tr_groups)} (g0={tr_g0}, g1={tr_g1}, g2={tr_g2}) | "
            f"val G={len(va_groups)} (g0={va_g0}, g1={va_g1}, g2={va_g2})"
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

    parser.add_argument(
        "--zscorenormalization",
        type=str,
        required=True,
        choices=["ON", "OFF"],
        help="Whether or not we normalize input EEG data or rely on C++ processing"
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
    # make sure csv log doesn't deviate from this format!!
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
        if tf_hz < 0 and tf_hz != -1:
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

def compute_optimal_k_folds(
    *,
    n_groups_c0: int,
    n_groups_c1: int,
    n_groups_c2: int,
    n_windows_c0: int,
    n_windows_c1: int,
    n_windows_c2: int,
    min_groups_per_fold: int = 2,
    min_windows_per_fold: int = 12,
    preferred_k: int = 5,
    logger = None,
) -> tuple[int, str]:
    """
    Intelligently determines optimal number of CV folds based on data availability.
    
    Args:
        n_groups_cX: Number of groups for each class
        n_windows_cX: Total windows for each class
        min_groups_per_fold: Minimum groups per class per fold
        min_windows_per_fold: Minimum windows per class per fold
        preferred_k: Ideal number of folds (if data supports it)
        logger: Optional logger
    
    Returns:
        (k_optimal, reason_str)
    """
    
    k_limits = []
    reasons = []
    
    # Constraint 1: Groups per fold (with safety factor for variance)
    safety_factor = 1.5  # Account for group size variance
    
    for cls_idx, n_groups in enumerate([n_groups_c0, n_groups_c1, n_groups_c2]):
        if n_groups == 0:
            k_limits.append(0)
            reasons.append(f"c{cls_idx} has 0 groups")
            continue
        
        required_groups_per_fold = min_groups_per_fold * safety_factor
        max_k_groups = int(n_groups / required_groups_per_fold)
        k_limits.append(max_k_groups)
        
        if max_k_groups < preferred_k:
            reasons.append(
                f"c{cls_idx} ({n_groups} groups → max k={max_k_groups} with variance safety)"
            )
    
    # Constraint 2: Windows per fold
    for cls_idx, n_windows in enumerate([n_windows_c0, n_windows_c1, n_windows_c2]):
        if n_windows == 0:
            k_limits.append(0)
            reasons.append(f"c{cls_idx} has 0 windows")
            continue
        
        max_k_windows = n_windows // min_windows_per_fold
        k_limits.append(max_k_windows)
        
        if max_k_windows < preferred_k:
            reasons.append(
                f"c{cls_idx} windows ({n_windows} windows → max k={max_k_windows})"
            )
    
    # Take most restrictive constraint
    k_optimal = min(k_limits) if k_limits else 0
    k_optimal = max(3, min(k_optimal, preferred_k)) # min 3 folds
    
    # Build explanation
    if k_optimal == preferred_k:
        reason_str = f"Using preferred k={preferred_k}"
    elif k_optimal >= 2:
        limiting_reasons = [r for i, r in enumerate(reasons) if k_limits[i] == min(k_limits)]
        reason_str = f"Reduced to k={k_optimal}. {limiting_reasons[0] if limiting_reasons else ''}"
    else:
        reason_str = f"FATAL: Cannot create valid folds"
    
    if logger:
        logger.log(f"[K_FOLDS] Groups: c0={n_groups_c0} c1={n_groups_c1} c2={n_groups_c2}")
        logger.log(f"[K_FOLDS] Windows: c0={n_windows_c0} c1={n_windows_c1} c2={n_windows_c2}")
        logger.log(f"[K_FOLDS] Selected k={k_optimal} (preferred={preferred_k})")
        logger.log(f"[K_FOLDS] {reason_str}")
    
    return k_optimal, reason_str

def count_groups_per_class(
    *,
    yb: np.ndarray,
    groups: np.ndarray,
) -> tuple[int, int, int, np.ndarray, np.ndarray]:
    """
    Returns:
      n_groups_c0, n_groups_c1, n_groups_c2
      uniq_groups (sorted),
      group_label aligned with uniq_groups (0/1/2)
    Assumes each group is label-pure -> ONE CLASS PER GROUP

    one group = BLOCK OF SAMPLES THAT MUST STAY TOGETHER, like:
    yb     = [0, 0, 0, 1, 1, 2, 2, 2]   # class labels  
    groups = [5, 5, 5, 8, 8, 9, 9, 9]   # group IDs
    """
    yb = np.asarray(yb).astype(np.int64)
    groups = np.asarray(groups).astype(np.int64)

    uniq_groups = np.unique(groups)
    group_label = np.empty(len(uniq_groups), dtype=np.int64)

    for gi, g in enumerate(uniq_groups):
        idx_g = np.where(groups == g)[0] # np. where is boolean mask: true at idxs where groups == g, and [0] returns first element of array
        if idx_g.size == 0:
            utils.abort("FOLDS", "Empty group encountered (should be impossible)")
        group_label[gi] = int(yb[idx_g[0]]) # assign the class label for each unique group based on first idx we found for it (assumes label purity)

    # count the total num of groups in each class
    n_groups_c0 = int((group_label == 0).sum())
    n_groups_c1 = int((group_label == 1).sum())
    n_groups_c2 = int((group_label == 2).sum())
    return n_groups_c0, n_groups_c1, n_groups_c2, uniq_groups, group_label

def validate_folds_strict(
    *,
    yb: np.ndarray,
    groups: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    hz_a: int | None = None,  
    hz_b: int | None = None,
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
    ok = True

    if len(folds) < 2:
        issues.append(utils.issue("FOLDS", "len(folds) < 2", {"n_folds": int(len(folds))}))
        return False, issues

    max_imbalance = 0.0
    max_imbalance_tr = 0.0
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
            ok = False
            continue

        # class presence check (val) 
        va0 = int((yb[va_idx] == 0).sum())
        va1 = int((yb[va_idx] == 1).sum())
        va2 = int((yb[va_idx] == 2).sum())
        if va0 == 0 or va1 == 0 or va2 == 0:
            issues.append(utils.issue(
                "FOLDS",
                "Validation split is missing a class",
                {"fold": int(fi), "va0": va0, "va1": va1, "va2 rest": va2},
            ))
            ok = False
        # minimum windows per class in validation
        for cls, count in [(0, va0), (1, va1), (2, va2)]:
            if count < utils.MIN_WINDOWS_PER_CLASS_VAL and count > 0:
                issues.append(utils.issue(
                    "FOLDS",
                    f"Too few validation samples for class {cls} in fold {fi}",
                    {"fold": fi, "class": cls, "count": count, "min": utils.MIN_WINDOWS_PER_CLASS_VAL},
                    data_insufficiency={
                        "frequency_hz": hz_a if cls == 0 else hz_b if cls == 1 else -1,
                        "metric": "windows",
                        "required": utils.MIN_WINDOWS_PER_CLASS_VAL,
                        "actual": count,
                    }
                ))
        # minimum groups per class in validation
        va_groups = groups[va_idx]
        for cls in [0, 1, 2]:
            cls_mask = yb[va_idx] == cls
            n_groups = len(np.unique(va_groups[cls_mask]))
            if n_groups < utils.MIN_GROUPS_PER_CLASS_FOR_CV and n_groups > 0:
                issues.append(utils.issue(
                    "FOLDS",
                    f"Too few validation groups for class {cls} in fold {fi}",
                    {"fold": fi, "class": cls, "n_groups": n_groups, "min": utils.MIN_GROUPS_PER_CLASS_FOR_CV},
                    data_insufficiency={
                        "frequency_hz": hz_a if cls == 0 else hz_b if cls == 1 else -1,
                        "metric": "groups",
                        "required": utils.MIN_GROUPS_PER_CLASS_FOR_CV,
                        "actual": n_groups,
                    }
                ))

        # class presence check (train)
        tr0 = int((yb[tr_idx] == 0).sum())
        tr1 = int((yb[tr_idx] == 1).sum())
        tr2 = int((yb[tr_idx] == 2).sum())
        tr_total = tr0+tr1+tr2
        if tr0 == 0 or tr1 == 0 or tr2 == 0:
            issues.append(utils.issue(
                "FOLDS",
                "Training split is missing a class",
                {"fold": int(fi), "tr0": tr0, "tr1": tr1, "tr2 rest": tr2},
            ))
            ok = False
        # minimum windows per class in training
        for cls, count in [(0, tr0), (1, tr1), (2, tr2)]:
            if count < utils.MIN_WINDOWS_PER_CLASS_TRAIN and count > 0:
                issues.append(utils.issue(
                    "FOLDS",
                    f"Too few training samples for class {cls} in fold {fi}",
                    {"fold": fi, "class": cls, "count": count, "min": utils.MIN_WINDOWS_PER_CLASS_TRAIN},
                    data_insufficiency={
                        "frequency_hz": hz_a if cls == 0 else hz_b if cls == 1 else -1,
                        "metric": "windows",
                        "required": utils.MIN_WINDOWS_PER_CLASS_TRAIN,
                        "actual": count,
                    }
                ))

        # check class balance in train
        if tr_total > 0:
            fracs_tr = [tr0/tr_total, tr1/tr_total, tr2/tr_total]
            max_frac_tr = max(fracs_tr)
            min_frac_tr = min(fracs_tr)
            imbalance_tr = max_frac_tr - min_frac_tr
            max_imbalance_tr = max(max_imbalance_tr, imbalance_tr)

        # split size sanity
        if tr_idx.size == 0 or va_idx.size == 0:
            issues.append(utils.issue(
                "FOLDS",
                "Fold has empty train or val split",
                {"fold": int(fi), "train_size": int(tr_idx.size), "val_size": int(va_idx.size)},
            ))
            ok = False

    # check class balance in val
    imbalance_pct, _ = mean_val_imbalance_pct(yb, folds, n_classes=3)  # e.g. returns 0..100
    max_imbalance = max(max_imbalance, imbalance_pct)

    if max_imbalance > 15: # in percent
        # keep ok = true but warn...
        issues.append(utils.issue(
                "FOLDS",
                "Imbalance in val fold",
            ))
    if max_imbalance_tr > 0.2: # in decimal
        # keep ok = true but warn...
        issues.append(utils.issue(
                "FOLDS",
                "Imbalance in train fold",
            ))
    
    return ok, issues

def make_cv_folds_by_blocked_windows(
    *,
    yb: np.ndarray,
    trial_ids: np.ndarray,
    window_ids: np.ndarray,
    k: int | None = None, # can autodetect
    seed: int,
    n_time: int,
    hop_samples: int,
    debug_logger: utils.DebugLogger | None = None,
    hz_a: int,
    hz_b: int,
    debug_tag: str = "CV",
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[utils.TrainIssue], np.ndarray, np.ndarray]:
    """
    Stratified CV on *blocked window groups*:
      - windows close in time (overlapping) are grouped into the same "block" group
        block_size_windows ~= ceil(W/H) where W=n_time and H=hop_samples
    This is desired compromise:
      random-ish windows in each fold, but overlap leakage is prevented because
      overlapping neighbors live in the same group.
    
    Strategy:
    1. Create overlap-safe groups (same as before)
    2. Compute window count per group
    3. Use weighted sampling to ensure val folds have ~equal windows per class

    Guarantees (if returns non-empty):
      - leak_groups=0 in every fold
      - each fold val has both classes
      - each fold train has both classes
    """
    issues: list[utils.TrainIssue] = []
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
    block_size_windows = compute_block_size_windows(
        n_time=int(n_time),
        hop_samples=int(hop_samples),
    )
    groups, keep = make_block_groups_within_trials_by_window_idx(
        trial_ids=trial_ids,
        yb=yb,
        window_ids=window_ids,
        block_size_windows=int(block_size_windows),
        seed=seed,
    )
    # apply keep mmask to all window-level arrays
    yb  = yb[keep==1]
    trial_ids  = trial_ids[keep==1]
    window_ids = window_ids[keep==1]
    groups = groups[keep==1]
    # remap for cleanliness so we don't have gaps in group numbers
    groups = pd.factorize(groups)[0].astype(np.int64)

    # 2) Build group metadata & Compute k bound from groups-per-class
    uniq_groups = np.unique(groups)
    group_label = np.zeros(len(uniq_groups), dtype=np.int64)
    group_size = np.zeros(len(uniq_groups), dtype=np.int64)
    
    for gi, g in enumerate(uniq_groups):
        idx_g = np.where(groups == g)[0]
        group_label[gi] = int(yb[idx_g[0]])
        group_size[gi] = len(idx_g)
    
    # Count groups and windows per class
    groups_c0 = uniq_groups[group_label == 0]
    groups_c1 = uniq_groups[group_label == 1]
    groups_c2 = uniq_groups[group_label == 2]
    n_windows_c0 = int((yb == 0).sum())
    n_windows_c1 = int((yb == 1).sum())
    n_windows_c2 = int((yb == 2).sum())
    # shuffle for randomness
    rng = np.random.default_rng(seed)
    rng.shuffle(groups_c0)
    rng.shuffle(groups_c1)
    rng.shuffle(groups_c2)
    
    if k is None or k <= 0:
        # Auto-detect optimal k
        k_eff, k_reason = compute_optimal_k_folds(
            n_groups_c0=len(groups_c0),
            n_groups_c1=len(groups_c1),
            n_groups_c2=len(groups_c2),
            n_windows_c0=n_windows_c0,
            n_windows_c1=n_windows_c1,
            n_windows_c2=n_windows_c2,
            min_groups_per_fold=2,
            min_windows_per_fold=12,
            preferred_k=5,  # Default preference
            logger=debug_logger,
        )
    else:
        # User specified k, but still bound by available groups
        k_requested = k
        k_eff = min(k, len(groups_c0), len(groups_c1), len(groups_c2))
        
        if k_eff < k_requested and debug_logger:
            debug_logger.log(
                f"[{debug_tag}] Requested k={k_requested} reduced to k={k_eff} "
                f"(limited by groups: c0={len(groups_c0)} c1={len(groups_c1)} c2={len(groups_c2)})"
            )
    
    if k_eff < 2:
        issues.append({"stage": "FOLDS", "message": "Not enough groups per class"})
        return [], issues, keep, []

    # 3) Distribute groups with size-awareness
    # This helps distribute large and small groups more evenly
    fold_groups = [[] for _ in range(k_eff)]

    def assign_groups_lfd(groups_list, k_folds):
        """Assign groups to folds using largest-first strategy"""
        # Get sizes
        group_info = [(g, int(group_size[uniq_groups == g][0])) for g in groups_list]
        # Sort by size descending
        group_info.sort(key=lambda x: -x[1])
        
        # Initialize folds
        fold_assignments = [[] for _ in range(k_folds)]
        fold_totals = [0] * k_folds
        
        # Assign each group to fold with minimum current total
        for g, size in group_info:
            min_fold_idx = min(range(k_folds), key=lambda i: fold_totals[i])
            fold_assignments[min_fold_idx].append(int(g))
            fold_totals[min_fold_idx] += size
        
        return fold_assignments
    
    # Assign each class with LFD
    c0_assignments = assign_groups_lfd(list(groups_c0), k_eff)
    c1_assignments = assign_groups_lfd(list(groups_c1), k_eff)
    c2_assignments = assign_groups_lfd(list(groups_c2), k_eff)
    
    # Combine into fold_groups
    fold_groups = []
    for fold_i in range(k_eff):
        combined = c0_assignments[fold_i] + c1_assignments[fold_i] + c2_assignments[fold_i]
        fold_groups.append(combined)
    
    # 4) Create folds
    folds = []
    all_indices = np.arange(len(yb), dtype=np.int64)
    
    for fold_i in range(k_eff):
        val_groups_set = set(fold_groups[fold_i])
        val_mask = np.isin(groups, list(val_groups_set))
        val_idx = all_indices[val_mask]
        train_idx = all_indices[~val_mask]
        folds.append((train_idx, val_idx))

    # 5) Validation & Logging (Check window balance)
    if debug_logger is not None:
        debug_logger.log(f"[{debug_tag}] ===== Fold builder (blocked groups) =====")
        debug_logger.log(f"[{debug_tag}] N_windows={len(yb)}")
        debug_logger.log(f"[{debug_tag}] block_size_windows=ceil(W/H)={block_size_windows}")
        debug_logger.log(f"[{debug_tag}] n_groups_total={len(uniq_groups)}")
        debug_logger.log(f"[{debug_tag}] k_eff (group-limited)={k_eff}")
        debug_logger.log()

    # group_label aligns with uniq_groups (one label per group)
    g0 = int((group_label == 0).sum())
    g1 = int((group_label == 1).sum())
    g2 = int((group_label == 2).sum())
    # group sizes (how many windows per group)
    group_sizes = np.bincount(groups.astype(np.int64))
    gsize_min = int(group_sizes.min()) if group_sizes.size else 0
    gsize_med = int(np.median(group_sizes)) if group_sizes.size else 0
    gsize_max = int(group_sizes.max()) if group_sizes.size else 0
    def _class_gsizes(cls: int) -> tuple[int,int,int]:
        cls_groups = uniq_groups[group_label == cls]
        if cls_groups.size == 0:
            return (0,0,0)
        sizes = np.array([group_sizes[int(g)] for g in cls_groups], dtype=np.int64)
        return (int(sizes.min()), int(np.median(sizes)), int(sizes.max()))
    c0_min, c0_med, c0_max = _class_gsizes(0)
    c1_min, c1_med, c1_max = _class_gsizes(1)
    c2_min, c2_med, c2_max = _class_gsizes(2)
    msg = (
        f"[{debug_tag}] GROUPS per class: g0={g0} g1={g1} g2={g2} | "
        f"group_size(min/med/max)={gsize_min}/{gsize_med}/{gsize_max} | "
        f"c0(min/med/max)={c0_min}/{c0_med}/{c0_max} "
        f"c1(min/med/max)={c1_min}/{c1_med}/{c1_max} "
        f"c2(min/med/max)={c2_min}/{c2_med}/{c2_max}"
    )
    if debug_logger is None:
        print(msg)
    else:
        debug_logger.log(msg)

    # Validate call
    ok, reasons = validate_folds_strict(yb=yb, groups=groups, folds=folds, hz_a=hz_a, hz_b=hz_b)
    
    if not ok:
        issues.extend(reasons)
        if debug_logger:
            debug_logger.log(f"[{debug_tag}] Fold validation FAILED")
            for r in reasons[:10]:
                debug_logger.log(f"[{debug_tag}]   {r.stage}: {r.message}")
        return [], issues, keep, groups
    
    # Success - summarize
    if debug_logger:
        summarize_blocked_folds(
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
        debug_logger.log(f"[{debug_tag}] SUCCESS: Created {k_eff} balanced folds (manual round-robin)")
    
    return folds, issues, keep, groups


def compute_block_size_windows(
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
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
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

    Returns:
      groups: (N,) int group id per window
      keep_mask: (N,) bool, True for windows we keep (drops only REST if needed)
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
    groups = pd.factorize(key)[0].astype(np.int64) # array of group numbers

    # (2) downsample REST group (because protocol is such that we likely we have way more rest examples, and we don't want imbalance)
    # REST can be downsampled randomly rather than by trial, since all background frequency pairs are random anyways
    # to minimize eventual size differences within rest groups & maximize training diversity, we structurally select REST examples from one group at a time
    
    rest_group_to_indices: dict[int, np.ndarray] = {} # dictionary mapping rest group (g) -> rest indices [x1,x2..]
    
    REST_RATIO = 1.1 # amount of rest examples relative to max(a,b)
    num_examples_a = (yb == 0).sum()
    num_examples_b = (yb == 1).sum()
    num_examples_rest = (yb == 2).sum()
    rest_target = int(min(REST_RATIO*max(num_examples_a,num_examples_b), num_examples_rest))
    if(rest_target >= num_examples_rest):
        return groups, np.ones(len(yb), dtype = bool)
    
    # Random shuffling
    rest_idx = np.where(yb == 2)[0].astype(np.int64) # get the array of idx elements
    rng = np.random.default_rng(int(seed))
    # for each rest group g, collect global idxs (build dict)
    rest_grps = np.unique(groups[rest_idx])
    rng.shuffle(rest_grps) # random order of groups
    for g in rest_grps:
        rest_grp_idxs = np.where((groups == g) & (yb==2))[0] # assuming groups are label pure
        rest_grp_idxs = rest_grp_idxs.astype(np.int64).copy()
        rng.shuffle(rest_grp_idxs) # random order of idxs per group
        rest_group_to_indices[int(g)] = rest_grp_idxs

    # Round-robin approach
    keep_rest_mask: list[int] = [] # indices
    keep_rest_bool_mask = np.ones(len(yb),dtype=bool) # should be 1 at selected indices
    selected_pos_in_group: dict[int,int] = {int(g): 0 for g in rest_group_to_indices.keys()} # init all values 0, group keys
    while len(keep_rest_mask) < rest_target:
        made_progress = False
        for g in rest_grps:
            if len(keep_rest_mask) >= rest_target: # recheck
                break
            g = int(g)
            idx_list = rest_group_to_indices.get(g) # alr randomly shuffled
            if idx_list is None or idx_list.size == 0:
                continue
            p = selected_pos_in_group[g]
            if p < idx_list.size: # less than members per group
                keep_rest_mask.append(int(idx_list[p]))
                # next pass when choosing from this group, we'll take next idx
                selected_pos_in_group[g] = p + 1
                made_progress = True
        
        if not made_progress: # after having iterated through all groups
            # we've exhausted all groups
            break
    
    keep_rest_bool_mask[rest_idx] = 0
    keep_rest_bool_mask[keep_rest_mask] = 1 # only select these from REST idxs, the rest stay 0

    return groups, keep_rest_bool_mask

# -----------------------------
# Best Pair Selection Logic
# -----------------------------
def make_binary_pair_dataset(X: np.ndarray, y_hz: np.ndarray, trial_ids: np.ndarray, window_ids: np.ndarray, hz_a: int, hz_b: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Filters to only windows in {hz_a, hz_b, -1} and returns:
      Xp: (Npair, C, T)
      yb: (Npair,) where hz_a -> 0, hz_b -> 1, hz_rest -> 2
    """
    mask = (y_hz == hz_a) | (y_hz == hz_b) | (y_hz == -1)
    Xp = X[mask]
    yp = y_hz[mask]
    tp = trial_ids[mask]
    wp = window_ids[mask]
    
    # remap
    a_mask = (yp == hz_a)
    b_mask = (yp == hz_b)
    # init new array all 2s (rest = default)
    yb = np.full((yp.shape), 2)
    yb[a_mask] = 0
    yb[b_mask] = 1

    return Xp, yb, tp, wp

def shortlist_freqs(y_hz: np.ndarray, pick_top_k: int) -> list[int]:
    """
    Keep only frequencies with enough windows, then take the top-K by count.
    """
    k_folds = 5 # REQUIRE AT LEAST K_FOLDS WINDOWS PER CLASS (bare minimum) and 20 for reasonability
    min_windows_per_class = max(utils.MIN_WINDOWS_PER_FREQ_FOR_SHORTLIST, int(k_folds * 2))
    
    y_hz = np.asarray(y_hz, dtype=np.int64)
    y_hz = y_hz[y_hz!=-1] # don't count rest as candidate frequency
    
    vals, counts = np.unique(y_hz, return_counts=True)
    keep = [(int(v), int(c)) for v, c in zip(vals, counts) if int(c) >= int(min_windows_per_class)]
    if len(keep) < 2:
        return []

    # Sort by count desc (more data = more stable), then by freq asc
    keep.sort(key=lambda t: (-t[1], t[0]))
    freqs = [hz for hz, _ in keep]
    return freqs[: min(pick_top_k, len(freqs))]

def mean_val_imbalance_pct(y: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]], n_classes: int = 3) -> tuple[float, list[float]]:
    imbs = []
    for _, va_idx in folds:
        yv = y[va_idx]
        counts = np.bincount(yv.astype(np.int64), minlength=n_classes)
        n = int(counts.sum())
        if n == 0:
            imbs.append(100.0)
            continue
        pct = counts / n
        imbs.append(float((pct.max() - pct.min()) * 100.0))
    return float(np.mean(imbs)) if imbs else 100.0, imbs

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
    rejected_pairs = []
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
                },
                issue.data_insufficiency
            ))
            else:
                pair_issues.append(utils.issue(
                    "PAIR_SEARCH", "Pair dataset invalid for unknown reason",
                    {"hz_a": int(hz_a), "hz_b": int(hz_b), "n_pair": int(len(yb))}
                ))
            continue

        k = int(gen_cfg.number_cross_val_folds)
        folds, fold_issues, keep, groups_pair = make_cv_folds_by_blocked_windows(
            yb=yb,
            trial_ids=tp,
            window_ids=wp,
            k=None,
            seed=0,
            n_time=info.n_time,    # window length in samples
            hop_samples=utils.HOP_SAMPLES,
            debug_logger=debug_logger,
            hz_a=hz_a,
            hz_b=hz_b,
            debug_tag=f"PAIR {hz_a}vs{hz_b}",
        )
        # re-index based on keep
        Xp = Xp[keep]
        yb = yb[keep]
        tp = tp[keep]
        wp = wp[keep]

        if len(folds) < 2:
            pair_issues.append(utils.issue(
                "PAIR_SEARCH",
                "Fold build failed for pair",
                {"hz_a": int(hz_a), "hz_b": int(hz_b), "n_fold_issues": int(len(fold_issues))},
            ))
            pair_issues.extend(fold_issues)
            continue # skip this pair

        # reject based on val imbalance before training
        mean_imb, per_fold_imbs = mean_val_imbalance_pct(yb, folds, n_classes=3)
        if mean_imb >= 15.0:
            rejected_pairs.append({
                "hz_a": int(hz_a),
                "hz_b": int(hz_b),
                "mean_val_imbalance_pct": float(mean_imb),
                "per_fold_val_imbalance_pct": [float(x) for x in per_fold_imbs],
                "n": int(len(yb)),
            })
            if debug_logger:
                debug_logger.log(
                    f"[PAIR_SEARCH] REJECT ({hz_a},{hz_b}) mean_val_imbalance={mean_imb:.1f}% (>=15%) "
                    f"per_fold={','.join(f'{x:.1f}' for x in per_fold_imbs)}"
                )
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
                zscorearg=args.zscorenormalization,
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
        debug_logger.log(f"[PAIR_SEARCH] pair ({hz_a},{hz_b}) mean_bal_acc={score:.3f} (+/-{std_score:.3f})")

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
    # Convention: left = freq a, right = freq b
    left_hz = a
    right_hz = b

    debug = {
        "rejected_pairs_from_class_skew": rejected_pairs,
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
                            rejected_pairs_from_class_skew: list[str], elapsed_time: float = 0.0,
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
        "rejected_pairs_from_class_skew": rejected_pairs_from_class_skew,
        "training_time": elapsed_time,
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
    start_time = time.time()
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
    zscore_arg = args.zscorenormalization
    
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
        rejected_pairs_from_class_skew = debug["rejected_pairs_from_class_skew"]
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
        c2 = int((yb == 2).sum())
        k_final = int(gen_cfg.number_cross_val_folds) #TODO: might not need anymore w adaptive k??
        folds_final: list[tuple[np.ndarray, np.ndarray]] = []
        fold_issues: list[utils.TrainIssue] = []
        if c0 >= 2 and c1 >= 2 and c2 >= 2: # simple guard for min 2 folds
            folds_final, fold_issues, keep, groups_pair = make_cv_folds_by_blocked_windows(
                yb=yb,
                trial_ids=tp,
                window_ids=wp,
                k=None,
                seed=0,
                n_time=info.n_time,
                hop_samples=utils.HOP_SAMPLES,
                debug_logger=debug_log,
                hz_a=best_left_hz,
                hz_b=best_right_hz,
                debug_tag=f"FINAL {best_left_hz}vs{best_right_hz}",
            )
            # re-index based on keep
            Xp = Xp[keep]
            yb = yb[keep]
            tp = tp[keep]
            wp = wp[keep]
        else:
            fold_issues.append(utils.issue("FINAL", "Not enough windows in final fold construction", {"c0": c0, "c1": c1, "c2": c2}))

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
                group_pair=groups_pair,
                folds=folds_final,
                n_ch=info.n_ch,
                n_time=info.n_time,
                out_onnx_path=(out_dir / "ssvep_model.onnx"),
                hparam_tuning=hparam_arg,
                zscore_norm=zscore_arg,
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

        elapsed = time.time() - start_time
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
            rejected_pairs_from_class_skew=rejected_pairs_from_class_skew,
            elapsed_time=elapsed,
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
    left_e=hz2e.get(best_left_hz, -1)
    right_e=hz2e.get(best_right_hz, -1)
    elapsed = time.time() - start_time

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
        rejected_pairs_from_class_skew=[-1],
        elapsed_time=elapsed,
        extra={
            "fatal_stage": issues[-1].stage if issues else "UNKNOWN",
            "n_windows_total": int(X.shape[0]) if X is not None else 0,
        },
    )
    return 1


# ENTRY POINT
if __name__ == "__main__":
    sys.exit(main())

