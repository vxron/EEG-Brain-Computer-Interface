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

HOP_SAMPLES = 80

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
    logger: "DebugLogger",
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

    # --- group purity checks + build group->label and group->trial maps
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

        # even if not pure, choose first for logging (but we'll warn)
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

    # --- overall class balance (windows)
    c0 = int((yb == 0).sum())
    c1 = int((yb == 1).sum())
    logger.log(f"[{tag}] overall class counts: c0={c0}, c1={c1}")
    logger.log()

    # --- per-fold stats
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
# TODO: Make it move upward dynamically (as it stands, breaks if we move folders)
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

    raise ValueError(f"Unknown calibsetting: {calibsetting}")

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

def add_trial_ids_per_window(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds an integer trial_id per window based on contiguous runs of testfreq_hz within each _src.
    Trial increments when testfreq_hz changes (in time order).
    This is essential so that we can batch per trial instead of per window in CNN arch (to optimize learning steps with richer info).
    """
    # window-level metadata
    win = (
        df.groupby(["_src", "window_idx"], sort=True)
          .agg(
              testfreq_hz=("testfreq_hz", "first"),
              start_sample=("sample_idx", "min"),
          )
          .reset_index()
    )

    # sort windows by time within each session
    win = win.sort_values(["_src", "start_sample", "window_idx"], kind="stable")

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

    win = win.groupby("_src", group_keys=False, sort=False).apply(_assign_trials)

    # Make trial ids globally unique ints across sessions:
    # factorize on (_src, trial_local)
    trial_key = list(zip(win["_src"].astype(str), win["trial_local"].astype(int)))
    win["trial_id"] = pd.factorize(trial_key)[0].astype(np.int64)

    # merge trial_id back onto per-sample rows
    df = df.merge(win[["_src", "window_idx", "trial_id"]], on=["_src", "window_idx"], how="left")
    if df["trial_id"].isna().any():
        raise RuntimeError("trial_id merge failed for some rows.")
    df["trial_id"] = df["trial_id"].astype(np.int64)
    return df

def load_windows_csv(sources: list[CsvSource]) -> tuple[np.ndarray, np.ndarray, utils.DatasetInfo]:
    """
    BUILD TRAINING DATA
    Reads window-level CSV(s) and returns:
      X: (N, C, T) = (num_windows, num_channels, window_len_samples)
      y_hz: (N,)
      trial_ids_np: single array with the trial ids for each window so that we can shuffle by trial if desired
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

        # LOG: per-file read success + per-frequency window counts (after filters)
        n_rows = int(len(df))
        if n_rows == 0:
            raise RuntimeError(
                f"[PY] eeg_windows.csv has 0 usable rows after filters "
                f"(is_trimmed==1, is_bad!=1) "
                f"for session '{src.src_id}' at {src.path}"
            )
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
        raise FileNotFoundError("No usable eeg_windows.csv found for requested setting.")

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
        raise ValueError("No EEG channel columns detected between sample_idx and testfreq_e.")

    windows: list[np.ndarray] = []
    labels_hz: list[int] = []
    trial_ids: list[int] = []

    # group by (_src, window_idx) to avoid collisions
    grouped = df.groupby(["_src", "window_idx"], sort=True)

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
        trial_ids.append(trial_id)

    if not windows:
        raise RuntimeError("No valid windows found after grouping/filters.")

    X = np.stack(windows, axis=0).astype(np.float32)  # (N, C, T)
    y_hz = np.array(labels_hz, dtype=np.int64)
    trial_ids_np = np.array(trial_ids, dtype=np.int64)

    info = utils.DatasetInfo(
        ch_cols=ch_cols,
        n_ch=n_ch,
        n_time=int(X.shape[2]),
        classes_hz=sorted(set(y_hz.tolist())),
    )
    return X, y_hz, trial_ids_np, info

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
            raise RuntimeError("Impossible: empty group encountered.")
        group_label[gi] = int(yb[idx_g[0]])

    n_groups_c0 = int((group_label == 0).sum())
    n_groups_c1 = int((group_label == 1).sum())
    return n_groups_c0, n_groups_c1, uniq_groups, group_label

def _validate_folds_strict(
    *,
    yb: np.ndarray,
    groups: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[bool, list[str]]:
    """
    Strict validation:
      - at least 2 folds
      - no group leakage between train and val
      - each fold has both classes in val AND train
    Returns (ok, reasons)
    """
    yb = np.asarray(yb).astype(np.int64)
    groups = np.asarray(groups).astype(np.int64)

    reasons: list[str] = []
    if len(folds) < 2:
        reasons.append("len(folds) < 2")
        return False, reasons

    for fi, (tr_idx, va_idx) in enumerate(folds):
        tr_idx = np.asarray(tr_idx, dtype=np.int64)
        va_idx = np.asarray(va_idx, dtype=np.int64)

        # group leakage check
        inter = set(groups[tr_idx]).intersection(set(groups[va_idx]))
        if inter:
            reasons.append(f"fold {fi}: group leakage (n={len(inter)})")
            continue

        # class presence check (val)
        va0 = int((yb[va_idx] == 0).sum())
        va1 = int((yb[va_idx] == 1).sum())
        if va0 == 0 or va1 == 0:
            reasons.append(f"fold {fi}: val single-class (va0={va0}, va1={va1})")

        # class presence check (train)
        tr0 = int((yb[tr_idx] == 0).sum())
        tr1 = int((yb[tr_idx] == 1).sum())
        if tr0 == 0 or tr1 == 0:
            reasons.append(f"fold {fi}: train single-class (tr0={tr0}, tr1={tr1})")

        # size sanity
        if tr_idx.size == 0 or va_idx.size == 0:
            reasons.append(f"fold {fi}: empty split (train={tr_idx.size}, val={va_idx.size})")

    ok = (len(reasons) == 0)
    return ok, reasons


def make_cv_folds_binary_by_blocked_windows(
    *,
    yb: np.ndarray,
    trial_ids: np.ndarray,
    k: int,
    seed: int,
    n_time: int,
    hop_samples: int,
    fs_hz: float = 250.0,
    debug_logger: utils.DebugLogger | None = None,
    debug_tag: str = "CV",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
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
    yb = np.asarray(yb).astype(np.int64)
    trial_ids = np.asarray(trial_ids).astype(np.int64)

    if yb.shape[0] != trial_ids.shape[0]:
        raise ValueError("yb and trial_ids must have same length")

    # 1) Build block groups
    block_size_windows = _compute_block_size_windows(
        n_time=int(n_time),
        hop_samples=int(hop_samples),
    )
    groups = _make_block_groups_within_trials(
        trial_ids=trial_ids,
        yb=yb,
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
        if debug_logger is not None:
            debug_logger.log(f"[{debug_tag}] FAIL: k_eff < 2 (not enough groups per class)")
            debug_logger.log(f"[{debug_tag}] ========================================\n")
        return []

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
                if debug_logger is not None:
                    debug_logger.log(f"[{debug_tag}]   seed={s}: EXCEPTION during split: {e}")
                continue

            ok, reasons = _validate_folds_strict(yb=yb, groups=groups, folds=folds)

            if debug_logger is not None:
                debug_logger.log(f"[{debug_tag}]   seed={s}: built {len(folds)} folds -> ok={ok}")
                if not ok:
                    for r in reasons[:6]:
                        debug_logger.log(f"[{debug_tag}]     reason: {r}")
                    if len(reasons) > 6:
                        debug_logger.log(f"[{debug_tag}]     ... ({len(reasons)-6} more)")
                else:
                    debug_logger.log(f"[{debug_tag}]   seed={s}: SUCCESS ✓")

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
                return folds

        if debug_logger is not None:
            debug_logger.log(f"[{debug_tag}] No valid folds for k_try={k_try}. Degrading k...\n")

    if debug_logger is not None:
        debug_logger.log(f"[{debug_tag}] FAIL: Could not produce valid folds for any k>=2")
        debug_logger.log(f"[{debug_tag}] ========================================\n")

    return []



def make_cv_folds_binary_by_window_old(
    yb: np.ndarray, trial_ids: np.ndarray, k: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Standard stratified k-fold on WINDOWS (not trials).
    Accepts window overlap as standard practice in SSVEP research.
    trial_ids parameter kept for API compatibility but not used.
    """
    
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = []
    
    for train_idx, val_idx in skf.split(np.arange(len(yb)), yb):
        folds.append((train_idx, val_idx))
    
    return folds

def make_cv_folds_binary_by_trial(
    yb: np.ndarray, trial_ids: np.ndarray, k: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    K-fold CV where entire trials are assigned to folds.
    Ensures no trial appears in both train and val.
    Stratifies approximately by class by assigning trials based on their majority label.
    """
    rng = np.random.default_rng(seed)

    trial_ids = np.asarray(trial_ids).astype(np.int64)
    yb = np.asarray(yb).astype(np.int64)

    uniq_trials = np.unique(trial_ids)
    # Determine a label per trial (majority vote)
    trial_label = {}
    trial_to_indices = {}

    for t in uniq_trials:
        idx = np.where(trial_ids == t)[0]
        trial_to_indices[int(t)] = idx
        # majority label
        lab = int(np.round(yb[idx].mean()))  # works for binary 0/1
        trial_label[int(t)] = lab

    trials0 = [t for t in uniq_trials if trial_label[int(t)] == 0]
    trials1 = [t for t in uniq_trials if trial_label[int(t)] == 1]
    rng.shuffle(trials0)
    rng.shuffle(trials1)

    folds0 = np.array_split(np.array(trials0, dtype=np.int64), k)
    folds1 = np.array_split(np.array(trials1, dtype=np.int64), k)

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(k):
        val_trials = np.concatenate([folds0[i], folds1[i]]) if (len(folds0[i]) + len(folds1[i])) else np.array([], dtype=np.int64)
        if val_trials.size == 0:
            continue

        val_idx = np.concatenate([trial_to_indices[int(t)] for t in val_trials])
        train_trials = np.setdiff1d(uniq_trials, val_trials, assume_unique=False)
        train_idx = np.concatenate([trial_to_indices[int(t)] for t in train_trials])

        if val_idx.size == 0 or train_idx.size == 0:
            continue
        folds.append((train_idx.astype(np.int64), val_idx.astype(np.int64)))

    return folds

def make_cv_folds_binary(yb: np.ndarray, k: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Returns list of (train_idx, val_idx) for k-fold CV.
    Ensures each fold gets some samples from each class when possible.
    OBSOLETE !
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
        raise ValueError(f"hop_samples must be > 0, got {hop_samples}")
    if n_time <= 0:
        raise ValueError(f"n_time must be > 0, got {n_time}")
    # fs_hz isn't mathematically required for W/H, but we keep it in signature
    # because you'll likely want to log/validate it.
    block = int(np.ceil(float(n_time) / float(hop_samples)))
    return max(1, block)


def _make_block_groups_within_trials(
    *,
    trial_ids: np.ndarray,
    yb: np.ndarray,
    block_size_windows: int,
) -> np.ndarray:
    """
    Create "group" ids so that:
      - groups never cross trial boundaries
      - groups never cross label boundaries (safety)
      - within each (trial_id, label) segment, consecutive windows are chunked into blocks
        of size block_size_windows

    IMPORTANT: this assumes the incoming arrays are already in time order
    (which they are, because your windows are built in time order in load_windows_csv()).
    """
    trial_ids = np.asarray(trial_ids).astype(np.int64)
    yb = np.asarray(yb).astype(np.int64)
    if trial_ids.shape[0] != yb.shape[0]:
        raise ValueError("trial_ids and yb must have same length")

    N = int(len(yb))
    groups = np.empty(N, dtype=np.int64)

    g = -1
    pos_in_block = 0

    for i in range(N):
        # start a new group when:
        # - first sample
        # - trial changes
        # - label changes (extra guard)
        # - current block is full
        new_group = False
        if i == 0:
            new_group = True
        else:
            if trial_ids[i] != trial_ids[i - 1]:
                new_group = True
            elif yb[i] != yb[i - 1]:
                new_group = True
            elif pos_in_block >= block_size_windows:
                new_group = True

        if new_group:
            g += 1
            pos_in_block = 1
        else:
            pos_in_block += 1

        groups[i] = g

    return groups


def make_cv_folds_binary_by_blocked_windows_old(
    *,
    yb: np.ndarray,
    trial_ids: np.ndarray,
    k: int,
    seed: int,
    n_time: int,
    hop_samples: int,
    fs_hz: float = 250.0,
    debug_logger: utils.DebugLogger | None = None,
    debug_tag: str = "CV",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Stratified CV on *blocked window groups*:
      - windows close in time (overlapping) are grouped into the same "block" group
        block_size_windows ~= ceil(W/H) where W=n_time and H=hop_samples
      - we then do StratifiedGroupKFold so each fold has both classes, but blocks
        never get split across train/val.

    This is desired compromise:
      random-ish windows in each fold, but overlap leakage is prevented because
      overlapping neighbors live in the same group.

    NOTE: This relies on arrays being in time order within each trial_id.
    """
    yb = np.asarray(yb).astype(np.int64)
    trial_ids = np.asarray(trial_ids).astype(np.int64)

    # 1) Build groups = block ids
    block_size_windows = _compute_block_size_windows(
        n_time=int(n_time),
        hop_samples=int(hop_samples),
    )
    groups = _make_block_groups_within_trials(
        trial_ids=trial_ids,
        yb=yb,
        block_size_windows=int(block_size_windows),
    )

    # 2) reduce k based on BLOCKS (groups)
    uniq_groups = np.unique(groups)
    group_label = np.empty(len(uniq_groups), dtype=np.int64)
    for gi, g in enumerate(uniq_groups):
        idx_g = np.where(groups == g)[0]
        group_label[gi] = int(yb[idx_g[0]])
    n_groups_c0 = int((group_label == 0).sum())
    n_groups_c1 = int((group_label == 1).sum())
    k_req = int(k)
    k_eff = int(min(k_req, n_groups_c0, n_groups_c1))

    if debug_logger is not None:
        debug_logger.log(
            f"[{debug_tag}] k requested={k_req}, "
            f"groups per class: c0={n_groups_c0}, c1={n_groups_c1} -> k_eff={k_eff}"
        )
    if k_eff < 2:
        if debug_logger is not None:
            debug_logger.log(f"[{debug_tag}] fold build failed: k_eff < 2 (not enough groups).")
        return []

    # 3) StratifiedGroupKFold on windows with "groups" constraint
    # This ensures each fold has both classes (when possible) AND groups aren't split.
    sgkf = StratifiedGroupKFold(n_splits=int(k_eff), shuffle=True, random_state=int(seed))
    idx = np.arange(len(yb), dtype=np.int64)

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for tr_idx, va_idx in sgkf.split(idx, yb, groups):
        folds.append((tr_idx.astype(np.int64), va_idx.astype(np.int64)))

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

    return folds

# -----------------------------
# Best Pair Selection Logic
# -----------------------------
def make_binary_pair_dataset(X: np.ndarray, y_hz: np.ndarray, trial_ids: np.ndarray, hz_a: int, hz_b: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Filters to only windows in {hz_a, hz_b} and returns:
      Xp: (Npair, C, T)
      yb: (Npair,) where hz_a -> 0, hz_b -> 1
    """
    mask = (y_hz == hz_a) | (y_hz == hz_b)
    Xp = X[mask]
    yp = y_hz[mask]
    tp = trial_ids[mask]
    yb = np.where(yp == hz_a, 0, 1).astype(np.int64)
    return Xp, yb, tp

def shortlist_freqs(y_hz: np.ndarray, pick_top_k: int) -> list[int]:
    """
    Keep only frequencies with enough windows, then take the top-K by count.
    """
    k_folds = 5 # REQUIRE AT LEAST K_FOLDS WINDOWS PER CLASS (bare minimum) and 20 for reasonability
    min_windows_per_class = max(20, int(k_folds * 2))
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
    debug_logger: utils.DebugLogger | None = None,
    args,
) -> tuple[int, int, dict[str, Any]]:
    """
    Returns: (best_left_hz, best_right_hz, debug_info)
    Scoring metric: mean CV balanced accuracy.
    """
    cand_freqs = shortlist_freqs(
        y_hz,
        pick_top_k=6,
    )
    if len(cand_freqs) < 2:
        raise RuntimeError("Not enough usable frequencies after min_windows_per_class filtering.")

    pairs = [(cand_freqs[i], cand_freqs[j]) for i in range(len(cand_freqs)) for j in range(i + 1, len(cand_freqs))]

    best_metrics = None
    best_score = -1.0
    all_metrics = []

    print(f"[PY] Pair search candidates: freqs={cand_freqs} -> {len(pairs)} pairs, arch={args.arch}")

    for hz_a, hz_b in pairs:

        # ====== 1) build binary dataset for this candidate pair =====
        Xp, yb, tp = make_binary_pair_dataset(X, y_hz, trial_ids, hz_a, hz_b)

        # Guard: need enough samples per class for k-fold
        c0 = int((yb == 0).sum())
        c1 = int((yb == 1).sum())
        if c0 < 2 or c1 < 2:
            print(f"[PY] pair ({hz_a},{hz_b}) skipped: not enough data (c0={c0}, c1={c1})")
            continue

        k = int(gen_cfg.number_cross_val_folds)
        folds = make_cv_folds_binary_by_blocked_windows(
            yb=yb,
            trial_ids=tp,
            k=k,
            seed=0,
            n_time=info.n_time,    # window length in samples
            hop_samples=HOP_SAMPLES,
            fs_hz=250.0,
            debug_logger=debug_logger,
            debug_tag=f"PAIR {hz_a}vs{hz_b}",
        )
        if len(folds) < 2:
            print(f"[PY] pair ({hz_a},{hz_b}) skipped: fold build failed")
            continue

        # ===== 2) score the pair using trainer API =====
        if args.arch == "CNN":
            metrics = score_pair_cv_cnn(
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
        elif args.arch == "SVM":
            metrics = utils.ModelMetrics(0, -99, -99)
            print("hadeel todo for svm")
        else:
            raise ValueError(f"Unknown arch: {args.arch}")

        all_metrics.append(metrics)

        if not metrics.cv_ok:
            print(f"[PY] pair ({hz_a},{hz_b}) skipped due to cv error")
            continue

        score = metrics.avg_fold_balanced_accuracy # current scoring metric we're using
        std_score = metrics.std_fold_balanced_accuracy
        print(f"[PY] pair ({hz_a},{hz_b}) mean_bal_acc={score:.3f} (+/-{std_score:.3f})")

        if score > best_score:
            best_score = score
            best_metrics = metrics

    if best_metrics is None:
        raise RuntimeError("Failed to select any valid pair (all candidates lacked data).")

    a = best_metrics.freq_a_hz
    b = best_metrics.freq_b_hz

    # Make left/right deterministic
    # Convention: left = lower Hz, right = higher Hz
    left_hz, right_hz = (a, b) if a < b else (b, a)

    debug = {
        "candidate_freqs": cand_freqs,
        "pair_scores": [asdict(m) for m in all_metrics], # dictionary format for JSON serialization
        "best_score_mean_bal_acc": best_score,
    }
    return left_hz, right_hz, debug


# -----------------------------
# JSON Contract & ONNX Export
# -----------------------------
def write_train_result_json(model_dir: Path, *, train_ok: bool, onnx_ok: bool, arch: str, calibsetting: str,
                            left_hz: int, right_hz: int,
                            left_e: int, right_e: int,
                            extra: dict[str, Any] | None = None) -> Path:
    payload: dict[str, Any] = {
        "train_ok": bool(train_ok),
        "onnx_ok": bool(onnx_ok),
        "arch": arch,
        "calibsetting": calibsetting,
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

    # 0: RESOLVE PATHS FROM ARGS
    out_dir_arg = Path(args.model)
    out_dir = out_dir_arg if out_dir_arg.is_absolute() else (MODELS_ROOT / out_dir_arg)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[PY] model output dir:", out_dir.resolve())
    debug_log = utils.DebugLogger(out_dir / "train_ssvep_debug.txt")
    print(f"[PY] CV debug log -> {debug_log.path.resolve()}")

    data_arg = Path(args.data)
    data_session_dir = data_arg if data_arg.is_absolute() else (DATA_ROOT / data_arg)
    data_session_dir = data_session_dir.resolve()

    hparam_arg = args.tunehparams

    gen_cfg = utils.GeneralTrainingConfigs()

    # 1) Resolve sources based on calibsetting
    sources = list_window_csvs(data_session_dir, args.calibsetting)
    print(f"[PY] loading {len(sources)} eeg_windows.csv sources")

    # 2) Load data
    X, y_hz, trial_ids, info = load_windows_csv(sources)
    print("[PY] Loaded X:", X.shape, "y_hz:", y_hz.shape, "classes_hz:", info.classes_hz)

    # 3) Select best pair using CV scoring with same arch
    best_left_hz, best_right_hz, debug = select_best_pair(X=X, y_hz=y_hz, trial_ids=trial_ids, info=info, gen_cfg=gen_cfg, debug_logger=debug_log, args=args)
    print(f"[PY] BEST PAIR: {best_left_hz}Hz vs {best_right_hz}Hz")

    # 4) Train final model on winning pair
    Xp, yb, tp = make_binary_pair_dataset(X, y_hz, trial_ids, best_left_hz, best_right_hz)
    # Build folds for the winning pair (used for final CV reporting)
    c0 = int((yb == 0).sum())
    c1 = int((yb == 1).sum())
    k_final = int(gen_cfg.number_cross_val_folds)
    folds_final: list[tuple[np.ndarray, np.ndarray]] = []
    if c0 >= 2 and c1 >= 2:
        folds_final = make_cv_folds_binary_by_blocked_windows(
            yb=yb,
            trial_ids=tp,
            k=k_final,
            seed=0,
            n_time=info.n_time,
            hop_samples=HOP_SAMPLES,
            fs_hz=250.0,
            debug_logger=debug_log,
            debug_tag=f"FINAL {best_left_hz}vs{best_right_hz}",
        )
    if len(folds_final) < 2:
        print(f"[PY] warning: final folds not usable (k_final={k_final}, folds={len(folds_final)}). "
              f"Final training will still run; CV metrics may be 0.")

    if args.arch == "CNN":
        final = train_final_cnn_and_export(
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

    elif args.arch == "SVM":
        raise NotImplementedError("Wire SVM final training + export here.")

    # 6) Write train_result.json (contract consumed by C++)
    hz2e = hz_to_enum_mapping()
    if best_left_hz not in hz2e or best_right_hz not in hz2e:
        raise ValueError(f"Best pair ({best_left_hz},{best_right_hz}) not in hz_to_enum_mapping()")

    write_train_result_json(
        out_dir,
        train_ok=final.train_ok,
        onnx_ok=final.onnx_export_ok,
        arch=args.arch,
        calibsetting=args.calibsetting,
        left_hz=best_left_hz,
        right_hz=best_right_hz,
        left_e=hz2e[best_left_hz],
        right_e=hz2e[best_right_hz],
        extra={
            "n_windows_total": int(X.shape[0]),
            "n_windows_pair": int(Xp.shape[0]),
            "pair_selection_metric": "mean_cv_balanced_accuracy",
            "candidate_freqs": debug.get("candidate_freqs"),
            "best_score_mean_bal_acc": debug.get("best_score_mean_bal_acc"),
            "pair_scores": debug.get("pair_scores"),
            "final_train_acc": final.final_train_acc,
            "final_val_acc": final.final_val_acc,
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
