#!/usr/bin/env python3
"""
train_ssvep.py

Script for training an SSVEP model from calibration data.
This script is called automatically by the C++ backend after a calibration
session. It MUST match the CLI interface expected by C++. 
(see training manager thread in CapstoneProject.cpp for details)

- Loads windowed EEG from CSVs with columns:
  window_idx, is_trimmed, is_bad, sample_idx, eeg1..eeg8, testfreq_hz
- Trains either:
  --arch svm     : linear SVM (HADEEL)
  --arch eegnet  : compact CNN (EEGNet) in PyTorch
- Selects best frequencies:
  1) compute per-class (per-frequency) validation accuracy
  2) pick top-K freqs (--pick_top_k_freqs)
  3) evaluate all pairs among top-K and choose best pair
- Saves:
  - model artifact (onnx)
  - metadata.json (freq list, mappings, top-K, best pair)

Expected args:
    --data <path>      directory containing calibration data
    --model <path>     directory where ONNX + meta.json should be written
    --subject <str>    subject ID
    --session <str>    session ID
    --arch    <str>    'SVM' or 'CNN'
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from dataclasses import dataclass
from __future__ import annotations

# Trainers live in trainers/
from trainers.cnn_trainer import train_cnn, export_cnn_onnx
from trainers.svm_trainer import train_svm_and_export

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
        help="Path to directory containing calibration data (windows + labels).",
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

    # shared pipeline knobs
    parser.add_argument("--val_ratio", type=float, default=0.2) # 20% val, 80% train
    parser.add_argument("--seed", type=int, default=0)

    # CNN knobs (passed through)
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)

    return parser.parse_args()

# -----------------------------
# Shared data + preprocess
# -----------------------------
@dataclass
class DatasetInfo:
    ch_cols: list[str]
    n_ch: int
    n_time: int
    classes_hz: list[int]


class LabelEncoderHz:
    """Maps Hz labels -> class indices 0..K-1 (and back)."""
    def __init__(self, classes_hz: list[int]):
        self.classes_hz = list(classes_hz)
        self.hz_to_class = {hz: i for i, hz in enumerate(self.classes_hz)}
        self.class_to_hz = {i: hz for hz, i in self.hz_to_class.items()}

    def encode(self, y_hz: np.ndarray) -> np.ndarray:
        return np.array([self.hz_to_class[int(h)] for h in y_hz], dtype=np.int64)

    def decode(self, y_cls: np.ndarray) -> np.ndarray:
        return np.array([self.class_to_hz[int(c)] for c in y_cls], dtype=np.int64)


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


# BUILD TRAINING DATA
# We read CSV and get the shape of X
# (N, C, T) = (num_windows, num_channels, window_len_samples)
def load_windows_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray, DatasetInfo]:
    """
    Reads window-level CSV and returns:
      X: (N, C, T)
      y_hz: (N,)
      info: DatasetInfo
    Expected columns:
      window_idx, ui_state, is_trimmed, is_bad, sample_idx, eeg1..eeg8, testfreq_e, testfreq_hz
    """
    df = pd.read_csv(csv_path)
    cols = list(df.columns)

    # Find channel columns by slicing between sample_idx and testfreq_e
    tf_e_i = cols.index("testfreq_e")
    sample_i = cols.index("sample_idx")
    ch_cols = cols[sample_i + 1 : tf_e_i]
    n_ch = len(ch_cols)

    # Filters
    df = df[df["is_trimmed"] == 1].copy()
    df = df[df["is_bad"] != 1].copy()  # keep only NOT-bad

    grouped = df.groupby("window_idx", sort=True)

    windows = []
    labels_hz = []

    for wid, g in grouped:
        g = g.sort_values("sample_idx")
        tf_hz = int(g["testfreq_hz"].iloc[0])
        if tf_hz < 0:
            continue

        x_tc = g[ch_cols].to_numpy(dtype=np.float32)  # (T, C)
        if x_tc.ndim != 2 or x_tc.shape[1] != n_ch:
            continue

        windows.append(x_tc.T)       # (C, T)
        labels_hz.append(tf_hz)

    X = np.stack(windows, axis=0)          # (N, C, T)
    y_hz = np.array(labels_hz, dtype=np.int64)

    classes_hz = sorted(set(y_hz.tolist()))
    info = DatasetInfo(
        ch_cols=ch_cols,
        n_ch=n_ch,
        n_time=int(X.shape[2]),
        classes_hz=classes_hz,
    )
    return X, y_hz, info


def write_meta(out_dir: Path, *, arch: str, info: DatasetInfo, norm: NormalizerSpec, subject: str, session: str):
    meta = {
        "arch": arch,
        "subject": subject,
        "session": session,
        "input": {
            "format": "N1CT",
            "C": info.n_ch,
            "T": info.n_time,
        },
        "classes_hz": info.classes_hz,
        "normalization": {
            "type": norm.type,
            "eps": norm.eps,
        },
    }
    out_path = out_dir / "meta.json"
    out_path.write_text(json.dumps(meta, indent=2))
    print(f"[PY] wrote meta -> {out_path.resolve()}")

# ONNX export will happen by designated trainers

# -----------------------------
# Orchestrator
# -----------------------------
def main():
    args = get_args()
    csv_path = Path(args.data)
    out_dir = Path(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 0) Device (pipeline owns)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[PY] Using hw device:", device)

    # 1) Load
    X, y_hz, info = load_windows_csv(csv_path)
    print("[PY] X:", X.shape, "y_hz:", y_hz.shape, "classes_hz:", info.classes_hz)

    # 2) Preprocess
    norm = NormalizerSpec()
    X = norm.apply(X).astype(np.float32)
    encoder = LabelEncoderHz(info.classes_hz)
    y_cls = encoder.encode(y_hz)

    # 3) Train + export (delegated to arch-specific trainer)
    if args.arch == "CNN":
        model, best_state, history = train_cnn(
            X=X,
            y_cls=y_cls,
            n_ch=info.n_ch,
            n_time=info.n_time,
            n_classes=len(info.classes_hz),
            seed=args.seed,
            val_ratio=args.val_ratio,
            batch_size=args.batch_size,
            lr=args.lr,
            max_epochs=args.max_epochs,
            patience=args.patience,
            min_delta=args.min_delta,
            device=device,
        )

        onnx_path = export_cnn_onnx(
            model=model,
            n_ch=info.n_ch,
            n_time=info.n_time,
            out_path=(out_dir / "ssvep_model.onnx"),
        )
        print(f"[PY] wrote onnx -> {onnx_path.resolve()}")

    elif args.arch == "SVM":
        raise NotImplementedError("SVM path not wired in this script yet.")
    
    else:
        raise ValueError(f"Unknown --arch '{args.arch}'")
    
    # 4) Shared meta
    write_meta(
        out_dir,
        arch=args.arch,
        info=info,
        norm=norm,
        subject=args.subject,
        session=args.session,
        encoder=encoder,
    )

    return 0


# ENTRY POINT
if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"[PY] FATAL TRAINING ERROR: {e}", file=sys.stderr)
        sys.exit(1)
