#!/usr/bin/env python3
"""
Convert one subject from the Tsinghua SSVEP benchmark dataset (S01.mat ... S35.mat)
into:
  - float32 binary: concatenated trials, scan-major interleaved channels
  - JSON metadata: trial list, freq/phase mapping, segment boundaries, channel labels

Keeps all targets/blocks so we can choose which freqs to use at runtime in C++.

Expected MAT content:
  data: shape [64, 1500, 40, 6]  (ch, time, target, block)
  Sampling rate in distributed epochs: 250 Hz
  Epoch: 6 s = 0.5 s pre-stim + 5.5 s post-onset (stim lasts 5.0 s, then 0.5 s blank)

This script keeps the full 1500 samples per trial.
C++ can stream:
  - pre-stim samples [0 : pre_len) as "NONE"
  - stim-on samples [stim_on_start : stim_on_start+stim_on_len) as target label
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

def load_mat_any(path: Path) -> Dict:
    """
    Loads .mat files. Supports:
      - MATLAB v7: scipy.io.loadmat
      - MATLAB v7.3 (HDF5): h5py

    Returns a dict-like object with 'data' etc.
    """
    try:
        import scipy.io
        md = scipy.io.loadmat(str(path), squeeze_me=False, struct_as_record=False)
        return md
    except Exception:
        pass

    # Fallback to v7.3 via h5py
    import h5py
    md = {}
    with h5py.File(str(path), "r") as f:
        # Convert all top-level datasets to numpy arrays
        for k in f.keys():
            try:
                md[k] = f[k][()]
            except Exception:
                # Skip groups or non-array nodes
                continue
    return md

def load_freq_phase(freq_phase_mat: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Freq_Phase.mat which typically contains arrays for stimulus frequency and phase.
    Returns:
      freqs_hz: shape [40]
      phases_rad: shape [40]
    """
    md = load_mat_any(freq_phase_mat)

    # Commonly: 'freqs' / 'phases' or 'freq' / 'phase' etc.
    cand_freq_keys = ["freqs", "freq", "Freq", "FREQ", "frequency", "Frequency"]
    cand_phase_keys = ["phases", "phase", "Phase", "PHASE"]

    freq_arr = None
    phase_arr = None

    for k in cand_freq_keys:
        if k in md:
            freq_arr = np.array(md[k]).squeeze()
            break
    for k in cand_phase_keys:
        if k in md:
            phase_arr = np.array(md[k]).squeeze()
            break

    if freq_arr is None:
        raise KeyError(f"Could not find frequency array in {freq_phase_mat}. Keys={list(md.keys())}")
    if phase_arr is None:
        # Phase isn't strictly required
        phase_arr = np.full_like(freq_arr, fill_value=np.nan, dtype=np.float64)

    # Ensure length 40
    if freq_arr.size != 40:
        raise ValueError(f"Expected 40 freqs, got {freq_arr.size}")
    if phase_arr.size != 40:
        # allow scalar/empty; broadcast to 40
        if phase_arr.size == 1:
            phase_arr = np.full((40,), float(phase_arr.item()), dtype=np.float64)
        else:
            phase_arr = np.full((40,), np.nan, dtype=np.float64)

    return freq_arr.astype(np.float64), phase_arr.astype(np.float64)

def parse_64ch_loc(loc_path: Path) -> List[str]:
    """
    Parse a '64-channels.loc' file.
    These files vary in format. We'll be tolerant:
      - Skip empty/comment lines
      - Tokenize by whitespace
      - Choose the first token that looks like a channel label (contains letters)
        typically token[0] or token[3]/token[4] depending on format.

    Returns a list of 64 labels in dataset order.
    """
    labels: List[str] = []
    with open(loc_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#") or s.startswith("%"):
                continue
            toks = s.replace(",", " ").split()
            if len(toks) < 2:
                continue

            cand = None
            for t in toks:
                if any(c.isalpha() for c in t):
                    t2 = t.strip("\"'").strip()
                    if 1 <= len(t2) <= 6:  # typical EEG labels: Fz, PO7, Oz...
                        cand = t2
                        break
            if cand is None:
                continue

            labels.append(cand)

    if len(labels) < 64:
        raise ValueError(f"Parsed only {len(labels)} labels from {loc_path}, expected 64. "
                         f"Check file format; you may need to adjust parser.")
    if len(labels) > 64:
        labels = labels[:64]
    return labels

def find_channel_indices(all_labels: List[str], desired: List[str]) -> List[int]:
    """
    Find indices of desired labels in all_labels (case-insensitive, strips punctuation).
    """
    def norm(x: str) -> str:
        return x.strip().lower().replace(".", "").replace("_", "")

    map_norm = {norm(lbl): i for i, lbl in enumerate(all_labels)}
    idxs: List[int] = []
    missing: List[str] = []

    for d in desired:
        dn = norm(d)
        if dn in map_norm:
            idxs.append(map_norm[dn])
            continue

        # Try partial match fallback:
        hit = None
        for k, i in map_norm.items():
            if k == dn:
                hit = i
                break
            if k.startswith(dn) or dn.startswith(k):
                hit = i
                break
        if hit is not None:
            idxs.append(hit)
        else:
            missing.append(d)

    if missing:
        raise KeyError(f"Could not find channels {missing} in loc labels: {all_labels}")
    return idxs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject_mat", required=True, type=str, help="Path to Sxx.mat file")
    ap.add_argument("--freq_phase_mat", required=True, type=str, help="Path to Freq_Phase.mat")
    ap.add_argument("--loc", required=True, type=str, help="Path to 64-channels.loc")
    ap.add_argument("--out_dir", required=True, type=str, help="Output directory")
    ap.add_argument("--subject_id", default=None, type=str, help="Override subject ID (e.g., S01)")
    ap.add_argument("--dtype", default="float32", choices=["float32"], help="Binary output dtype")
    ap.add_argument("--keep_full_trial", action="store_true",
                    help="Keep full 1500 samples (default true behavior)")
    ap.add_argument("--stim_on_seconds", type=float, default=5.5,
                    help="How many seconds of stim-on to mark (default 5.5s)")
    ap.add_argument("--pre_stim_seconds", type=float, default=0.5,
                    help="How many seconds pre-stim to mark (default 0.5s)")
    args = ap.parse_args()

    subject_mat = Path(args.subject_mat)
    freq_phase_mat = Path(args.freq_phase_mat)
    loc_path = Path(args.loc)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Subject ID
    sid = args.subject_id
    if sid is None:
        sid = subject_mat.stem  # e.g., "S01"
    sid = str(sid)

    # Load dataset arrays
    md = load_mat_any(subject_mat)
    if "data" not in md:
        raise KeyError(f"'data' not found in {subject_mat}. Keys={list(md.keys())}")
    data = np.array(md["data"])

    # Ensure shape is [64, 1500, 40, 6]
    if data.ndim != 4:
        raise ValueError(f"Expected 4-D data, got shape {data.shape}")
    # Some loaders yield [1500,64,40,6]; try to detect:
    if data.shape[0] != 64 and data.shape[1] == 64:
        # swap axes 0 and 1
        data = np.swapaxes(data, 0, 1)

    if data.shape[0] != 64 or data.shape[1] != 1500 or data.shape[2] != 40 or data.shape[3] != 6:
        raise ValueError(f"Unexpected data shape after normalization: {data.shape} (expected [64,1500,40,6])")

    fs = 250  # per dataset description (epochs already downsampled)
    pre_len = int(round(args.pre_stim_seconds * fs))       # 0.5s -> 125
    stim_on_len = int(round(args.stim_on_seconds * fs))    # 5.0s -> 1250
    stim_on_start = pre_len                                 # start right after pre-stim

    if pre_len <= 0 or stim_on_len <= 0:
        raise ValueError("pre_len and stim_on_len must be positive")
    if stim_on_start + stim_on_len > data.shape[1]:
        raise ValueError(f"Stim-on segment exceeds trial length: start={stim_on_start} len={stim_on_len} "
                         f"trial_len={data.shape[1]}")

    # Channel mapping: dataset 64 labels -> unicorn 8
    all_labels = parse_64ch_loc(loc_path)
    desired_unicorn = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
    ch_idxs = find_channel_indices(all_labels, desired_unicorn)
    chosen_labels = [all_labels[i] for i in ch_idxs]
    n_ch = len(ch_idxs)

    # Load freq/phase mapping
    freqs_hz, phases = load_freq_phase(freq_phase_mat)

    # Output paths
    bin_path = out_dir / f"{sid}_unicorn8_trials.bin"
    json_path = out_dir / f"{sid}_unicorn8_trials.json"

    # We will write trials in this fixed order:
    # for block in 0..5:
    #   for target in 0..39:
    #     trial_idx++
    # And within each trial, we store samples scan-major:
    #   sample 0: ch0,ch1,...ch7
    #   sample 1: ch0,ch1,...ch7
    # to match the acq provider's layout dest[NUM_CH*i + ch]
    blocks = 6
    targets = 40
    trial_len = data.shape[1]  # 1500

    trial_list = []
    dtype = np.float32
    # Precompute total floats: trials * samples * channels
    n_trials = blocks * targets
    total_floats = n_trials * trial_len * n_ch

    # Stream-write to avoid huge RAM usage
    with open(bin_path, "wb") as fb:
        trial_idx = 0
        for b in range(blocks):
            for t in range(targets):
                # Extract [64,1500] for this target+block
                x = data[:, :, t, b]  # [64,1500]
                # Select unicorn channels
                x8 = x[ch_idxs, :]    # [8,1500]
                # Convert to float32
                x8 = np.asarray(x8, dtype=dtype)

                # Reorder to scan-major interleaved channels:
                # Current x8 is [ch, time]. We want [time, ch] then flatten row-major.
                interleaved = x8.T.reshape(-1)  # time-major, contiguous

                fb.write(interleaved.tobytes(order="C"))

                trial_list.append({
                    "trial_idx": trial_idx,
                    "block_idx": b,
                    "target_idx": t,
                    "freq_hz": float(freqs_hz[t]),
                    "phase": float(phases[t]) if np.isfinite(phases[t]) else None,
                    # Offsets in the binary file (in float32 units):
                    "start_float_index": int(trial_idx * trial_len * n_ch),
                    "n_samples": int(trial_len),
                })
                trial_idx += 1

    meta = {
        "dataset": "Tsinghua_SSVEP_Benchmark",
        "subject_id": sid,
        "fs_hz": fs,
        "trial_len_samples": int(trial_len),     # 1500
        "n_channels": int(n_ch),                 # 8
        "n_targets":  int(data.shape[2]),        # frequencies = 40
        "n_blocks":   int(data.shape[3]),        # 6
        "n_trials_total": int(data.shape[2]*data.shape[3]),
        "channel_labels": chosen_labels,
        "channel_labels_requested": desired_unicorn,
        "layout": {
            "storage": "trial_concatenated",
            "sample_order": "time_major_interleaved_channels",
            "per_sample_stride": int(n_ch),
            "dtype": "float32",
        },
        "segments": {
            "pre_stim": {
                "start": 0,
                "len": int(pre_len),
                "note": "Use this for NoSSVEP class (baseline)."
            },
            "stim_on": {
                "start": int(stim_on_start),
                "len": int(stim_on_len),
                "note": "Use this for labeled SSVEP windows."
            }
        },
        "targets": [{
            "target_idx": int(i),
            "freq_hz": float(freqs_hz[i]),
            "phase": float(phases[i]) if np.isfinite(phases[i]) else None
        } for i in range(targets)],
        "trials": trial_list
    }

    with open(json_path, "w", encoding="utf-8") as fj:
        json.dump(meta, fj, indent=2)

    print(f"[OK] Wrote binary: {bin_path}")
    print(f"[OK] Wrote metadata: {json_path}")
    print(f"[INFO] Trials: {n_trials}  |  Trial samples: {trial_len}  |  Channels: {n_ch}")
    print(f"[INFO] pre_stim: {pre_len} samples ({pre_len/fs:.3f}s), stim_on: {stim_on_len} samples ({stim_on_len/fs:.3f}s)")

if __name__ == "__main__":
    main()
