#!/usr/bin/env python3
"""
analyze_run_log.py

Run from the python training directory:
  cd "C:\\Users\\fsdma\\capstone\\capstone\\CapstoneProject\\model train\\python"
  python analyze_run_log.py

Analyzes run_classifier_log.csv to calibrate softmax thresholds.
"""

import pandas as pd
import numpy as np
from pathlib import Path

LOG_PATH = r"C:\Users\fsdma\capstone\capstone\data\DEMO\2026-03-24_11-16-26\run_classifier_log.csv"

df = pd.read_csv(LOG_PATH)
df = df[df["was_used"] == 1].copy()  # only windows where inference ran

print(f"\nTotal inference windows: {len(df)}")
print(f"Columns: {list(df.columns)}\n")

# ── 1) Overall prediction distribution ──────────────────────────────────────
print("=" * 60)
print("1) RAW MODEL PREDICTION DISTRIBUTION (onnx_class_raw)")
print("=" * 60)
counts = df["onnx_class_raw"].value_counts().sort_index()
for cls, cnt in counts.items():
    label = {0: "LEFT", 1: "RIGHT", 2: "REST"}.get(int(cls), f"cls{cls}")
    print(f"  class {int(cls)} ({label}): {cnt} windows ({100*cnt/len(df):.1f}%)")

# ── 2) Per stim_state breakdown ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("2) PREDICTION BREAKDOWN BY STIM STATE (ground truth)")
print("=" * 60)
for state in ["left", "right", "none", "other"]:
    sub = df[df["stim_state"] == state]
    if len(sub) == 0:
        continue
    print(f"\n  stim_state='{state}' ({len(sub)} windows):")
    for cls in [0, 1, 2]:
        cnt = (sub["onnx_class_raw"] == cls).sum()
        label = {0: "LEFT", 1: "RIGHT", 2: "REST"}[cls]
        print(f"    predicted {label}: {cnt} ({100*cnt/len(sub):.1f}%)")
    print(f"    softmax_2 (REST): mean={sub['softmax_2'].mean():.3f}  "
          f"median={sub['softmax_2'].median():.3f}  "
          f"p10={sub['softmax_2'].quantile(0.10):.3f}  "
          f"p90={sub['softmax_2'].quantile(0.90):.3f}")
    print(f"    softmax_0 (LEFT): mean={sub['softmax_0'].mean():.3f}  "
          f"median={sub['softmax_0'].median():.3f}")
    print(f"    softmax_1 (RIGHT): mean={sub['softmax_1'].mean():.3f}  "
          f"median={sub['softmax_1'].median():.3f}")

# ── 3) Threshold sweep ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3) THRESHOLD SWEEP — what happens at different settings?")
print("   (only for windows where stim_state is known left/right/none)")
print("=" * 60)

known = df[df["stim_state"].isin(["left", "right", "none"])].copy()
if len(known) == 0:
    print("  No windows with known stim_state — log may be demo REST-only.")
    print("  Run a hardware session or a demo session with active SSVEP periods.")
else:
    for ssvep_thresh in [0.50, 0.60, 0.70, 0.80]:
        for rest_veto in [0.15, 0.20, 0.25, 0.30]:
            # Simulate the new logic
            def simulate(row):
                s0, s1, s2 = row["softmax_0"], row["softmax_1"], row["softmax_2"]
                idx_max = int(np.argmax([s0, s1, s2]))
                max_s = max(s0, s1, s2)
                if idx_max in (0, 1):
                    if max_s >= ssvep_thresh and s2 < rest_veto:
                        return idx_max
                    else:
                        return 2
                else:
                    return 2  # always REST when model predicts REST or uncertain

            known["sim_pred"] = known.apply(simulate, axis=1)

            # Accuracy per class
            left_rows  = known[known["stim_state"] == "left"]
            right_rows = known[known["stim_state"] == "right"]
            none_rows  = known[known["stim_state"] == "none"]

            left_acc  = (left_rows["sim_pred"] == 0).mean()  if len(left_rows)  else float("nan")
            right_acc = (right_rows["sim_pred"] == 1).mean() if len(right_rows) else float("nan")
            none_acc  = (none_rows["sim_pred"] == 2).mean()  if len(none_rows)  else float("nan")
            bal_acc   = np.nanmean([left_acc, right_acc, none_acc])

            print(f"  ssvep≥{ssvep_thresh:.2f} rest_veto<{rest_veto:.2f}  →  "
                  f"left={left_acc:.2f}  right={right_acc:.2f}  none={none_acc:.2f}  "
                  f"bal={bal_acc:.2f}")

# ── 4) Softmax distribution during REST periods ──────────────────────────────
print("\n" + "=" * 60)
print("4) SOFTMAX_2 (REST) DISTRIBUTION DURING 'none' STIM PERIODS")
print("   (how often does REST softmax exceed various veto thresholds?)")
print("=" * 60)
none_rows = df[df["stim_state"] == "none"]
if len(none_rows) == 0:
    # Fall back to all windows if no labelled none
    none_rows = df
    print("  (no labelled 'none' windows found — using all windows as proxy)")
for thresh in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    frac = (none_rows["softmax_2"] >= thresh).mean()
    print(f"  softmax_2 >= {thresh:.2f}: {100*frac:.1f}% of REST windows")

# ── 5) Raw softmax histograms (text) ─────────────────────────────────────────
print("\n" + "=" * 60)
print("5) SOFTMAX PERCENTILES BY CLASS (all windows)")
print("=" * 60)
for col, label in [("softmax_0","LEFT"), ("softmax_1","RIGHT"), ("softmax_2","REST")]:
    s = df[col]
    print(f"  {label}: p5={s.quantile(.05):.3f}  p25={s.quantile(.25):.3f}  "
          f"p50={s.quantile(.50):.3f}  p75={s.quantile(.75):.3f}  "
          f"p95={s.quantile(.95):.3f}  mean={s.mean():.3f}")

print("\nDone. Paste the output above back to Claude for threshold recommendations.\n")