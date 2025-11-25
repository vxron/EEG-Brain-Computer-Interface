# HADEEL EDIT HERE !!!!

#!/usr/bin/env python3
"""
train_svm.py

Script for training an SSVEP model from calibration data.
This script is called automatically by the C++ backend after a calibration
session. It MUST match the CLI interface expected by C++.

Expected args:
    --data <path>      directory containing calibration data
    --model <path>     directory where ONNX + meta.json should be written
    --subject <str>    subject ID
    --session <str>    session ID
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


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

    return parser.parse_args()


# ------------------------------
# DATA LOADING (stub)
# ------------------------------
def load_data(data_dir: Path):
    """
    Load calibration dataset
    This is a csv outputted from c++ after calib session
    Directory should be data/calib/<subject_id>
    """
    print(f"[PY] Loading data from: {data_dir}")



# ------------------------------
# TRAINING LOGIC (stub)
# ------------------------------
def train_model(X, y):
    print("[PY] Training model (stub)...")



# ------------------------------
# EXPORT ONNX + META (stub)
# ------------------------------
def export_model(model, out_dir: Path, subject: str, session: str):
    """
    Dump ONNX + metadata into out_dir/latest/
    """
    print(f"[PY] Exporting model to: {out_dir}")

    latest = out_dir / "latest"
    latest.mkdir(parents=True, exist_ok=True)

    # TODO: EXPORT REAL ONNX
    onnx_path = latest / "ssvep_model.onnx"
    with open(onnx_path, "w") as f:
        f.write("DUMMY ONNX CONTENT\n")

    # Meta file
    meta = {
        "subject_id": subject,
        "session_id": session,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "description": "TEMP PLACEHOLDER — replace with real metadata",
    }

    meta_path = latest / "ssvep_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("[PY] Export complete.")
    print(f"[PY] ONNX: {onnx_path}")
    print(f"[PY] META: {meta_path}")


# ------------------------------
# MAIN
# ------------------------------
def main():
    args = get_args()

    # Resolve paths
    data_dir = Path(args.data)
    model_dir = Path(args.model)

    print(f"[PY] ================= START TRAINING =================")
    print(f"[PY] Subject:     {args.subject}")
    print(f"[PY] Session:     {args.session}")
    print(f"[PY] Data dir:    {data_dir}")
    print(f"[PY] Model dir:   {model_dir}")

    # Step 1: load dataset
    X, y, meta = load_data(data_dir)

    # Step 2: train
    model = train_model(X, y)

    # Step 3: export ONNX + meta
    export_model(model, model_dir, args.subject, args.session)

    print("[PY] =============== TRAINING DONE ================")
    return 0


# ------------------------------
# ENTRY POINT
# ------------------------------
if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"[PY] FATAL TRAINING ERROR: {e}", file=sys.stderr)
        sys.exit(1)
