"""
Reconstruct holdout set from calibration CSVs and evaluate ONNX model.
CSV: one row per sample, 560 rows per window, grouped by window_idx.
window_idx restarts per file so we make it globally unique.
"""

import argparse
import numpy as np
import pandas as pd
import onnxruntime as ort
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score

W          = 560
N_CHANNELS = 8
HOLDOUT_FRAC = 0.15
BLOCK_SIZE   = 7
RANDOM_SEED  = 42
EEG_COLS     = [f'eeg{i}' for i in range(1, 9)]


def load_and_build_windows(csv_paths, freq_left, freq_right):
    dfs = []
    offset = 0
    for p in csv_paths:
        df = pd.read_csv(p)
        df = df[df['is_bad'] == 0].copy()
        # Make window_idx globally unique across files
        df['window_idx'] = df['window_idx'] + offset
        offset += df['window_idx'].max() + 1
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total rows after bad-sample filter: {len(df)}")
    print(f"Unique window_idx values: {df['window_idx'].nunique()}")
    print(f"testfreq_hz distribution:\n{df['testfreq_hz'].value_counts()}")

    grouped = df.groupby('window_idx')
    windows, labels = [], []
    skipped = 0

    for widx, grp in grouped:
        if len(grp) != W:
            skipped += 1
            continue
        freq = grp['testfreq_hz'].iloc[0]
        if freq == freq_left:
            label = 0
        elif freq == freq_right:
            label = 1
        else:
            label = 2  # rest / other freq
        eeg = grp[EEG_COLS].values.astype(np.float32)  # [560, 8]
        windows.append(eeg.T)  # [8, 560]
        labels.append(label)

    if skipped:
        print(f"Skipped {skipped} windows with wrong row count (expected {W})")

    X = np.stack(windows, axis=0)
    y = np.array(labels, dtype=np.int64)
    print(f"Windows built: {len(y)} (c0={(y==0).sum()} c1={(y==1).sum()} c2={(y==2).sum()})")
    return X, y


def holdout_split(X, y):
    np.random.seed(RANDOM_SEED)
    n = len(y)
    groups = np.arange(n) // BLOCK_SIZE
    all_groups = np.unique(groups)
    np.random.shuffle(all_groups)

    target_total = int(np.round(n * HOLDOUT_FRAC))
    holdout_mask = np.zeros(n, dtype=bool)

    for g in all_groups:
        g_idx = np.where(groups == g)[0]
        if holdout_mask.sum() + len(g_idx) <= target_total * 1.05:
            holdout_mask[g_idx] = True
        if holdout_mask.sum() >= target_total:
            break

    X_h, y_h = X[holdout_mask], y[holdout_mask]
    print(f"Holdout: {len(y_h)} windows (c0={(y_h==0).sum()} c1={(y_h==1).sum()} c2={(y_h==2).sum()})")
    print(f"Expected from log:          (c0=75  c1=67  c2=113, total=255)")
    return X_h, y_h


def run_onnx(model_path, X):
    sess = ort.InferenceSession(str(model_path))
    inp  = sess.get_inputs()[0]
    print(f"ONNX input: {inp.name} shape={inp.shape}")
    Xb = X[:, np.newaxis, :, :]  # [N,1,8,560]
    if len(inp.shape) == 4 and inp.shape[2] == W and inp.shape[3] == N_CHANNELS:
        print("Transposing to [N,1,T,C]")
        Xb = Xb.transpose(0, 1, 3, 2)
    out = []
    for i in range(0, len(Xb), 256):
        out.append(sess.run(None, {inp.name: Xb[i:i+256]})[0])
    return np.concatenate(out, axis=0)


def plot_cm(cm, save_path, title, freq_left, freq_right):
    labels = [f'Left ({int(freq_left)} Hz)', f'Right ({int(freq_right)} Hz)', 'None']
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    for i in range(3):
        for j in range(3):
            ax.text(j+0.5, i+0.38, str(cm[i,j]), ha='center', va='center', fontsize=13,
                    fontweight='bold', color='white' if cm_norm[i,j] > 0.5 else 'black')
            ax.text(j+0.5, i+0.68, f'({cm_norm[i,j]*100:.1f}%)', ha='center', va='center',
                    fontsize=9, color='white' if cm_norm[i,j] > 0.5 else 'gray')
    ax.set_ylabel('True Label'); ax.set_xlabel('Predicted Label'); ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',      required=True)
    parser.add_argument('--csvs',       nargs='+', required=True)
    parser.add_argument('--freq_left',  type=float, default=14)
    parser.add_argument('--freq_right', type=float, default=10)
    parser.add_argument('--output_dir', default='.')
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("\n=== Loading CSVs and building windows ===")
    X, y = load_and_build_windows(args.csvs, args.freq_left, args.freq_right)

    print("\n=== Splitting holdout ===")
    X_hold, y_hold = holdout_split(X, y)

    print("\n=== Running ONNX inference ===")
    logits = run_onnx(args.model, X_hold)
    y_pred = logits.argmax(axis=1)

    present_classes = np.unique(np.concatenate([y_hold, y_pred]))
    all3 = [0, 1, 2]
    tnames_all = [f'Left({int(args.freq_left)}Hz)', f'Right({int(args.freq_right)}Hz)', 'None']

    print("\n=== Results ===")
    cm      = confusion_matrix(y_hold, y_pred, labels=all3)
    acc     = (y_hold == y_pred).mean()
    bal_acc = balanced_accuracy_score(y_hold, y_pred)

    print(f"Overall accuracy:  {acc*100:.2f}%")
    print(f"Balanced accuracy: {bal_acc*100:.2f}%")
    report = classification_report(y_hold, y_pred, labels=all3, target_names=tnames_all, digits=4, zero_division=0)
    print(report)
    print("Confusion matrix (rows=True, cols=Predicted):")
    for i, lbl in enumerate(tnames_all):
        print(f"  {lbl:<18}: {cm[i]}")

    plot_cm(cm, out / 'confusion_matrix_holdout.png',
            title=f"Holdout Confusion Matrix — Veronica ({int(args.freq_left)} vs {int(args.freq_right)} Hz)\n"
                  f"Bal. Acc = {bal_acc*100:.2f}%  |  Acc = {acc*100:.2f}%",
            freq_left=args.freq_left, freq_right=args.freq_right)

    with open(out / 'holdout_report.txt', 'w') as f:
        f.write(f"Overall accuracy:  {acc*100:.4f}%\n")
        f.write(f"Balanced accuracy: {bal_acc*100:.4f}%\n\n")
        f.write(report)
        f.write(f"\nConfusion matrix:\n{cm}\n")

    print(f"\nAll results saved to {out}")


if __name__ == '__main__':
    main()