"""
Real-Subject SSVEP Run Mode Analysis (no ground truth labels)
Usage: python analyze_real_subject_run_log.py <run_classifier_log.csv> --output-dir ./results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

sns.set_style("whitegrid")


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df['time_s'] = df['timestamp_ms'] / 1000.0
    df['onnx_class_raw'] = pd.to_numeric(df['onnx_class_raw'], errors='coerce')
    df['num_stable_windows'] = pd.to_numeric(df['num_stable_windows'], errors='coerce')
    for col in ['softmax_0', 'softmax_1', 'softmax_2', 'logit_0', 'logit_1', 'logit_2']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def analyze_windows(df):
    total       = len(df)
    artifact    = (df['is_artifactual'] == 1).sum()
    used        = (df['was_used'] == 1).sum()
    duration    = df['time_s'].max()
    return {
        'total': total,
        'artifact': artifact,
        'used': used,
        'artifact_rate_pct': 100 * artifact / total if total else 0,
        'duration_s': duration,
    }


def analyze_predictions(df):
    used = df[df['was_used'] == 1].copy()
    used['max_softmax'] = used[['softmax_0','softmax_1','softmax_2']].max(axis=1)

    class_counts = used['predicted_state'].value_counts().to_dict()

    # Confidence stats
    conf = used['max_softmax']
    below_thresh = (used['onnx_class_raw'] == -1).sum()  # unknown predictions

    # Prediction change rate (proxy for stability)
    pred_changes = (used['predicted_state'].shift(1) != used['predicted_state']).sum()

    return {
        'class_distribution': class_counts,
        'mean_confidence': conf.mean(),
        'median_confidence': conf.median(),
        'p25_confidence': conf.quantile(0.25),
        'p75_confidence': conf.quantile(0.75),
        'p5_confidence':  conf.quantile(0.05),
        'below_threshold_count': below_thresh,
        'below_threshold_pct': 100 * below_thresh / len(used) if len(used) else 0,
        'prediction_changes': pred_changes,
        'changes_per_minute': pred_changes / (df['time_s'].max() / 60) if df['time_s'].max() > 0 else 0,
        'used_df': used,
    }


def analyze_debounce(df):
    acts = df[df['actuation_requested'] == 1].copy()
    left_acts  = (acts['actuation_direction'] == 'left').sum()
    right_acts = (acts['actuation_direction'] == 'right').sum()

    intervals = None
    if len(acts) > 1:
        intervals = acts['time_s'].diff().dropna()

    # How often does debounce counter reach target without firing?
    # A reset is when num_stable_windows drops after being > 0 but no actuation
    used = df[df['was_used'] == 1].copy()
    counter = used['num_stable_windows'].values
    resets = 0
    for i in range(1, len(counter)):
        if counter[i] < counter[i-1] and counter[i-1] > 0:
            # Check no actuation at this point
            resets += 1

    return {
        'total_actuations': len(acts),
        'left_actuations': left_acts,
        'right_actuations': right_acts,
        'mean_interval_s': intervals.mean() if intervals is not None else None,
        'min_interval_s': intervals.min() if intervals is not None else None,
        'max_interval_s': intervals.max() if intervals is not None else None,
        'debounce_resets': resets,
    }


def plot_timeline(df, save_path):
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    used = df[df['was_used'] == 1]

    # 1. Predicted class over time
    ax = axes[0]
    class_map = {'left': 0, 'right': 1, 'none': 2, 'unknown': -1}
    pred_numeric = used['predicted_state'].map(class_map).fillna(-1)
    ax.plot(used['time_s'], pred_numeric, '.', markersize=2, alpha=0.5, color='steelblue')
    ax.set_yticks([-1, 0, 1, 2])
    ax.set_yticklabels(['Unknown', 'Left', 'Right', 'None'])
    ax.set_ylabel('Predicted Class')
    ax.set_title('Predicted State Over Time (Real Subject)')

    # Mark actuations
    acts = df[df['actuation_requested'] == 1]
    for _, row in acts.iterrows():
        color = 'blue' if row['actuation_direction'] == 'left' else 'green'
        ax.axvline(x=row['time_s'], color=color, alpha=0.7, linewidth=1.5)

    # 2. Debounce counter
    ax = axes[1]
    ax.plot(used['time_s'], used['num_stable_windows'], color='darkorange', linewidth=1)
    threshold = df['stable_target'].mode()[0] if 'stable_target' in df.columns else 10
    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5, label=f'Threshold = {threshold}')
    ax.fill_between(used['time_s'], 0, used['num_stable_windows'], alpha=0.2, color='darkorange')
    ax.set_ylabel('Stable Windows')
    ax.set_title('Debounce Counter')
    ax.legend()

    # 3. Max softmax confidence
    ax = axes[2]
    max_sm = used[['softmax_0','softmax_1','softmax_2']].max(axis=1)
    ax.plot(used['time_s'], max_sm, color='seagreen', linewidth=0.8, alpha=0.8)
    ax.fill_between(used['time_s'], 0, max_sm, alpha=0.15, color='seagreen')
    ax.axhline(y=0.75, color='red', linestyle='--', linewidth=1.5, label='75% threshold')
    ax.set_ylabel('Max Softmax')
    ax.set_xlabel('Time (s)')
    ax.set_title('Prediction Confidence')
    ax.set_ylim(0, 1.05)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Timeline saved: {save_path}")
    plt.close()


def plot_confidence_and_logits(used_df, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Max softmax histogram
    ax = axes[0]
    max_sm = used_df[['softmax_0','softmax_1','softmax_2']].max(axis=1)
    ax.hist(max_sm, bins=60, color='steelblue', edgecolor='black', alpha=0.8)
    ax.axvline(max_sm.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {max_sm.mean():.3f}')
    ax.axvline(0.75, color='orange', linestyle='--', linewidth=2, label='Threshold = 0.75')
    ax.set_xlabel('Max Softmax')
    ax.set_ylabel('Count')
    ax.set_title('Confidence Distribution')
    ax.legend()

    # Per-class softmax distributions
    ax = axes[1]
    colors = ['blue', 'green', 'gray']
    labels = ['Softmax(Left)', 'Softmax(Right)', 'Softmax(None)']
    for i, (col, color, label) in enumerate(zip(['softmax_0','softmax_1','softmax_2'], colors, labels)):
        ax.hist(used_df[col].dropna(), bins=40, alpha=0.4, color=color, label=label, edgecolor='black')
    ax.set_xlabel('Softmax Value')
    ax.set_ylabel('Count')
    ax.set_title('Per-Class Softmax Distributions')
    ax.legend()

    # Logit distributions
    ax = axes[2]
    for i, (col, color, label) in enumerate(zip(['logit_0','logit_1','logit_2'], colors,
                                                  ['Logit(Left)', 'Logit(Right)', 'Logit(None)'])):
        ax.hist(used_df[col].dropna(), bins=40, alpha=0.4, color=color, label=label, edgecolor='black')
    ax.set_xlabel('Logit Value')
    ax.set_ylabel('Count')
    ax.set_title('Raw Logit Distributions')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Confidence plot saved: {save_path}")
    plt.close()


def plot_prediction_class_distribution(pred_stats, save_path):
    dist = pred_stats['class_distribution']
    labels = list(dist.keys())
    counts = list(dist.values())

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, counts, color=['steelblue','seagreen','gray','salmon'])
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('Window Count')
    ax.set_title('Prediction Class Distribution (Real Subject Run Mode)')
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                str(count), ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Class distribution saved: {save_path}")
    plt.close()


def generate_report(win, pred, deb):
    lines = []
    lines.append("=" * 70)
    lines.append("REAL-SUBJECT RUN MODE ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"SESSION DURATION:   {win['duration_s']:.1f} s  ({win['duration_s']/60:.1f} min)")
    lines.append(f"TOTAL WINDOWS:      {win['total']}")
    lines.append(f"  Artifactual:      {win['artifact']}  ({win['artifact_rate_pct']:.1f}%)")
    lines.append(f"  Used (clean):     {win['used']}  ({100*win['used']/win['total']:.1f}%)")
    lines.append("")
    lines.append("PREDICTION CLASS DISTRIBUTION:")
    for cls, count in sorted(pred['class_distribution'].items()):
        pct = 100 * count / win['used'] if win['used'] else 0
        lines.append(f"  {cls:<10}: {count:>7}  ({pct:.1f}%)")
    lines.append("")
    lines.append("SOFTMAX CONFIDENCE (used windows):")
    lines.append(f"  Mean:    {pred['mean_confidence']:.4f}")
    lines.append(f"  Median:  {pred['median_confidence']:.4f}")
    lines.append(f"  P25:     {pred['p25_confidence']:.4f}")
    lines.append(f"  P75:     {pred['p75_confidence']:.4f}")
    lines.append(f"  P5:      {pred['p5_confidence']:.4f}")
    lines.append(f"  Below 0.75 threshold: {pred['below_threshold_count']}  ({pred['below_threshold_pct']:.1f}%)")
    lines.append("")
    lines.append("PREDICTION STABILITY:")
    lines.append(f"  Prediction changes: {pred['prediction_changes']}")
    lines.append(f"  Changes per minute: {pred['changes_per_minute']:.1f}")
    lines.append(f"  Debounce resets (approx): {deb['debounce_resets']}")
    lines.append("")
    lines.append("ACTUATION EVENTS:")
    lines.append(f"  Total:  {deb['total_actuations']}")
    lines.append(f"  Left:   {deb['left_actuations']}")
    lines.append(f"  Right:  {deb['right_actuations']}")
    if deb['mean_interval_s'] is not None:
        lines.append(f"  Mean interval: {deb['mean_interval_s']:.1f} s")
        lines.append(f"  Min interval:  {deb['min_interval_s']:.1f} s")
        lines.append(f"  Max interval:  {deb['max_interval_s']:.1f} s")
    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path')
    parser.add_argument('--output-dir', default='.')
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_data(args.csv_path)

    win  = analyze_windows(df)
    pred = analyze_predictions(df)
    deb  = analyze_debounce(df)

    plot_timeline(df, out / 'timeline_real.png')
    plot_confidence_and_logits(pred['used_df'], out / 'confidence_real.png')
    plot_prediction_class_distribution(pred, out / 'class_distribution_real.png')

    report = generate_report(win, pred, deb)
    print(report)
    (out / 'analysis_report_real.txt').write_text(report)
    print(f"\nResults saved to {out}")


if __name__ == '__main__':
    main()