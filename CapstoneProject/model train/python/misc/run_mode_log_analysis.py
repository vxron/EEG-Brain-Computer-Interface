"""
SSVEP Classifier Run Mode Analysis
Analyzes run_classifier_log.csv to validate:
- Prediction accuracy vs ground truth stimulus
- Debounce behavior and stability
- Softmax confidence distributions
- Actuation timing and correctness
- Window quality (artifact rate)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def load_data(csv_path):
    """Load and preprocess the run log CSV."""
    df = pd.read_csv(csv_path)
    
    # Convert timestamp to seconds
    df['time_s'] = df['timestamp_ms'] / 1000.0
    
    # Create ground truth label from stim_state
    label_map = {'left': 0, 'right': 1, 'none': 2, 'other': -1}
    df['ground_truth'] = df['stim_state'].map(label_map)
    
    # Ensure numeric columns are proper types
    df['onnx_class_raw'] = pd.to_numeric(df['onnx_class_raw'], errors='coerce')
    df['num_stable_windows'] = pd.to_numeric(df['num_stable_windows'], errors='coerce')
    
    return df


def compute_accuracy_metrics(df):
    """Compute prediction accuracy vs ground truth."""
    # Only use clean windows with valid predictions
    valid_mask = (df['was_used'] == 1) & (df['ground_truth'] >= 0) & (df['onnx_class_raw'] >= 0)
    valid_df = df[valid_mask].copy()
    
    if len(valid_df) == 0:
        print("WARNING: No valid windows for accuracy computation")
        return None
    
    # Overall accuracy
    correct = (valid_df['onnx_class_raw'] == valid_df['ground_truth']).sum()
    total = len(valid_df)
    accuracy = correct / total if total > 0 else 0
    
    # Per-class accuracy
    class_acc = {}
    for cls_name, cls_idx in [('left', 0), ('right', 1), ('none', 2)]:
        cls_mask = valid_df['ground_truth'] == cls_idx
        if cls_mask.sum() > 0:
            cls_correct = ((valid_df.loc[cls_mask, 'onnx_class_raw'] == cls_idx).sum())
            class_acc[cls_name] = cls_correct / cls_mask.sum()
        else:
            class_acc[cls_name] = np.nan
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(valid_df['ground_truth'], valid_df['onnx_class_raw'], 
                          labels=[0, 1, 2])
    
    results = {
        'overall_accuracy': accuracy,
        'total_windows': total,
        'correct_windows': correct,
        'class_accuracy': class_acc,
        'confusion_matrix': cm
    }
    
    return results


def analyze_debounce(df, threshold=10):
    """Analyze debounce behavior and actuation timing."""
    # Find actuation events
    actuations = df[df['actuation_requested'] == 1].copy()
    
    debounce_stats = {
        'total_actuations': len(actuations),
        'left_actuations': (actuations['actuation_direction'] == 'left').sum(),
        'right_actuations': (actuations['actuation_direction'] == 'right').sum(),
        'avg_stable_at_actuation': actuations['num_stable_windows'].mean() if len(actuations) > 0 else 0,
        'min_stable_at_actuation': actuations['num_stable_windows'].min() if len(actuations) > 0 else 0,
    }
    
    # Check if actuations happened when stable count exceeded threshold
    if len(actuations) > 0:
        valid_actuations = (actuations['num_stable_windows'] > threshold).sum()
        debounce_stats['valid_actuation_timing'] = valid_actuations / len(actuations)
    else:
        debounce_stats['valid_actuation_timing'] = np.nan
    
    # Analyze stability transitions (prediction changes)
    pred_changes = (df['predicted_state'].shift(1) != df['predicted_state']).sum()
    debounce_stats['total_prediction_changes'] = pred_changes
    
    # Time between actuations
    if len(actuations) > 1:
        actuation_intervals = actuations['time_s'].diff().dropna()
        debounce_stats['mean_actuation_interval_s'] = actuation_intervals.mean()
        debounce_stats['min_actuation_interval_s'] = actuation_intervals.min()
    
    return debounce_stats


def analyze_confidence(df):
    """Analyze softmax confidence distributions."""
    valid_df = df[df['was_used'] == 1].copy()
    
    if len(valid_df) == 0:
        return None
    
    # Max softmax per window
    valid_df['max_softmax'] = valid_df[['softmax_0', 'softmax_1', 'softmax_2']].max(axis=1)
    
    # Confidence when correct vs incorrect
    correct_mask = valid_df['onnx_class_raw'] == valid_df['ground_truth']
    
    stats = {
        'mean_confidence_overall': valid_df['max_softmax'].mean(),
        'mean_confidence_correct': valid_df.loc[correct_mask, 'max_softmax'].mean() if correct_mask.sum() > 0 else np.nan,
        'mean_confidence_incorrect': valid_df.loc[~correct_mask, 'max_softmax'].mean() if (~correct_mask).sum() > 0 else np.nan,
        'windows_below_threshold': (valid_df['onnx_class_raw'] == -1).sum(),  # Below confidence threshold
    }
    
    return stats, valid_df


def plot_timeline(df, save_path):
    """Plot prediction timeline vs ground truth."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    
    # 1. Ground truth vs prediction
    ax = axes[0]
    ax.plot(df['time_s'], df['ground_truth'], 'k-', label='Ground Truth', linewidth=2, alpha=0.7)
    ax.plot(df['time_s'], df['onnx_class_raw'], 'r.', label='Prediction', markersize=4, alpha=0.6)
    ax.axhline(y=0, color='blue', linestyle='--', alpha=0.3, label='Left')
    ax.axhline(y=1, color='green', linestyle='--', alpha=0.3, label='Right')
    ax.axhline(y=2, color='gray', linestyle='--', alpha=0.3, label='None')
    ax.set_ylabel('Class')
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Left', 'Right', 'None'])
    ax.legend(loc='upper right')
    ax.set_title('Prediction vs Ground Truth Timeline')
    ax.grid(True, alpha=0.3)
    
    # 2. Debounce counter
    ax = axes[1]
    ax.plot(df['time_s'], df['num_stable_windows'], 'b-', linewidth=1.5)
    threshold = df['stable_target'].iloc[0] if 'stable_target' in df.columns else 10
    ax.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    ax.fill_between(df['time_s'], 0, df['num_stable_windows'], alpha=0.3)
    ax.set_ylabel('Stable Windows')
    ax.set_title('Debounce Counter')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Actuation events
    ax = axes[2]
    actuations = df[df['actuation_requested'] == 1]
    if len(actuations) > 0:
        left_act = actuations[actuations['actuation_direction'] == 'left']
        right_act = actuations[actuations['actuation_direction'] == 'right']
        ax.scatter(left_act['time_s'], [0]*len(left_act), c='blue', s=100, marker='^', label='Left Actuation', zorder=3)
        ax.scatter(right_act['time_s'], [0]*len(right_act), c='green', s=100, marker='v', label='Right Actuation', zorder=3)
    ax.set_ylabel('Actuation')
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_title('Actuation Events')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Confidence (max softmax)
    ax = axes[3]
    valid_mask = df['was_used'] == 1
    max_softmax = df.loc[valid_mask, ['softmax_0', 'softmax_1', 'softmax_2']].max(axis=1)
    ax.plot(df.loc[valid_mask, 'time_s'], max_softmax, 'g-', linewidth=1, alpha=0.7)
    ax.fill_between(df.loc[valid_mask, 'time_s'], 0, max_softmax, alpha=0.3, color='green')
    ax.axhline(y=0.5, color='orange', linestyle='--', label='50% confidence')
    ax.set_ylabel('Max Softmax')
    ax.set_xlabel('Time (seconds)')
    ax.set_title('Prediction Confidence')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Timeline plot saved to {save_path}")
    plt.close()


def plot_confusion_matrix(cm, save_path):
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Left', 'Right', 'None'],
                yticklabels=['Left', 'Right', 'None'],
                cbar_kws={'label': 'Count'})
    
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_confidence_distributions(df_valid, save_path):
    """Plot softmax confidence distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Overall max softmax distribution
    ax = axes[0, 0]
    max_softmax = df_valid[['softmax_0', 'softmax_1', 'softmax_2']].max(axis=1)
    ax.hist(max_softmax, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=max_softmax.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {max_softmax.mean():.3f}')
    ax.set_xlabel('Max Softmax')
    ax.set_ylabel('Count')
    ax.set_title('Overall Confidence Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Per-class softmax distributions
    ax = axes[0, 1]
    for i, (name, color) in enumerate([('Left', 'blue'), ('Right', 'green'), ('None', 'gray')]):
        class_mask = df_valid['ground_truth'] == i
        if class_mask.sum() > 0:
            class_softmax = df_valid.loc[class_mask, f'softmax_{i}']
            ax.hist(class_softmax, bins=30, alpha=0.5, label=name, color=color, edgecolor='black')
    ax.set_xlabel('Softmax Value')
    ax.set_ylabel('Count')
    ax.set_title('Per-Class Softmax Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Correct vs Incorrect confidence
    ax = axes[1, 0]
    correct_mask = df_valid['onnx_class_raw'] == df_valid['ground_truth']
    correct_conf = df_valid.loc[correct_mask, ['softmax_0', 'softmax_1', 'softmax_2']].max(axis=1)
    incorrect_conf = df_valid.loc[~correct_mask, ['softmax_0', 'softmax_1', 'softmax_2']].max(axis=1)
    
    ax.hist(correct_conf, bins=30, alpha=0.6, label=f'Correct (n={len(correct_conf)})', color='green', edgecolor='black')
    ax.hist(incorrect_conf, bins=30, alpha=0.6, label=f'Incorrect (n={len(incorrect_conf)})', color='red', edgecolor='black')
    ax.set_xlabel('Max Softmax')
    ax.set_ylabel('Count')
    ax.set_title('Confidence: Correct vs Incorrect Predictions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Logit distributions (raw outputs before softmax)
    ax = axes[1, 1]
    for i, (name, color) in enumerate([('Logit 0 (Left)', 'blue'), ('Logit 1 (Right)', 'green'), ('Logit 2 (None)', 'gray')]):
        logit_vals = df_valid[f'logit_{i}'].dropna()
        ax.hist(logit_vals, bins=40, alpha=0.5, label=name, color=color, edgecolor='black')
    ax.set_xlabel('Logit Value')
    ax.set_ylabel('Count')
    ax.set_title('Raw Logit Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Confidence distributions saved to {save_path}")
    plt.close()


def generate_report(df, acc_metrics, debounce_stats, conf_stats):
    """Generate text report."""
    report = []
    report.append("=" * 80)
    report.append("SSVEP CLASSIFIER RUN MODE ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Dataset info
    report.append("DATASET OVERVIEW:")
    report.append(f"  Total windows logged: {len(df)}")
    report.append(f"  Windows used (clean): {(df['was_used'] == 1).sum()}")
    report.append(f"  Windows rejected (artifacts): {(df['is_artifactual'] == 1).sum()}")
    report.append(f"  Duration: {df['time_s'].max():.1f} seconds")
    report.append("")
    
    # Accuracy
    if acc_metrics:
        report.append("PREDICTION ACCURACY:")
        report.append(f"  Overall: {acc_metrics['overall_accuracy']*100:.2f}% ({acc_metrics['correct_windows']}/{acc_metrics['total_windows']})")
        report.append(f"  Per-class accuracy:")
        for cls_name, acc in acc_metrics['class_accuracy'].items():
            if not np.isnan(acc):
                report.append(f"    {cls_name.capitalize():6s}: {acc*100:.2f}%")
        report.append("")
    
    # Debounce
    report.append("DEBOUNCE & ACTUATION:")
    report.append(f"  Total actuations: {debounce_stats['total_actuations']}")
    report.append(f"    Left:  {debounce_stats['left_actuations']}")
    report.append(f"    Right: {debounce_stats['right_actuations']}")
    if debounce_stats['total_actuations'] > 0:
        report.append(f"  Avg stable windows at actuation: {debounce_stats['avg_stable_at_actuation']:.1f}")
        report.append(f"  Min stable windows at actuation: {debounce_stats['min_stable_at_actuation']}")
        if not np.isnan(debounce_stats.get('valid_actuation_timing', np.nan)):
            report.append(f"  Valid actuation timing: {debounce_stats['valid_actuation_timing']*100:.1f}%")
    report.append(f"  Total prediction changes: {debounce_stats['total_prediction_changes']}")
    if 'mean_actuation_interval_s' in debounce_stats:
        report.append(f"  Mean time between actuations: {debounce_stats['mean_actuation_interval_s']:.2f}s")
    report.append("")
    
    # Confidence
    if conf_stats:
        report.append("CONFIDENCE METRICS:")
        report.append(f"  Mean confidence overall: {conf_stats['mean_confidence_overall']:.3f}")
        if not np.isnan(conf_stats['mean_confidence_correct']):
            report.append(f"  Mean confidence (correct): {conf_stats['mean_confidence_correct']:.3f}")
        if not np.isnan(conf_stats['mean_confidence_incorrect']):
            report.append(f"  Mean confidence (incorrect): {conf_stats['mean_confidence_incorrect']:.3f}")
        report.append(f"  Windows below threshold: {conf_stats['windows_below_threshold']}")
        report.append("")
    
    # Quality metrics
    artifact_rate = (df['is_artifactual'] == 1).sum() / len(df) if len(df) > 0 else 0
    report.append("SIGNAL QUALITY:")
    report.append(f"  Artifact rate: {artifact_rate*100:.2f}%")
    report.append("")
    
    report.append("=" * 80)
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='Analyze SSVEP run mode classifier log')
    parser.add_argument('csv_path', type=str, help='Path to run_classifier_log.csv')
    parser.add_argument('--output-dir', type=str, default='.', help='Directory for output plots and report')
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from {csv_path}...")
    df = load_data(csv_path)
    
    print("Computing accuracy metrics...")
    acc_metrics = compute_accuracy_metrics(df)
    
    print("Analyzing debounce behavior...")
    debounce_stats = analyze_debounce(df)
    
    print("Analyzing confidence distributions...")
    conf_stats, df_valid = analyze_confidence(df)
    
    print("Generating plots...")
    plot_timeline(df, output_dir / 'timeline.png')
    
    if acc_metrics and acc_metrics['confusion_matrix'] is not None:
        plot_confusion_matrix(acc_metrics['confusion_matrix'], output_dir / 'confusion_matrix.png')
    
    if df_valid is not None and len(df_valid) > 0:
        plot_confidence_distributions(df_valid, output_dir / 'confidence_distributions.png')
    
    print("Generating report...")
    report = generate_report(df, acc_metrics, debounce_stats, conf_stats)
    
    report_path = output_dir / 'analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(report)
    print(f"\nReport saved to {report_path}")
    print(f"Plots saved to {output_dir}")


if __name__ == '__main__':
    main()