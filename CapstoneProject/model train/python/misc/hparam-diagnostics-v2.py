import re
import numpy as np
from collections import defaultdict, Counter
import sys

# Check for matplotlib availability
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    import seaborn as sns
    HAS_MATPLOTLIB = True
    HAS_SEABORN = True
except ImportError:
    HAS_MATPLOTLIB = False
    HAS_SEABORN = False
    print("Warning: matplotlib not available - visualizations disabled")

def parse_log_file(log_path=None, log_text=None):
    """
    Parse debug log from file or text string.
    Returns structured data for analysis.
    """
    if log_path:
        with open(log_path, 'r') as f:
            log_text = f.read()
    elif not log_text:
        raise ValueError("Must provide either log_path or log_text")
    
    return log_text


def analyze_hyperparameter_tuning(log_text, verbose=True):
    if verbose:
        print("\n" + "=" * 80)
        print("HYPERPARAMETER TUNING ANALYSIS")
        print("=" * 80)

    htune_pattern = (
        r'\[HTUNE\]\s+'
        r'score=([\d.]+)\s+'
        r'F1=(\d+)\s+'
        r'D=(\d+)\s+'
        r'k=(\d+)\s+'
        r'p1=(\d+)\s+'
        r'p2=(\d+)\s+'
        r'bs=(\d+)\s+'
        r'lr=([\d.]+)'
        r'(?:\s*norm_mode=([A-Za-z0-9_]+))?'
        r'(?:\s*dropout=([\d.]+))?'
        r'(?:\s*patience=(\d+))?'
        r'(?:\s*val_batch_mode=([A-Za-z0-9_]+))?'
    )

    matches = re.findall(htune_pattern, log_text)
    if not matches:
        if verbose:
            print("No hyperparameter tuning data found in log")
        return None, [], {}

    configs = []
    for m in matches:
        score, F1, D, k, p1, p2, bs, lr, norm_mode, dropout, patience, val_batch_mode = m

        config = {
            "score": float(score),
            "F1": int(F1),
            "D": int(D),
            "k": int(k),
            "p1": int(p1),
            "p2": int(p2),
            "bs": int(bs),
            "lr": float(lr),
        }
        if norm_mode:
            config["norm_mode"] = norm_mode
        if dropout:
            config["dropout"] = float(dropout)
        if patience:
            config["patience"] = int(patience)
        if val_batch_mode:
            config["val_batch_mode"] = val_batch_mode

        configs.append(config)

    if not configs:
        return None, [], {}

    best_config = max(configs, key=lambda c: c["score"])
    
    if verbose:
        print(f"\nFound {len(configs)} hyperparameter configurations")
        print(f"\n🏆 BEST CONFIGURATION (score={best_config['score']:.4f}):")
        print(f"  F1 (filters):     {best_config['F1']}")
        print(f"  D (depth mult):   {best_config['D']}")
        print(f"  k (kernel size):  {best_config['k']}")
        print(f"  p1 (pool 1):      {best_config['p1']}")
        print(f"  p2 (pool 2):      {best_config['p2']}")
        print(f"  batch size:       {best_config['bs']}")
        print(f"  learning rate:    {best_config['lr']}")
        if 'norm_mode' in best_config:
            print(f"  norm mode:        {best_config['norm_mode']}")
        if 'dropout' in best_config:
            print(f"  dropout:          {best_config['dropout']}")
        if 'patience' in best_config:
            print(f"  patience:         {best_config['patience']}")
        if 'val_batch_mode' in best_config:
            print(f"  val_batch_mode:   {best_config['val_batch_mode']}")
    
    # Analyze hyperparameter trends
    analysis = {}
    
    # Group by each hyperparameter (include new ones)
    params = ['F1', 'D', 'k', 'p1', 'p2', 'bs', 'lr']
    if any('norm_mode' in c for c in configs):
        params.append('norm_mode')
    if any('dropout' in c for c in configs):
        params.append('dropout')
    if any('patience' in c for c in configs):
        params.append('patience')
    if any('val_batch_mode' in c for c in configs):
        params.append('val_batch_mode')
    
    param_scores = {p: defaultdict(list) for p in params}
    
    for config in configs:
        for param in params:
            if param in config:
                param_scores[param][config[param]].append(config['score'])
    
    # Compute average score per value
    param_avg_scores = {}
    for param in params:
        if param_scores[param]:  # Only include if we have data
            param_avg_scores[param] = {
                val: np.mean(scores) 
                for val, scores in param_scores[param].items()
            }
    
    if verbose:
        print(f"\n📊 HYPERPARAMETER IMPACT ANALYSIS:")
        print(f"  (showing average score for each parameter value)\n")
        
        for param in params:
            if param not in param_avg_scores:
                continue
                
            avg_scores = param_avg_scores[param]
            if len(avg_scores) > 1:
                best_val = max(avg_scores, key=avg_scores.get)
                worst_val = min(avg_scores, key=avg_scores.get)
                best_score = avg_scores[best_val]
                worst_score = avg_scores[worst_val]
                improvement = best_score - worst_score
                
                print(f"  {param:12s}: best={best_val} (score={best_score:.4f}), worst={worst_val} (score={worst_score:.4f}), Δ={improvement:.4f}")
                
                # Show all values
                sorted_vals = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
                vals_str = ", ".join([f"{val}→{score:.3f}" for val, score in sorted_vals])
                print(f"               All: {vals_str}")
        
        # Score statistics
        scores = [c['score'] for c in configs]
        print(f"\n📈 SCORE STATISTICS:")
        print(f"  Best:    {max(scores):.4f}")
        print(f"  Worst:   {min(scores):.4f}")
        print(f"  Mean:    {np.mean(scores):.4f}")
        print(f"  Std:     {np.std(scores):.4f}")
        print(f"  Range:   {max(scores) - min(scores):.4f}")
    
    analysis = {
        'best_config': best_config,
        'n_configs': len(configs),
        'score_range': (min(c['score'] for c in configs), max(c['score'] for c in configs)),
        'param_avg_scores': param_avg_scores,
        'param_scores': param_scores,
        'all_scores': [c['score'] for c in configs],
    }
    
    return best_config, configs, analysis


def visualize_hyperparameter_tuning(configs, analysis, save_path=None):
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available - skipping visualization")
        return
    
    if not configs:
        print("No configurations to visualize")
        return
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid' if HAS_SEABORN else 'default')
    
    # Create larger figure with better spacing
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.35, 
                  left=0.06, right=0.98, top=0.94, bottom=0.05)
    
    scores = [c['score'] for c in configs]
    best_config = analysis['best_config']
    param_avg_scores = analysis['param_avg_scores']
    param_scores = analysis['param_scores']
    
    # Color palette
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    # ========================
    # ROW 1: Overview
    # ========================
    
    # 1. Score distribution histogram
    ax1 = fig.add_subplot(gs[0, 0])
    n_bins = min(30, len(configs)//3)
    n, bins, patches = ax1.hist(scores, bins=n_bins, edgecolor='black', alpha=0.7, color='#3498db')
    
    # Color bins by performance
    cm = plt.cm.RdYlGn
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = (bin_centers - min(scores)) / (max(scores) - min(scores))
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))
    
    ax1.axvline(best_config['score'], color='red', linestyle='--', linewidth=2.5, label=f'Best: {best_config["score"]:.4f}')
    ax1.axvline(np.mean(scores), color='darkgreen', linestyle='--', linewidth=2.5, label=f'Mean: {np.mean(scores):.4f}')
    ax1.set_xlabel('Score', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Score Distribution', fontsize=12, fontweight='bold', pad=10)
    ax1.legend(fontsize=10, framealpha=0.9)
    ax1.grid(alpha=0.3, linestyle='--')
    
    # 2. Parameter impact bar chart
    ax2 = fig.add_subplot(gs[0, 1:3])
    
    # Get all available parameters
    params = list(param_avg_scores.keys())
    param_impacts = []
    param_labels = []
    
    for param in params:
        avg_scores_dict = param_avg_scores[param]
        if len(avg_scores_dict) > 1:
            impact = max(avg_scores_dict.values()) - min(avg_scores_dict.values())
            param_impacts.append(impact)
            param_labels.append(param)
    
    if param_impacts:
        y_pos = np.arange(len(param_labels))
        bars = ax2.barh(y_pos, param_impacts, color=colors[:len(param_impacts)], 
                       edgecolor='black', linewidth=1.5, alpha=0.8)
        
        # Add value labels
        for i, (bar, impact) in enumerate(zip(bars, param_impacts)):
            ax2.text(impact + max(param_impacts)*0.01, bar.get_y() + bar.get_height()/2, 
                    f'{impact:.4f}', ha='left', va='center', fontsize=10, fontweight='bold')
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(param_labels, fontsize=11, fontweight='bold')
        ax2.set_xlabel('Score Impact (max - min across values)', fontsize=11, fontweight='bold')
        ax2.set_title('Hyperparameter Sensitivity Analysis', fontsize=12, fontweight='bold', pad=10)
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        ax2.set_xlim(0, max(param_impacts) * 1.15)
    
    # 3. Best config summary
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.axis('off')
    
    summary_text = f"""
🏆 BEST CONFIGURATION
Score: {best_config['score']:.4f}

F1 (filters):    {best_config['F1']}
D (depth):       {best_config['D']}
k (kernel):      {best_config['k']}
p1 (pool 1):     {best_config['p1']}
p2 (pool 2):     {best_config['p2']}
batch size:      {best_config['bs']}
learning rate:   {best_config['lr']:.6f}
"""
    
    # Add new parameters if present
    if 'norm_mode' in best_config:
        summary_text += f"norm mode:       {best_config['norm_mode']}\n"
    if 'dropout' in best_config:
        summary_text += f"dropout:         {best_config['dropout']}\n"
    if 'patience' in best_config:
        summary_text += f"patience:       {best_config['patience']}\n"
    if 'val_batch_mode' in best_config:
        summary_text += f"val_batch_mode:  {best_config['val_batch_mode']}\n"
    
    summary_text += f"""
Total configs:   {len(configs)}
Score range:     {min(scores):.4f} - {max(scores):.4f}
    """.strip()
    
    ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ========================
    # ROWS 2-4: Individual parameter analysis
    # ========================
    
    # Build plot configurations for all available parameters
    plot_positions = [
        gs[1, 0], gs[1, 1], gs[1, 2], gs[1, 3],
        gs[2, 0], gs[2, 1], gs[2, 2], gs[2, 3],
        gs[3, 0],
    ]
    
    param_titles = {
        'F1': 'Filters (F1)',
        'D': 'Depth Multiplier (D)',
        'k': 'Kernel Size (k)',
        'p1': 'Pool Size 1 (p1)',
        'p2': 'Pool Size 2 (p2)',
        'bs': 'Batch Size (bs)',
        'lr': 'Learning Rate (lr)',
        'norm_mode': 'Normalization Mode',
        'dropout': 'Dropout Rate',
        'patience': 'Patience',
        'val_batch_mode' : 'Val Batch Mode',
    }
    
    for idx, param in enumerate(params):
        if idx >= len(plot_positions):
            break  # No more subplot positions
            
        ax = fig.add_subplot(plot_positions[idx])
        
        # Get unique values for this parameter
        unique_vals = sorted(set(c[param] for c in configs if param in c))
        
        # For categorical parameters (like norm_mode), use string positions
        is_categorical = isinstance(unique_vals[0], str)
        
        # Detect log-scale parameters (learning rate, dropout)
        use_log_positions = param in ['lr', 'dropout'] and not is_categorical
        
        if is_categorical or use_log_positions:
            # Use integer positions for categorical OR log-scale numeric params
            val_to_pos = {val: i for i, val in enumerate(unique_vals)}
            positions = list(range(len(unique_vals)))
        else:
            positions = unique_vals
        
        # Prepare data for box plot
        data_for_box = [param_scores[param][val] for val in unique_vals]
        
        # Create box plot - always use the positions array
        bp = ax.boxplot(data_for_box, positions=positions, 
                       widths=0.6 if not (is_categorical or use_log_positions) else 0.4, 
                       patch_artist=True,
                       medianprops=dict(color='red', linewidth=2),
                       boxprops=dict(facecolor='lightblue', alpha=0.7, edgecolor='black'),
                       whiskerprops=dict(color='black', linewidth=1.5),
                       capprops=dict(color='black', linewidth=1.5))
        
        # Overlay scatter plot with jitter
        for i, val in enumerate(unique_vals):
            scores_for_val = param_scores[param][val]
            # Always use integer position for jittering
            x_pos = i if (is_categorical or use_log_positions) else val
            jittered_x = np.random.normal(x_pos, 0.08, len(scores_for_val))
            ax.scatter(jittered_x, scores_for_val, alpha=0.4, s=30, color='navy', 
                      edgecolors='black', linewidth=0.5)
        
        # Mark best config
        if param in best_config:
            best_val = best_config[param]
            best_x_pos = val_to_pos[best_val] if (is_categorical or use_log_positions) else best_val
            ax.scatter([best_x_pos], [best_config['score']], 
                      marker='D', c='red', s=150,
                      edgecolors='darkred', linewidth=2, 
                      label='Best Config', zorder=10)
        
        # Plot average trend line
        avg_scores_list = [param_avg_scores[param][val] for val in unique_vals]
        if not (is_categorical or use_log_positions):
            ax.plot(unique_vals, avg_scores_list, 
                   'g--', linewidth=2.5, alpha=0.7, label='Average', zorder=5)
        else:
            ax.plot(range(len(unique_vals)), avg_scores_list, 
                   'g--', linewidth=2.5, alpha=0.7, label='Average', zorder=5)
        
        # Formatting
        if is_categorical or use_log_positions:
            ax.set_xticks(range(len(unique_vals)))
            # Format learning rate in scientific notation if needed
            if param == 'lr':
                labels = [f'{v:.0e}' if v < 0.01 else f'{v:.4f}' for v in unique_vals]
            else:
                labels = [str(v) for v in unique_vals]
            ax.set_xticklabels(labels, fontsize=9, 
                              rotation=45 if (is_categorical and len(str(unique_vals[0])) > 8) else 0)
        else:
            ax.set_xticks(unique_vals)
            ax.set_xticklabels([str(v) for v in unique_vals], fontsize=9)
        
        ax.set_ylabel('Score', fontsize=10, fontweight='bold')
        ax.set_xlabel(param, fontsize=10, fontweight='bold')
        ax.set_title(param_titles.get(param, param), fontsize=11, fontweight='bold', pad=8)
        ax.grid(alpha=0.3, linestyle='--', axis='y')
        ax.legend(fontsize=8, loc='best', framealpha=0.9)
        
        # Set y-limits with padding
        y_min, y_max = min(scores), max(scores)
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.05*y_range, y_max + 0.05*y_range)
    
    # ========================
    # Additional Analysis Plots
    # ========================
    
    # Score vs Config Index (temporal pattern)
    ax_temporal = fig.add_subplot(gs[3, 1])
    config_indices = range(len(configs))
    ax_temporal.scatter(config_indices, scores, c=scores, cmap='RdYlGn', 
                       s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax_temporal.axhline(best_config['score'], color='red', linestyle='--', linewidth=2, label='Best')
    ax_temporal.axhline(np.mean(scores), color='green', linestyle='--', linewidth=2, label='Mean')
    best_idx = scores.index(best_config['score'])
    ax_temporal.scatter([best_idx], [best_config['score']], 
                       marker='D', c='red', s=150, edgecolors='darkred', linewidth=2, zorder=10)
    ax_temporal.set_xlabel('Configuration Index', fontsize=10, fontweight='bold')
    ax_temporal.set_ylabel('Score', fontsize=10, fontweight='bold')
    ax_temporal.set_title('Score Progression', fontsize=11, fontweight='bold', pad=8)
    ax_temporal.grid(alpha=0.3, linestyle='--')
    ax_temporal.legend(fontsize=8, framealpha=0.9)
    
    # Main title
    fig.suptitle(f'Hyperparameter Tuning Analysis (n={len(configs)} configs, best={best_config["score"]:.4f})', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"\n📊 Visualization saved to: {save_path}")
    else:
        plt.savefig('htune_analysis_v2.png', dpi=200, bbox_inches='tight', facecolor='white')
        print(f"\n📊 Visualization saved to: htune_analysis_v2.png")
    
    plt.close()


def analyze_fold_balance(log_text, verbose=True):
    """
    Comprehensive fold balance analysis with scoring.
    Returns: (is_ok, issues, stats)
    """
    if verbose:
        print("=" * 80)
        print("FOLD BALANCE ANALYSIS")
        print("=" * 80)
    
    fold_pattern = r'\[([^\]]+)\] Fold (\d+): train N=(\d+) \(c0=(\d+), c1=(\d+), c2=(\d+)\) \| val N=(\d+) \(c0=(\d+), c1=(\d+), c2=(\d+)\)'
    matches = re.findall(fold_pattern, log_text)
    
    fold_stats = defaultdict(list)
    issues = []
    
    for match in matches:
        tag, fold_id, train_n, tr_c0, tr_c1, tr_c2, val_n, va_c0, va_c1, va_c2 = match
        fold_id = int(fold_id)
        val_n = int(val_n)
        va_c0, va_c1, va_c2 = int(va_c0), int(va_c1), int(va_c2)
        tr_c0, tr_c1, tr_c2 = int(tr_c0), int(tr_c1), int(tr_c2)
        
        # Validation balance
        if val_n > 0:
            pct_c0 = 100 * va_c0 / val_n
            pct_c1 = 100 * va_c1 / val_n
            pct_c2 = 100 * va_c2 / val_n
            
            val_imbalance = max(pct_c0, pct_c1, pct_c2) - min(pct_c0, pct_c1, pct_c2)
            
            # Training balance
            train_n = int(train_n)
            tr_pct_c0 = 100 * tr_c0 / train_n
            tr_pct_c1 = 100 * tr_c1 / train_n
            tr_pct_c2 = 100 * tr_c2 / train_n
            tr_imbalance = max(tr_pct_c0, tr_pct_c1, tr_pct_c2) - min(tr_pct_c0, tr_pct_c1, tr_pct_c2)
            
            fold_stats[tag].append({
                'fold_id': fold_id,
                'val_imbalance': val_imbalance,
                'train_imbalance': tr_imbalance,
                'val_n': val_n,
                'val_dist': (pct_c0, pct_c1, pct_c2),
                'train_dist': (tr_pct_c0, tr_pct_c1, tr_pct_c2),
            })
            
            if verbose:
                val_status = "❌ IMBALANCED" if val_imbalance > 20 else ("⚠️  SKEWED" if val_imbalance > 15 else "✅ OK")
                print(f"\n[{tag}] Fold {fold_id:02d} {val_status}")
                print(f"  Val:   N={val_n:3d} c0={va_c0:2d}({pct_c0:5.1f}%) c1={va_c1:2d}({pct_c1:5.1f}%) c2={va_c2:2d}({pct_c2:5.1f}%) | imbalance={val_imbalance:.1f}%")
                print(f"  Train: N={train_n:3d} c0={tr_c0:2d}({tr_pct_c0:5.1f}%) c1={tr_c1:2d}({tr_pct_c1:5.1f}%) c2={tr_c2:2d}({tr_pct_c2:5.1f}%) | imbalance={tr_imbalance:.1f}%")
            
            # Collect issues
            if val_imbalance > 20:
                issues.append(f"[{tag}] Fold {fold_id}: SEVERE val imbalance ({val_imbalance:.1f}%)")
            if tr_imbalance > 20:
                issues.append(f"[{tag}] Fold {fold_id}: SEVERE train imbalance ({tr_imbalance:.1f}%)")
    
    # Summary statistics
    if verbose and fold_stats:
        print("\n" + "-" * 80)
        print("FOLD BALANCE SUMMARY")
        print("-" * 80)
        
        for tag, folds in fold_stats.items():
            val_imbalances = [f['val_imbalance'] for f in folds]
            train_imbalances = [f['train_imbalance'] for f in folds]
            
            print(f"\n[{tag}]:")
            print(f"  Val imbalance:   min={min(val_imbalances):.1f}% max={max(val_imbalances):.1f}% mean={np.mean(val_imbalances):.1f}%")
            print(f"  Train imbalance: min={min(train_imbalances):.1f}% max={max(train_imbalances):.1f}% mean={np.mean(train_imbalances):.1f}%")
            
            if max(val_imbalances) < 15:
                print(f"  ✅ EXCELLENT fold balance")
            elif max(val_imbalances) < 20:
                print(f"  ✅ GOOD fold balance (some skew)")
            else:
                print(f"  ❌ POOR fold balance (severe imbalance)")
    
    is_ok = all(f['val_imbalance'] < 20 for folds in fold_stats.values() for f in folds)
    
    return is_ok, issues, fold_stats


def analyze_batch_quality(log_text, verbose=True):
    """
    Analyze batch construction quality from [BATCH] logs.
    """
    if verbose:
        print("\n" + "=" * 80)
        print("BATCH QUALITY ANALYSIS")
        print("=" * 80)
    
    # Pattern for batch summary
    batch_summary_pattern = r'\[BATCH\] Total batches: (\d+)\s+\[BATCH\] Batch sizes: min=(\d+) max=(\d+) mean=([\d.]+)\s+\[BATCH\] Class 0: min=(\d+) max=(\d+) mean=([\d.]+)\s+\[BATCH\] Class 1: min=(\d+) max=(\d+) mean=([\d.]+)\s+\[BATCH\] Class 2: min=(\d+) max=(\d+) mean=([\d.]+)'
    
    # Pattern for individual batch warnings
    warning_pattern = r'\[BATCH\] ⚠️  Batch (\d+): IMBALANCED size=(\d+) c0=(\d+) c1=(\d+) c2=(\d+)'
    
    summaries = re.findall(batch_summary_pattern, log_text)
    warnings = re.findall(warning_pattern, log_text)
    
    issues = []
    
    if verbose:
        print(f"\nFound {len(summaries)} batch construction reports")
        print(f"Found {len(warnings)} imbalanced batch warnings")
    
    for i, summary in enumerate(summaries):
        n_batches, size_min, size_max, size_mean, c0_min, c0_max, c0_mean, c1_min, c1_max, c1_mean, c2_min, c2_max, c2_mean = summary
        
        n_batches = int(n_batches)
        size_min, size_max = int(size_min), int(size_max)
        c0_min, c0_max = int(c0_min), int(c0_max)
        c1_min, c1_max = int(c1_min), int(c1_max)
        c2_min, c2_max = int(c2_min), int(c2_max)
        
        # Check for perfect balance
        perfect_balance = (c0_min == c0_max == c1_min == c1_max == c2_min == c2_max)
        all_same_size = (size_min == size_max)
        
        if verbose and i < 5:  # Only show first 5 for brevity
            print(f"\nBatch Set #{i+1}:")
            print(f"  Total batches: {n_batches}")
            print(f"  Batch sizes: min={size_min} max={size_max} mean={size_mean}")
            print(f"  Class balance: c0=[{c0_min}-{c0_max}] c1=[{c1_min}-{c1_max}] c2=[{c2_min}-{c2_max}]")
            
            if perfect_balance and all_same_size:
                print(f"  ✅ PERFECT: All batches perfectly balanced")
            elif perfect_balance:
                print(f"  ✅ GOOD: Perfect class balance (varying batch sizes)")
            else:
                print(f"  ⚠️  IMPERFECT: Some batches have uneven class distribution")
        
        if not perfect_balance:
            issues.append(f"Batch set #{i+1}: Imperfect class balance detected")
    
    if verbose and len(summaries) > 5:
        print(f"\n  ... and {len(summaries) - 5} more batch sets")
    
    if warnings and verbose:
        print(f"\n⚠️  IMBALANCED BATCHES DETECTED:")
        for batch_id, size, c0, c1, c2 in warnings[:10]:  # Show first 10
            print(f"  Batch {batch_id}: size={size} c0={c0} c1={c1} c2={c2}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more")
    
    is_ok = len(warnings) == 0
    
    return is_ok, issues, {
        'n_sets': len(summaries),
        'n_warnings': len(warnings),
        'summaries': summaries,
    }


def analyze_convergence(log_text, verbose=True):
    """
    Comprehensive convergence analysis.
    """
    if verbose:
        print("\n" + "=" * 80)
        print("CONVERGENCE ANALYSIS")
        print("=" * 80)
    
    # Find early stop lines
    stop_pattern = r'Early stop at epoch (\d+) \(best val loss ([\d.]+)\)'
    stops = re.findall(stop_pattern, log_text)
    
    if not stops:
        if verbose:
            print("No convergence data found in log")
        return False, ["No convergence data"], {}
    
    epochs = [int(e) for e, _ in stops]
    losses = [float(l) for _, l in stops]
    
    # Statistics
    epoch_min, epoch_max, epoch_mean = min(epochs), max(epochs), np.mean(epochs)
    epoch_std = np.std(epochs)
    loss_min, loss_max, loss_mean = min(losses), max(losses), np.mean(losses)
    loss_std = np.std(losses)
    
    # Identify issues
    issues = []
    
    # High variance in epochs
    if epoch_std > 40:
        issues.append(f"HIGH epoch variance (std={epoch_std:.1f}) - suggests unstable training")
    
    # High average loss
    if loss_mean > 0.8:
        issues.append(f"HIGH average val loss ({loss_mean:.3f}) - model not learning effectively")
    
    # High loss variance
    if loss_std > 0.2:
        issues.append(f"HIGH loss variance (std={loss_std:.3f}) - inconsistent convergence")
    
    # Many early stops (< 50 epochs)
    early_stops = sum(1 for e in epochs if e < 50)
    if early_stops > len(epochs) * 0.3:
        issues.append(f"{early_stops}/{len(epochs)} folds stopped before epoch 50 - possible overfitting or bad init")
    
    if verbose:
        print(f"\nConvergence Statistics (n={len(stops)} folds):")
        print(f"  Epochs: min={epoch_min} max={epoch_max} mean={epoch_mean:.1f} std={epoch_std:.1f}")
        print(f"  Val Loss: min={loss_min:.3f} max={loss_max:.3f} mean={loss_mean:.3f} std={loss_std:.3f}")
        
        # Categorize
        print(f"\nEpoch Distribution:")
        print(f"  Early (<50):  {sum(1 for e in epochs if e < 50):2d} folds ({100*sum(1 for e in epochs if e < 50)/len(epochs):.0f}%)")
        print(f"  Medium (50-150): {sum(1 for e in epochs if 50 <= e < 150):2d} folds ({100*sum(1 for e in epochs if 50 <= e < 150)/len(epochs):.0f}%)")
        print(f"  Late (150+):  {sum(1 for e in epochs if e >= 150):2d} folds ({100*sum(1 for e in epochs if e >= 150)/len(epochs):.0f}%)")
        
        print(f"\nLoss Distribution:")
        print(f"  Good (<0.6):    {sum(1 for l in losses if l < 0.6):2d} folds")
        print(f"  Medium (0.6-0.9): {sum(1 for l in losses if 0.6 <= l < 0.9):2d} folds")
        print(f"  Poor (0.9+):    {sum(1 for l in losses if l >= 0.9):2d} folds")
        
        if issues:
            print(f"\n⚠️  ISSUES DETECTED:")
            for issue in issues:
                print(f"  • {issue}")
        else:
            print(f"\n✅ Convergence looks healthy")
    
    is_ok = len(issues) == 0
    
    return is_ok, issues, {
        'epochs': epochs,
        'losses': losses,
        'stats': {
            'epoch_mean': epoch_mean,
            'epoch_std': epoch_std,
            'loss_mean': loss_mean,
            'loss_std': loss_std,
        }
    }


def analyze_holdout(log_text, verbose=True):
    """
    Analyze holdout split balance.
    """
    if verbose:
        print("\n" + "=" * 80)
        print("HOLDOUT SPLIT ANALYSIS")
        print("=" * 80)
    
    # Find holdout lines
    holdout_pattern = r'\[HOLDOUT[^\]]*\] (train|hold)\s+windows=(\d+) c0=(\d+) c1=(\d+) c2=(\d+)'
    matches = re.findall(holdout_pattern, log_text)
    
    if not matches:
        if verbose:
            print("No holdout data found in log")
        return False, ["No holdout data"], {}
    
    data = {}
    for split, n, c0, c1, c2 in matches:
        data[split] = {
            'n': int(n),
            'c0': int(c0),
            'c1': int(c1),
            'c2': int(c2),
        }
    
    issues = []
    
    if 'train' in data and 'hold' in data:
        train = data['train']
        hold = data['hold']
        
        # Check ratios
        total_c0 = train['c0'] + hold['c0']
        total_c1 = train['c1'] + hold['c1']
        total_c2 = train['c2'] + hold['c2']
        
        if total_c0 == 0 or total_c1 == 0 or total_c2 == 0:
            issues.append("Missing class in train+hold combined!")
            if verbose:
                print("❌ CRITICAL: Missing entire class!")
            return False, issues, {}
        
        hold_pct_c0 = 100 * hold['c0'] / total_c0 if total_c0 > 0 else 0
        hold_pct_c1 = 100 * hold['c1'] / total_c1 if total_c1 > 0 else 0
        hold_pct_c2 = 100 * hold['c2'] / total_c2 if total_c2 > 0 else 0
        
        # Check balance
        max_pct = max(hold_pct_c0, hold_pct_c1, hold_pct_c2)
        min_pct = min(hold_pct_c0, hold_pct_c1, hold_pct_c2)
        imbalance = max_pct - min_pct
        
        if verbose:
            print(f"\nTrain: N={train['n']:3d} c0={train['c0']:2d} c1={train['c1']:2d} c2={train['c2']:2d}")
            print(f"Hold:  N={hold['n']:3d} c0={hold['c0']:2d} c1={hold['c1']:2d} c2={hold['c2']:2d}")
            
            print(f"\nHoldout percentages (of total per class):")
            print(f"  c0: {hold_pct_c0:5.1f}% ({hold['c0']}/{total_c0})")
            print(f"  c1: {hold_pct_c1:5.1f}% ({hold['c1']}/{total_c1})")
            print(f"  c2: {hold_pct_c2:5.1f}% ({hold['c2']}/{total_c2})")
            print(f"  Imbalance: {imbalance:.1f}%")
        
        # Check for issues
        if imbalance > 30:
            issues.append(f"SEVERE holdout imbalance ({imbalance:.1f}%)")
            if verbose:
                print(f"\n❌ SEVERE IMBALANCE: One class has {max_pct:.1f}% in holdout, another only {min_pct:.1f}%")
        elif imbalance > 15:
            issues.append(f"Moderate holdout imbalance ({imbalance:.1f}%)")
            if verbose:
                print(f"\n⚠️  MODERATE IMBALANCE")
        else:
            if verbose:
                print(f"\n✅ Holdout split looks balanced")
        
        # Check for extreme cases
        if hold_pct_c0 > 40 or hold_pct_c1 > 40 or hold_pct_c2 > 40:
            issues.append("One class has >40% in holdout - too much held out!")
        if hold_pct_c0 < 5 or hold_pct_c1 < 5 or hold_pct_c2 < 5:
            issues.append("One class has <5% in holdout - too little held out!")
        
        is_ok = len(issues) == 0
        
        return is_ok, issues, {
            'train': train,
            'hold': hold,
            'percentages': (hold_pct_c0, hold_pct_c1, hold_pct_c2),
            'imbalance': imbalance,
        }
    else:
        issues.append("Missing train or hold split in log")
        return False, issues, {}

def analyze_group_sizes(log_text, verbose=True):
    if verbose:
        print("\n" + "=" * 80)
        print("GROUP SIZE ANALYSIS")
        print("=" * 80)

    pattern = (
        r'\[([^\]]+)\] GROUPS per class: g0=(\d+) g1=(\d+) g2=(\d+) \| '
        r'group_size\(min/med/max\)=(\d+)/(\d+)/(\d+) \| '
        r'c0\(min/med/max\)=(\d+)/(\d+)/(\d+) '
        r'c1\(min/med/max\)=(\d+)/(\d+)/(\d+) '
        r'c2\(min/med/max\)=(\d+)/(\d+)/(\d+)'
    )
    matches = re.findall(pattern, log_text)

    issues = []
    if not matches:
        return True, [], {"n_pairs": 0}

    for match in matches:
        (tag, g0, g1, g2,
         gmin, gmed, gmax,
         c0min, c0med, c0max,
         c1min, c1med, c1max,
         c2min, c2med, c2max) = match

        g0, g1, g2 = int(g0), int(g1), int(g2)
        gmin, gmed, gmax = int(gmin), int(gmed), int(gmax)
        c0min, c0med, c0max = int(c0min), int(c0med), int(c0max)
        c1min, c1med, c1max = int(c1min), int(c1med), int(c1max)
        c2min, c2med, c2max = int(c2min), int(c2med), int(c2max)

        if verbose:
            print(f"\n[{tag}]:")
            print(f"  Groups per class: c0={g0} c1={g1} c2={g2}")
            print(f"  Overall group sizes: min={gmin} med={gmed} max={gmax}")
            print(f"  Class 0 group sizes: min={c0min} med={c0med} max={c0max}")
            print(f"  Class 1 group sizes: min={c1min} med={c1med} max={c1max}")
            print(f"  Class 2 group sizes: min={c2min} med={c2med} max={c2max}")

        if gmin > 0 and gmax / gmin >= 7:
            msg = f"[{tag}] Large group size range (max/min={gmax}/{gmin}={gmax/gmin:.1f}x). This is only problematic if batching assumes groups are equal-sized."
            issues.append(msg)
            if verbose:
                print("  ⚠️  Wide size range (this is a batching concern, not automatically a data-quality problem)")

        # Size-1 groups: be accurate + non-alarmist
        if gmin == 1:
            msg = f"[{tag}] Size-1 groups exist. This is OK if rare; it mainly hurts if you do group-balanced batching or if many folds end up with tiny per-class validation."
            # Treat as INFO/WARN, not "severe"
            issues.append(msg)
            if verbose:
                print("  ℹ️  Size-1 groups exist (not necessarily bad; only an issue if frequent or if batching assumes equal group sizes)")

    # Deduplicate repeated issues so summary doesn’t spam
    issues = list(dict.fromkeys(issues))

    is_ok = True 
    return is_ok, issues, {"n_pairs": len(matches)}


def generate_report(log_text, verbose=True, save_plots=True):
    """
    Generate comprehensive report on training quality.
    """
    print("=" * 80)
    print("COMPREHENSIVE TRAINING DIAGNOSTIC REPORT")
    print("=" * 80)
    
    results = {}
    all_issues = []
    
    # Run all analyses
    fold_ok, fold_issues, fold_stats = analyze_fold_balance(log_text, verbose)
    results['folds'] = {'ok': fold_ok, 'issues': fold_issues, 'stats': fold_stats}
    all_issues.extend(fold_issues)
    
    batch_ok, batch_issues, batch_stats = analyze_batch_quality(log_text, verbose)
    results['batches'] = {'ok': batch_ok, 'issues': batch_issues, 'stats': batch_stats}
    all_issues.extend(batch_issues)
    
    conv_ok, conv_issues, conv_stats = analyze_convergence(log_text, verbose)
    results['convergence'] = {'ok': conv_ok, 'issues': conv_issues, 'stats': conv_stats}
    all_issues.extend(conv_issues)
    
    hold_ok, hold_issues, hold_stats = analyze_holdout(log_text, verbose)
    results['holdout'] = {'ok': hold_ok, 'issues': hold_issues, 'stats': hold_stats}
    all_issues.extend(hold_issues)
    
    group_ok, group_issues, group_stats = analyze_group_sizes(log_text, verbose)
    results['groups'] = {'ok': group_ok, 'issues': group_issues, 'stats': group_stats}
    all_issues.extend(group_issues)
    
    # Hyperparameter tuning analysis
    best_config, all_configs, htune_analysis = analyze_hyperparameter_tuning(log_text, verbose)
    results['hyperparameters'] = {
        'best_config': best_config,
        'all_configs': all_configs,
        'analysis': htune_analysis,
    }
    
    # Generate visualizations if hyperparameter data exists
    if all_configs and save_plots:
        visualize_hyperparameter_tuning(all_configs, htune_analysis)
    
    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    
    print(f"\nComponent Health:")
    print(f"  Fold Balance:     {'✅ PASS' if fold_ok else '❌ FAIL'}")
    print(f"  Batch Quality:    {'✅ PASS' if batch_ok else '❌ FAIL'}")
    print(f"  Convergence:      {'✅ PASS' if conv_ok else '❌ FAIL'}")
    print(f"  Holdout Split:    {'✅ PASS' if hold_ok else '❌ FAIL'}")
    print(f"  Group Sizes:      {'✅ PASS' if group_ok else '❌ FAIL'}")
    
    if best_config:
        print(f"  Hyperparam Tuning: ✅ COMPLETE (n={len(all_configs)} configs)")
    
    overall_ok = all([fold_ok, batch_ok, conv_ok, hold_ok, group_ok])
    
    if overall_ok:
        print(f"\n✅ ALL CHECKS PASSED - Training looks healthy!")
    else:
        print(f"\n❌ ISSUES DETECTED - See details above")
        print(f"\nTotal issues found: {len(all_issues)}")
        if all_issues:
            print("\nTop Issues:")
            for issue in all_issues[:10]:
                print(f"  • {issue}")
    
    return results


def quick_check(log_text_or_path):
    """
    Quick pass/fail check without verbose output.
    Returns: (overall_ok, summary_dict)
    """
    if isinstance(log_text_or_path, str) and '\n' not in log_text_or_path and len(log_text_or_path) < 300:
        # Looks like a path
        log_text = parse_log_file(log_text_or_path)
    else:
        log_text = log_text_or_path
    
    fold_ok, _, _ = analyze_fold_balance(log_text, verbose=False)
    batch_ok, _, _ = analyze_batch_quality(log_text, verbose=False)
    conv_ok, _, _ = analyze_convergence(log_text, verbose=False)
    hold_ok, _, _ = analyze_holdout(log_text, verbose=False)
    group_ok, _, _ = analyze_group_sizes(log_text, verbose=False)
    
    overall = fold_ok and batch_ok and conv_ok and hold_ok and group_ok
    
    return overall, {
        'overall': overall,
        'folds': fold_ok,
        'batches': batch_ok,
        'convergence': conv_ok,
        'holdout': hold_ok,
        'groups': group_ok,
    }


if __name__ == "__main__":
    
    print("CNN Training Diagnostic Tool - COMPLETE v2")
    print("=" * 80)
    
    if len(sys.argv) > 1:
        # Read from file
        log_path = sys.argv[1]
        print(f"Reading log from: {log_path}\n")
        log_text = parse_log_file(log_path)
        generate_report(log_text, verbose=True, save_plots=True)