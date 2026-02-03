import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# ---------------------------- parsing structs ----------------------------

@dataclass
class RunRecord:
    run_idx: int
    stage: str            # "PAIR" or "FINAL" (from [PAIR xvsY] vs [FINAL xvsY])
    pair: str             # "11vs8"
    fold: Optional[int]   # 0..4 when present
    early_stop_epoch: Optional[int]
    best_val_loss: Optional[float]
    train_counts: Optional[Tuple[int, int, int]]  # c0,c1,c2
    val_counts: Optional[Tuple[int, int, int]]
    batch_avail: Optional[Tuple[int, int, int]]   # avail c0,c1,c2 at stop
    batches_created: Optional[int]

@dataclass
class PairSearchRecord:
    pair: str
    mean_bal_acc: float
    std_bal_acc: float

def parse_log(text: str) -> Tuple[List[RunRecord], List[PairSearchRecord]]:
    # Patterns
    p_header = re.compile(r'^\[(PAIR|FINAL)\s+(\d+vs\d+)\]')
    p_fold = re.compile(r'Fold\s+(\d{2}):')
    p_split_train = re.compile(r'^\[SPLIT\]\s+train windows=\d+\s+c0=(\d+)\s+c1=(\d+)\s+c2=(\d+)')
    p_split_val   = re.compile(r'^\[SPLIT\]\s+val\s+windows=\d+\s+c0=(\d+)\s+c1=(\d+)\s+c2=(\d+)')
    p_early = re.compile(r'Early stop at epoch\s+(\d+)\s+\(best val loss\s+([0-9.]+)\)\.')
    p_batch_stop = re.compile(r'^\[CNN_BATCH\]\s+Stopping:\s+avail c0=(\d+)\s+c1=(\d+)\s+c2=(\d+),')
    p_batches_created = re.compile(r'^\[CNN_BATCH\]\s+Created\s+(\d+)\s+batches:')
    p_pair_search = re.compile(r'^\[PAIR_SEARCH\]\s+pair\s+\((\d+),(\d+)\)\s+mean_bal_acc=([0-9.]+)\s+\(\+/-\s*([0-9.]+)\)')

    runs: List[RunRecord] = []
    pair_search: List[PairSearchRecord] = []

    cur_stage = None
    cur_pair = None
    cur_fold = None

    # We'll create a new RunRecord whenever we hit an Early stop line
    # (because that's the closest "end-of-run" marker in your log).
    run_idx = 0

    # Keep latest split + batch info encountered prior to early stop
    last_train = None
    last_val = None
    last_avail = None
    last_batches = None

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        m = p_header.match(line)
        if m:
            cur_stage = m.group(1)   # PAIR or FINAL
            cur_pair = m.group(2)    # 11vs8
            # do not reset fold here; fold lines will set it
            continue

        m = p_fold.search(line)
        if m:
            cur_fold = int(m.group(1))
            continue

        m = p_batches_created.match(line)
        if m:
            last_batches = int(m.group(1))
            continue

        m = p_batch_stop.match(line)
        if m:
            last_avail = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            continue

        m = p_split_train.match(line)
        if m:
            last_train = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            continue

        m = p_split_val.match(line)
        if m:
            last_val = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            continue

        m = p_early.search(line)
        if m:
            epoch = int(m.group(1))
            best_loss = float(m.group(2))

            # Sometimes fold isn't explicitly restated right before early stop;
            # but in your log it usually is. We'll still allow None.
            rec = RunRecord(
                run_idx=run_idx,
                stage=cur_stage or "UNKNOWN",
                pair=cur_pair or "UNKNOWN",
                fold=cur_fold,
                early_stop_epoch=epoch,
                best_val_loss=best_loss,
                train_counts=last_train,
                val_counts=last_val,
                batch_avail=last_avail,
                batches_created=last_batches,
            )
            runs.append(rec)
            run_idx += 1

            # Reset "per-run" transient things that should not leak too far:
            # (but keep stage/pair/fold context until changed by new headers)
            last_train = None
            last_val = None
            last_avail = None
            last_batches = None
            continue

        m = p_pair_search.match(line)
        if m:
            a = m.group(1)
            b = m.group(2)
            mean = float(m.group(3))
            std = float(m.group(4))
            pair_search.append(PairSearchRecord(pair=f"{a}vs{b}", mean_bal_acc=mean, std_bal_acc=std))
            continue

    return runs, pair_search


# ---------------------------- plotting helpers ----------------------------

from collections import defaultdict

def plot_best_val_loss_by_pair(runs: List[RunRecord]) -> None:
    if not runs:
        print("No runs found.")
        return

    # Group (x, y) by frequency pair
    by_pair = defaultdict(list)
    for r in runs:
        if r.best_val_loss is None:
            continue
        by_pair[r.pair].append((r.run_idx, r.best_val_loss))

    plt.figure()
    for pair, pts in sorted(by_pair.items()):
        pts.sort(key=lambda t: t[0])  # sort by run order
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        # Each pair plotted separately -> different default color per pair
        plt.plot(xs, ys, marker="o", linestyle="-", label=pair)

    plt.title("Best validation loss per run (colored by frequency pair)")
    plt.xlabel("Run index (appearance order in log)")
    plt.ylabel("Best val loss")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Pair", fontsize=8, ncol=2)

def plot_early_stop(runs: List[RunRecord]) -> None:
    if not runs:
        print("No runs found.")
        return

    xs = [r.run_idx for r in runs]
    epochs = [r.early_stop_epoch for r in runs]
    losses = [r.best_val_loss for r in runs]

    # annotate labels like "11vs8 F0" or "17vs20 F3"
    labels = []
    for r in runs:
        f = f"F{r.fold:02d}" if r.fold is not None else "F??"
        labels.append(f"{r.pair} {f}")

    plt.figure()
    plt.plot(xs, epochs, marker="o")
    plt.title("Early-stop epoch per run (in log order)")
    plt.xlabel("Run index (appearance order in log)")
    plt.ylabel("Early-stop epoch")
    plt.grid(True, alpha=0.3)

    # add light annotation every N points to keep readable
    step = max(1, len(xs) // 12)
    for i in range(0, len(xs), step):
        plt.annotate(labels[i], (xs[i], epochs[i]), textcoords="offset points", xytext=(5, 5), fontsize=8)

    plt.figure()
    plt.plot(xs, losses, marker="o")
    plt.title("Best validation loss per run (lower is better)")
    plt.xlabel("Run index (appearance order in log)")
    plt.ylabel("Best val loss")
    plt.grid(True, alpha=0.3)

def plot_pair_search(pair_search: List[PairSearchRecord]) -> None:
    if not pair_search:
        print("No pair-search records found.")
        return

    # sort by mean acc ascending to highlight "worst" at top
    pair_search = sorted(pair_search, key=lambda r: r.mean_bal_acc)

    pairs = [r.pair for r in pair_search]
    means = [r.mean_bal_acc for r in pair_search]
    stds = [r.std_bal_acc for r in pair_search]

    plt.figure()
    plt.errorbar(range(len(pairs)), means, yerr=stds, fmt="o")
    plt.title("Pair-search mean balanced accuracy (with +/- std)")
    plt.xlabel("Pair (sorted by mean)")
    plt.ylabel("Mean balanced accuracy")
    plt.xticks(range(len(pairs)), pairs, rotation=45, ha="right")
    plt.ylim(0.0, 1.02)
    plt.grid(True, alpha=0.3)

def plot_fold_balance(runs: List[RunRecord]) -> None:
    # Only include runs that have split counts
    rows = [r for r in runs if r.train_counts and r.val_counts]
    if not rows:
        print("No split counts found.")
        return

    xs = [r.run_idx for r in rows]

    # plot class imbalance ratios for train + val
    def imbalance_ratio(counts: Tuple[int,int,int]) -> float:
        mn = min(counts)
        mx = max(counts)
        return mx / mn if mn > 0 else float("inf")

    train_ir = [imbalance_ratio(r.train_counts) for r in rows]
    val_ir = [imbalance_ratio(r.val_counts) for r in rows]

    plt.figure()
    plt.plot(xs, train_ir, marker="o", label="train imbalance ratio (max/min)")
    plt.plot(xs, val_ir, marker="o", label="val imbalance ratio (max/min)")
    plt.title("Fold class-balance sanity (max/min across c0,c1,c2)")
    plt.xlabel("Run index")
    plt.ylabel("Imbalance ratio (1.0 is perfectly balanced)")
    plt.grid(True, alpha=0.3)
    plt.legend()

def plot_batch_leftovers(runs: List[RunRecord]) -> None:
    rows = [r for r in runs if r.batch_avail is not None]
    if not rows:
        print("No CNN_BATCH stopping lines found.")
        return

    xs = [r.run_idx for r in rows]
    c0 = [r.batch_avail[0] for r in rows]
    c1 = [r.batch_avail[1] for r in rows]
    c2 = [r.batch_avail[2] for r in rows]

    plt.figure()
    plt.plot(xs, c0, marker="o", label="avail c0 at stop")
    plt.plot(xs, c1, marker="o", label="avail c1 at stop")
    plt.plot(xs, c2, marker="o", label="avail c2 at stop")
    plt.title("Batcher leftovers at stop (shows class starvation)")
    plt.xlabel("Run index")
    plt.ylabel("Remaining windows per class when batching stops")
    plt.grid(True, alpha=0.3)
    plt.legend()


def save_all_figs(out_dir: Path, *, dpi: int = 180) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    figs = [plt.figure(n) for n in plt.get_fignums()]
    for i, fig in enumerate(figs, start=1):
        fig.tight_layout()
        fig.savefig(out_dir / f"plot_{i:02d}.png", dpi=dpi)
        fig.savefig(out_dir / f"plot_{i:02d}.pdf")  # vector-friendly

    print(f"[export] saved {len(figs)} figures to: {out_dir}")

# ---------------------------- main entrypoint ----------------------------

def main():
    # Option A: paste log into this triple-quoted string
    LOG_TEXT = r"""
==== DEBUG LOG START ====
created: 2026-02-02T15:22:16.982018
path: C:\Users\fsdma\capstone\capstone\models\alexandra3\2026-02-02_15-02-55\train_ssvep_debug.txt
=========================

[K_FOLDS] Groups: c0=1074 c1=1058 c2=4719
[K_FOLDS] Windows: c0=7277 c1=7115 c2=8004
[K_FOLDS] Selected k=5 (preferred=5)
[K_FOLDS] Using preferred k=5
[PAIR 11vs8] ===== Fold builder (blocked groups) =====
[PAIR 11vs8] N_windows=22396
[PAIR 11vs8] block_size_windows=ceil(W/H)=7
[PAIR 11vs8] n_groups_total=6851
[PAIR 11vs8] k_eff (group-limited)=5

[PAIR 11vs8] GROUPS per class: g0=1074 g1=1058 g2=4719 | group_size(min/med/max)=1/2/7 | c0(min/med/max)=1/7/7 c1(min/med/max)=1/7/7 c2(min/med/max)=1/2/2
[PAIR 11vs8] ===== Blocked-fold summary =====
[PAIR 11vs8] N_windows=22396
[PAIR 11vs8] n_time(W)=560 samples
[PAIR 11vs8] hop_samples(H)=80
[PAIR 11vs8] block_size_windows=ceil(W/H)=7
[PAIR 11vs8] n_groups=6851
[PAIR 11vs8] k_used=5

[PAIR 11vs8] group purity OK ✓

[PAIR 11vs8] overall class counts: c0=7277, c1=7115, c2=8004

[PAIR 11vs8] Fold 00: train N=17916 (c0=5821, c1=5692, c2=6403) | val N=4480 (c0=1456, c1=1423, c2=1601) | leak_groups=0
[PAIR 11vs8]   groups: train G=5481 (g0=859, g1=847, g2=3775) | val G=1370 (g0=215, g1=211, g2=944)
[PAIR 11vs8] Fold 01: train N=17917 (c0=5822, c1=5692, c2=6403) | val N=4479 (c0=1455, c1=1423, c2=1601) | leak_groups=0
[PAIR 11vs8]   groups: train G=5482 (g0=860, g1=847, g2=3775) | val G=1369 (g0=214, g1=211, g2=944)
[PAIR 11vs8] Fold 02: train N=17916 (c0=5821, c1=5692, c2=6403) | val N=4480 (c0=1456, c1=1423, c2=1601) | leak_groups=0
[PAIR 11vs8]   groups: train G=5480 (g0=859, g1=846, g2=3775) | val G=1371 (g0=215, g1=212, g2=944)
[PAIR 11vs8] Fold 03: train N=17917 (c0=5822, c1=5692, c2=6403) | val N=4479 (c0=1455, c1=1423, c2=1601) | leak_groups=0
[PAIR 11vs8]   groups: train G=5480 (g0=859, g1=846, g2=3775) | val G=1371 (g0=215, g1=212, g2=944)
[PAIR 11vs8] Fold 04: train N=17918 (c0=5822, c1=5692, c2=6404) | val N=4478 (c0=1455, c1=1423, c2=1600) | leak_groups=0
[PAIR 11vs8]   groups: train G=5481 (g0=859, g1=846, g2=3776) | val G=1370 (g0=215, g1=212, g2=943)
[PAIR 11vs8] ===== End summary =====
[PAIR 11vs8] SUCCESS: Created 5 balanced folds (manual round-robin)
[CNN_BATCH] Stopping: avail c0=129 c1=0 c2=711, need full=6 or small>=4
[CNN_BATCH] Created 949 batches: sizes min=12 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17916 c0=5821 c1=5692 c2=6403
[SPLIT] val   windows=4480   c0=1456 c1=1423 c2=1601
Early stop at epoch 26 (best val loss 0.0216).
[CNN_BATCH] Stopping: avail c0=130 c1=0 c2=711, need full=6 or small>=4
[CNN_BATCH] Created 949 batches: sizes min=12 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17917 c0=5822 c1=5692 c2=6403
[SPLIT] val   windows=4479   c0=1455 c1=1423 c2=1601
Early stop at epoch 30 (best val loss 0.0223).
[CNN_BATCH] Stopping: avail c0=129 c1=0 c2=711, need full=6 or small>=4
[CNN_BATCH] Created 949 batches: sizes min=12 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17916 c0=5821 c1=5692 c2=6403
[SPLIT] val   windows=4480   c0=1456 c1=1423 c2=1601
Early stop at epoch 26 (best val loss 0.0042).
[CNN_BATCH] Stopping: avail c0=130 c1=0 c2=711, need full=6 or small>=4
[CNN_BATCH] Created 949 batches: sizes min=12 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17917 c0=5822 c1=5692 c2=6403
[SPLIT] val   windows=4479   c0=1455 c1=1423 c2=1601
Early stop at epoch 46 (best val loss 0.0138).
[CNN_BATCH] Stopping: avail c0=130 c1=0 c2=712, need full=6 or small>=4
[CNN_BATCH] Created 949 batches: sizes min=12 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17918 c0=5822 c1=5692 c2=6404
[SPLIT] val   windows=4478   c0=1455 c1=1423 c2=1600
Early stop at epoch 26 (best val loss 0.0213).
[PAIR_SEARCH] pair (11,8) mean_bal_acc=0.998 (+/-0.001)
[K_FOLDS] Groups: c0=1074 c1=1055 c2=4719
[K_FOLDS] Windows: c0=7277 c1=7088 c2=8004
[K_FOLDS] Selected k=5 (preferred=5)
[K_FOLDS] Using preferred k=5
[PAIR 11vs17] ===== Fold builder (blocked groups) =====
[PAIR 11vs17] N_windows=22369
[PAIR 11vs17] block_size_windows=ceil(W/H)=7
[PAIR 11vs17] n_groups_total=6848
[PAIR 11vs17] k_eff (group-limited)=5

[PAIR 11vs17] GROUPS per class: g0=1074 g1=1055 g2=4719 | group_size(min/med/max)=1/2/7 | c0(min/med/max)=1/7/7 c1(min/med/max)=1/7/7 c2(min/med/max)=1/2/2
[PAIR 11vs17] ===== Blocked-fold summary =====
[PAIR 11vs17] N_windows=22369
[PAIR 11vs17] n_time(W)=560 samples
[PAIR 11vs17] hop_samples(H)=80
[PAIR 11vs17] block_size_windows=ceil(W/H)=7
[PAIR 11vs17] n_groups=6848
[PAIR 11vs17] k_used=5

[PAIR 11vs17] group purity OK ✓

[PAIR 11vs17] overall class counts: c0=7277, c1=7088, c2=8004

[PAIR 11vs17] Fold 00: train N=17894 (c0=5821, c1=5670, c2=6403) | val N=4475 (c0=1456, c1=1418, c2=1601) | leak_groups=0
[PAIR 11vs17]   groups: train G=5478 (g0=859, g1=844, g2=3775) | val G=1370 (g0=215, g1=211, g2=944)
[PAIR 11vs17] Fold 01: train N=17895 (c0=5822, c1=5670, c2=6403) | val N=4474 (c0=1455, c1=1418, c2=1601) | leak_groups=0
[PAIR 11vs17]   groups: train G=5479 (g0=860, g1=844, g2=3775) | val G=1369 (g0=214, g1=211, g2=944)
[PAIR 11vs17] Fold 02: train N=17894 (c0=5821, c1=5670, c2=6403) | val N=4475 (c0=1456, c1=1418, c2=1601) | leak_groups=0
[PAIR 11vs17]   groups: train G=5478 (g0=859, g1=844, g2=3775) | val G=1370 (g0=215, g1=211, g2=944)
[PAIR 11vs17] Fold 03: train N=17896 (c0=5822, c1=5671, c2=6403) | val N=4473 (c0=1455, c1=1417, c2=1601) | leak_groups=0
[PAIR 11vs17]   groups: train G=5478 (g0=859, g1=844, g2=3775) | val G=1370 (g0=215, g1=211, g2=944)
[PAIR 11vs17] Fold 04: train N=17897 (c0=5822, c1=5671, c2=6404) | val N=4472 (c0=1455, c1=1417, c2=1600) | leak_groups=0
[PAIR 11vs17]   groups: train G=5479 (g0=859, g1=844, g2=3776) | val G=1369 (g0=215, g1=211, g2=943)
[PAIR 11vs17] ===== End summary =====
[PAIR 11vs17] SUCCESS: Created 5 balanced folds (manual round-robin)
[CNN_BATCH] Stopping: avail c0=151 c1=0 c2=733, need full=6 or small>=4
[CNN_BATCH] Created 945 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17894 c0=5821 c1=5670 c2=6403
[SPLIT] val   windows=4475   c0=1456 c1=1418 c2=1601
Early stop at epoch 42 (best val loss 0.0239).
[CNN_BATCH] Stopping: avail c0=152 c1=0 c2=733, need full=6 or small>=4
[CNN_BATCH] Created 945 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17895 c0=5822 c1=5670 c2=6403
[SPLIT] val   windows=4474   c0=1455 c1=1418 c2=1601
Early stop at epoch 39 (best val loss 0.0163).
[CNN_BATCH] Stopping: avail c0=151 c1=0 c2=733, need full=6 or small>=4
[CNN_BATCH] Created 945 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17894 c0=5821 c1=5670 c2=6403
[SPLIT] val   windows=4475   c0=1456 c1=1418 c2=1601
Early stop at epoch 47 (best val loss 0.0113).
[CNN_BATCH] Stopping: avail c0=152 c1=1 c2=733, need full=6 or small>=4
[CNN_BATCH] Created 945 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17896 c0=5822 c1=5671 c2=6403
[SPLIT] val   windows=4473   c0=1455 c1=1417 c2=1601
Early stop at epoch 30 (best val loss 0.0172).
[CNN_BATCH] Stopping: avail c0=152 c1=1 c2=734, need full=6 or small>=4
[CNN_BATCH] Created 945 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17897 c0=5822 c1=5671 c2=6404
[SPLIT] val   windows=4472   c0=1455 c1=1417 c2=1600
Early stop at epoch 26 (best val loss 0.0138).
[PAIR_SEARCH] pair (11,17) mean_bal_acc=0.998 (+/-0.001)
[K_FOLDS] Groups: c0=1074 c1=1021 c2=4719
[K_FOLDS] Windows: c0=7277 c1=6806 c2=8004
[K_FOLDS] Selected k=5 (preferred=5)
[K_FOLDS] Using preferred k=5
[PAIR 11vs14] ===== Fold builder (blocked groups) =====
[PAIR 11vs14] N_windows=22087
[PAIR 11vs14] block_size_windows=ceil(W/H)=7
[PAIR 11vs14] n_groups_total=6814
[PAIR 11vs14] k_eff (group-limited)=5

[PAIR 11vs14] GROUPS per class: g0=1074 g1=1021 g2=4719 | group_size(min/med/max)=1/2/7 | c0(min/med/max)=1/7/7 c1(min/med/max)=1/7/7 c2(min/med/max)=1/2/2
[PAIR 11vs14] ===== Blocked-fold summary =====
[PAIR 11vs14] N_windows=22087
[PAIR 11vs14] n_time(W)=560 samples
[PAIR 11vs14] hop_samples(H)=80
[PAIR 11vs14] block_size_windows=ceil(W/H)=7
[PAIR 11vs14] n_groups=6814
[PAIR 11vs14] k_used=5

[PAIR 11vs14] group purity OK ✓

[PAIR 11vs14] overall class counts: c0=7277, c1=6806, c2=8004

[PAIR 11vs14] Fold 00: train N=17668 (c0=5821, c1=5444, c2=6403) | val N=4419 (c0=1456, c1=1362, c2=1601) | leak_groups=0
[PAIR 11vs14]   groups: train G=5450 (g0=859, g1=816, g2=3775) | val G=1364 (g0=215, g1=205, g2=944)
[PAIR 11vs14] Fold 01: train N=17670 (c0=5822, c1=5445, c2=6403) | val N=4417 (c0=1455, c1=1361, c2=1601) | leak_groups=0
[PAIR 11vs14]   groups: train G=5452 (g0=860, g1=817, g2=3775) | val G=1362 (g0=214, g1=204, g2=944)
[PAIR 11vs14] Fold 02: train N=17669 (c0=5821, c1=5445, c2=6403) | val N=4418 (c0=1456, c1=1361, c2=1601) | leak_groups=0
[PAIR 11vs14]   groups: train G=5451 (g0=859, g1=817, g2=3775) | val G=1363 (g0=215, g1=204, g2=944)
[PAIR 11vs14] Fold 03: train N=17670 (c0=5822, c1=5445, c2=6403) | val N=4417 (c0=1455, c1=1361, c2=1601) | leak_groups=0
[PAIR 11vs14]   groups: train G=5451 (g0=859, g1=817, g2=3775) | val G=1363 (g0=215, g1=204, g2=944)
[PAIR 11vs14] Fold 04: train N=17671 (c0=5822, c1=5445, c2=6404) | val N=4416 (c0=1455, c1=1361, c2=1600) | leak_groups=0
[PAIR 11vs14]   groups: train G=5452 (g0=859, g1=817, g2=3776) | val G=1362 (g0=215, g1=204, g2=943)
[PAIR 11vs14] ===== End summary =====
[PAIR 11vs14] SUCCESS: Created 5 balanced folds (manual round-robin)
[CNN_BATCH] Stopping: avail c0=379 c1=2 c2=961, need full=6 or small>=4
[CNN_BATCH] Created 907 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17668 c0=5821 c1=5444 c2=6403
[SPLIT] val   windows=4419   c0=1456 c1=1362 c2=1601
Early stop at epoch 41 (best val loss 0.0237).
[CNN_BATCH] Stopping: avail c0=380 c1=3 c2=961, need full=6 or small>=4
[CNN_BATCH] Created 907 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17670 c0=5822 c1=5445 c2=6403
[SPLIT] val   windows=4417   c0=1455 c1=1361 c2=1601
Early stop at epoch 27 (best val loss 0.0116).
[CNN_BATCH] Stopping: avail c0=379 c1=3 c2=961, need full=6 or small>=4
[CNN_BATCH] Created 907 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17669 c0=5821 c1=5445 c2=6403
[SPLIT] val   windows=4418   c0=1456 c1=1361 c2=1601
Early stop at epoch 26 (best val loss 0.0150).
[CNN_BATCH] Stopping: avail c0=380 c1=3 c2=961, need full=6 or small>=4
[CNN_BATCH] Created 907 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17670 c0=5822 c1=5445 c2=6403
[SPLIT] val   windows=4417   c0=1455 c1=1361 c2=1601
Early stop at epoch 26 (best val loss 0.0224).
[CNN_BATCH] Stopping: avail c0=380 c1=3 c2=962, need full=6 or small>=4
[CNN_BATCH] Created 907 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17671 c0=5822 c1=5445 c2=6404
[SPLIT] val   windows=4416   c0=1455 c1=1361 c2=1600
Early stop at epoch 31 (best val loss 0.0240).
[PAIR_SEARCH] pair (11,14) mean_bal_acc=0.998 (+/-0.001)
[K_FOLDS] Groups: c0=1074 c1=1023 c2=4719
[K_FOLDS] Windows: c0=7277 c1=6781 c2=8004
[K_FOLDS] Selected k=5 (preferred=5)
[K_FOLDS] Using preferred k=5
[PAIR 11vs20] ===== Fold builder (blocked groups) =====
[PAIR 11vs20] N_windows=22062
[PAIR 11vs20] block_size_windows=ceil(W/H)=7
[PAIR 11vs20] n_groups_total=6816
[PAIR 11vs20] k_eff (group-limited)=5

[PAIR 11vs20] GROUPS per class: g0=1074 g1=1023 g2=4719 | group_size(min/med/max)=1/2/7 | c0(min/med/max)=1/7/7 c1(min/med/max)=1/7/7 c2(min/med/max)=1/2/2
[PAIR 11vs20] ===== Blocked-fold summary =====
[PAIR 11vs20] N_windows=22062
[PAIR 11vs20] n_time(W)=560 samples
[PAIR 11vs20] hop_samples(H)=80
[PAIR 11vs20] block_size_windows=ceil(W/H)=7
[PAIR 11vs20] n_groups=6816
[PAIR 11vs20] k_used=5

[PAIR 11vs20] group purity OK ✓

[PAIR 11vs20] overall class counts: c0=7277, c1=6781, c2=8004

[PAIR 11vs20] Fold 00: train N=17648 (c0=5821, c1=5424, c2=6403) | val N=4414 (c0=1456, c1=1357, c2=1601) | leak_groups=0
[PAIR 11vs20]   groups: train G=5452 (g0=859, g1=818, g2=3775) | val G=1364 (g0=215, g1=205, g2=944)
[PAIR 11vs20] Fold 01: train N=17650 (c0=5822, c1=5425, c2=6403) | val N=4412 (c0=1455, c1=1356, c2=1601) | leak_groups=0
[PAIR 11vs20]   groups: train G=5454 (g0=860, g1=819, g2=3775) | val G=1362 (g0=214, g1=204, g2=944)
[PAIR 11vs20] Fold 02: train N=17649 (c0=5821, c1=5425, c2=6403) | val N=4413 (c0=1456, c1=1356, c2=1601) | leak_groups=0
[PAIR 11vs20]   groups: train G=5453 (g0=859, g1=819, g2=3775) | val G=1363 (g0=215, g1=204, g2=944)
[PAIR 11vs20] Fold 03: train N=17650 (c0=5822, c1=5425, c2=6403) | val N=4412 (c0=1455, c1=1356, c2=1601) | leak_groups=0
[PAIR 11vs20]   groups: train G=5452 (g0=859, g1=818, g2=3775) | val G=1364 (g0=215, g1=205, g2=944)
[PAIR 11vs20] Fold 04: train N=17651 (c0=5822, c1=5425, c2=6404) | val N=4411 (c0=1455, c1=1356, c2=1600) | leak_groups=0
[PAIR 11vs20]   groups: train G=5453 (g0=859, g1=818, g2=3776) | val G=1363 (g0=215, g1=205, g2=943)
[PAIR 11vs20] ===== End summary =====
[PAIR 11vs20] SUCCESS: Created 5 balanced folds (manual round-robin)
[CNN_BATCH] Stopping: avail c0=397 c1=0 c2=979, need full=6 or small>=4
[CNN_BATCH] Created 904 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17648 c0=5821 c1=5424 c2=6403
[SPLIT] val   windows=4414   c0=1456 c1=1357 c2=1601
Early stop at epoch 31 (best val loss 0.0204).
[CNN_BATCH] Stopping: avail c0=398 c1=1 c2=979, need full=6 or small>=4
[CNN_BATCH] Created 904 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17650 c0=5822 c1=5425 c2=6403
[SPLIT] val   windows=4412   c0=1455 c1=1356 c2=1601
Early stop at epoch 26 (best val loss 0.0106).
[CNN_BATCH] Stopping: avail c0=397 c1=1 c2=979, need full=6 or small>=4
[CNN_BATCH] Created 904 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17649 c0=5821 c1=5425 c2=6403
[SPLIT] val   windows=4413   c0=1456 c1=1356 c2=1601
Early stop at epoch 26 (best val loss 0.0098).
[CNN_BATCH] Stopping: avail c0=398 c1=1 c2=979, need full=6 or small>=4
[CNN_BATCH] Created 904 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17650 c0=5822 c1=5425 c2=6403
[SPLIT] val   windows=4412   c0=1455 c1=1356 c2=1601
Early stop at epoch 32 (best val loss 0.0089).
[CNN_BATCH] Stopping: avail c0=398 c1=1 c2=980, need full=6 or small>=4
[CNN_BATCH] Created 904 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17651 c0=5822 c1=5425 c2=6404
[SPLIT] val   windows=4411   c0=1455 c1=1356 c2=1600
Early stop at epoch 26 (best val loss 0.0317).
[PAIR_SEARCH] pair (11,20) mean_bal_acc=0.998 (+/-0.001)
[K_FOLDS] Groups: c0=1058 c1=1055 c2=4719
[K_FOLDS] Windows: c0=7115 c1=7088 c2=7826
[K_FOLDS] Selected k=5 (preferred=5)
[K_FOLDS] Using preferred k=5
[PAIR 8vs17] ===== Fold builder (blocked groups) =====
[PAIR 8vs17] N_windows=22029
[PAIR 8vs17] block_size_windows=ceil(W/H)=7
[PAIR 8vs17] n_groups_total=6832
[PAIR 8vs17] k_eff (group-limited)=5

[PAIR 8vs17] GROUPS per class: g0=1058 g1=1055 g2=4719 | group_size(min/med/max)=1/2/7 | c0(min/med/max)=1/7/7 c1(min/med/max)=1/7/7 c2(min/med/max)=1/2/2
[PAIR 8vs17] ===== Blocked-fold summary =====
[PAIR 8vs17] N_windows=22029
[PAIR 8vs17] n_time(W)=560 samples
[PAIR 8vs17] hop_samples(H)=80
[PAIR 8vs17] block_size_windows=ceil(W/H)=7
[PAIR 8vs17] n_groups=6832
[PAIR 8vs17] k_used=5

[PAIR 8vs17] group purity OK ✓

[PAIR 8vs17] overall class counts: c0=7115, c1=7088, c2=7826

[PAIR 8vs17] Fold 00: train N=17622 (c0=5692, c1=5670, c2=6260) | val N=4407 (c0=1423, c1=1418, c2=1566) | leak_groups=0
[PAIR 8vs17]   groups: train G=5466 (g0=847, g1=844, g2=3775) | val G=1366 (g0=211, g1=211, g2=944)
[PAIR 8vs17] Fold 01: train N=17623 (c0=5692, c1=5670, c2=6261) | val N=4406 (c0=1423, c1=1418, c2=1565) | leak_groups=0
[PAIR 8vs17]   groups: train G=5467 (g0=847, g1=844, g2=3776) | val G=1365 (g0=211, g1=211, g2=943)
[PAIR 8vs17] Fold 02: train N=17623 (c0=5692, c1=5670, c2=6261) | val N=4406 (c0=1423, c1=1418, c2=1565) | leak_groups=0
[PAIR 8vs17]   groups: train G=5465 (g0=846, g1=844, g2=3775) | val G=1367 (g0=212, g1=211, g2=944)
[PAIR 8vs17] Fold 03: train N=17624 (c0=5692, c1=5671, c2=6261) | val N=4405 (c0=1423, c1=1417, c2=1565) | leak_groups=0
[PAIR 8vs17]   groups: train G=5465 (g0=846, g1=844, g2=3775) | val G=1367 (g0=212, g1=211, g2=944)
[PAIR 8vs17] Fold 04: train N=17624 (c0=5692, c1=5671, c2=6261) | val N=4405 (c0=1423, c1=1417, c2=1565) | leak_groups=0
[PAIR 8vs17]   groups: train G=5465 (g0=846, g1=844, g2=3775) | val G=1367 (g0=212, g1=211, g2=944)
[PAIR 8vs17] ===== End summary =====
[PAIR 8vs17] SUCCESS: Created 5 balanced folds (manual round-robin)
[CNN_BATCH] Stopping: avail c0=22 c1=0 c2=590, need full=6 or small>=4
[CNN_BATCH] Created 945 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17622 c0=5692 c1=5670 c2=6260
[SPLIT] val   windows=4407   c0=1423 c1=1418 c2=1566
Early stop at epoch 26 (best val loss 0.0228).
[CNN_BATCH] Stopping: avail c0=22 c1=0 c2=591, need full=6 or small>=4
[CNN_BATCH] Created 945 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17623 c0=5692 c1=5670 c2=6261
[SPLIT] val   windows=4406   c0=1423 c1=1418 c2=1565
Early stop at epoch 26 (best val loss 0.0089).
[CNN_BATCH] Stopping: avail c0=22 c1=0 c2=591, need full=6 or small>=4
[CNN_BATCH] Created 945 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17623 c0=5692 c1=5670 c2=6261
[SPLIT] val   windows=4406   c0=1423 c1=1418 c2=1565
Early stop at epoch 26 (best val loss 0.0261).
[CNN_BATCH] Stopping: avail c0=22 c1=1 c2=591, need full=6 or small>=4
[CNN_BATCH] Created 945 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17624 c0=5692 c1=5671 c2=6261
[SPLIT] val   windows=4405   c0=1423 c1=1417 c2=1565
Early stop at epoch 39 (best val loss 0.0202).
[CNN_BATCH] Stopping: avail c0=22 c1=1 c2=591, need full=6 or small>=4
[CNN_BATCH] Created 945 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17624 c0=5692 c1=5671 c2=6261
[SPLIT] val   windows=4405   c0=1423 c1=1417 c2=1565
Early stop at epoch 26 (best val loss 0.0128).
[PAIR_SEARCH] pair (8,17) mean_bal_acc=0.998 (+/-0.001)
[K_FOLDS] Groups: c0=1058 c1=1021 c2=4719
[K_FOLDS] Windows: c0=7115 c1=6806 c2=7826
[K_FOLDS] Selected k=5 (preferred=5)
[K_FOLDS] Using preferred k=5
[PAIR 8vs14] ===== Fold builder (blocked groups) =====
[PAIR 8vs14] N_windows=21747
[PAIR 8vs14] block_size_windows=ceil(W/H)=7
[PAIR 8vs14] n_groups_total=6798
[PAIR 8vs14] k_eff (group-limited)=5

[PAIR 8vs14] GROUPS per class: g0=1058 g1=1021 g2=4719 | group_size(min/med/max)=1/2/7 | c0(min/med/max)=1/7/7 c1(min/med/max)=1/7/7 c2(min/med/max)=1/2/2
[PAIR 8vs14] ===== Blocked-fold summary =====
[PAIR 8vs14] N_windows=21747
[PAIR 8vs14] n_time(W)=560 samples
[PAIR 8vs14] hop_samples(H)=80
[PAIR 8vs14] block_size_windows=ceil(W/H)=7
[PAIR 8vs14] n_groups=6798
[PAIR 8vs14] k_used=5

[PAIR 8vs14] group purity OK ✓

[PAIR 8vs14] overall class counts: c0=7115, c1=6806, c2=7826

[PAIR 8vs14] Fold 00: train N=17396 (c0=5692, c1=5444, c2=6260) | val N=4351 (c0=1423, c1=1362, c2=1566) | leak_groups=0
[PAIR 8vs14]   groups: train G=5438 (g0=847, g1=816, g2=3775) | val G=1360 (g0=211, g1=205, g2=944)
[PAIR 8vs14] Fold 01: train N=17398 (c0=5692, c1=5445, c2=6261) | val N=4349 (c0=1423, c1=1361, c2=1565) | leak_groups=0
[PAIR 8vs14]   groups: train G=5440 (g0=847, g1=817, g2=3776) | val G=1358 (g0=211, g1=204, g2=943)
[PAIR 8vs14] Fold 02: train N=17398 (c0=5692, c1=5445, c2=6261) | val N=4349 (c0=1423, c1=1361, c2=1565) | leak_groups=0
[PAIR 8vs14]   groups: train G=5438 (g0=846, g1=817, g2=3775) | val G=1360 (g0=212, g1=204, g2=944)
[PAIR 8vs14] Fold 03: train N=17398 (c0=5692, c1=5445, c2=6261) | val N=4349 (c0=1423, c1=1361, c2=1565) | leak_groups=0
[PAIR 8vs14]   groups: train G=5438 (g0=846, g1=817, g2=3775) | val G=1360 (g0=212, g1=204, g2=944)
[PAIR 8vs14] Fold 04: train N=17398 (c0=5692, c1=5445, c2=6261) | val N=4349 (c0=1423, c1=1361, c2=1565) | leak_groups=0
[PAIR 8vs14]   groups: train G=5438 (g0=846, g1=817, g2=3775) | val G=1360 (g0=212, g1=204, g2=944)
[PAIR 8vs14] ===== End summary =====
[PAIR 8vs14] SUCCESS: Created 5 balanced folds (manual round-robin)
[CNN_BATCH] Stopping: avail c0=250 c1=2 c2=818, need full=6 or small>=4
[CNN_BATCH] Created 907 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17396 c0=5692 c1=5444 c2=6260
[SPLIT] val   windows=4351   c0=1423 c1=1362 c2=1566
Early stop at epoch 34 (best val loss 0.0207).
[CNN_BATCH] Stopping: avail c0=250 c1=3 c2=819, need full=6 or small>=4
[CNN_BATCH] Created 907 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17398 c0=5692 c1=5445 c2=6261
[SPLIT] val   windows=4349   c0=1423 c1=1361 c2=1565
Early stop at epoch 26 (best val loss 0.0093).
[CNN_BATCH] Stopping: avail c0=250 c1=3 c2=819, need full=6 or small>=4
[CNN_BATCH] Created 907 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17398 c0=5692 c1=5445 c2=6261
[SPLIT] val   windows=4349   c0=1423 c1=1361 c2=1565
Early stop at epoch 37 (best val loss 0.0279).
[CNN_BATCH] Stopping: avail c0=250 c1=3 c2=819, need full=6 or small>=4
[CNN_BATCH] Created 907 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17398 c0=5692 c1=5445 c2=6261
[SPLIT] val   windows=4349   c0=1423 c1=1361 c2=1565
Early stop at epoch 26 (best val loss 0.0195).
[CNN_BATCH] Stopping: avail c0=250 c1=3 c2=819, need full=6 or small>=4
[CNN_BATCH] Created 907 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17398 c0=5692 c1=5445 c2=6261
[SPLIT] val   windows=4349   c0=1423 c1=1361 c2=1565
Early stop at epoch 26 (best val loss 0.0184).
[PAIR_SEARCH] pair (8,14) mean_bal_acc=0.998 (+/-0.001)
[K_FOLDS] Groups: c0=1058 c1=1023 c2=4719
[K_FOLDS] Windows: c0=7115 c1=6781 c2=7826
[K_FOLDS] Selected k=5 (preferred=5)
[K_FOLDS] Using preferred k=5
[PAIR 8vs20] ===== Fold builder (blocked groups) =====
[PAIR 8vs20] N_windows=21722
[PAIR 8vs20] block_size_windows=ceil(W/H)=7
[PAIR 8vs20] n_groups_total=6800
[PAIR 8vs20] k_eff (group-limited)=5

[PAIR 8vs20] GROUPS per class: g0=1058 g1=1023 g2=4719 | group_size(min/med/max)=1/2/7 | c0(min/med/max)=1/7/7 c1(min/med/max)=1/7/7 c2(min/med/max)=1/2/2
[PAIR 8vs20] ===== Blocked-fold summary =====
[PAIR 8vs20] N_windows=21722
[PAIR 8vs20] n_time(W)=560 samples
[PAIR 8vs20] hop_samples(H)=80
[PAIR 8vs20] block_size_windows=ceil(W/H)=7
[PAIR 8vs20] n_groups=6800
[PAIR 8vs20] k_used=5

[PAIR 8vs20] group purity OK ✓

[PAIR 8vs20] overall class counts: c0=7115, c1=6781, c2=7826

[PAIR 8vs20] Fold 00: train N=17376 (c0=5692, c1=5424, c2=6260) | val N=4346 (c0=1423, c1=1357, c2=1566) | leak_groups=0
[PAIR 8vs20]   groups: train G=5440 (g0=847, g1=818, g2=3775) | val G=1360 (g0=211, g1=205, g2=944)
[PAIR 8vs20] Fold 01: train N=17378 (c0=5692, c1=5425, c2=6261) | val N=4344 (c0=1423, c1=1356, c2=1565) | leak_groups=0
[PAIR 8vs20]   groups: train G=5442 (g0=847, g1=819, g2=3776) | val G=1358 (g0=211, g1=204, g2=943)
[PAIR 8vs20] Fold 02: train N=17378 (c0=5692, c1=5425, c2=6261) | val N=4344 (c0=1423, c1=1356, c2=1565) | leak_groups=0
[PAIR 8vs20]   groups: train G=5440 (g0=846, g1=819, g2=3775) | val G=1360 (g0=212, g1=204, g2=944)
[PAIR 8vs20] Fold 03: train N=17378 (c0=5692, c1=5425, c2=6261) | val N=4344 (c0=1423, c1=1356, c2=1565) | leak_groups=0
[PAIR 8vs20]   groups: train G=5439 (g0=846, g1=818, g2=3775) | val G=1361 (g0=212, g1=205, g2=944)
[PAIR 8vs20] Fold 04: train N=17378 (c0=5692, c1=5425, c2=6261) | val N=4344 (c0=1423, c1=1356, c2=1565) | leak_groups=0
[PAIR 8vs20]   groups: train G=5439 (g0=846, g1=818, g2=3775) | val G=1361 (g0=212, g1=205, g2=944)
[PAIR 8vs20] ===== End summary =====
[PAIR 8vs20] SUCCESS: Created 5 balanced folds (manual round-robin)
[CNN_BATCH] Stopping: avail c0=268 c1=0 c2=836, need full=6 or small>=4
[CNN_BATCH] Created 904 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17376 c0=5692 c1=5424 c2=6260
[SPLIT] val   windows=4346   c0=1423 c1=1357 c2=1566
Early stop at epoch 26 (best val loss 0.0264).
[CNN_BATCH] Stopping: avail c0=268 c1=1 c2=837, need full=6 or small>=4
[CNN_BATCH] Created 904 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17378 c0=5692 c1=5425 c2=6261
[SPLIT] val   windows=4344   c0=1423 c1=1356 c2=1565
Early stop at epoch 26 (best val loss 0.0119).
[CNN_BATCH] Stopping: avail c0=268 c1=1 c2=837, need full=6 or small>=4
[CNN_BATCH] Created 904 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17378 c0=5692 c1=5425 c2=6261
[SPLIT] val   windows=4344   c0=1423 c1=1356 c2=1565
Early stop at epoch 26 (best val loss 0.0611).
[CNN_BATCH] Stopping: avail c0=268 c1=1 c2=837, need full=6 or small>=4
[CNN_BATCH] Created 904 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17378 c0=5692 c1=5425 c2=6261
[SPLIT] val   windows=4344   c0=1423 c1=1356 c2=1565
Early stop at epoch 26 (best val loss 0.0167).
[CNN_BATCH] Stopping: avail c0=268 c1=1 c2=837, need full=6 or small>=4
[CNN_BATCH] Created 904 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17378 c0=5692 c1=5425 c2=6261
[SPLIT] val   windows=4344   c0=1423 c1=1356 c2=1565
Early stop at epoch 26 (best val loss 0.0106).
[PAIR_SEARCH] pair (8,20) mean_bal_acc=0.996 (+/-0.005)
[K_FOLDS] Groups: c0=1055 c1=1021 c2=4719
[K_FOLDS] Windows: c0=7088 c1=6806 c2=7796
[K_FOLDS] Selected k=5 (preferred=5)
[K_FOLDS] Using preferred k=5
[PAIR 17vs14] ===== Fold builder (blocked groups) =====
[PAIR 17vs14] N_windows=21690
[PAIR 17vs14] block_size_windows=ceil(W/H)=7
[PAIR 17vs14] n_groups_total=6795
[PAIR 17vs14] k_eff (group-limited)=5

[PAIR 17vs14] GROUPS per class: g0=1055 g1=1021 g2=4719 | group_size(min/med/max)=1/2/7 | c0(min/med/max)=1/7/7 c1(min/med/max)=1/7/7 c2(min/med/max)=1/2/2
[PAIR 17vs14] ===== Blocked-fold summary =====
[PAIR 17vs14] N_windows=21690
[PAIR 17vs14] n_time(W)=560 samples
[PAIR 17vs14] hop_samples(H)=80
[PAIR 17vs14] block_size_windows=ceil(W/H)=7
[PAIR 17vs14] n_groups=6795
[PAIR 17vs14] k_used=5

[PAIR 17vs14] group purity OK ✓

[PAIR 17vs14] overall class counts: c0=7088, c1=6806, c2=7796

[PAIR 17vs14] Fold 00: train N=17350 (c0=5670, c1=5444, c2=6236) | val N=4340 (c0=1418, c1=1362, c2=1560) | leak_groups=0
[PAIR 17vs14]   groups: train G=5435 (g0=844, g1=816, g2=3775) | val G=1360 (g0=211, g1=205, g2=944)
[PAIR 17vs14] Fold 01: train N=17352 (c0=5670, c1=5445, c2=6237) | val N=4338 (c0=1418, c1=1361, c2=1559) | leak_groups=0
[PAIR 17vs14]   groups: train G=5437 (g0=844, g1=817, g2=3776) | val G=1358 (g0=211, g1=204, g2=943)
[PAIR 17vs14] Fold 02: train N=17352 (c0=5670, c1=5445, c2=6237) | val N=4338 (c0=1418, c1=1361, c2=1559) | leak_groups=0
[PAIR 17vs14]   groups: train G=5436 (g0=844, g1=817, g2=3775) | val G=1359 (g0=211, g1=204, g2=944)
[PAIR 17vs14] Fold 03: train N=17353 (c0=5671, c1=5445, c2=6237) | val N=4337 (c0=1417, c1=1361, c2=1559) | leak_groups=0
[PAIR 17vs14]   groups: train G=5436 (g0=844, g1=817, g2=3775) | val G=1359 (g0=211, g1=204, g2=944)
[PAIR 17vs14] Fold 04: train N=17353 (c0=5671, c1=5445, c2=6237) | val N=4337 (c0=1417, c1=1361, c2=1559) | leak_groups=0
[PAIR 17vs14]   groups: train G=5436 (g0=844, g1=817, g2=3775) | val G=1359 (g0=211, g1=204, g2=944)
[PAIR 17vs14] ===== End summary =====
[PAIR 17vs14] SUCCESS: Created 5 balanced folds (manual round-robin)
[CNN_BATCH] Stopping: avail c0=228 c1=2 c2=794, need full=6 or small>=4
[CNN_BATCH] Created 907 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17350 c0=5670 c1=5444 c2=6236
[SPLIT] val   windows=4340   c0=1418 c1=1362 c2=1560
Early stop at epoch 47 (best val loss 0.0245).
[CNN_BATCH] Stopping: avail c0=228 c1=3 c2=795, need full=6 or small>=4
[CNN_BATCH] Created 907 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17352 c0=5670 c1=5445 c2=6237
[SPLIT] val   windows=4338   c0=1418 c1=1361 c2=1559
Early stop at epoch 47 (best val loss 0.0147).
[CNN_BATCH] Stopping: avail c0=228 c1=3 c2=795, need full=6 or small>=4
[CNN_BATCH] Created 907 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17352 c0=5670 c1=5445 c2=6237
[SPLIT] val   windows=4338   c0=1418 c1=1361 c2=1559
Early stop at epoch 50 (best val loss 0.0164).
[CNN_BATCH] Stopping: avail c0=229 c1=3 c2=795, need full=6 or small>=4
[CNN_BATCH] Created 907 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17353 c0=5671 c1=5445 c2=6237
[SPLIT] val   windows=4337   c0=1417 c1=1361 c2=1559
Early stop at epoch 32 (best val loss 0.0232).
[CNN_BATCH] Stopping: avail c0=229 c1=3 c2=795, need full=6 or small>=4
[CNN_BATCH] Created 907 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17353 c0=5671 c1=5445 c2=6237
[SPLIT] val   windows=4337   c0=1417 c1=1361 c2=1559
Early stop at epoch 36 (best val loss 0.0136).
[PAIR_SEARCH] pair (17,14) mean_bal_acc=0.998 (+/-0.001)
[K_FOLDS] Groups: c0=1055 c1=1023 c2=4719
[K_FOLDS] Windows: c0=7088 c1=6781 c2=7796
[K_FOLDS] Selected k=5 (preferred=5)
[K_FOLDS] Using preferred k=5
[PAIR 17vs20] ===== Fold builder (blocked groups) =====
[PAIR 17vs20] N_windows=21665
[PAIR 17vs20] block_size_windows=ceil(W/H)=7
[PAIR 17vs20] n_groups_total=6797
[PAIR 17vs20] k_eff (group-limited)=5

[PAIR 17vs20] GROUPS per class: g0=1055 g1=1023 g2=4719 | group_size(min/med/max)=1/2/7 | c0(min/med/max)=1/7/7 c1(min/med/max)=1/7/7 c2(min/med/max)=1/2/2
[PAIR 17vs20] ===== Blocked-fold summary =====
[PAIR 17vs20] N_windows=21665
[PAIR 17vs20] n_time(W)=560 samples
[PAIR 17vs20] hop_samples(H)=80
[PAIR 17vs20] block_size_windows=ceil(W/H)=7
[PAIR 17vs20] n_groups=6797
[PAIR 17vs20] k_used=5

[PAIR 17vs20] group purity OK ✓

[PAIR 17vs20] overall class counts: c0=7088, c1=6781, c2=7796

[PAIR 17vs20] Fold 00: train N=17330 (c0=5670, c1=5424, c2=6236) | val N=4335 (c0=1418, c1=1357, c2=1560) | leak_groups=0
[PAIR 17vs20]   groups: train G=5437 (g0=844, g1=818, g2=3775) | val G=1360 (g0=211, g1=205, g2=944)
[PAIR 17vs20] Fold 01: train N=17332 (c0=5670, c1=5425, c2=6237) | val N=4333 (c0=1418, c1=1356, c2=1559) | leak_groups=0
[PAIR 17vs20]   groups: train G=5439 (g0=844, g1=819, g2=3776) | val G=1358 (g0=211, g1=204, g2=943)
[PAIR 17vs20] Fold 02: train N=17332 (c0=5670, c1=5425, c2=6237) | val N=4333 (c0=1418, c1=1356, c2=1559) | leak_groups=0
[PAIR 17vs20]   groups: train G=5438 (g0=844, g1=819, g2=3775) | val G=1359 (g0=211, g1=204, g2=944)
[PAIR 17vs20] Fold 03: train N=17333 (c0=5671, c1=5425, c2=6237) | val N=4332 (c0=1417, c1=1356, c2=1559) | leak_groups=0
[PAIR 17vs20]   groups: train G=5437 (g0=844, g1=818, g2=3775) | val G=1360 (g0=211, g1=205, g2=944)
[PAIR 17vs20] Fold 04: train N=17333 (c0=5671, c1=5425, c2=6237) | val N=4332 (c0=1417, c1=1356, c2=1559) | leak_groups=0
[PAIR 17vs20]   groups: train G=5437 (g0=844, g1=818, g2=3775) | val G=1360 (g0=211, g1=205, g2=944)
[PAIR 17vs20] ===== End summary =====
[PAIR 17vs20] SUCCESS: Created 5 balanced folds (manual round-robin)
[CNN_BATCH] Stopping: avail c0=246 c1=0 c2=812, need full=6 or small>=4
[CNN_BATCH] Created 904 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17330 c0=5670 c1=5424 c2=6236
[SPLIT] val   windows=4335   c0=1418 c1=1357 c2=1560
Early stop at epoch 29 (best val loss 0.0222).
[CNN_BATCH] Stopping: avail c0=246 c1=1 c2=813, need full=6 or small>=4
[CNN_BATCH] Created 904 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17332 c0=5670 c1=5425 c2=6237
[SPLIT] val   windows=4333   c0=1418 c1=1356 c2=1559
Early stop at epoch 26 (best val loss 0.0144).
[CNN_BATCH] Stopping: avail c0=246 c1=1 c2=813, need full=6 or small>=4
[CNN_BATCH] Created 904 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17332 c0=5670 c1=5425 c2=6237
[SPLIT] val   windows=4333   c0=1418 c1=1356 c2=1559
Early stop at epoch 26 (best val loss 0.0159).
[CNN_BATCH] Stopping: avail c0=247 c1=1 c2=813, need full=6 or small>=4
[CNN_BATCH] Created 904 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17333 c0=5671 c1=5425 c2=6237
[SPLIT] val   windows=4332   c0=1417 c1=1356 c2=1559
Early stop at epoch 31 (best val loss 0.0126).
[CNN_BATCH] Stopping: avail c0=247 c1=1 c2=813, need full=6 or small>=4
[CNN_BATCH] Created 904 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17333 c0=5671 c1=5425 c2=6237
[SPLIT] val   windows=4332   c0=1417 c1=1356 c2=1559
Early stop at epoch 40 (best val loss 0.0149).
[PAIR_SEARCH] pair (17,20) mean_bal_acc=0.998 (+/-0.000)
[K_FOLDS] Groups: c0=1021 c1=1023 c2=4719
[K_FOLDS] Windows: c0=6806 c1=6781 c2=7486
[K_FOLDS] Selected k=5 (preferred=5)
[K_FOLDS] Using preferred k=5
[PAIR 14vs20] ===== Fold builder (blocked groups) =====
[PAIR 14vs20] N_windows=21073
[PAIR 14vs20] block_size_windows=ceil(W/H)=7
[PAIR 14vs20] n_groups_total=6763
[PAIR 14vs20] k_eff (group-limited)=5

[PAIR 14vs20] GROUPS per class: g0=1021 g1=1023 g2=4719 | group_size(min/med/max)=1/2/7 | c0(min/med/max)=1/7/7 c1(min/med/max)=1/7/7 c2(min/med/max)=1/2/2
[PAIR 14vs20] ===== Blocked-fold summary =====
[PAIR 14vs20] N_windows=21073
[PAIR 14vs20] n_time(W)=560 samples
[PAIR 14vs20] hop_samples(H)=80
[PAIR 14vs20] block_size_windows=ceil(W/H)=7
[PAIR 14vs20] n_groups=6763
[PAIR 14vs20] k_used=5

[PAIR 14vs20] group purity OK ✓

[PAIR 14vs20] overall class counts: c0=6806, c1=6781, c2=7486

[PAIR 14vs20] Fold 00: train N=16856 (c0=5444, c1=5424, c2=5988) | val N=4217 (c0=1362, c1=1357, c2=1498) | leak_groups=0
[PAIR 14vs20]   groups: train G=5409 (g0=816, g1=818, g2=3775) | val G=1354 (g0=205, g1=205, g2=944)
[PAIR 14vs20] Fold 01: train N=16859 (c0=5445, c1=5425, c2=5989) | val N=4214 (c0=1361, c1=1356, c2=1497) | leak_groups=0
[PAIR 14vs20]   groups: train G=5412 (g0=817, g1=819, g2=3776) | val G=1351 (g0=204, g1=204, g2=943)
[PAIR 14vs20] Fold 02: train N=16859 (c0=5445, c1=5425, c2=5989) | val N=4214 (c0=1361, c1=1356, c2=1497) | leak_groups=0
[PAIR 14vs20]   groups: train G=5411 (g0=817, g1=819, g2=3775) | val G=1352 (g0=204, g1=204, g2=944)
[PAIR 14vs20] Fold 03: train N=16859 (c0=5445, c1=5425, c2=5989) | val N=4214 (c0=1361, c1=1356, c2=1497) | leak_groups=0
[PAIR 14vs20]   groups: train G=5410 (g0=817, g1=818, g2=3775) | val G=1353 (g0=204, g1=205, g2=944)
[PAIR 14vs20] Fold 04: train N=16859 (c0=5445, c1=5425, c2=5989) | val N=4214 (c0=1361, c1=1356, c2=1497) | leak_groups=0
[PAIR 14vs20]   groups: train G=5410 (g0=817, g1=818, g2=3775) | val G=1353 (g0=204, g1=205, g2=944)
[PAIR 14vs20] ===== End summary =====
[PAIR 14vs20] SUCCESS: Created 5 balanced folds (manual round-robin)
[CNN_BATCH] Stopping: avail c0=20 c1=0 c2=564, need full=6 or small>=4
[CNN_BATCH] Created 904 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=16856 c0=5444 c1=5424 c2=5988
[SPLIT] val   windows=4217   c0=1362 c1=1357 c2=1498
Early stop at epoch 26 (best val loss 0.0043).
[CNN_BATCH] Stopping: avail c0=21 c1=1 c2=565, need full=6 or small>=4
[CNN_BATCH] Created 904 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=16859 c0=5445 c1=5425 c2=5989
[SPLIT] val   windows=4214   c0=1361 c1=1356 c2=1497
Early stop at epoch 26 (best val loss 0.0407).
[CNN_BATCH] Stopping: avail c0=21 c1=1 c2=565, need full=6 or small>=4
[CNN_BATCH] Created 904 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=16859 c0=5445 c1=5425 c2=5989
[SPLIT] val   windows=4214   c0=1361 c1=1356 c2=1497
Early stop at epoch 34 (best val loss 0.0075).
[CNN_BATCH] Stopping: avail c0=21 c1=1 c2=565, need full=6 or small>=4
[CNN_BATCH] Created 904 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=16859 c0=5445 c1=5425 c2=5989
[SPLIT] val   windows=4214   c0=1361 c1=1356 c2=1497
Early stop at epoch 32 (best val loss 0.0204).
[CNN_BATCH] Stopping: avail c0=21 c1=1 c2=565, need full=6 or small>=4
[CNN_BATCH] Created 904 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=16859 c0=5445 c1=5425 c2=5989
[SPLIT] val   windows=4214   c0=1361 c1=1356 c2=1497
Early stop at epoch 30 (best val loss 0.0157).
[PAIR_SEARCH] pair (14,20) mean_bal_acc=0.998 (+/-0.001)
[K_FOLDS] Groups: c0=1055 c1=1023 c2=4719
[K_FOLDS] Windows: c0=7088 c1=6781 c2=7796
[K_FOLDS] Selected k=5 (preferred=5)
[K_FOLDS] Using preferred k=5
[FINAL 17vs20] ===== Fold builder (blocked groups) =====
[FINAL 17vs20] N_windows=21665
[FINAL 17vs20] block_size_windows=ceil(W/H)=7
[FINAL 17vs20] n_groups_total=6797
[FINAL 17vs20] k_eff (group-limited)=5

[FINAL 17vs20] GROUPS per class: g0=1055 g1=1023 g2=4719 | group_size(min/med/max)=1/2/7 | c0(min/med/max)=1/7/7 c1(min/med/max)=1/7/7 c2(min/med/max)=1/2/2
[FINAL 17vs20] ===== Blocked-fold summary =====
[FINAL 17vs20] N_windows=21665
[FINAL 17vs20] n_time(W)=560 samples
[FINAL 17vs20] hop_samples(H)=80
[FINAL 17vs20] block_size_windows=ceil(W/H)=7
[FINAL 17vs20] n_groups=6797
[FINAL 17vs20] k_used=5

[FINAL 17vs20] group purity OK ✓

[FINAL 17vs20] overall class counts: c0=7088, c1=6781, c2=7796

[FINAL 17vs20] Fold 00: train N=17330 (c0=5670, c1=5424, c2=6236) | val N=4335 (c0=1418, c1=1357, c2=1560) | leak_groups=0
[FINAL 17vs20]   groups: train G=5437 (g0=844, g1=818, g2=3775) | val G=1360 (g0=211, g1=205, g2=944)
[FINAL 17vs20] Fold 01: train N=17332 (c0=5670, c1=5425, c2=6237) | val N=4333 (c0=1418, c1=1356, c2=1559) | leak_groups=0
[FINAL 17vs20]   groups: train G=5439 (g0=844, g1=819, g2=3776) | val G=1358 (g0=211, g1=204, g2=943)
[FINAL 17vs20] Fold 02: train N=17332 (c0=5670, c1=5425, c2=6237) | val N=4333 (c0=1418, c1=1356, c2=1559) | leak_groups=0
[FINAL 17vs20]   groups: train G=5438 (g0=844, g1=819, g2=3775) | val G=1359 (g0=211, g1=204, g2=944)
[FINAL 17vs20] Fold 03: train N=17333 (c0=5671, c1=5425, c2=6237) | val N=4332 (c0=1417, c1=1356, c2=1559) | leak_groups=0
[FINAL 17vs20]   groups: train G=5437 (g0=844, g1=818, g2=3775) | val G=1360 (g0=211, g1=205, g2=944)
[FINAL 17vs20] Fold 04: train N=17333 (c0=5671, c1=5425, c2=6237) | val N=4332 (c0=1417, c1=1356, c2=1559) | leak_groups=0
[FINAL 17vs20]   groups: train G=5437 (g0=844, g1=818, g2=3775) | val G=1360 (g0=211, g1=205, g2=944)
[FINAL 17vs20] ===== End summary =====
[FINAL 17vs20] SUCCESS: Created 5 balanced folds (manual round-robin)
[CNN_BATCH] Stopping: avail c0=246 c1=0 c2=812, need full=6 or small>=4
[CNN_BATCH] Created 904 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17330 c0=5670 c1=5424 c2=6236
[SPLIT] val   windows=4335   c0=1418 c1=1357 c2=1560
Early stop at epoch 29 (best val loss 0.0222).
[CNN_BATCH] Stopping: avail c0=246 c1=1 c2=813, need full=6 or small>=4
[CNN_BATCH] Created 904 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17332 c0=5670 c1=5425 c2=6237
[SPLIT] val   windows=4333   c0=1418 c1=1356 c2=1559
Early stop at epoch 26 (best val loss 0.0144).
[CNN_BATCH] Stopping: avail c0=246 c1=1 c2=813, need full=6 or small>=4
[CNN_BATCH] Created 904 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17332 c0=5670 c1=5425 c2=6237
[SPLIT] val   windows=4333   c0=1418 c1=1356 c2=1559
Early stop at epoch 26 (best val loss 0.0159).
[CNN_BATCH] Stopping: avail c0=247 c1=1 c2=813, need full=6 or small>=4
[CNN_BATCH] Created 904 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17333 c0=5671 c1=5425 c2=6237
[SPLIT] val   windows=4332   c0=1417 c1=1356 c2=1559
Early stop at epoch 31 (best val loss 0.0126).
[CNN_BATCH] Stopping: avail c0=247 c1=1 c2=813, need full=6 or small>=4
[CNN_BATCH] Created 904 batches: sizes min=18 max=18 mean=18.0
[CNN_BATCH] Batch 0: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 1: size=18 c0=6 c1=6 c2=6
[CNN_BATCH] Batch 2: size=18 c0=6 c1=6 c2=6
[BATCH_DEBUG] Inspecting first 3 batches...
[BATCH_DEBUG] Batch 0: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 1: size=18 | c0=6 c1=6 c2=6
[BATCH_DEBUG] Batch 2: size=18 | c0=6 c1=6 c2=6
[SPLIT] train windows=17333 c0=5671 c1=5425 c2=6237
[SPLIT] val   windows=4332   c0=1417 c1=1356 c2=1559
Early stop at epoch 40 (best val loss 0.0149).
[HOLDOUT] Total windows=21665 (c0=7088 c1=6781 c2=7796)
[HOLDOUT] Target holdout ~= 3250 windows (0.15 of total)
[HOLDOUT] Selected groups: 610 | hold windows=3255 (c0=1428 c1=1421 c2=406) | train windows=18410 (c0=5660 c1=5360 c2=7390)
[HOLDOUT] train windows=18410 c0=5660 c1=5360 c2=7390
[HOLDOUT] hold  windows=3255 c0=1428 c1=1421 c2=406
"""

    runs, pair_search = parse_log(LOG_TEXT)

    # 1) early stop epoch + best val loss over run order
    plot_early_stop(runs)

    # 2) pair-search summary (mean +/- std)
    plot_pair_search(pair_search)

    # 3) fold balance sanity (max/min ratio)
    plot_fold_balance(runs)

    # 4) batch leftovers (why c1 hits 0 constantly)
    plot_batch_leftovers(runs)

    plot_best_val_loss_by_pair(runs)

    out_dir = Path("train_debug_plots") / datetime.now().strftime("%Y%m%d_%H%M%S")
    save_all_figs(out_dir)
    plt.show()

if __name__ == "__main__":
    main()
