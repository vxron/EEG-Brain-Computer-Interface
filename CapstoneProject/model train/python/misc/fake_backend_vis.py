"""
Visualize simulated EEG calibration data from eeg_calib_data.csv
Outputs:
 - Time-domain plots for each stimulus frequency in the calib protocol (all 8 channels)
 - PSD plots showing SSVEP peaks per frequency for a chosen channel
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch

FS = 250.0           # sampling rate [Hz]
N_CHANNELS = 8       # eeg1..eeg8


def compute_psd(x: np.ndarray, fs: float):
    """
    Simple PSD estimate using FFT (periodogram).

    Returns:
        freqs: frequency axis (Hz)
        psd:   power spectral density (µV^2/Hz)
    """
    print("debug")
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = len(x)
    if n < 2:
        return np.array([0.0]), np.array([0.0])

    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_vals = np.fft.rfft(x)
    psd = (np.abs(fft_vals) ** 2) / (fs * n)
    return freqs, psd


def plot_time_domain(df: pd.DataFrame, out_dir: Path, seconds: float):
    """
    For each stimulus frequency, plot a short time segment for all 8 channels.
    """
    print("debug 2")
    # Only keep rows with a valid stimulus
    df = df[df["testfreq_hz"] > 0].copy()
    if df.empty:
        print("No labeled rows with testfreq_hz > 0 found. Skipping time-domain plots.")
        return

    stim_freqs = sorted(df["testfreq_hz"].unique())
    n_samples_segment = int(seconds * FS)

    print(f"Time-domain: found stimulus frequencies: {stim_freqs}")

    for fstim in stim_freqs:
        g = df[df["testfreq_hz"] == fstim].copy()
        g = g.sort_values(["chunk_tick", "sample_idx"]).reset_index(drop=True)

        if len(g) < n_samples_segment:
            seg = g
            actual_seconds = len(g) / FS
            print(f"  {fstim} Hz: only {actual_seconds:.2f} s of data; plotting all of it.")
        else:
            seg = g.iloc[:n_samples_segment]
            actual_seconds = seconds

        t = np.arange(len(seg)) / FS

        fig, axes = plt.subplots(
            N_CHANNELS, 1, figsize=(10, 2 * N_CHANNELS), sharex=True
        )

        for ch_idx in range(N_CHANNELS):
            col = f"eeg{ch_idx + 1}"
            ax = axes[ch_idx]
            ax.plot(t, seg[col].values)
            ax.set_ylabel(f"Ch {ch_idx + 1}\n(µV)", rotation=0, labelpad=25)
            ax.grid(True, alpha=0.2)

        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(f"Time-domain EEG – first {actual_seconds:.2f} s @ {fstim:.0f} Hz")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        fname = out_dir / f"time_all_channels_{int(round(fstim))}Hz.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  Saved {fname}")


def plot_psd_per_freq(
    df: pd.DataFrame, out_dir: Path, channel: int, fmax: float = 40.0
):
    """
    For a chosen channel, overlay PSD curves for each stimulus frequency.
    This is where we should see clear peaks at 8/9/10/11/12 Hz, etc.
    """
    print("debug 3")
    df = df[df["testfreq_hz"] > 0].copy()
    if df.empty:
        print("No labeled rows with testfreq_hz > 0 found. Skipping PSD plots.")
        return

    stim_freqs = sorted(df["testfreq_hz"].unique())
    col = f"eeg{channel}"

    # Larger figure so the PSD uses more space
    fig, ax = plt.subplots(figsize=(10, 6))
    print(f"PSD: plotting for channel {channel}, freqs: {stim_freqs}")

    for fstim in stim_freqs:
        g = df[df["testfreq_hz"] == fstim].copy()
        g = g.sort_values(["chunk_tick", "sample_idx"])
        x = g[col].values

        if len(x) < int(FS):  # need at least ~1 second of data
            print(f"  {fstim} Hz: not enough data for PSD (len={len(x)}). Skipping.")
            continue

        freqs, psd = compute_psd(x, FS)
        ax.semilogy(
            freqs,
            psd,
            label=f"Stim freq on UI: {fstim:.0f} Hz",
        )

    ax.set_xlim(0, fmax)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (µV²/Hz)")
    ax.set_title(f"PSD – Channel {channel}")
    ax.grid(True, which="both", alpha=0.3)

    # More explicit legend title + bottom-left placement
    ax.legend(
        title="Stim frequency being displayed on UI",
        loc="lower left",
    )

    # Use more of the figure area while leaving room for title/labels
    fig.tight_layout(rect=[0.04, 0.04, 0.98, 0.96])

    fname = out_dir / f"psd_channel{channel}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")



def plot_psd_grid_all_channels(df, fs=250.0, max_freq=40.0):
    """
    Make a single figure with a 2x4 grid of PSDs,
    one subplot per EEG channel (eeg1..eeg8).
    """
    eeg_cols = [f"eeg{i}" for i in range(1, 9)]
    stim_values = sorted(df["testfreq_hz"].unique())

    fig, axes = plt.subplots(
        2, 4, figsize=(16, 8), sharex=True, sharey=True
    )

    for ch_idx, col in enumerate(eeg_cols):
        ax = axes[ch_idx // 4, ch_idx % 4]

        for stim in stim_values:
            sub = df[df["testfreq_hz"] == stim]
            x = sub[col].values
            if x.size < 2:
                continue

            f, Pxx = welch(x, fs=fs, nperseg=1024)
            mask = (f >= 0) & (f <= max_freq)
            ax.semilogy(
                f[mask],
                Pxx[mask],
                label=f"Stim freq on UI: {int(stim)} Hz",
            )

        ax.set_title(f"Channel {ch_idx + 1}")
        if ch_idx // 4 == 1:
            ax.set_xlabel("Freq (Hz)")
        if ch_idx % 4 == 0:
            ax.set_ylabel("PSD (µV²/Hz)")

    # One legend for the whole figure, *below* the subplots
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Condition",
        loc="lower center",
        ncol=len(stim_values),
        frameon=True,
    )

    fig.suptitle("PSD – All Channels", y=0.97)
    fig.tight_layout(rect=[0.02, 0.06, 0.98, 0.92])

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize simulated SSVEP calibration data."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default = r"C:\Users\fsdma\capstone\capstone\out\build\x64-release-vcpkg-vs\CapstoneProject\Release\eeg_calib_data.csv",
        help="Path to eeg_calib_data.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="plots",
        help="Directory to save output figures",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=3.0,
        help="Seconds of data to show in time-domain plots (per frequency)",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=1,
        help="Channel index (1–8) to use for PSD overlay plots",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=40.0,
        help="Max frequency (Hz) to show on PSD x-axis",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        return

    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)

    expected_cols = ["chunk_tick", "sample_idx", "testfreq_e", "testfreq_hz"]
    eeg_cols = [f"eeg{i}" for i in range(1, N_CHANNELS + 1)]
    missing = [c for c in expected_cols + eeg_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Missing expected columns in CSV: {missing}")
        return

    # Time-domain visualization: all channels, per stim frequency
    plot_time_domain(df, out_dir, seconds=args.seconds)

    # Frequency-domain visualization: one channel, overlay all stim freqs
    plot_psd_per_freq(df, out_dir, channel=args.channel, fmax=args.fmax)

    plot_psd_grid_all_channels(df, fs=250.0)

    print("Done.")


if __name__ == "__main__":
    main()