import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch

# ===================== CONFIG =====================
CSV_PATH = "eeg_windows_2.csv"   # change if needed
FS = 250.0                     # sampling rate
EEG_CHANNELS = [f"eeg{i}" for i in range(1, 9)]
TARGET_FREQS = [10, 12]        # <-- put your SSVEP freqs here
# =================================================


def load_and_group(csv_path):
    df = pd.read_csv(csv_path)

    # Keep only trimmed + labeled + non-bad windows
    df = df[(df["is_trimmed"] == 1) & (df["is_bad"] == 0)]
    df = df[df["testfreq_hz"] > 0]

    grouped = dict(tuple(df.groupby(["window_idx", "testfreq_hz"])))
    print(f"Loaded {len(grouped)} usable windows")
    return grouped


def window_to_matrix(df):
    """Return shape (n_ch, n_samples)"""
    X = df[EEG_CHANNELS].values.T
    return X


# =================================================
# 1) TIME-DOMAIN EXPLORATION
# =================================================
def plot_time_domain(grouped):
    for (win_idx, freq), df in grouped.items():
        X = window_to_matrix(df)

        plt.figure(figsize=(10, 4))
        for ch in range(X.shape[0]):
            plt.plot(X[ch], alpha=0.5, label=f"Ch{ch+1}")
        plt.title(f"Window {win_idx} – Time Domain – {freq} Hz")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude (µV)")
        plt.legend(ncol=4, fontsize=8)
        plt.tight_layout()
        plt.show()

        break  # show one example only


# =================================================
# 2) PSD (WELCH)
# =================================================
def compute_psd(X):
    psds = []
    for ch in range(X.shape[0]):
        f, Pxx = welch(X[ch], fs=FS, nperseg=256)
        psds.append(Pxx)
    return f, np.mean(psds, axis=0)


def plot_psd_by_class(grouped):
    psd_by_freq = {}

    for (win_idx, freq), df in grouped.items():
        X = window_to_matrix(df)
        f, psd = compute_psd(X)

        if freq not in psd_by_freq:
            psd_by_freq[freq] = []
        psd_by_freq[freq].append(psd)

    plt.figure(figsize=(10, 5))
    for freq, psds in psd_by_freq.items():
        mean_psd = np.mean(psds, axis=0)
        plt.semilogy(f, mean_psd, label=f"{freq} Hz")

    for tf in TARGET_FREQS:
        plt.axvline(tf, color="k", linestyle="--", alpha=0.3)

    plt.title("Mean PSD per SSVEP Class")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.xlim(0, 60)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =================================================
# 3) POWER AT TARGET FREQUENCIES
# =================================================
def extract_band_power(f, psd, f0, bw=0.5):
    mask = (f >= f0 - bw) & (f <= f0 + bw)
    return np.trapz(psd[mask], f[mask])


def plot_ssvep_power(grouped):
    rows = []

    for (win_idx, freq), df in grouped.items():
        X = window_to_matrix(df)
        f, psd = compute_psd(X)

        for tf in TARGET_FREQS:
            power = extract_band_power(f, psd, tf)
            rows.append({
                "true_freq": freq,
                "target_freq": tf,
                "power": power
            })

    power_df = pd.DataFrame(rows)

    plt.figure(figsize=(8, 5))
    for tf in TARGET_FREQS:
        subset = power_df[power_df["target_freq"] == tf]
        plt.scatter(subset["true_freq"], subset["power"], label=f"@{tf} Hz")

    plt.xlabel("True Stimulus Frequency (Hz)")
    plt.ylabel("Band Power")
    plt.title("SSVEP Power at Target Frequencies")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =================================================
# 4) SIMPLE FEATURE SEPARABILITY CHECK
# =================================================
def plot_feature_scatter(grouped):
    feats = []

    for (win_idx, freq), df in grouped.items():
        X = window_to_matrix(df)
        f, psd = compute_psd(X)

        p10 = extract_band_power(f, psd, 10)
        p12 = extract_band_power(f, psd, 12)

        feats.append([p10, p12, freq])

    feats = np.array(feats)

    plt.figure(figsize=(6, 6))
    for freq in np.unique(feats[:, 2]):
        mask = feats[:, 2] == freq
        plt.scatter(
            feats[mask, 0],
            feats[mask, 1],
            label=f"{int(freq)} Hz"
        )

    plt.xlabel("Power @ 10 Hz")
    plt.ylabel("Power @ 12 Hz")
    plt.title("SSVEP Feature Space (Exploratory)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =================================================
# MAIN
# =================================================
if __name__ == "__main__":
    grouped = load_and_group(CSV_PATH)

    plot_time_domain(grouped)
    plot_psd_by_class(grouped)
    plot_ssvep_power(grouped)
    plot_feature_scatter(grouped)
