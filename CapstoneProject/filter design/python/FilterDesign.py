"""
Design FIR bandpass filters and Savitzky–Golay smoothing filter
for SSVEP-EEG

Outputs:
 - Magnitude response plot for the FIR bandpass candidates
 - Printed FIR taps in C++-ready format
 - Recommended Savitzky–Golay parameters
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz, savgol_filter, butter, savgol_filter, savgol_coeffs

FS = 250.0  # sampling rate [Hz]
F_LO = 2.0   # bandpass low cutoff [Hz]
F_HI = 35.0  # bandpass high cutoff [Hz]

# ============ (1) 2nd ORDER BANDPASS FILTER =======================
def design_fir(numtaps: int, window: str):
    """Design linear-phase FIR bandpass."""
    taps = firwin(
        numtaps=numtaps,
        cutoff=[F_LO, F_HI],
        pass_zero=False,
        fs=FS,
        window=window,
    )
    b = taps
    a = np.array([1.0])  # FIR denominator
    return b, a


def design_iir_butter(order: int):
    """Design IIR Butterworth bandpass."""
    nyq = FS / 2.0
    Wn = [F_LO / nyq, F_HI / nyq]
    b, a = butter(order, Wn, btype="bandpass")
    return b, a


def mag_response_db(b, a, worN=4096):
    """Return (freqs, magnitude_dB)."""
    w, H = freqz(b, a, worN=worN, fs=FS)
    mag_db = 20 * np.log10(np.maximum(np.abs(H), 1e-12))
    return w, mag_db

def print_cpp_vector(var_name: str, coeffs, as_double: bool = False):
    """
    Print coefficients as a C++ static constexpr array.

    Example (float):
        static constexpr float fir_hamming_61_b[61] = {
            0.000123f, 0.000456f, ...
        };
    """
    coeffs = np.asarray(coeffs, dtype=float)
    c_type = "double" if as_double else "float"
    suffix = "" if as_double else "f"

    print(f"static constexpr {c_type} {var_name}[{len(coeffs)}] = {{")
    line = "    "
    for i, c in enumerate(coeffs):
        token = f"{c:.9g}{suffix}"
        # wrap lines to keep them readable
        if len(line) + len(token) + 2 > 100:
            print(line)
            line = "    " + token
        else:
            if i > 0:
                line += ", "
            line += token
    print(line)
    print("};\n")


# ============ (2) savitzky-golay filter ===================================
def demo_savgol_params():
    """
    Just prints a suggested Savitzky–Golay configuration and shows
    what it looks like on a noisy sinusoid (for intuition).
    """
    fs = FS
    # Example: window length ~ 80 ms (must be odd)
    # 0.08 s * 250 Hz = 20 samples -> closest odd is 21
    window_length = 21
    polyorder = 3

    print("Suggested Savitzky–Golay parameters:")
    print(f"  window_length = {window_length} samples "
          f"(~{window_length / fs:.3f} s)")
    print(f"  polyorder     = {polyorder}")

    # 1) Generate FIR coefficients (use='conv' for FIR form)
    taps = savgol_coeffs(window_length, polyorder, use="conv")

    print("\nC++: Savitzky–Golay FIR taps (window=21, poly=3):\n")
    print_cpp_vector("sg_21_3_b", taps, as_double=False)

    # 2) Demo on synthetic noisy signal
    t = np.arange(0, 2.0, 1.0 / fs)
    x = np.sin(2 * np.pi * 10 * t) + 0.5 * np.random.randn(len(t))
    y = savgol_filter(x, window_length=window_length, polyorder=polyorder)

    plt.figure(figsize=(8, 4))
    plt.plot(t, x, label="noisy 10 Hz", alpha=0.4)
    plt.plot(t, y, label="after Savitzky–Golay", linewidth=1.5)
    plt.xlim(0, 0.6)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Savitzky–Golay smoothing demo")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============== helper: group delay =======================
# compare the latency cost of diff tap count to assess real-time responsiveness
def group_delay_seconds(numtaps: int, fs: float) -> float:
    return ((numtaps - 1) / 2) / fs


# ====================== MAIN ============================================
def main():

    # Show suggested Savitzky–Golay behavior
    demo_savgol_params()

    # --------- 1) Define candidate bandpass filters ---------
    candidates = []

    # FIR candidates
    candidates.append({
        "id": "fir_hamming_101",
        "name": "FIR Hamming, 101 taps",
        "type": "FIR",
        "design": lambda: design_fir(numtaps=101, window="hamming"),
    })
    candidates.append({
        "id": "fir_hamming_201",
        "name": "FIR Hamming, 201 taps",
        "type": "FIR",
        "design": lambda: design_fir(numtaps=201, window="hamming"),
    })
    candidates.append({
        "id": "fir_blackman_201",
        "name": "FIR Blackman, 201 taps",
        "type": "FIR",
        "design": lambda: design_fir(numtaps=201, window="blackman"),
    })
    candidates.append({
        "id": "fir_hann_201",
        "name": "FIR Hann, 201 taps",
        "type": "FIR",
        "design": lambda: design_fir(numtaps=201, window="hann"),
    })

    # IIR candidate (for reference)
    candidates.append({
        "id": "iir_butter_2",
        "name": "IIR Butterworth, order 2",
        "type": "IIR",
        "design": lambda: design_iir_butter(order=2),
    })

    # Design and store responses
    for c in candidates:
        b, a = c["design"]() # FIR: a=None, IIR: b,a
        w, mag_db = mag_response_db(b, a, worN=4096)
        c["b"] = b
        c["a"] = a
        c["w"] = w
        c["mag_db"] = mag_db

    # Group (TRANSPORT) delay:
    # Each output sample corresponds to an input sample that occured this much earlier in time
    if c["type"] == "FIR":
        gd_s = group_delay_seconds(len(b), FS)
        print(f"{c['id']}: group delay ≈ {gd_s*1000:.1f} ms")

    # --------- 2) Side-by-side subplot figure ---------
    n_filters = len(candidates)
    n_rows = 2
    n_cols = 3  # 2x3 grid; last cell will be empty if n_filters < 6

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(16, 8), sharex=True, sharey=True
    )
    axes = axes.ravel()

    for idx, c in enumerate(candidates):
        ax = axes[idx]
        w = c["w"]
        mag_db = c["mag_db"]

        ax.plot(w, mag_db)
        ax.axvline(F_LO, color="k", linestyle="--", linewidth=0.8)
        ax.axvline(F_HI, color="k", linestyle="--", linewidth=0.8)
        ax.set_xlim(0, 60)
        ax.set_ylim(-100, 5)
        ax.set_title(c["name"])
        ax.grid(True, alpha=0.3)
        if idx // n_cols == n_rows - 1:
            ax.set_xlabel("Frequency (Hz)")
        if idx % n_cols == 0:
            ax.set_ylabel("Magnitude (dB)")

    for k in range(n_filters, n_rows * n_cols):
        fig.delaxes(axes[k])

    fig.suptitle("Candidate Bandpass Filters (fs = 250 Hz)", y=0.98)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])

    # --------- 3) Overlay all candidates in 0–40 Hz range ---------
    plt.figure(figsize=(8, 4))
    for c in candidates:
        w = c["w"]
        mag_db = c["mag_db"]
        mask = (w >= 0) & (w <= 40)
        plt.plot(w[mask], mag_db[mask], label=c["name"])

    # Draw key frequencies
    plt.axvline(F_LO, color="k", linestyle="--", linewidth=0.8, label="low cut")
    plt.axvline(F_HI, color="k", linestyle="--", linewidth=0.8, label="high cut")
    for f in [8, 9, 10, 11, 12]:
        plt.axvline(f, color="r", linestyle=":", linewidth=0.7)

    plt.ylim(-60, 5)
    plt.xlim(0, 40)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Passband comparison (0–40 Hz)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()

    plt.show()

    print("\n// ===== C++ filter coefficients =====\n")
    for cand in candidates:
        b = cand["b"]
        a = cand["a"]

        if cand["type"] == "FIR":
            # Only numerator taps
            print_cpp_vector(f"{cand['id']}_b", b, as_double=False)
        else:
            # IIR: numerator and denominator
            print_cpp_vector(f"{cand['id']}_b", b, as_double=False)
            print_cpp_vector(f"{cand['id']}_a", a, as_double=False)



if __name__ == "__main__":
    main()