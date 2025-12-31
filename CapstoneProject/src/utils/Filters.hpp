#pragma once
#include <array>
#include <cstddef>
#include "Types.h"
#include <numeric>   // std::accumulate
#include <cmath>     // std::fabs
#include "../utils/Logger.hpp"
#include <cstddef>
#include <algorithm>

/* ALL PREPROCESSING LIVES HERE */
/* MUTATES CHUNKS IN PLACE:
1) Bandpass FIR Filter from 0.1 to 35Hz
2) DC removal
3) Artifact rejection
*/

// EEG signals are typically <100uV. Artifacts produce the huge swings, so we want to look out for those.
static constexpr float MAX_SPIKE_AMP_UV = 175; 
// Point-to-point jumps are also likely artifactual, can't be real change in EEG
static constexpr float MAX_BTWN_SAMPLE_STEP_UV = 100;

struct DcBlocker1P {
    // y[n] = x[n] - x[n-1] + a*y[n-1]
    // a close to 1 => lower cutoff (slower drift removed)
    float a = 0.995f;
    float x1 = 0.0f;
    float y1 = 0.0f;

    void reset(float x0 = 0.0f) { x1 = x0; y1 = 0.0f; }

    float process(float x) {
        float y = (x - x1) + a * y1;
        x1 = x;
        y1 = y;
        return y;
    }
};

// Linear-phase FIR filter template with N taps:
//
//   y[n] = Σ b[k] · x[n−k]
//
// Delay line mapping:
//   state[0] = x[n], state[1] = x[n−1], ...
template<std::size_t N>
struct FirFilter_T {
    std::array<float, N> taps{};   // coefficients (filter taps)
    std::array<float, N> state{};  // delay line

    FirFilter_T() = default;

    // Initialize from a C-style array of coeffs (output from python script)
    void init_from_taps(const float (&coeffs)[N]) {
        for (std::size_t i = 0; i < N; ++i) {
            taps[i]  = coeffs[i];
            state[i] = 0.0f;
        }
    }

    // process new sample x in chunk
    inline float process(float x) {
        // Shift state (simple FIR delay line)
        for (std::size_t i = N - 1; i > 0; --i) {
            state[i] = state[i - 1];
        }
        state[0] = x; // get output for state like [C,B,A(...)] where A is oldest

        float y = 0.0f;
        // 
        for (std::size_t i = 0; i < N; ++i) {
            y += taps[i] * state[i];
        }
        return y;
    }
};

class EegFilterBank_C {
public:
    EegFilterBank_C();
    // Entry point: preprocessing pipeline
    void process_chunk(bufferChunk_S& chunk);
private:
    static constexpr std::size_t BP_TAPS = 201;
    using BandpassFilter = FirFilter_T<BP_TAPS>;
    using SmoothFilter   = FirFilter_T<21>;   // Savitzky–Golay, 21-point

    // todo: read ch from statestore instead of using num_ch_chunk
    std::array<BandpassFilter, NUM_CH_CHUNK> bandpass_; // one for each channel of data
    std::array<SmoothFilter,   NUM_CH_CHUNK> smooth_;
    DcBlocker1P dc_[NUM_CH_CHUNK];

    // Preprocessing pipeline:
    void apply_bandpass(bufferChunk_S& chunk);
    void remove_common_mode_noise(bufferChunk_S& chunk);
};
