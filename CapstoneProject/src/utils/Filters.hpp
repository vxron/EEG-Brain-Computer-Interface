#pragma once
#include <array>
#include <cstddef>
#include "Types.h"

// Generic FIR filter template with N taps
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

    inline float process(float x) {
        // Shift state (simple FIR delay line)
        for (std::size_t i = N - 1; i > 0; --i) {
            state[i] = state[i - 1];
        }
        state[0] = x;

        float y = 0.0f;
        for (std::size_t i = 0; i < N; ++i) {
            y += taps[i] * state[i];
        }
        return y;
    }
};

class EegFilterBank_C {
public:
    EegFilterBank_C();
    void process_chunk(bufferChunk_S& chunk);  // mutates in-place

private:
    using BandpassFilter = FirFilter_T<201>;  // 5–25 Hz, Blackman, 201 taps
    using SmoothFilter   = FirFilter_T<21>;   // Savitzky–Golay, 21-point

    std::array<BandpassFilter, NUM_CH_CHUNK> bandpass_; // one for each channel of data
    std::array<SmoothFilter,   NUM_CH_CHUNK> smooth_;
};
