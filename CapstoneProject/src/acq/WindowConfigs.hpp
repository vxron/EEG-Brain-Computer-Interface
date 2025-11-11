#pragma once
#include <cstddef>

// central place for sizes used across headers
// unicorn sampling rate of 250 Hz means 1 scan is about 4ms (or, 32 scans per getData() call is about 128ms)
inline constexpr std::size_t RING_BUFFER_CAPACITY       = 1200;              // ring buffer holds individual scans (added in bouts of 128ms)
inline constexpr std::size_t WINDOW_SAMPLES    = RING_BUFFER_CAPACITY/3;     // 400 samples @250Hz (sampling period 4ms), this is 1.6 s
inline constexpr std::size_t WINDOW_HOP        = RING_BUFFER_CAPACITY/6;     // 50% overlap