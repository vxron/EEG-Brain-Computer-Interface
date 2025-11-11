#pragma once
#include <cstddef>

// central place for sizes used across headers
// unicorn sampling rate of 250 Hz means 1 scan is about 4ms (or, 32 scans per getData() call is about 128ms)
// for 1.2s windows -> we need 300 individual scans with hop 38 (every 0.152s)
inline constexpr std::size_t WINDOW_SAMPLES       = 300;     // 300 samples @250Hz (sampling period 4ms), this is 1.2 s
inline constexpr std::size_t WINDOW_HOP           = 38;      // every 0.152s (~87% overlap)  

// ur gonna be building these from across DISTINCT CHUNKS IN RAW BUFFER !