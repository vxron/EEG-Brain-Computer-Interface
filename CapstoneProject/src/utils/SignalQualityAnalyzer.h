#pragma once
#include <array>
#include <vector>
#include <algorithm>
#include <cmath>
#include "../acq/WindowConfigs.hpp"
#include "../shared/StateStore.hpp"

// THIS WILL SERVE FOR ARTIFACT REMOVAL (BAD SEGMENT REMOVER)
// Commonly: Eye blinking, heartbeat, jaw movement, etc

class SignalQualityAnalyzer_C {
public:
    void check_artifact_and_flag_window(sliding_window_t& window);
    void update_statestore(StateStore_s &stateStoreRef);
private:
    void update_stats_with_new_win();

    struct WindowMetrics_t{
        std::array<float, NUM_CH_CHUNK> rms_uv_;
        std::array<float, NUM_CH_CHUNK> mad_uv_;
        std::array<float, NUM_CH_CHUNK> max_abs_uv_;
        std::array<float, NUM_CH_CHUNK> max_step_uv_;
        std::array<float, NUM_CH_CHUNK> kurt_;
        std::array<float, NUM_CH_CHUNK> entropy_;
        bool isBad;
    };

    std::array<float, NUM_CH_CHUNK> bad_win_rate_; // 0..1
    float global_bad_win_rate_;
    float baseline_window_sec_ = 45.0;

    // counter for internally processed windows -> want to have ~45s worth to make stats calculations
    // take 17 windows for this purpose
    float hop_sec = static_cast<float>(WINDOW_HOP_SCANS) / 250.0f;
    size_t curr_win_acq_ = 0;
    RingBuffer_C<WindowMetrics_t> win_hist_{NEEDED_WIN_};
    size_t NEEDED_WIN_ = ceil( baseline_window_sec_ / hop_sec ); // = 45 / hop 
};
