#pragma once
#include <array>
#include <vector>
#include <algorithm>
#include <cmath>
#include "../acq/WindowConfigs.hpp"
#include "../shared/StateStore.hpp"
#include <numeric>

// THIS WILL SERVE FOR 
// (1) ARTIFACT REMOVAL (BAD SEGMENT REMOVER)
// Commonly: Eye blinking, heartbeat, jaw movement, etc
// (2) to publish general statistics to state store for UI publishing
// (3) to save to/pull from state store saved sessions to ensure mean value is sufficiently similar to calib session; otherwise add offset

    // (a) Max absolute amplitude = 200uV for >= 2 samples on any channel
    
    // (b) Max point-to-point step = 100uV on any channel

    // (c) Excess kurtosis (eye blinks)

    // (d) Entropy (eye blinks)


// thresholds
static constexpr float MAX_ABS_UV = 200.0f;     // amplitude threshold
static constexpr float MAX_STEP_UV = 100.0f;    // point-to-point threshold
static constexpr int   AMP_PERSIST_SAMPLES = 2; // require >=2 samples over MAX_ABS_UV
static constexpr int   STEP_PERSIST_SAMPLES = 2; // require >=1 big step
static constexpr int   UI_UPDATE_EVERY_WIN = 10; // for ~3s updates

// SHOULD BE A SINGLETON
class SignalQualityAnalyzer_C {
public:
    explicit SignalQualityAnalyzer_C(StateStore_s* stateStoreRef);
    void check_artifact_and_flag_window(sliding_window_t& window);
    void update_statestore();
private:
    void update_stats_with_new_win();

    StateStore_s* stateStoreRef_{nullptr};

    std::vector<float> win_snapshot_;     // reused buffer (snapshot of RB contents)
    size_t global_win_acq_ = 0; // total windows passed through analyzer

    // averages we keep track of with each new window for eventual statestore publishing
    Stats_s RollingSums_{}; // contains sum for each metric for all currently appended windows! (when we publish to state store we avg over all windows acq in the last 45s period)
    Stats_s evicted_ {}; // keep last evicted from ring buffer for ref
    std::vector<Stats_s> tempWinStats_; // reused for recompute max

    std::array<int, NUM_CH_CHUNK> isGreaterThanMaxUvCount_{};
    std::array<int, NUM_CH_CHUNK> surpassesMaxStepCount_{};

    // set in constructor
    float baseline_window_sec_ = 45.0;
    float hop_sec;
    size_t NEEDED_WIN_; // = 45 / hop 
    RingBuffer_C<Stats_s> RollingWinStatsBuf;

    // helpers for temp storage
    size_t ui_tick_ = 0;
	size_t current_bad_win_num_ = 0;
	size_t overall_bad_win_num_ = 0;
    size_t num_win_in_rolling_ = 0;
};