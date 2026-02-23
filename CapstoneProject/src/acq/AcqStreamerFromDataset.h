#pragma once
#include "IAcqProvider.h" // IAcqProvider_S
#include <vector>
#include <string>
#include "../shared/StateStore.hpp"
#include <fstream>
#include <filesystem>
#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <cstdint>
#include <map>
#include <random>

namespace fs = std::filesystem;

namespace AcqStreamerFromDataset {

struct Target_S {
    int target_idx = 0;
    float freq_hz = 0;
};

struct Trial_S{
    int trial_idx = 0;
    AcqStreamerFromDataset::Target_S target;
    std::size_t start_float_idx = 0;
    std::size_t n_samples = 0;
    int block_idx = 0; // 0, 1, 2... for each trial of given target
};

// Leakage prevention
inline const std::vector<int> CALIB_BLOCKS = {0, 1, 2, 3, 4};
inline const std::vector<int> RUN_TEST_BLOCKS = {5};
}


class AcqStreamerFromDataset_C : public IAcqProvider_S {
public:
    explicit AcqStreamerFromDataset_C(StateStore_s* stateStoreRef);

    // Public Interface must implement IAcqProvider_S
    bool getData(std::size_t const numberOfScans, float* dest) override; // return last amount of stream
    bool unicorn_start_acq(bool testMode) override; // reset stream pos + collect freqs and series and sequences for this run & start streaming
    bool unicorn_demo_stop_acq(); // indep of parent class
    void setActiveStimulus(double fStimHz); // sets the active stim based on curr streaming freq
    bool unicorn_stop_and_close() override; // close files
	bool unicorn_init() override; // open dataset + meta 
	bool dump_config_and_indices() override { return true; }; // nothing to dump

	// getters
    int  getNumChannels() const override;
    void getChannelLabels(std::vector<std::string>& out) const override;
    DemoStreamerSnapshot_s getStreamerSnapshot() const;

private:    
    bool load_metadata(const std::string& json_path);
    bool open_bin(const std::string& bin_path);
    std::size_t read_floats_at(std::uint64_t start_float_index, float* dest, std::size_t n_floats);
    void schedule_reset(bool reshuffle);

    void find_closest_targets(std::vector<int>& desired_targets) const; // map any g_freq_hz to nearest available dataset target
    int map_freq_hz_to_target(double hz); // take desired Hz -> return closest target match by abs diff
    int pick_next_trial_for_target(int target_idx); // target idx -> trial idx to stream next

    StateStore_s* stateStoreRef_ = nullptr;
    int fs_hz_;
    std::string subject_id_;
    int n_channels_;
    int pre_start_, pre_len_;
    int stim_start_, stim_len_;
    std::vector<std::string> channel_labels_;
    bool initialized_ = false;
    bool started_ = false;
    bool stopped_ = true;
    std::mt19937 rng_{0xC0FFEEu};
    std::ifstream bin_;

    std::vector<AcqStreamerFromDataset::Trial_S> trials_;
    std::vector<AcqStreamerFromDataset::Target_S> target_freqs_;
    // trials_by_target_ contains the list of trial indices for a given target
    // ex: trials_by_target[17] returns list of trial indices for target 17
    std::vector<std::vector<int>> trials_by_target_;

    void build_run_schedule(const std::vector<int>& allowed_targets);
    void build_calib_schedule(trainingProto_S trainingProto);
    
    bool isCalibMode_ = false; // run vs calib mode

    // for getData streaming purposes
    int active_target_idx_ = -999;
    int active_trial_idx_ = -1;
    std::size_t start_float_offset_ = 0; // how far into SEGMENT we are

    // run mode scheduler
    std::vector<int> run_no_ssvep_per_target_order_; 
    std::vector<int> run_left_target_order_;
    std::vector<int> run_right_target_order_; // trial indices order for right target
    std::size_t run_no_ssvep_trial_idx_ = 0; // which trial in run_order_ we're currently on for no ssvep (to use nex for pre_stim segments)
    std::size_t run_target_left_trial_idx_ = 0;
    std::size_t run_target_right_trial_idx_ = 0;
    int run_target_left_idx_ = -1; // actual left target freq idx
    int run_target_right_idx_ = -1;
    std::vector<bool> run_mode_cycling_trials_; // 0: no_ssvep, 1: left, 2: right

    // calib scheduler
    std::vector<std::vector<int>> calib_per_target_deque_; // calib: trial indices order for each target (maybe just reuse trials_by_target_)
    std::vector<int> calib_idxs_; // one idx for each per target deque, should include no_ssvep (-1)
    std::vector<int> calib_no_ssvep_trials_; // separate vector for no_ssvep deck
    std::size_t calib_no_ssvep_idx_ = 0;
    std::vector<bool> cycling_trials_per_target_; // true for a target_idx if we've started having to cycle the trials
    bool cycling_trials_no_ssvep_ = false; // extra flag for no_ssvep case
};