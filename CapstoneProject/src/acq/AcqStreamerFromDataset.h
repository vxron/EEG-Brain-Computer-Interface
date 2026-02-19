#pragma once
#include "IAcqProvider.h" // IAcqProvider_S
#include <vector>
#include <string>
#include "../shared/StateStore.hpp"
#include <fstream>
#include <filesystem>
#include <iostream>
#include <cstdint>

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
};

}


class AcqStreamerFromDataset_C : IAcqProvider_S {
public:
    explicit AcqStreamerFromDataset_C(StateStore_s* stateStoreRef);

    // Public Interface must implement IAcqProvider_S
    bool getData(std::size_t const numberOfScans, float* dest) override; // return last amount of stream
    bool unicorn_start_acq(bool testMode) override; // reset stream pos + collect freqs and series and sequences for this run & start streaming
    void setActiveStimulus(double fStimHz); // sets the active stim based on curr streaming freq
    bool unicorn_stop_and_close() override; // close files
	bool unicorn_init() override; // open dataset + meta 
	bool dump_config_and_indices() override { return true; }; // nothing to dump

	// channel metadata
    int  getNumChannels() const override;
    void getChannelLabels(std::vector<std::string>& out) const override;

private:    
    bool load_metadata(const std::string& json_path);
    bool open_bin(const std::string& bin_path);
    std::size_t read_floats_at(std::uint64_t start_float_index, float* dst, std::size_t n_floats);
    void schedule_reset(bool reshuffle);

    void find_closest_targets(std::vector<int>& desired_targets) const; // map any g_freq_hz to nearest available dataset target

    StateStore_s* stateStoreRef_ = nullptr;
    
    std::ifstream bin_;
    std::vector<AcqStreamerFromDataset::Trial_S> trials_;
    std::vector<AcqStreamerFromDataset::Target_S> target_freqs_;
    // trials_by_target_ contains the list of trial indices for a given target
    // ex: trials_by_target[17] returns list of trial indices for target 17
    std::vector<std::vector<int>> trials_by_target_;
    std::vector<int> run_order_; // per run schedule (trial indices)
    std::size_t run_trial_idx_; // which trial in run_order_ we're currently on
    std::size_t trial_sample_idx_; // which sample within current trial segment

    void build_run_schedule(const std::vector<int>& allowed_targets, bool balanced);
    void build_calib_schedule(trainingProto_S trainingProto);
    
    int fs_hz_;
    std::string subject_id_;
    int n_channels_;
    int pre_start_, pre_len_;
    int stim_start_, stim_len_;
    std::vector<std::string> channel_labels_;
    bool initialized_ = false;

};