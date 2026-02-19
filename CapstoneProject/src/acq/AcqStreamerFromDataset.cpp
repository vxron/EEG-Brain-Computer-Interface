// TODO: Stream from existing SSVEP dataset for proof-of-concept / testing / demo
// (In case Unicorn EEG Headset is simply insufficient at achieving high enough accuracy)
#include "AcqStreamerFromDataset.h"
#include "../utils/json.hpp"
#include <algorithm>
#include <cctype>
#include "WindowConfigs.hpp"
#include <cmath>

// --------------------------------- Helpers for json parsing ---------------------------------------
static std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
        [](unsigned char c){ return (char)std::tolower(c); });
    return s;
}

// Canonicalize channel labels so "FZ" / "Fz" / "fz" all compare equal
static std::string canonLabel(std::string s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        if (c == '.' || c == '_' || c == ' ' || c == '\t') continue;
        out.push_back((char)std::tolower(c));
    }
    return out;
}
// ------------------------------ End helpers -------------------------------------------

// ------------------------------------- unicorn_init func ------------------------------------

bool AcqStreamerFromDataset_C::unicorn_init(){
    // Load/parse JSON meta
    if(initialized_ == true){
        // already init
        return true;
    }

    // 1) resolve dataset directory relative to source file
    fs::path datasetDir;
    try {
        datasetDir = fs::path(__FILE__).parent_path() / "Tsinghua-SSVEP-Dataset";
    } catch (...) {
        std::cerr << "[AcqStreamerFromDataset] unicorn_init: failed to build datasetDir from __FILE__\n";
        return false; // TODO: more graceful handling (e.g. switch to regular fake acq)
    }
    if (!fs::exists(datasetDir) || !fs::is_directory(datasetDir)) {
        std::cerr << "[AcqStreamerFromDataset] unicorn_init: datasetDir not found: "
                  << datasetDir.string() << "\n";
        return false;
    }

    // 2) choose subject_id & resolve offline python export
    std::string sid = "S1"; // can change it later
    fs::path jsonPath = datasetDir / (sid + "_unicorn8_trials.json");
    fs::path binPath  = datasetDir / (sid + "_unicorn8_trials.bin");
    if (!fs::exists(jsonPath)) {
        std::cerr << "[AcqStreamerFromDataset] unicorn_init: missing JSON: "
                  << jsonPath.string() << "\n";
        return false;
    }
    if (!fs::exists(binPath)) {
        std::cerr << "[AcqStreamerFromDataset] unicorn_init: missing BIN: "
                  << binPath.string() << "\n";
        return false;
    }

    // 3) read meta & load into class members
    if (!load_metadata(jsonPath.string())) {
        std::cerr << "[AcqStreamerFromDataset] unicorn_init: load_metadata failed\n";
        return false;
    }

    // 4) open binary stream
    if (!open_bin(binPath.string())) {
        std::cerr << "[AcqStreamerFromDataset] unicorn_init: open_bin failed\n";
        return false;
    }

    // 5) validation
    if (n_channels_ != NUM_CH_CHUNK) {
        std::cerr << "[AcqStreamerFromDataset] invalid n_channels_=" << n_channels_ << "\n";
        return false;
    }
    if ((int)channel_labels_.size() != n_channels_) {
        std::cerr << "[AcqStreamerFromDataset] channel label count mismatch: labels="
                  << channel_labels_.size() << " n_channels_=" << n_channels_ << "\n";
        return false;
    }
    if (fs_hz_ <= 0) {
        std::cerr << "[AcqStreamerFromDataset] invalid fs_hz_=" << fs_hz_ << "\n";
        return false;
    }
    if (target_freqs_.empty()) {
        std::cerr << "[AcqStreamerFromDataset] target_freqs_ is empty\n";
        return false;
    }
    if (trials_.empty()) {
        std::cerr << "[AcqStreamerFromDataset] trials_ is empty\n";
        return false;
    }
    if (trials_by_target_.size() != target_freqs_.size()) {
        std::cerr << "[AcqStreamerFromDataset] trials_by_target size mismatch: "
                  << trials_by_target_.size() << " vs targets " << target_freqs_.size() << "\n";
        return false;
    }
    
    // Segment sanity: stim segment must fit inside trial
    const std::size_t trial_len = trials_[0].n_samples; 
    if (trial_len == 0) {
        std::cerr << "[AcqStreamerFromDataset] trial_len=0\n";
        return false;
    }
    if (pre_start_ < 0 || pre_len_ <= 0 || stim_start_ < 0 || stim_len_ <= 0) {
        std::cerr << "[AcqStreamerFromDataset] invalid segments\n";
        return false;
    }
    if ((std::size_t)pre_start_ + (std::size_t)pre_len_ > trial_len ||
        (std::size_t)stim_start_ + (std::size_t)stim_len_ > trial_len) {
        std::cerr << "[AcqStreamerFromDataset] segments exceed trial_len=" << trial_len
                  << " pre=(" << pre_start_ << "," << pre_len_ << ")"
                  << " stim=(" << stim_start_ << "," << stim_len_ << ")\n";
        return false;
    }

    // Verify BIN size matches what JSON implies
    // expected floats in bin = sum(trial.n_samples * n_channels)
    std::uint64_t expectedFloats = 0;
    for(const auto& tr : trials_){
        expectedFloats += (std::uint64_t)tr.n_samples * (std::uint64_t)n_channels_;
    }
    const std::uint64_t expectedBytes = expectedFloats*sizeof(float);
    std::uint64_t actualBytes = 0;
    try {
        actualBytes = (std::uint64_t)fs::file_size(binPath);
    } catch (...) {
        std::cerr << "[AcqStreamerFromDataset] could not stat BIN size\n";
        return false;
    }
    if(actualBytes != expectedBytes){
        // this could happen if JSON/BIN don't match, e.g. wrong subject, stale files, etc
         std::cerr << "[AcqStreamerFromDataset] BIN size mismatch.\n"
                  << "  expectedBytes=" << expectedBytes << "\n"
                  << "  actualBytes  =" << actualBytes << "\n";
        return false;
    }

    // 6) re-initialize runtime vars
    run_order_.clear();
    run_trial_idx_ = 0;
    trial_sample_idx_ = 0;
    initialized_ = true;
    return true;
}

bool AcqStreamerFromDataset_C::load_metadata(const std::string& json_path){
    std::ifstream f(json_path);
    if (!f.is_open()) {
        std::cerr << "[AcqStreamerFromDataset] failed to open json: " << json_path << "\n";
        return false;
    }
    
    nlohmann::json j;
    try {
        f >> j;
    } catch (const std::exception& e) {
        std::cerr << "[AcqStreamerFromDataset] json parse error: " << e.what() << "\n";
        return false;
    }

    try {
        fs_hz_ = j.at("fs_hz").get<int>();
        subject_id_ = j.at("subject_id").get<std::string>();
        n_channels_ = j.at("n_channels").get<int>();
        channel_labels_.clear();
        channel_labels_ = j.at("channel_labels").get<std::vector<std::string>>();

        // segments
        // pre segment = no ssvep (no target)
        pre_start_ = j.at("segments").at("pre_stim").at("start").get<int>(); 
        pre_len_ = j.at("segments").at("pre_stim").at("len").get<int>();
        // stim segment = target
        stim_start_ = j.at("segments").at("stim_on").at("start").get<int>();
        stim_len_   = j.at("segments").at("stim_on").at("len").get<int>();

        // targets
        const auto& jt = j.at("targets"); // & ensures we don't copy, just refer to exisiting JSON obj inside j
        int n_targets = (int)jt.size();
        target_freqs_.assign(n_targets, AcqStreamerFromDataset::Target_S{}); // wipe old and assign new targets
        for(const auto& t : jt){
            AcqStreamerFromDataset::Target_S target;
            target.target_idx = t.at("target_idx").get<int>();
            target.freq_hz = t.at("freq_hz").get<float>();
            if(target.target_idx >= n_targets || target.target_idx < 0){
                std::cerr << "[AcqStreamerFromDataset] bad target_idx " << target.target_idx << "\n";
                return false;
            }
            target_freqs_[target.target_idx] = target; // will be the right numner eventually since it's alr sized to target len
        }

        // trials
        const auto& jtr = j.at("trials");
        int n_trials = (int)jtr.size();
        trials_.assign(n_trials, AcqStreamerFromDataset::Trial_S{});
        trials_by_target_.assign(n_targets, {}); // clear + resize
        for (const auto& tr : jtr){
            AcqStreamerFromDataset::Trial_S trial;
            trial.n_samples = tr.at("n_samples").get<std::size_t>();
            const std::size_t target_idx = tr.at("target_idx").get<int>();
            trial.target = target_freqs_[target_idx];
            trial.trial_idx = tr.at("trial_idx").get<int>();
            trial.start_float_idx = tr.at("start_float_index").get<std::size_t>();
            if(trial.trial_idx >= n_trials || trial.trial_idx < 0){
                std::cerr << "[AcqStreamerFromDataset] bad trial_idx " << trial.trial_idx << "\n";
                return false;
            }
            trials_[trial.trial_idx] = trial;
            // add trial to correct target idx vector
            trials_by_target_[target_idx].push_back(trial.trial_idx);
        }

    } catch (const std::exception& e) {
        std::cerr << "[AcqStreamerFromDataset] metadata missing/invalid fields: " << e.what() << "\n";
        return false;
    }

    return true;
}

// Open binary file containing all dataset floats in raw format 
bool AcqStreamerFromDataset_C::open_bin(const std::string& bin_path){
    if(bin_.is_open()) bin_.close();
    bin_.open(bin_path, std::ios::binary);
    if(!bin_.is_open()){
        std::cerr << "[AcqStreamerFromDataset] failed to open binary: " << bin_path << "\n";
        return false;
    }
    bin_.exceptions(std::ifstream::badbit);
    return true;
}

// ----------------------------------- End everything for unicorn_init func ----------------------------------


// ----------------------------------- unicorn_start_acq function ------------------------------------------------
// Goal of this section: Prepare a run so that subsquent getData() calls can stream samples continuously with correct labels if needed.
bool AcqStreamerFromDataset_C::unicorn_start_acq(bool testMode){
    // (1) preconditions
    if (!initialized_) return false;
    if (!bin_.is_open()) return false;
    if (trials_.empty() || target_freqs_.empty()) return false;
    
    // (2) treat the bool testMode in this case as calib vs run
    // calib = testMode 1
    // run = testMode 0
    // ensure they match what state store expects
    UIState_E currState = stateStoreRef_->g_ui_state.load(std::memory_order_acquire);
    if(testMode == 1 && !(currState == UIState_Active_Calib || currState == UIState_Instructions || currState == UIState_NoSSVEP_Test)){
        // mismatch
        std::cerr << "[AcqStreamerFromDataset] mismatch at unicorn_start_acq time in expected calib mode";
        return false;
    }
    if(testMode == 0 && !(currState == UIState_Active_Run)){
        // mismatch
        std::cerr << "[AcqStreamerFromDataset] mismatch at unicorn_start_acq time in expected run mode";
        return false;
    }

    // (3) subsequent path dep on calib vs run mode
    if(testMode == 0){
        // run mode
        // can use g_freq_hz from stim controller OR make own scheduler here
        // should probably make own scheduler because we have less no_ssvep
        std::vector<int> allowed_targets_for_subject = {-1};
        {
            std::lock_guard<std::mutex> lock_sess(stateStoreRef_->saved_sessions_mutex);
            int currSessIdx = stateStoreRef_->currentSessionIdx.load(std::memory_order_acquire);
            allowed_targets_for_subject.push_back(stateStoreRef_->saved_sessions[currSessIdx].freq_left_hz);
            allowed_targets_for_subject.push_back(stateStoreRef_->saved_sessions[currSessIdx].freq_right_hz);
        }
        // 
        build_run_schedule(allowed_targets_for_subject, true);
    }
    else {
        // calib mode
        trainingProto_S trainingProto{};
        {
            std::lock_guard<std::mutex> train_proto_lock(stateStoreRef_->mtx_streaming_request);
            trainingProto = stateStoreRef_->training_proto;
        }
        build_calib_schedule(trainingProto);
    }

    return true;
}

void AcqStreamerFromDataset_C::find_closest_targets(std::vector<int>& desired_targets) const {
    float currBestDiff = 10000;
    float currDiff = 0;
    int closestIdx = -1;
    int outer_idx = 0;
    for(const auto& freq : desired_targets) {
        // guard against -1 targets
        if(freq <= 0){
            continue;
        }

        for(const auto& target : target_freqs_){
            currDiff = std::abs(target.freq_hz - freq);
            if(currDiff < currBestDiff) {
                closestIdx = target.target_idx;
                currBestDiff = currDiff;
            }
        }
        // mutate vec passed in place -> pass target idx back instead of raw freq
        desired_targets[outer_idx] = closestIdx;
        outer_idx++;
        // reset for next desired target
        currBestDiff = 10000;
        closestIdx = -1;
    }

    // pass to statestore final freqs we acc use
    {
        std::lock_guard<std::mutex> lock_stream(stateStoreRef_->mtx_streamer_freqs);
        // clear any stale entries
        stateStoreRef_->acc_freqs_in_use_by_streamer.clear();
        for(int i =0; i < desired_targets.size(); i++){
            stateStoreRef_->acc_freqs_in_use_by_streamer.push_back(target_freqs_[desired_targets[i]].freq_hz);
        }
    }
}

void AcqStreamerFromDataset_C::build_calib_schedule(trainingProto_S trainingProto){
    // we will rely on g_freq_hz in real time however we will set up order here 
    // and only use g_freq_hz to determine transitions
    // since the freqs may not match exactly 
    std::vector<int> desired_freqs_for_calib;
    for(const auto& freq_e : trainingProto.freqsToTest){
        desired_freqs_for_calib.push_back(TestFreqEnumToInt(freq_e));
    }
    find_closest_targets(desired_freqs_for_calib);
    // schedule trials to cycle for each desired freq
    // cannot reuse data in the same calib session because it would cheat model training (leakage) -> must guard against this

}

// ----------------------------------- End everything for unicorn_start_acq function ----------------------------------


/*
bool getData(std::size_t const numberOfScans, float* dest) override; // return last amount of stream





bool unicorn_start_acq(bool testMode) override; // reset stream pos + collect freqs and series and sequences for this run & start streaming
    void setActiveStimulus(double fStimHz); // sets the active stim based on curr streaming freq
    bool unicorn_stop_and_close() override; // close files

	bool dump_config_and_indices() override { return true; }; // nothing to dump

	// channel metadata
    int  getNumChannels() const override;
    void getChannelLabels(std::vector<std::string>& out) const override;

private:    
    bool load_metadata(const std::string& json_path);
    bool open_bin(const std::string& bin_path);
    std::size_t read_floats_at(std::uint64_t start_float_index, float* dst, std::size_t n_floats);
    void schedule_reset(bool reshuffle);
    */