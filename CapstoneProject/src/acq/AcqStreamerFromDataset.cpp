// TODO: Stream from existing SSVEP dataset for proof-of-concept / testing / demo
// (In case Unicorn EEG Headset is simply insufficient at achieving high enough accuracy)
#include "AcqStreamerFromDataset.h"
#include "../utils/json.hpp"
#include <algorithm>
#include <cctype>
#include "WindowConfigs.hpp"
#include <cmath>
#include "../utils/Logger.hpp"

AcqStreamerFromDataset_C::AcqStreamerFromDataset_C(StateStore_s* stateStoreRef) 
: stateStoreRef_(stateStoreRef)
{}

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
// ------------------------------------ End helpers ------------------------------------------------------


// ---------------------------------------- Getters ----------------------------------------------
void AcqStreamerFromDataset_C::getChannelLabels(std::vector<std::string>& out) const {
    out = channel_labels_;
}

int AcqStreamerFromDataset_C::getNumChannels() const {
    return n_channels_;
}

DemoStreamerSnapshot_s AcqStreamerFromDataset_C::getStreamerSnapshot() const {
    DemoStreamerSnapshot_s s;
    s.active_target_idx = active_target_idx_;
    s.active_trial_idx = active_trial_idx_;
    // block_idx from curr trial
    if(active_trial_idx_ >= 0 && active_trial_idx_ < trials_.size()){
         s.block_idx = trials_[active_trial_idx_].block_idx;
    }
    // cycling if any of the run_mode_cycling flags are set (for any one of the three categories...)
    if(!run_mode_cycling_trials_.empty()){
        for(bool c: run_mode_cycling_trials_) {
            if (c==true){
                s.is_cycling = true;
                break;
            }
        }
    }
    // target_hz from target_idx
    s.active_target_hz = target_freqs_[active_target_idx_].freq_hz;
    return s;
}
// -------------------------------------- End getters ------------------------------------------------


// ------------------------------------ unicorn_init func --------------------------------------------
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
            trial.block_idx = tr.at("block_idx").get<int>();
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
    // (1) preconditions & inits
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
        isCalibMode_ = false;
        run_no_ssvep_trial_idx_ = 0;
        run_target_left_trial_idx_ = 0;
        run_target_right_trial_idx_ = 0;
        // can use g_freq_hz from stim controller
        std::vector<int> allowed_targets_for_subject = {};
        int left_hz = -1, right_hz = -1;
        {
            std::lock_guard<std::mutex> lock_sess(stateStoreRef_->saved_sessions_mutex);
            int currSessIdx = stateStoreRef_->currentSessionIdx.load(std::memory_order_acquire);
            left_hz  = stateStoreRef_->saved_sessions[currSessIdx].freq_left_hz;
            right_hz = stateStoreRef_->saved_sessions[currSessIdx].freq_right_hz;
            allowed_targets_for_subject.push_back(left_hz);
            allowed_targets_for_subject.push_back(right_hz);
        }
        run_target_left_idx_  = map_freq_hz_to_target(left_hz);
        run_target_right_idx_ = map_freq_hz_to_target(right_hz);
        build_run_schedule(allowed_targets_for_subject);
    }

    else {
        // calib mode
        isCalibMode_ = true;
        trainingProto_S trainingProto{};
        {
            std::lock_guard<std::mutex> train_proto_lock(stateStoreRef_->mtx_streaming_request);
            trainingProto = stateStoreRef_->training_proto;
        }
        build_calib_schedule(trainingProto);
    }

    start_float_offset_ = 0;
    stopped_ = false;

    return true;
}

bool AcqStreamerFromDataset_C::unicorn_demo_stop_acq(){
    stopped_ = true;
    started_ = false;
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
            outer_idx++;
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
}

int AcqStreamerFromDataset_C::map_freq_hz_to_target(double hz){
    if(hz == -1){ return -1; }
    float currBestDiff = 10000;
    float currDiff = 0;
    int bestIdx = -1;
    for(const auto& target : target_freqs_){
        currDiff = std::abs(hz - target.freq_hz);
        if(currDiff < currBestDiff){
            bestIdx = target.target_idx;
            currBestDiff = currDiff;
        }
    }
    return bestIdx;
}

void AcqStreamerFromDataset_C::build_calib_schedule(trainingProto_S trainingProto){
    // 0) reset vectors + streaming cursors + flags
    const int n_targets = (int)target_freqs_.size();
    calib_per_target_deque_.assign(n_targets, {}); // resize to num targets
    calib_idxs_.assign(calib_per_target_deque_.size(), 0);
    cycling_trials_per_target_.assign(n_targets, false);
    calib_no_ssvep_trials_.clear();
    calib_no_ssvep_idx_ = 0;
    cycling_trials_no_ssvep_ = false;

    // 1) collect desired Hz list from trainingProto
    std::vector<int> desired_freqs_for_calib;
    desired_freqs_for_calib.reserve(trainingProto.freqsToTest.size());
    for(const auto& freq_e : trainingProto.freqsToTest){
        desired_freqs_for_calib.push_back(TestFreqEnumToInt(freq_e));
    }
    find_closest_targets(desired_freqs_for_calib); // now desired_freqs_for_calib holds target_idx values

    // 2) avoid duplicates + invalid target indices in desired_freqs_for_calib
    // schedule trials to cycle for each desired freq
    // cannot reuse data in the same calib session because it would cheat model training (leakage) -> must guard against this
    std::vector<int> unique_targets;
    unique_targets.reserve(desired_freqs_for_calib.size());
    std::unordered_set<int> seen;
    for(int tg_Idx : desired_freqs_for_calib){
        if(tg_Idx < 0 || tg_Idx >= n_targets) continue;
        if(seen.insert(tg_Idx).second) unique_targets.push_back(tg_Idx); // if it gets inserted -> returns true (not a dupe)
    }

    // 3) fill per-target trial lists + shuffle
    // eliminate blocks we don't want to use in calib
    for(const int target_idx : unique_targets){
        for(int trial_idx : trials_by_target_[target_idx]){
            int block_idx = trials_[trial_idx].block_idx;
            bool isBlockCalib = (std::find(AcqStreamerFromDataset::CALIB_BLOCKS.begin(), AcqStreamerFromDataset::CALIB_BLOCKS.end(), block_idx) != AcqStreamerFromDataset::CALIB_BLOCKS.end());
            if (isBlockCalib) {
                // only push back if it's a calib trial block
                calib_per_target_deque_[target_idx].push_back(trial_idx);
            }
        }
        if(!calib_per_target_deque_[target_idx].empty()){
            std::shuffle(calib_per_target_deque_[target_idx].begin(), calib_per_target_deque_[target_idx].end(), rng_);
        }
    }

    // 4) create no_ssvep deck
    std::vector<int> all_trials;
    for(int i = 0; i < calib_per_target_deque_.size(); i++){
        for(int j = 0; j < calib_per_target_deque_[i].size(); j++){
            all_trials.push_back(calib_per_target_deque_[i][j]);
        }
    }
    calib_no_ssvep_trials_= std::move(all_trials); // all trials for all targets have the no stim segment
    std::shuffle(calib_no_ssvep_trials_.begin(), calib_no_ssvep_trials_.end(), rng_);
    calib_no_ssvep_idx_ = 0;

    // 5) pass to statestore final freqs we acc use
    {
        std::lock_guard<std::mutex> lock_stream(stateStoreRef_->mtx_streamer_freqs);
        // clear any stale entries
        stateStoreRef_->acc_freqs_in_use_by_streamer.clear();
        for(int i =0; i < unique_targets.size(); i++){
            stateStoreRef_->acc_freqs_in_use_by_streamer.push_back(target_freqs_[unique_targets[i]].freq_hz);
        }
    }
    
}

void AcqStreamerFromDataset_C::build_run_schedule(const std::vector<int>& allowed_targets){
    // 0) reset vectors + streaming cursors + flags
    run_no_ssvep_per_target_order_.clear();
    run_left_target_order_.clear();
    run_right_target_order_.clear();
    run_no_ssvep_trial_idx_ = 0; 
    run_target_left_trial_idx_ = 0;
    run_target_right_trial_idx_ = 0;
    run_mode_cycling_trials_.assign(3, false);
    const int n_targets = (int)target_freqs_.size();

    std::vector<int> allowed_target_idx;
    // 1) convert allowed_targets to target indices
    for(const int tg_hz : allowed_targets){
        int bestIdx = map_freq_hz_to_target(tg_hz);
        if(bestIdx >= 0) allowed_target_idx.push_back(bestIdx);
    }
    if (allowed_target_idx.empty()) {
        std::cerr << "[AcqStreamerFromDataset] build_run_schedule: no allowed targets\n";
        return;
    }

    // 2) avoid duplicates + invalid target indices in allowed_target_idx
    std::vector<int> unique_targets;
    unique_targets.reserve(allowed_target_idx.size());
    std::unordered_set<int> seen;
    for(int tg_Idx : allowed_target_idx){
        if(tg_Idx < 0 || tg_Idx >= n_targets) continue;
        if(seen.insert(tg_Idx).second) unique_targets.push_back(tg_Idx); // if it gets inserted -> returns true (not a dupe)
        else { 
            LOG_ALWAYS("[AcqStreamerFromDataset] Error when building the run schedule: no unique targets available in the dataset for the desired frequency-pair \n");
        }
    }

    // 3) fill per-target trial lists + shuffle
    // eliminate blocks we don't want to use in run
    for(const int target_idx : unique_targets){
        for(int trial_idx : trials_by_target_[target_idx]){
            int block_idx = trials_[trial_idx].block_idx;
            bool isBlockRun = (std::find(AcqStreamerFromDataset::RUN_TEST_BLOCKS.begin(), AcqStreamerFromDataset::RUN_TEST_BLOCKS.end(), block_idx) != AcqStreamerFromDataset::RUN_TEST_BLOCKS.end());
            if (isBlockRun) {
                if(target_idx == run_target_left_idx_){
                    run_left_target_order_.push_back(trial_idx);
                } else if (target_idx == run_target_right_idx_) {
                    run_right_target_order_.push_back(trial_idx);
                } else {
                    LOG_ALWAYS("[AcqStreamerFromDataset] Unrecognized target_idx found in build_run_schedule");
                }
            }
        }
    }
    // Shuffle left/right once
    if(!run_left_target_order_.empty()){
        std::shuffle(run_left_target_order_.begin(), run_left_target_order_.end(), rng_);
    }
    if(!run_right_target_order_.empty()){
        std::shuffle(run_right_target_order_.begin(), run_right_target_order_.end(), rng_);
    }

    // 4) build no_ssvep deck as union of all L/R trials
    run_no_ssvep_per_target_order_.insert(run_no_ssvep_per_target_order_.end(), run_left_target_order_.begin(), run_left_target_order_.end());
    run_no_ssvep_per_target_order_.insert(run_no_ssvep_per_target_order_.end(), run_right_target_order_.begin(), run_right_target_order_.end());
    if(!run_no_ssvep_per_target_order_.empty()){
        std::shuffle(run_no_ssvep_per_target_order_.begin(), run_no_ssvep_per_target_order_.end(), rng_);
    }

    // 5) pass to statestore final freqs we acc use
    {
        std::lock_guard<std::mutex> lock_stream(stateStoreRef_->mtx_streamer_freqs);
        // clear any stale entries
        stateStoreRef_->acc_freqs_in_use_by_streamer.clear();
        for(int i =0; i < unique_targets.size(); i++){
            stateStoreRef_->acc_freqs_in_use_by_streamer.push_back(target_freqs_[unique_targets[i]].freq_hz);
        }
    }
    
}

int AcqStreamerFromDataset_C::pick_next_trial_for_target(int target_idx){
    if(isCalibMode_){

        if((target_idx < 0 && target_idx != -1)||target_idx >= (int)calib_per_target_deque_.size()) {
            return -1; 
        }

        // no ssvep case
        if(target_idx == -1){
            if(calib_no_ssvep_trials_.empty()) return -1;
            const int next = calib_no_ssvep_trials_[(std::size_t)calib_no_ssvep_idx_];
            calib_no_ssvep_idx_++;
            if(calib_no_ssvep_idx_ >= calib_no_ssvep_trials_.size()) {
                // cycle (exhausted for no_ssvep)
                cycling_trials_no_ssvep_ = true;
                calib_no_ssvep_idx_ = 0;
            }
            return next;
        }

        if (calib_per_target_deque_[target_idx].empty()) return -1;
        int nextTrial = calib_per_target_deque_[target_idx][calib_idxs_[target_idx]];
        calib_idxs_[target_idx]++;
        if(calib_idxs_[target_idx] >= calib_per_target_deque_[target_idx].size()){
            // all data exhausted for this target
            // cycle (wraparound)
            calib_idxs_[target_idx] = 0;
            cycling_trials_per_target_[target_idx] = true;
        }

        return nextTrial;
    }

    else {
        // for run mode, target_idx must either be -1 (no_ssvep) or the target freq idx
        if(target_idx == -1) {
            // no ssvep
            if(run_no_ssvep_trial_idx_ >= run_no_ssvep_per_target_order_.size()){
                // exhausted no ssvep
                run_mode_cycling_trials_[0] = true;
                run_no_ssvep_trial_idx_ = 0;
            }
            int nextTrial = run_no_ssvep_per_target_order_[run_no_ssvep_trial_idx_];
            run_no_ssvep_trial_idx_++;
            return nextTrial;
        } else if (target_idx ==  run_target_left_idx_){
            if(run_target_left_trial_idx_ >= run_left_target_order_.size()){
                // exhausted left
                run_mode_cycling_trials_[1] = true;
                run_target_left_trial_idx_ = 0;
            }
            int nextTrial = run_left_target_order_[run_target_left_trial_idx_];
            run_target_left_trial_idx_++;
            return nextTrial;
        } else if (target_idx == run_target_right_idx_) {
            if(run_target_right_trial_idx_ >= run_right_target_order_.size()){
                // exhausted right
                run_mode_cycling_trials_[2] = true;
                run_target_right_trial_idx_ = 0;
            }
            int nextTrial = run_right_target_order_[run_target_right_trial_idx_];
            run_target_right_trial_idx_++;
            return nextTrial;
        } else {
            LOG_ALWAYS("[AcqStreamerFromDataset] Incorrect target_idx passed to pick_next_trial_for_target in run mode");
            return -1;
        }
    }
}

bool AcqStreamerFromDataset_C::unicorn_stop_and_close() {
    stopped_ = true;
    started_ = false;
    if (bin_.is_open()) {
        bin_.close();
    }
    return true;
}
// ----------------------------------- End everything for unicorn_start_acq function ----------------------------------

// ----------------------------------- getData function ----------------------------------
bool AcqStreamerFromDataset_C::getData(std::size_t const numberOfScans, float* dest) {
    if (!initialized_ || !bin_.is_open() || dest == nullptr || numberOfScans == 0) return false;
    if(n_channels_ != NUM_CH_CHUNK) return false;
    // 1 scan = 8 measurements (all channels at 1 time point)
    // check which mode we started in (calib or run)
    int desired_floats_per_getData = numberOfScans * n_channels_; // 8 floats per scan
    std::size_t out_floats_read = 0;
    int guard_against_huge_loop = 100;
    int guard = 0;

    // Helpers to segment parameters based on "target" (-1 => pre_stim, else stim_on)
    auto seg_start_samples = [&](int target_idx) -> std::size_t {
        return (target_idx == -1) ? (std::size_t)pre_start_ : (std::size_t)stim_start_;
    };
    auto seg_len_samples = [&](int target_idx) -> std::size_t {
        return (target_idx == -1) ? (std::size_t)pre_len_ : (std::size_t)stim_len_;
    };

    // which target idx are we on
    // check g_freq_hz from statestore
    int desired_hz = stateStoreRef_->g_freq_hz.load(std::memory_order_acquire);
    int target_idx = map_freq_hz_to_target(desired_hz);

    // if target freq changed since last call, reset segment offset and force switch target freq trial vectors
    if(target_idx != active_target_idx_) {
        active_trial_idx_ = -1;
        active_target_idx_ = target_idx;
        start_float_offset_ = 0;
    }

    while(out_floats_read < desired_floats_per_getData){
        if(guard > guard_against_huge_loop){
            LOG_ALWAYS("[AcqStreamerFromDataset] Unknown crash in GetData loop");
            break;
        }
        guard++;

        // ensure we have a trial to read from in active idx, or pick new one
        if(active_trial_idx_ < 0){
            active_trial_idx_ = pick_next_trial_for_target(target_idx);
            start_float_offset_ = 0;
            if (active_trial_idx_ < 0) {
                LOG_ALWAYS("[AcqStreamerFromDataset] no trial available for target_idx=" << target_idx);
                return false;
            }
        }

        const AcqStreamerFromDataset::Trial_S& tr = trials_[(std::size_t)active_trial_idx_];
        
        // Compute segment base and bounds (in floats) for this trial
        const std::size_t seg_base_float = tr.start_float_idx + seg_start_samples(target_idx)*(std::size_t)n_channels_; // samples = timepoints 
        const std::size_t seg_len_float = seg_len_samples(target_idx)*(std::size_t)n_channels_;

        if(start_float_offset_ >= seg_len_float){
            // segment exhausted -> move to next trial
            active_trial_idx_ = -1;
            start_float_offset_ = 0;
            continue;
        }

        std::size_t floats_needed = desired_floats_per_getData - out_floats_read;
        std::size_t seg_left = seg_len_float - start_float_offset_;
        std::size_t floats_to_read = std::min(seg_left, floats_needed); // take all that we have left to try and reach needed

        std::size_t floats_read = read_floats_at(
            (seg_base_float + start_float_offset_), 
            (dest + out_floats_read), 
            (floats_to_read));

        if (floats_read == 0) {
            LOG_ALWAYS("[AcqStreamerFromDataset] read_floats_at returned 0");
            return false;
        }
       
        out_floats_read += floats_read;
        start_float_offset_ += floats_read;

        // if we finished the segment, force next iter to choose new trial
        if(start_float_offset_ >= seg_len_float){
            active_trial_idx_ = -1; 
            start_float_offset_ = 0;
        }
    }

    if (out_floats_read != (std::size_t)desired_floats_per_getData) {
        LOG_ALWAYS("[AcqStreamerFromDataset] getData short read: got=" << out_floats_read
            << " need=" << desired_floats_per_getData);
        return false;
    }
    return true;
}

std::size_t AcqStreamerFromDataset_C::read_floats_at(std::uint64_t start_float_index, float* dest, std::size_t n_floats){
    if(!bin_.is_open() || dest == nullptr || n_floats == 0){
        return 0;
    }
    bin_.clear();
    
    // convert float index -> byte offset
    const std::uint64_t byte_off = start_float_index*sizeof(float);

    // seek to the right spot
    bin_.seekg((std::streamoff)byte_off, std::ios::beg); // moves in bytes
    if (!bin_) return 0;

    // read raw bytes into dest
    const std::size_t bytes_to_read = n_floats * sizeof(float);
    bin_.read(reinterpret_cast<char*>(dest), (std::streamsize)bytes_to_read); // .read expects char* ptr & reaches EOF gracefully

    // return number of floats read (will return less than desired n_flaots if we reached EOF)
    const std::streamsize bytes_read = bin_.gcount(); // rtn type is std::streamsize (signed version of std::size_t)
    if (bytes_read <= 0) { return 0 ;}
    std::size_t floats_read = bytes_read/(std::streamsize)sizeof(float);

    return floats_read;
}
// ----------------------------------- End everything for getData function ----------------------------------