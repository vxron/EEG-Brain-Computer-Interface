#include "SignalQualityAnalyzer.h"

// ========================= INLINE STATS HELPERS =====================
static inline float safe_sqrt(float x) { return std::sqrt(std::max(0.0f, x)); }

static float median_inplace(std::vector<float>& v) {
    if (v.empty()) return 0.0f;
    size_t n = v.size();
    size_t mid = n / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    float med = v[mid];
    if ((n % 2) == 0) {
        std::nth_element(v.begin(), v.begin() + (mid - 1), v.end());
        med = 0.5f * (med + v[mid - 1]);
    }
    return med;
}

// Histogram entropy (time-domain). 
// TODO: replace with spectral entropy later (when we compute ftrs anyways)
static float hist_entropy_channel(const std::vector<float>& snap, size_t ch,
                                 int bins = 64, float minv = -200.0f, float maxv = 200.0f) {
    if (!(maxv > minv) || bins <= 1) return 0.0f;
    std::vector<int> h((size_t)bins, 0);
    float inv = 1.0f / (maxv - minv);
    for (size_t s = 0; s < WINDOW_SCANS; ++s) {
        float v = snap[s * NUM_CH_CHUNK + ch];
        float t = (v - minv) * inv;
        int b = (int)(t * bins);
        b = std::max(0, std::min(b, bins - 1));
        h[(size_t)b]++;
    }
    float H = 0.0f;
    float n = (float)WINDOW_SCANS;
    for (int c : h) {
        if (c == 0) continue;
        float p = (float)c / n;
        H += -p * std::log(p);
    }
    return H;
}

// Excess kurtosis using mean and m2/m4
static float excess_kurtosis_channel(const std::vector<float>& snap, size_t ch, float mean) {
    double m2 = 0.0, m4 = 0.0;
    for (size_t s = 0; s < WINDOW_SCANS; ++s) {
        double d = (double)snap[s * NUM_CH_CHUNK + ch] - (double)mean;
        double d2 = d * d;
        m2 += d2;
        m4 += d2 * d2;
    }
    m2 /= (double)WINDOW_SCANS;
    m4 /= (double)WINDOW_SCANS;
    if (m2 < 1e-12) return 0.0f;
    return (float)(m4 / (m2 * m2) - 3.0);
}

// ===================== CLASS FUNCTIONS ================================
SignalQualityAnalyzer_C::SignalQualityAnalyzer_C(StateStore_s* stateStoreRef)
    : stateStoreRef_(stateStoreRef)
    , hop_sec(static_cast<float>(WINDOW_HOP_SCANS) / 250.0f)
    , NEEDED_WIN_(static_cast<size_t>(std::ceil(baseline_window_sec_ / hop_sec)))
    , RollingWinStatsBuf(NEEDED_WIN_)
{
    win_snapshot_.reserve(WINDOW_SCANS * NUM_CH_CHUNK);
    tempWinStats_.reserve(NEEDED_WIN_);
}


void SignalQualityAnalyzer_C::update_statestore(){
    const size_t numWins = RollingWinStatsBuf.get_count();
    if(numWins==0) {return;}

    Stats_s rolling_avg{};
    float numWins_inv = 1.0f / (float)numWins;

    // this is how we avg across all window history
    for (size_t ch = 0; ch < NUM_CH_CHUNK; ++ch) {
        rolling_avg.mean_uv[ch]  = RollingSums_.mean_uv[ch] * numWins_inv;
        rolling_avg.std_uv[ch]   = RollingSums_.std_uv[ch]  * numWins_inv;
        rolling_avg.rms_uv[ch]   = RollingSums_.rms_uv[ch]  * numWins_inv;
        rolling_avg.kurt[ch]     = RollingSums_.kurt[ch]    * numWins_inv;
        rolling_avg.entropy[ch]  = RollingSums_.entropy[ch] * numWins_inv;

        // these are rolling maxima we maintained
        rolling_avg.max_abs_uv[ch]  = RollingSums_.max_abs_uv[ch];
        rolling_avg.max_step_uv[ch] = RollingSums_.max_step_uv[ch];
    }
    
    std::lock_guard<std::mutex> lock(stateStoreRef_->signal_stats_mtx);
    stateStoreRef_->SignalStats.num_win_in_rolling = numWins;
    stateStoreRef_->SignalStats.rollingStats = rolling_avg;
    const float denom = (global_win_acq_ > 0) ? static_cast<float>(global_win_acq_) : 1.0f;
    stateStoreRef_->SignalStats.overall_bad_win_rate = (global_win_acq_ > 0) ? (float)overall_bad_win_num_ / (float)global_win_acq_ : 0.0f;
    stateStoreRef_->SignalStats.current_bad_win_rate = (numWins > 0) ? (float)current_bad_win_num_ / (float)numWins : 0.0f;
    return;
}

void SignalQualityAnalyzer_C::check_artifact_and_flag_window(sliding_window_t& window){
    Stats_s winStats {};
    bool didEvictThisRound = false;
    
    // rolling update (1): evict oldest if full, subtract contributions
    if(RollingWinStatsBuf.get_count() == NEEDED_WIN_){ // RB is full
        // increment rb (pop)
        RollingWinStatsBuf.pop(&evicted_);
        if (evicted_.isBad) current_bad_win_num_--;
        if(current_bad_win_num_ < 0){
            // prevent underflow
            current_bad_win_num_ = 0;
        }
        // subtract evicted from rolling sums
        for (size_t ch = 0; ch < NUM_CH_CHUNK; ++ch) {
            RollingSums_.mean_uv[ch]    -= evicted_.mean_uv[ch];
            RollingSums_.std_uv[ch]     -= evicted_.std_uv[ch];
            RollingSums_.rms_uv[ch]     -= evicted_.rms_uv[ch];
            RollingSums_.kurt[ch]       -= evicted_.kurt[ch];
            RollingSums_.entropy[ch]    -= evicted_.entropy[ch];
            kurt_sumsq_[ch]             -= (double)evicted_.kurt[ch]    * (double)evicted_.kurt[ch];
            ent_sumsq_[ch]              -= (double)evicted_.entropy[ch] * (double)evicted_.entropy[ch];
            // max_abs/max_step handled separately (see below)
        }
        didEvictThisRound = true;
    }

    // HARD THRESHOLDS -> Any one can set bad window flag...
    isGreaterThanMaxUvCount_.fill(0);
    surpassesMaxStepCount_.fill(0);

    bool failsMaxTest = 0;
    bool failsStepTest = 0;
    int failsKurtTestCount = 0;
    int failsEntTestCount = 0;

    window.sliding_window.get_data_snapshot(win_snapshot_);
    if (win_snapshot_.size() < WINDOW_SCANS * NUM_CH_CHUNK) {
        return; // not enough samples yet
        // shouldn't reach here if it's placed properly in main
    }

    global_win_acq_++;

    for(size_t ch = 0; ch < NUM_CH_CHUNK; ch++){
        double sum = 0.0, sumsq = 0.0;
        float max_abs = 0.0f;
        float max_step = 0.0f;

        // init prev for each channel (gets updated)
        float prev = win_snapshot_[0 * NUM_CH_CHUNK + ch];

        for(size_t s = 0; s < WINDOW_SCANS; ++s){
            // to acquire all for one channel, its the base plus the offset
            float sample = win_snapshot_[s*NUM_CH_CHUNK + ch];

            // Sums for stats calcs
            sum += sample;
            sumsq += (double)sample * (double)sample;

            // Max abs
            float av = std::abs(sample);
            max_abs = std::max(max_abs, av);
            if(av > MAX_ABS_UV){ // max abs amplitude = 200uv
                isGreaterThanMaxUvCount_[ch]++;
            }
            
            // Point-to-Point Step
            if(s>0){ // not the first scan, so we have a prev
                float step = std::abs(sample - prev);
                max_step = std::max(max_step, step);
                if(step > MAX_STEP_UV){
                    surpassesMaxStepCount_[ch]++;
                }
            }
            // prev for next time
            prev = sample;
        }

        // one channel complete
        float chMean = (float)(sum / double(WINDOW_SCANS));
        float ex2 = (float)(sumsq / (double)WINDOW_SCANS);
        float var = ex2 - chMean*chMean;
        float chStdv = safe_sqrt(var);
        float chRms  = safe_sqrt(ex2);

        winStats.mean_uv[ch]    = chMean;
        winStats.std_uv[ch]     = chStdv;
        winStats.rms_uv[ch]     = chRms;
        winStats.max_abs_uv[ch] = max_abs;
        winStats.max_step_uv[ch]= max_step;
        winStats.kurt[ch]    = excess_kurtosis_channel(win_snapshot_, ch, chMean);
        winStats.entropy[ch] = hist_entropy_channel(win_snapshot_, ch);
        // don't do MAD for now cuz it's lowkey very computationally expensive, let's see how much processing time we're up to

        if (isGreaterThanMaxUvCount_[ch] >= AMP_PERSIST_SAMPLES)  failsMaxTest = true;
        if (surpassesMaxStepCount_[ch] >= STEP_PERSIST_SAMPLES) failsStepTest = true;

        // assess kurtosis and entropy for this channel
        const size_t numWinsBeforePush = RollingWinStatsBuf.get_count(); // baseline window count

        bool failsKurtTest = false;
        bool failsEntTest  = false;

        if (numWinsBeforePush >= MIN_BASELINE_WINS) {
            // Rolling mean
            const double invN = 1.0 / (double)numWinsBeforePush;
        
            const double muK = (double)RollingSums_.kurt[ch]    * invN;
            const double muE = (double)RollingSums_.entropy[ch] * invN;
        
            // Rolling variance: E[x^2] - (E[x])^2
            const double ex2K = kurt_sumsq_[ch] * invN;
            const double ex2E = ent_sumsq_[ch]  * invN;
        
            double varK = ex2K - muK * muK;
            double varE = ex2E - muE * muE;
        
            if (varK < 0.0) varK = 0.0;
            if (varE < 0.0) varE = 0.0;
        
            const double sdK = std::sqrt(varK) + EPS_STD;
            const double sdE = std::sqrt(varE) + EPS_STD;
        
            // Thresholds (kurt unusually HIGH, entropy unusually LOW)
            const double kurt_hi = muK + (double)KURT_Z * sdK; // high-kurt outlier
            const double ent_lo  = muE - (double)ENT_Z  * sdE; // low-entropy outlier
        
            if ((double)winStats.kurt[ch]    > kurt_hi) failsKurtTest = true;
            if ((double)winStats.entropy[ch] < ent_lo)  failsEntTest  = true;
        }

        if (failsKurtTest) failsKurtTestCount++;
        if (failsEntTest)  failsEntTestCount++;

    }

    window.isArtifactualWindow =
    failsMaxTest ||
    failsStepTest ||
    (failsKurtTestCount >= MIN_CH_FAIL_KURT) ||
    (failsEntTestCount  >= MIN_CH_FAIL_ENT);

    overall_bad_win_num_ += window.isArtifactualWindow;
    current_bad_win_num_ += window.isArtifactualWindow;
    winStats.isBad = window.isArtifactualWindow;

    // rolling update: (2) push new and ADD CONTRIBUTIONS
    RollingWinStatsBuf.push(winStats);
    for (size_t ch = 0; ch < NUM_CH_CHUNK; ++ch) {
        RollingSums_.mean_uv[ch]    += winStats.mean_uv[ch];
        RollingSums_.std_uv[ch]     += winStats.std_uv[ch];
        RollingSums_.rms_uv[ch]     += winStats.rms_uv[ch];
        RollingSums_.kurt[ch]       += winStats.kurt[ch];
        RollingSums_.entropy[ch]    += winStats.entropy[ch];
        kurt_sumsq_[ch]             += (double)winStats.kurt[ch]    * (double)winStats.kurt[ch];
        ent_sumsq_[ch]              += (double)winStats.entropy[ch] * (double)winStats.entropy[ch];
    }

    // rolling max: cheap approach (HELPER)
    // maintain rolling max; if we evicted the current max, recompute by scanning hist_ (rare)
    // (Initialize on ramp-up too)
    auto recompute_max_if_needed = [&](size_t ch) {
        float mabs = 0.0f, mstep = 0.0f;
        RollingWinStatsBuf.get_data_snapshot(tempWinStats_);
        // iterate through windows n keep max
        for (const auto& w : tempWinStats_) {
        mabs  = std::max(mabs,  w.max_abs_uv[ch]);
        mstep = std::max(mstep, w.max_step_uv[ch]);
    }
        RollingSums_.max_abs_uv[ch] = mabs;   // treat these fields as “rolling max”, not sums
        RollingSums_.max_step_uv[ch] = mstep;
    };

    // NORMAL UPDATE
    for (size_t ch = 0; ch < NUM_CH_CHUNK; ++ch) {
        
        // incorporate new window by max() --> currently could be evicted max
        RollingSums_.max_abs_uv[ch]  = std::max(RollingSums_.max_abs_uv[ch],  winStats.max_abs_uv[ch]);
        RollingSums_.max_step_uv[ch] = std::max(RollingSums_.max_step_uv[ch], winStats.max_step_uv[ch]);

        // handle last evicted if there -> recompute max if necessary
        if(didEvictThisRound){
            bool evicted_was_max_abs = (evicted_.max_abs_uv[ch] == RollingSums_.max_abs_uv[ch]);
            bool evicted_was_max_step = (evicted_.max_step_uv[ch] == RollingSums_.max_step_uv[ch]);
            if (evicted_was_max_abs || evicted_was_max_step) recompute_max_if_needed(ch);
        }
    }

    // see if we need to update stats now (enough windows)
    // *(update on cadence)
    ui_tick_++;
    if(ui_tick_ % UI_UPDATE_EVERY_WIN == 0){
        update_statestore();
        // reset for next set of windows
        //CurrSignalStats_ = SignalStats_s{}; // back to default params (HACKY??)
    }

}