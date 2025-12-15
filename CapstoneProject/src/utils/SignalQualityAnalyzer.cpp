#include "SignalQualityAnalyzer.h"

void SignalQualityAnalyzer_C::check_artifact_and_flag_window(sliding_window_t& window){
    // HARD THRESHOLDS -> Any one can set bad window flag...
    bool isBadWindow = 0;

    // (1) Max absolute amplitude = 200uV for >= 3 samples on any channel
    
    
    // (2) Max point-to-point step = 100uV on any channel

    // (3) Excess kurtosis 

    // (4) Entropy

    window.isArtifactualWindow = isBadWindow;

    // see if we need to compute stats now (enough windows)
    if(curr_win_acq_ >= NEEDED_WIN_){
        perform_stats_calcs_on_wins();
    }

    // consider EAWICA in the future
}

