/* STIMULUS CONTROLLER : STATE MACHINE */

#pragma once
#include "../utils/Types.h"
#include "../utils/SWTimer.hpp"
#include <optional>
#include "../shared/StateStore.hpp"
#include <deque>
#include <random>

#ifdef ACQ_BACKEND_FAKE
// Reps per class in each shuffle cycle 
constexpr int FAKE_ACTIVE_REPS = 2; // LEFT x 2, RIGHT x 2 
constexpr int FAKE_NO_SSVEP_REPS = 8; // 2x total active
// Duration ranges (ms)
constexpr int FAKE_ACTIVE_MIN_MS = 1000;
constexpr int FAKE_ACTIVE_MAX_MS = 6000;
constexpr int FAKE_REST_MIN_MS   = 1000;
constexpr int FAKE_REST_MAX_MS   = 8000;
#endif

class StimulusController_C{
public:
    explicit StimulusController_C(StateStore_s* stateStoreRef);
    UIState_E getUIState() const {return state_;};
    std::chrono::milliseconds getCurrentBlockTime() const;
    void runUIStateMachine();
    void stopStateMachine();

private:
    bool is_stopped_ = false;
    StateStore_s* stateStoreRef_;
    UIState_E state_;
    UIState_E prevState_;
    UIState_E pausedFromState_; // for clean returns from paused

    trainingProto_S trainingProtocol_; // requires a default upon construction
    std::deque<TestFreq_E> activeBlockQueue_;
    std::size_t activeQueueIdx_ = 0;

#ifdef ACQ_BACKEND_FAKE
    std::vector<int> emulatedFreqsForFakeAcq_;
    std::vector<int> fakeAcqShuffledSeq_; // current shuffled pool (Hz or -1 for rest)
    int fakeAcqSeqIdx_ = 0; // current position in pool
    std::mt19937 fakeAcqRng_{ std::random_device{}() }; // seeded RNG (one per instance)
#endif
    
    SW_Timer_C currentWindowTimer_;
    SW_Timer_C guardAgainstInfLoopTimer_;
#ifdef ACQ_BACKEND_FAKE
    SW_Timer_C fakeAcqRunModeTimer_;
#endif
    std::chrono::milliseconds activeBlockDur_ms_{0};
    std::chrono::milliseconds restBlockDur_ms_{0};
    std::chrono::milliseconds noSSVEPBlockDur_ms_{0};

    std::string pending_subject_name_ = ""; // for calib mode quick access
    EpilepsyRisk_E pending_epilepsy_ = EpilepsyRisk_Unknown; 

    // latches to make things edge-triggered :)
    bool end_calib_timeout_emitted_ = false;
    bool awaiting_calib_overwrite_confirm_ = false;
    bool awaiting_highfreq_confirm_ = false;

    void rebuild_protocol_from_settings();
    std::optional<UIStateEvent_E> detectEvent();
    void processEvent(UIStateEvent_E ev);
    void onStateEnter(UIState_E prevState, UIState_E newState, UIStateEvent_E ev);
    void onStateExit(UIState_E state, UIStateEvent_E ev);
    int checkStimFreqIsIntDivisorOfRefresh(bool isCalib, int desiredTestFreq);
    static bool has_divisor_6_to_20(int n);

#ifdef ACQ_BACKEND_FAKE
    void fakeAcq_buildSeqAndShuffle();
    void fakeAcq_advanceToNextSSVEP();
    std::chrono::milliseconds fakeAcq_getDurationForHz(int hz);
#endif

};



