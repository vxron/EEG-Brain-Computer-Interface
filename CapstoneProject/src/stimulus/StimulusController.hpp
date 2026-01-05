/*
STIMULUS CONTROLLER : writer
- builds the training schedule (STATE MACHINE - which phase we're in; home, instructions, calib block, run)
- every ~100ms, updates atomics in the shared StateStore and publishes a new 'seq' to server so HTML can see it & compare against lastSeq
- launches the HTML once at start, sets a neutral end state on finish
*/

#pragma once
#include "../utils/Types.h"
#include "../utils/SWTimer.hpp"
#include <optional>
#include "../shared/StateStore.hpp"
#include <deque>

// SINGLETON
class StimulusController_C{
public:
    explicit StimulusController_C(StateStore_s* stateStoreRef, std::optional<trainingProto_S> trainingProtocol = std::nullopt);
    UIState_E getUIState() const {return state_;};
    std::chrono::milliseconds getCurrentBlockTime() const;
    void runUIStateMachine();
    void stopStateMachine();
private:
    bool is_stopped_ = false;
    StateStore_s* stateStoreRef_;
    UIState_E state_;
    UIState_E prevState_;
    trainingProto_S trainingProtocol_; // requires a default upon construction

    std::deque<TestFreq_E> activeBlockQueue_;
    std::size_t activeQueueIdx_ = 0;
    SW_Timer_C currentWindowTimer_;
    std::chrono::milliseconds activeBlockDur_ms_{0};
    std::chrono::milliseconds restBlockDur_ms_{0};

    std::string pending_subject_name_ = ""; // for calib mode quick access
    EpilepsyRisk_E pending_epilepsy_ = EpilepsyRisk_Unknown; 

    // latches to make things edge-triggered :)
    bool end_calib_timeout_emitted_ = false;
    bool awaiting_calib_overwrite_confirm_ = false; // do we need to double check w user to enter calib sess?
    bool awaiting_highfreq_confirm_ = false;

    std::optional<UIStateEvent_E> detectEvent();
    void processEvent(UIStateEvent_E ev);
    void onStateEnter(UIState_E prevState, UIState_E newState);
    void onStateExit(UIState_E state, UIStateEvent_E ev);
    void runHomeWindow();
    void runActiveCalibWindow();
    void runInstructionsCalibWindow();
    void runModeWindow();
    void transitionBetweenWindows(); // maybe this should be handled on UI client side
    void manageTrainingProtocol(); // uses currentWindowTimer to control calib blocks, trigger timeout events for state machine
    void process_inputs();
    void process_events();
    int checkStimFreqIsIntDivisorOfRefresh(bool isCalib, int desiredTestFreq); // returns freq to use; require the flicker frequency to be an integer divisor of the refresh rate; else bump to next closest integer divisor
    static bool has_divisor_6_to_20(int n);

};



