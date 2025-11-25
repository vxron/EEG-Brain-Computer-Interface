#pragma once
#include "../utils/Types.h"
#include <atomic>
/* STATESTORE
--> A single source of truth for all main c++ threads + client (js) to read things like:
    1) current UI state
    2) current stimulus frequency label 
    3) associated metadata
    (...) and many more :,)
*/

struct StateStore_s{
    std::atomic<bool> g_is_calib{false};
    std::atomic<UIState_E> g_ui_state{UIState_None}; // which "screen" should showing
    std::atomic<int> g_ui_seq{0}; // increment each time a new state is published by server so html can detect quickly

    std::atomic<int> g_block_id{0}; // block index in protocol
    
    std::atomic<TestFreq_E> g_freq_hz_e{TestFreq_None};
    std::atomic<int> g_freq_hz{0};
    std::atomic<int> g_refresh_hz{0}; // monitor screen's refresh rate 
    // ^WHEN THIS IS SET -> we know UI has successfully connected (use this to determine start state)
    //std::atomic<bool> g_passed_start_guard{false};

    // So that UI can POST events to stimulus controller state machine:
    std::atomic<UIStateEvent_E> g_ui_event{UIStateEvent_None};

    // run mode frequency pair to be sent to ui
    std::atomic<TestFreq_E> g_freq_left_hz_e{TestFreq_None};
    std::atomic<TestFreq_E> g_freq_right_hz_e{TestFreq_None};
    std::atomic<int> g_freq_right_hz{0};
    std::atomic<int> g_freq_left_hz{0};

    // Training status (Python) + subject / session ID
    struct sessionInfo_s {
        std::atomic<bool> g_isModelReady{0};
        std::atomic<std::string> g_active_model_path{""}; // where we pull current classifier from
        std::atomic<std::string> g_active_subject_id{""};
        std::atomic<std::string> g_active_session_id{""};
    };
    
    sessionInfo_s sessionInfo{};

};


