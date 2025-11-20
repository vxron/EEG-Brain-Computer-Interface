#pragma once
#include "../utils/Types.h"
#include <atomic>
/* STATESTORE
--> A single source of truth for all main c++ threads + client (js) to read:
    1) current UI state
    2) current stimulus frequency label 
    3) associated metadata
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
};


