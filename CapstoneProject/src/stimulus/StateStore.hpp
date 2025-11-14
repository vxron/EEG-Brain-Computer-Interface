#pragma once
#include "../utils/Types.h"
#include <atomic>
/* STATESTORE
--> A single place where host (server)/client (html/js) read/write stimulus information 
*/


struct StateStore_s{
    std::atomic<StimulusState_E> g_stim_window{StimState_None}; // which "screen" should showing
#if CALIB_MODE
    std::atomic<int> g_block_id{0}; // block index in protocol
    std::atomic<TestFreq_E> g_freq_hz_e{TestFreq_None};
    std::atomic<int> g_freq_hz{0};
#endif
    std::atomic<int> g_ui_seq{0}; // increment each time a new state is published by server so html can detect quickly
    std::atomic<int> g_refresh_hz{0}; // monitor screen's refresh rate
};


