#pragma once
#include "../utils/Types.h"
#include <atomic>
#include <mutex>
#include <condition_variable>
/* STATESTORE
--> A single source of truth for all main c++ threads + client (js) to read things like:
    1) current UI state
    2) current stimulus frequency label 
    3) associated metadata
    (...) and many more :,)
*/

struct StateStore_s{

    // General info about active channels
    std::atomic<int> g_n_eeg_channels{NUM_CH_CHUNK};
    // per-channel names (size fixed at compile time)
    std::array<std::string, NUM_CH_CHUNK> eeg_channel_labels;
    // channel enabled mask
    std::array<bool, NUM_CH_CHUNK> eeg_channel_enabled;

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

    std::atomic<UIPopup_E> g_ui_popup{UIPopup_None};

    // For displaying signal in real-time on UI
    std::atomic<bool> g_hasEegChunk{false};

    // custom type requires mutex protection
    mutable std::mutex last_chunk_mutex;
    bufferChunk_S g_lastEegChunk;

    // UI reads last chunk
    bufferChunk_S get_lastEegChunk() const {
        std::lock_guard<std::mutex> lock(last_chunk_mutex);
        return g_lastEegChunk;  // return by value (copy)
    }

    // backend (producer) sets last chunk
    void set_lastEegChunk(const bufferChunk_S& v) {
        std::lock_guard<std::mutex> lock(last_chunk_mutex);
        g_lastEegChunk = v;
    }

    // Running statistic measures of signals (rolling 45s)
    // AFTER bandpass + CAR + artifact rejection
    SignalStats_s SignalStats;
    mutable std::mutex signal_stats_mtx;

    // Helper: get copy of signal stats for HTTP to read safely 
    SignalStats_s get_signal_stats(){
        std::lock_guard<std::mutex> lock(signal_stats_mtx);
        return SignalStats;
    }

    // Pending name / epilepsy risk from front end (UIState_Calib_Options) (disclaimer form)
    std::mutex calib_options_mtx;
    std::string pending_subject_name;
    EpilepsyRisk_E pending_epilepsy;

    struct sessionInfo_s {
        std::atomic<bool> g_isModelReady{0};
        // strings must be mutex-protected (proceed 1 at a time)
        mutable std::mutex mtx_;
        std::string g_active_model_path = "";
        std::string g_active_subject_id = "";
        std::string g_active_session_id = "";
        std::string g_active_data_path = "";
        EpilepsyRisk_E g_epilepsy_risk = EpilepsyRisk_Unknown;

        std::string get_active_model_path() const {
            std::lock_guard<std::mutex> lock(mtx_);
            return g_active_model_path; // automatically unlocks mtx_
        }

        std::string get_active_subject_id() const {
            std::lock_guard<std::mutex> lock(mtx_);
            return g_active_subject_id;
        }

        std::string get_active_session_id() const {
            std::lock_guard<std::mutex> lock(mtx_);
            return g_active_session_id;
        }

        std::string get_active_data_path() const {
            std::lock_guard<std::mutex> lock(mtx_);
            return g_active_data_path;
        }
    };
    sessionInfo_s currentSessionInfo{};

    // LIST OF SAVED SESSIONS
    struct SavedSession_s {
        std::string id;          // unique ID (e.g. "veronica_2025-11-25T14-20")
        std::string label;       // human label for UI list ("Nov 25, 14:20 (Veronica)")
        std::string subject;     // subject_id
        std::string session;     // session_id
        std::string created_at;  // ISO time string
        std::string model_dir;   // model dir/path to load from
        
        // run mode frequency pair to be sent to ui
        TestFreq_E freq_left_hz_e{TestFreq_None};
        TestFreq_E freq_right_hz_e{TestFreq_None};
        int freq_right_hz{0};
        int freq_left_hz{0};
    };
    // Build the default session entry
    SavedSession_s defaultStart{
        .id = "default",
        .label = "Default",
        .subject = "",
        .session = "",
        .created_at = "",
        .model_dir = "",
        .freq_left_hz_e = TestFreq_None,
        .freq_right_hz_e = TestFreq_None,
        .freq_right_hz = 0,
        .freq_left_hz = 0
    };
    
    // vector of all saved sessions for storage, guarded with blocking mutex (fine since infrequent updates)
    mutable std::mutex saved_sessions_mutex;
    std::vector<SavedSession_s> saved_sessions { defaultStart };
    std::atomic<int> currentSessionIdx{0}; // iterates through vector

    // Helper: snapshot the list for HTTP /state (for display on sessions page)
    std::vector<SavedSession_s> snapshot_saved_sessions() const {
        // blocks until mutex is available for acquiring
        std::lock_guard<std::mutex> lock(saved_sessions_mutex);
        return saved_sessions;
    }

    // (1) finalize request slot from stim controller -> consumer after calibration success
    // conditional variable! 
    std::mutex mtx_finalize_request;
    std::condition_variable cv_finalize_request;
    bool finalize_requested = false;

    // (2) train job request from consumer -> training manager after finalize success
    std::mutex mtx_train_job_request;
    std::condition_variable cv_train_job_request;
    bool train_job_requested = false;

};



