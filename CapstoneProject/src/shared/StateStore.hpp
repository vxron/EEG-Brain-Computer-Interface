#pragma once
#include "../utils/Types.h"
#include <atomic>
#include <mutex>
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

    // run mode frequency pair to be sent to ui
    std::atomic<TestFreq_E> g_freq_left_hz_e{TestFreq_None};
    std::atomic<TestFreq_E> g_freq_right_hz_e{TestFreq_None};
    std::atomic<int> g_freq_right_hz{0};
    std::atomic<int> g_freq_left_hz{0};

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

    // Training status (Python) + subject / session ID
    struct sessionInfo_s {
        std::atomic<bool> g_isModelReady{0};
        // strings must be mutex-protected (proceed 1 at a time)
        mutable std::mutex mtx_;
        std::string g_active_model_path;
        std::string g_active_subject_id;
        std::string g_active_session_id;

        void set_active_model_path(const std::string& v) {
            std::lock_guard<std::mutex> lock(mtx_);
            g_active_model_path = v; // automatically unlocks mtx_
        }
        std::string get_active_model_path() const {
            std::lock_guard<std::mutex> lock(mtx_);
            return g_active_model_path; // automatically unlocks mtx_
        }

        void set_active_subject_id(const std::string& v) {
            std::lock_guard<std::mutex> lock(mtx_);
            g_active_subject_id = v;
        }
        std::string get_active_subject_id() const {
            std::lock_guard<std::mutex> lock(mtx_);
            return g_active_subject_id;
        }
    };
    sessionInfo_s sessionInfo{};

    // LIST OF SAVED SESSIONS
    struct SavedSession_s {
        std::string id;          // unique ID (e.g. "veronica_2025-11-25T14-20")
        std::string label;       // human label for UI list ("Nov 25, 14:20 (Veronica)")
        std::string subject;     // subject_id
        std::string session;     // session_id
        std::string created_at;  // ISO time string
        std::string model_dir;   // model dir/path to load from
    };

    // vector of all saved sessions for storage, guarded with blocking mutex (fine since infrequent updates)
    mutable std::mutex saved_sessions_mutex;
    std::vector<SavedSession_s> saved_sessions;

    // Helper: add a new saved session (called from StimulusController / training success)
    void add_saved_session(const SavedSession_s& s) {
        // blocks until mutex is available for acquiring
        std::lock_guard<std::mutex> lock(saved_sessions_mutex);
        saved_sessions.push_back(s);
    } // <-- lock goes out of scope here, mutex is automatically released

    // Helper: snapshot the list for HTTP /state (for display on sessions page)
    std::vector<SavedSession_s> snapshot_saved_sessions() const {
        // blocks until mutex is available for acquiring
        std::lock_guard<std::mutex> lock(saved_sessions_mutex);
        return saved_sessions;
    }

};



