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

    // =============== General info about active channels ==================
    std::atomic<int> g_n_eeg_channels{NUM_CH_CHUNK};
    // per-channel names (size fixed at compile time)
    std::array<std::string, NUM_CH_CHUNK> eeg_channel_labels;
    // channel enabled mask
    std::array<bool, NUM_CH_CHUNK> eeg_channel_enabled;

    // =================== UI State Machine Info ==================================== 
    std::atomic<bool> g_is_calib{false};
    std::atomic<UIState_E> g_ui_state{UIState_None}; // which "screen" should showing
    std::atomic<int> g_ui_seq{0}; // increment each time a new state is published by server so html can detect quickly
    
    // So that UI can POST events to stimulus controller state machine:
    std::atomic<UIStateEvent_E> g_ui_event{UIStateEvent_None};
    
    std::atomic<UIPopup_E> g_ui_popup{UIPopup_None};
    
    std::atomic<int> g_refresh_hz{0}; // monitor screen's refresh rate 
    // ^WHEN THIS IS SET -> we know UI has successfully connected (use this to determine start state)

    // Pending name / epilepsy risk from front end (UIState_Calib_Options) (disclaimer form)
    std::mutex calib_options_mtx;
    std::string pending_subject_name;
    EpilepsyRisk_E pending_epilepsy;

    // backend msg strings (where applicable)
    std::mutex train_fail_mtx; 
    std::string train_fail_msg = ""; // used as final train fail msg
    std::vector<TrainingIssue_s> train_fail_issues{};

    // ==================== Training Protocol Info & Fake Acq ========================================
    std::atomic<int> g_block_id{0}; // block index in protocol
    std::atomic<TestFreq_E> g_freq_hz_e{TestFreq_None}; 
    std::atomic<int> g_freq_hz{0}; // ************USED FOR FAKE ACQ DURING RUN MODE
    trainingProto_S training_proto; // ***********USED FOR FAKE ACQ STREAMER DURING CALIB

    // ======================== For AcqStreamerFromDataset class =============================================
// cv from stim controller -> producer for notifying producer when its time to start calib/run mode dataset streaming
    // i.e. producer will then call unicorn_start_acq w appropriate settings
    std::mutex mtx_streaming_request;
    std::condition_variable streaming_request;
    bool streaming_requested = false;
    std::mutex mtx_streamer_freqs;
    std::vector<int> acc_freqs_in_use_by_streamer;
    bool test_mode_arg = 0; // 1 for calib, 0 for run

    // ============ For displaying signal in real-time on UI (hardware checks page) ============
    std::atomic<bool> g_hasEegChunk{false};
    // custom types require mutex protection
    mutable std::mutex last_chunk_mutex;
    bufferChunk_S g_lastEegChunk;
    // helper so UI reads last chunk
    bufferChunk_S get_lastEegChunk() const {
        std::lock_guard<std::mutex> lock(last_chunk_mutex);
        return g_lastEegChunk;  // return by value (copy)
    }
    // helper so backend (producer) sets last chunk
    void set_lastEegChunk(const bufferChunk_S& v) {
        std::lock_guard<std::mutex> lock(last_chunk_mutex);
        g_lastEegChunk = v;
    }

    // ============= Running statistic measures of EEG (rolling 45s) for bad window detection/removal ============================
    // AFTER bandpass + CAR + artifact rejection
    SignalStats_s SignalStats;
    mutable std::mutex signal_stats_mtx;
    // Helper: get copy of signal stats for HTTP to read safely 
    SignalStats_s get_signal_stats(){
        std::lock_guard<std::mutex> lock(signal_stats_mtx);
        return SignalStats;
    }

    // ======================= Template for any user session ====================
    struct sessionInfo_s {
        std::atomic<bool> g_isModelReady{0};

        mutable std::mutex mtx_;
        std::string g_active_model_path = "";
        std::string g_active_subject_id = "";
        std::string g_active_session_id = "";
        std::string g_active_data_path = "";
        EpilepsyRisk_E g_epilepsy_risk = EpilepsyRisk_Unknown;

        std::string get_active_model_path() const {
            std::lock_guard<std::mutex> lock(mtx_);
            return g_active_model_path;
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
    // Current session info for fast access
    sessionInfo_s currentSessionInfo{};

    // ======================== LIST OF SAVED SESSIONS ================================
    // this is what we use IN RUNMODE (NOT CURRENTSESSIONINFO^, we use that for calib, args to pass Python script)
    // TODO: add bool or flag that says whether savedsession is good or bad, automatically delete bad after displaying results to user
    struct SavedSession_s {
        std::string id;                  // unique ID (e.g. "veronica_2025-11-25T14-20")
        std::string label;               // human label for UI list ("Nov 25, 14:20 (Veronica)")
        std::string subject;             // subject_id
        std::string session;             // session_id
        std::string model_dir;           // model dir/path to load from
        std::string model_arch;          // CNN vs SVM
        std::vector<DataInsufficiency_s> data_insuff; // if there were missing windows/trials/batches etc
        bool has_data_insuff = false;

        double final_holdout_acc;
        double final_train_acc;
        
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
        .model_dir = "",
        .model_arch = "",
        .data_insuff = {},
        .has_data_insuff = false,
        .final_holdout_acc = 0.0,
        .final_train_acc = 0.0,
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

    std::atomic<int> g_onnx_session_is_reloading{0}; 
    // since it can be a very time-consuming process, let ui know
    // -1: "waiting for onnx session status from consumer thread" (since it can take it some time to reach that part of the code)
    // +1: consumer responds "yes i am reloading"
    // 0: consumer responds "no reloading" or "done reloading"

    // flag to tell consumer it should re-init onnx model -> notify on session change
    // - whenever currentSessionIdx changes (e.g. from sessions page, from training manager when training finishes)
    //std::atomic<bool> g_onnx_session_needs_reload{true}; // init true for first load
    // TODO: make GENERAL to all things that should reload on new session (e.g. UI as well...)
    // this can get set by stim controller when user selects a new or diff sess?

    // mtx protecting global json meta containing session list on disk for loading at startup
    std::mutex global_session_list_json_mtx; 

    // flag for when sessions have been loaded from disk at startup
    std::atomic<bool> g_sessions_loaded_from_disk{false};

    // ======================== Actuation thread Sync ======================
    // cv to notify thread when consumer makes non-neutral inference (either left or right ssvep)
    std::mutex mtx_actuation_request;
    std::condition_variable actuation_request;
    bool actuation_requested = false;
    SSVEPState_E actuation_direction = SSVEP_Unknown; // data that comes with req

    // atomic to notify consumer when actuator is actuating (consumer should sleep during this brief period)
    std::atomic<bool> g_is_currently_actuating{false}; // consumer should freeze during actuation
    std::atomic<bool> g_consumer_ack_actuation_stop{true}; // force consumer to ack actuation stop (this gets set to false when actuating starts and must be set to true again by cons)

    // =================== Multi-thread training request flow after calibration finishes ===================================
    // (1) finalize request slot from stim controller -> consumer after calibration success
    // conditional variable! 
    // TODO: FINALIZE SHOULDNT HAPPEN ON UI TRANSITION BCUZ THERE COULD STILL BE THINGS TO PULL FROM RB, OR FLUSH FROM CSV
    std::mutex mtx_finalize_request;
    std::condition_variable cv_finalize_request;
    bool finalize_requested = false;

    // (2) train job request from consumer -> training manager after finalize success
    std::mutex mtx_train_job_request;
    std::condition_variable cv_train_job_request;
    bool train_job_requested = false;

    // (3) train done (model ready) notif from training manager -> stim controller
    // (device becomes operable), can update sessionInfo model_ready bool
    std::mutex mtx_model_ready;
    // std::condition_variable cv_model_ready; <- add back if you need to block another thread on it but for rn we only POLL in stim controller for model being ready
    bool model_just_ready = false;

    // ========================== SETTINGS PAGE =========================
    struct Settings_s {
        std::atomic<SettingCalibData_E> calib_data_setting{CalibData_MostRecentOnly};
        std::atomic<SettingTrainArch_E> train_arch_setting{TrainArch_CNN};
        std::atomic<SettingStimMode_E> stim_mode{StimMode_Flicker};
        std::atomic<SettingWaveform_E> waveform{Waveform_Square};
        std::atomic<SettingHparam_E> hparam_setting{HPARAM_OFF};
        std::atomic<bool> demo_mode{false};
        // frequency pool: up to 6 (fixed size array)
        // these are the defaults:
        std::array<TestFreq_E, 6> selected_freqs_e{
            TestFreq_8_Hz, TestFreq_11_Hz, TestFreq_14_Hz,
            TestFreq_17_Hz, TestFreq_20_Hz, TestFreq_None
        };
        // array mutation requires mtx protection
        std::mutex selected_freq_array_mtx;
        std::atomic<int> selected_freqs_n{5};
        std::atomic<int> num_times_cycle_repeats{3};
        std::atomic<int> duration_active_s{11};
        std::atomic<int> duration_none_s{10};
        std::atomic<int> duration_rest_s{8};
    };
    Settings_s settings{}; // instantiate

};



