#include <thread>
#include <chrono>
#include <iostream>
#include "utils/RingBuffer.hpp"
#include "utils/Types.h"
#include "shared/StateStore.hpp"
#include "stimulus/HttpServer.hpp"
#include "acq/WindowConfigs.hpp"
#include <atomic>
#include "utils/Logger.hpp"
#include <csignal>
#include <vector>
#include <cstdint>
#include <iomanip>
#include <string_view>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include "stimulus/StimulusController.hpp"
#include "acq/UnicornDriver.h"
#include <fstream>
#include "utils/SignalQualityAnalyzer.h"
#include <filesystem>
#include <ctime>
#include "utils/SessionPaths.hpp"
#include "classifier/ONNXClassifier.hpp"
#include "actuation/ServoDriver.h"
#include "utils/JsonUtils.hpp"
#include "utils/json.hpp"
#include <unordered_set>

#ifdef USE_EEG_FILTERS
#include "utils/Filters.hpp"
#endif

#ifdef ACQ_BACKEND_FAKE
#include "acq/FakeAcquisition.h"
#endif

constexpr bool TEST_MODE = 0;

// Global "please stop" flag set by Ctrl+C (SIGINT) to shut down cleanly
static std::atomic<bool> g_stop{false};

// Interrupt signal sent when ctrl+c is pressed
void handle_sigint(int) {
    g_stop.store(true,std::memory_order_relaxed);
}

void producer_thread_fn(RingBuffer_C<bufferChunk_S>& rb, StateStore_s& stateStoreRef){
    using namespace std::chrono_literals;
    logger::tlabel = "producer";
try {
    LOG_ALWAYS("producer start");

#ifdef ACQ_BACKEND_FAKE
    LOG_ALWAYS("PATH=MOCK");
    FakeAcquisition_C::stimConfigs_S fakeCfg{};
    // Check FakeAcquisition.h for defaults...
    fakeCfg.dcDrift.enabled = true;
    fakeCfg.lineNoise.enabled = true;
    fakeCfg.alpha.enabled = true;
    fakeCfg.beta.enabled = true;
    // random artifacts, alpha and beta sources off for now

    FakeAcquisition_C acqDriver(fakeCfg);

#else
    LOG_ALWAYS("PATH=HARDWARE");
    UnicornDriver_C acqDriver{};

#endif

#ifdef USE_EEG_FILTERS
    EegFilterBank_C filterBank;
#endif

    // somthn to use iacqprovider_s instead of unicorndriver_c directly
    // then we can choose based on acq_backend_fake which provider btwn unicorn and fake to set th eobjec too?
    // also need to updat csv so it logs appropraite measures (all eeg channels) in the acq_bavkend_fake path

    if (acqDriver.unicorn_init() == false || acqDriver.dump_config_and_indices() == false || acqDriver.unicorn_start_acq(TEST_MODE) == false){
        LOG_ALWAYS("unicorn_init failed; exiting producer");
        rb.close();
        return;
    };

    size_t tick_count = 0;

    // Channel configs
    int n_ch = acqDriver.getNumChannels();
    if (n_ch <= 0 || n_ch > NUM_CH_CHUNK) {
        n_ch = NUM_CH_CHUNK; // clamp defensively
    }
    stateStoreRef.g_n_eeg_channels.store(n_ch, std::memory_order_release);

    std::vector<std::string> labels;
    acqDriver.getChannelLabels(labels);
    if (labels.size() < static_cast<size_t>(n_ch)) {
        // Fallback: synthesize generic labels for missing ones
        for (int i = static_cast<int>(labels.size()); i < n_ch; ++i) {
            labels.emplace_back("Ch" + std::to_string(i + 1));
        }
    }

    // assume labels.size() >= n_ch (checked above)
    for (int i = 0; i < n_ch; ++i) {
        stateStoreRef.eeg_channel_labels[i] = labels[i];
        stateStoreRef.eeg_channel_enabled[i] = true;
    }
    for (int i = n_ch; i < NUM_CH_CHUNK; ++i) {
        stateStoreRef.eeg_channel_enabled[i] = false;
    }
    
    // MAIN ACQUISITION LOOP
    while(!g_stop.load(std::memory_order_relaxed)){
        bufferChunk_S chunk{};

#ifdef ACQ_BACKEND_FAKE
	int currSimFreq = stateStoreRef.g_freq_hz.load(std::memory_order_acquire);
    acqDriver.setActiveStimulus(static_cast<double>(currSimFreq)); // if 0, backend won't produce sinusoid
#endif
        
        acqDriver.getData(NUM_SCANS_CHUNK, chunk.data.data()); // chunk.data.data() gives type float* (addr of first float in std::array obj)
        tick_count++;
        chunk.tick = tick_count;

#ifdef USE_EEG_FILTERS
        // before we create window: PREPROCESS CHUNK
        filterBank.process_chunk(chunk);
#endif
        // Update state store with this new chunk for UI vis
        stateStoreRef.g_hasEegChunk.store(true, std::memory_order_release);
        stateStoreRef.set_lastEegChunk(chunk);
        
        // blocking push function... will always (eventually) run and return true, 
        // unless queue is closed, in which case we want out
        if(!rb.push(chunk)){
            LOG_ALWAYS("RingBuffer closed while pushing; stopping producer");
            break;
        } 
    }
    // on thread shutdown, close queue to call release n unblock any consumer waiting for acquire
    LOG_ALWAYS("producer shutting down; stopping acquisition backend...");
    acqDriver.unicorn_stop_and_close();
    rb.close();
}
catch (const std::exception& e) {
    LOG_ALWAYS("producer: FATAL unhandled exception: " << e.what());
    rb.close();
    g_stop.store(true, std::memory_order_relaxed);
}
catch (...) {
    LOG_ALWAYS("producer: FATAL unknown exception");
    rb.close();
    g_stop.store(true, std::memory_order_relaxed);
}
}

void consumer_thread_fn(RingBuffer_C<bufferChunk_S>& rb, StateStore_s& stateStoreRef){
    using namespace std::chrono_literals;
    logger::tlabel = "consumer";
    LOG_ALWAYS("consumer start");
    size_t tick_count = 0; // global
    size_t tick_count_per_session = 0; // per session for logging
    size_t run_mode_bad_windows = 0;
    size_t run_mode_bad_window_count = 0;
    size_t run_mode_clean_window_count = 0;
    SW_Timer_C run_mode_bad_window_timer;
    SW_Timer_C reload_timer_guard;

    // run mode pseudo state machine
    int window_class = -1;
    SSVEPState_E last_stable_prediction = SSVEP_Unknown;
    SSVEPState_E curr_win_prediction = SSVEP_Unknown;
    SSVEPState_E last_win_prediction = SSVEP_Unknown;
    int num_windows_stable_prediction = 0;
    int num_windows_unknown = 0;
    int bad_win_from_onnx_ct = 0;
    // new window every 0.32s -> means we'll have 3s/0.32s = 9.375 windows necessary for 2s
    int NUM_REQ_STABLE_WINDOWS_FOR_ACTUATION = 10; // req users to look at stimulus for min 3s
    int NUM_ALLOWED_WRONG_WINDOWS_BEFORE_DEBOUNCE_RESET = 3; // TODO
    ClassifyResult_s currResults{};
    
    // single instance of onnx env
    ONNX_RT_C ONNX_RT;

    // actually don't require concurrent actuation... since it should occur right after consumer requests it, and consumer shouldn't do anything else while actuating
    // actually ACTUALLY haha we do want actuation on its own thread because we need to run it at a fixed frequency (control loop)
    // TODO: EXCEPT if user presses button on UI that's "STOP IMMEDIATELY" -> then it should get interrupted immediately (torque cmd to zero)
    // thus need to decide frequency at which controller is operating (sending out torque cmds) & that will be freq at which we poll atomic g_actuation_stop_requested from state store

try{
    SignalQualityAnalyzer_C SignalQualityAnalyzer(&stateStoreRef);

    sliding_window_t window_run(false); // should acquire the data for 1 window with that many pops n then increment by hop... 
    sliding_window_t window_calib(true);
    sliding_window_t* windowPtr = &window_run; // ptr to window were using -> default start to run mode
    bufferChunk_S temp; // placeholder

    namespace fs = std::filesystem;

    // Two independent CSV files, one at window level, one at chunk level
    std::ofstream csv_chunk;   // eeg_chunk_data.csv
    std::ofstream csv_win;     // eeg_windows.csv
    std::ofstream csv_run_log; // run mode csv log
    bool chunk_opened = false;
    bool win_opened   = false;
    bool run_log_opened = false;
    std::chrono::steady_clock::time_point run_log_t0; // start of run log (using steady_clock)
    size_t rows_written_chunk = 0;
    size_t rows_written_win   = 0;
    bool run_mode_onnx_ready = false;

    // Track which session these files belong to so we can reopen when session changes
    std::string active_session_id;
    std::string active_data_dir;
    bool settings_written = false;
    std::string settings_written_session_id;
    
    // follow the session the stim controller created
    auto refresh_active_session_paths = [&]() -> bool {
        std::string sid;
        std::string ddir;
        {
            // Your sessionInfo has its own mutex; use its getters if you wrote them.
            sid  = stateStoreRef.currentSessionInfo.get_active_session_id();
            ddir = stateStoreRef.currentSessionInfo.get_active_data_path();
        }

        // Not ready yet (StimulusController may not have created a session)
        if (sid.empty() || ddir.empty()) return false;

        // If same session, no change
        if (sid == active_session_id && ddir == active_data_dir) return true;

        // Session changed - close old files and reset flags
        if (chunk_opened) { csv_chunk.flush(); csv_chunk.close(); chunk_opened = false; }
        if (win_opened)   { csv_win.flush();   csv_win.close();   win_opened   = false; }

        active_session_id = sid;
        active_data_dir   = ddir;
        settings_written = false; // need to write
        settings_written_session_id.clear();

        LOG_ALWAYS("consumer: switched logging session to "
                   << "session_id=" << active_session_id
                   << " data_dir=" << active_data_dir);

        return true;
    };

    auto ensure_settings_meta_written = [&]() -> bool {
        if (settings_written && settings_written_session_id == active_session_id) return true;

        if (!refresh_active_session_paths()) return false;
        if (active_session_id.empty() || active_data_dir.empty()) return false;

        namespace fs = std::filesystem;
        fs::path out_path = fs::path(active_data_dir) / "settings_meta.json";

        std::ofstream js(out_path, std::ios::out | std::ios::trunc);
        if (!js.is_open()) {
            LOG_ALWAYS("ERROR: failed to open " << out_path.string());
            return false;
        }

        // Timestamp
        const std::time_t now = std::time(nullptr);

        // Snapshot key session info
        std::string subject_id, session_id;
        {
            std::lock_guard<std::mutex> lock(stateStoreRef.currentSessionInfo.mtx_);
            subject_id = stateStoreRef.currentSessionInfo.g_active_subject_id;
            session_id = stateStoreRef.currentSessionInfo.g_active_session_id;
        }

        // Snapshot settings (atomics)
        SettingTrainArch_E train_arch =
            stateStoreRef.settings.train_arch_setting.load(std::memory_order_acquire);

        SettingCalibData_E calib_data =
            stateStoreRef.settings.calib_data_setting.load(std::memory_order_acquire);

        SettingStimMode_E stim_mode =
            stateStoreRef.settings.stim_mode.load(std::memory_order_acquire);

        SettingWaveform_E waveform =
            stateStoreRef.settings.waveform.load(std::memory_order_acquire);
        
        SettingHparam_E hparam =
            stateStoreRef.settings.hparam_setting.load(std::memory_order_acquire);

        // snapshot selected frequency pool 
        std::array<TestFreq_E, 6> selected_freqs_e{};
        int selected_n = 0;
        {
            std::lock_guard<std::mutex> lk(stateStoreRef.settings.selected_freq_array_mtx);
            selected_freqs_e = stateStoreRef.settings.selected_freqs_e;
            selected_n = stateStoreRef.settings.selected_freqs_n.load(std::memory_order_acquire);
        }

        // Channels
        int n_ch_local = stateStoreRef.g_n_eeg_channels.load(std::memory_order_acquire);
        if (n_ch_local <= 0 || n_ch_local > NUM_CH_CHUNK) n_ch_local = NUM_CH_CHUNK;

        // Window config
        const std::size_t winLen = windowPtr->winLen;
        const std::size_t winHop = windowPtr->winHop;

        // Trim
        const std::size_t trim_scans_each_side = TRIM_SCANS_EACH_SIDE;
        const std::size_t trim_samples_each_side = trim_scans_each_side * static_cast<std::size_t>(n_ch_local);

        // Backend flags
#ifdef ACQ_BACKEND_FAKE
        const char* backend = "FAKE";
#else
        const char* backend = "HARDWARE";
#endif

#ifdef USE_EEG_FILTERS
        const bool use_filters = true;
#else
        const bool use_filters = false;
#endif

        // Write JSON manually
        js << "{\n";
        js << "  \"subject_id\": " << std::quoted(subject_id) << ",\n";
        js << "  \"session_id\": " << std::quoted(session_id) << ",\n";
        js << "  \"written_unix\": " << static_cast<long long>(now) << ",\n";

        js << "  \"acq\": {\n";
        js << "    \"backend\": " << std::quoted(backend) << ",\n";
        js << "    \"test_mode\": " << (TEST_MODE ? "true" : "false") << ",\n";
        js << "    \"use_eeg_filters\": " << (use_filters ? "true" : "false") << "\n";
        js << "  },\n";

        js << "  \"settings\": {\n";
        js << "    \"train_arch\": " << std::quoted(TrainArchEnumToString(train_arch)) << ",\n";
        js << "    \"calib_data_policy\": " << std::quoted(CalibDataEnumToString(calib_data)) << ",\n";
        js << "    \"stim_mode\": " << std::quoted(StimModeEnumToString(stim_mode)) << ",\n";
        js << "    \"waveform\": " << std::quoted(WaveformEnumToString(waveform)) << ",\n";
        js << "    \"selected_freqs_n\": " << selected_n << ",\n";
        js << "    \"selected_freqs_e\": [";
        {
            bool first = true;
            int emitted = 0;
            for (int i = 0; i < 6 && emitted < selected_n; ++i) {
                TestFreq_E f = selected_freqs_e[static_cast<size_t>(i)];
                if (f == TestFreq_None) continue;
                if (!first) js << ", ";
                js << static_cast<int>(f);
                first = false;
                ++emitted;
            }
        }
        js << "],\n";
        js << "    \"selected_freqs_hz\": [";
        {
            bool first = true;
            int emitted = 0;
            for (int i = 0; i < 6 && emitted < selected_n; ++i) {
                TestFreq_E f = selected_freqs_e[static_cast<size_t>(i)];
                if (f == TestFreq_None) continue;
                if (!first) js << ", ";
                js << TestFreqEnumToInt(f);
                first = false;
                ++emitted;
            }
        }
        js << "]\n";
        js << "  },\n";


        js << "  \"channels\": {\n";
        js << "    \"n_channels\": " << n_ch_local << ",\n";
        js << "    \"labels\": [";
        for (int ch = 0; ch < n_ch_local; ++ch) {
            if (ch) js << ", ";
            js << std::quoted(stateStoreRef.eeg_channel_labels[ch]);
        }
        js << "],\n";

        js << "    \"enabled\": [";
        for (int ch = 0; ch < n_ch_local; ++ch) {
            if (ch) js << ", ";
            js << (stateStoreRef.eeg_channel_enabled[ch] ? "true" : "false");
        }
        js << "]\n";
        js << "  },\n";

        js << "  \"windowing\": {\n";
        js << "    \"win_len_samples\": " << winLen << ",\n";
        js << "    \"win_hop_samples\": " << winHop << ",\n";
        js << "    \"trim_samples_each_side\": " << trim_samples_each_side << "\n";
        js << "  }\n";

        js << "}\n";

        js.flush();
        js.close();

        settings_written = true;
        settings_written_session_id = active_session_id;

        LOG_ALWAYS("wrote " << out_path.string());
        return true;
    };

    // TODO -> FIX LOGGING TO NOT HARDCODE NUM_CH_CHUNK
    auto ensure_csv_open_chunk = [&]() -> bool {
        if (chunk_opened) return true;

        if (!refresh_active_session_paths()) {
            // No active session yet; don’t write
            return false;
        }

        fs::path out_path = fs::path(active_data_dir) / "eeg_calib_data.csv";

        csv_chunk.open(out_path, std::ios::out | std::ios::trunc);
        if (!csv_chunk.is_open()) {
            LOG_ALWAYS("ERROR: failed to open " << out_path.string());
            return false;
        }

        // Header
        csv_chunk << "chunk_tick,sample_idx";
        for (std::size_t ch = 0; ch < NUM_CH_CHUNK; ++ch) {
            csv_chunk << ",eeg" << (ch + 1);
        }
        csv_chunk << ",testfreq_e,testfreq_hz\n";

        chunk_opened = true;
        rows_written_chunk = 0;

        LOG_ALWAYS("opened " << out_path.string());
        return true;
    };

    auto ensure_csv_open_window = [&]() -> bool {
        if (win_opened) return true;

        if (!refresh_active_session_paths()) {
            return false;
        }

        fs::path out_path = fs::path(active_data_dir) / "eeg_windows.csv";

        csv_win.open(out_path, std::ios::out | std::ios::trunc);
        if (!csv_win.is_open()) {
            LOG_ALWAYS("ERROR: failed to open " << out_path.string());
            return false;
        }

        int n_ch_local = stateStoreRef.g_n_eeg_channels.load(std::memory_order_acquire);
        if (n_ch_local <= 0 || n_ch_local > NUM_CH_CHUNK) n_ch_local = NUM_CH_CHUNK;

        // Header
        csv_win << "window_idx,ui_state,is_trimmed,is_bad,sample_idx";
        for (int ch = 0; ch < n_ch_local; ++ch) csv_win << ",eeg" << (ch + 1);
        csv_win << ",testfreq_e,testfreq_hz\n";

        win_opened = true;
        rows_written_win = 0;
        tick_count_per_session = 0;   // reset index when starting this file

        LOG_ALWAYS("opened " << out_path.string());
        return true;
    };

    // helper for WINDOW LEVEL ONLY
    auto log_window_snapshot = [&](const sliding_window_t& w,
                               UIState_E uiState,
                               std::size_t window_idx,
                               bool use_trimmed) {
        if (!ensure_csv_open_window()) {
            LOG_ALWAYS("WINLOG: csv_win not open inside log_window_snapshot (window_idx="
                << window_idx << "), skipping write");
            return;
        }

        int n_ch_local = stateStoreRef.g_n_eeg_channels.load(std::memory_order_acquire);
        if (n_ch_local <= 0 || n_ch_local > NUM_CH_CHUNK) n_ch_local = NUM_CH_CHUNK;

        // choose buffer
        std::vector<float> snap;                 // local storage when snapshotting
        const std::vector<float>* pBuf = nullptr;

        if (use_trimmed && w.isTrimmed && !w.trimmed_window.empty()) {
            pBuf = &w.trimmed_window;
        } else {
            w.sliding_window.get_data_snapshot(snap);
            pBuf = &snap;
        }
        const std::vector<float>& buf = *pBuf;

        if (buf.empty()) {
            LOG_ALWAYS("WARN: snapshot empty, skipping CSV");
            return;
        }
        if (buf.size() % static_cast<std::size_t>(n_ch_local) != 0) {
            LOG_ALWAYS("WARN: snapshot size not divisible by n_ch; skipping CSV");
            return;
        }

        const std::size_t n_scans = buf.size() / static_cast<std::size_t>(n_ch_local);

        // label fields (only meaningful in calib)
        int tf_e = static_cast<int>(w.testFreq);
        int tf_hz = (w.testFreq == TestFreq_None) ? -1 : TestFreqEnumToInt(w.testFreq);

        for (std::size_t s = 0; s < n_scans; ++s) {
            csv_win << window_idx
                    << "," << static_cast<int>(uiState)
                    << "," << (use_trimmed && w.isTrimmed ? 1 : 0)
                    << "," << (w.isArtifactualWindow ? 1 : 0)
                    << "," << s;

            const std::size_t base = s * static_cast<std::size_t>(n_ch_local);
            for (int ch = 0; ch < n_ch_local; ++ch) {
                csv_win << "," << buf[base + static_cast<std::size_t>(ch)];
            }

            csv_win << "," << tf_e << "," << tf_hz << "\n";
            ++rows_written_win;
        }

        if (!csv_win) {
            LOG_ALWAYS("WINLOG: stream error after writing window_idx=" << window_idx
                << " (n_scans=" << n_scans << ")");
        }


        if ((rows_written_win % 5000) == 0) csv_win.flush();
    };

    auto log_run_classifier_window = [&](
        int stim_hz,                      // g_freq_hz snapshotted at window-build time for fake acq, default -1 for real acq
        bool is_artifact,       
        bool was_used,                    // !artifactual && onnx ran
        const ClassifyResult_s& clf,
        int class_out,                    // actual current class
        SSVEPState_E pred_state,          // curr_win_prediction as string
        int stable_ct,                    // debounce
        bool act_fired,                   // whether actuation was fired this window
        SSVEPState_E act_dir
    ) {
        // Open log file on first call
        if (!run_log_opened) {
            // get data dir for active session
            std::string subject;
            std::string session;
            {
                std::lock_guard<std::mutex> savedsesslock(stateStoreRef.saved_sessions_mutex);
                int currIdx = stateStoreRef.currentSessionIdx.load(std::memory_order_acquire);
                subject = stateStoreRef.saved_sessions[currIdx].subject;
                session = stateStoreRef.saved_sessions[currIdx].session;
            }
            const fs::path PROJECT_ROOT = sesspaths::find_project_root();
            const fs::path data_dir = PROJECT_ROOT / "data";
            fs::path log_path;
            if(session != "" && !session.empty()){
                log_path = data_dir / subject / session / "run_classifier_log.csv";
            }
            // fallback to making new "unknown session" path -> overwrites
            else {
                log_path = data_dir / subject / "run_mode_logs" / "run_classifier_log.csv";
            }
            // Create directory if it doesn't exist
            std::error_code ec;
            fs::create_directories(log_path.parent_path(), ec);
            if (ec) {
                LOG_ALWAYS("RUN_LOG: failed to create directory: " << ec.message());
                return;
            }
            
            csv_run_log.open(log_path, std::ios::out | std::ios::trunc);
            LOG_ALWAYS("RUN_LOG: attempted open at " << log_path.string());
            if (!csv_run_log.is_open()) return;
            LOG_ALWAYS("RUN_LOG: opened at " << log_path.string());

            // Write header
            csv_run_log << "window_idx,timestamp_ms"
                        << ",stim_freq_hz,stim_state"
                        << ",is_artifactual,was_used"
                        << ",logit_0,logit_1,logit_2"
                        << ",softmax_0,softmax_1,softmax_2"
                        << ",onnx_class_raw,predicted_state"
                        << ",num_stable_windows,stable_target"
                        << ",actuation_requested,actuation_direction"
                        << ",bad_win_count_in_period,clean_win_count_in_period\n";

            run_log_opened = true;
            run_log_t0 = std::chrono::steady_clock::now();
        }

        // if we're in real mode, we acc don't know what "stim_hz" is (dont know where user is looking)
#ifndef ACQ_BACKEND_FAKE
        stim_hz = -1; // default
#endif

        // map hz to state string via session's known freqs
        std::string stim_state = "none";
        if (stim_hz != -1 && stim_hz != 0) {
            std::lock_guard<std::mutex> lk(stateStoreRef.saved_sessions_mutex);
            int idx = stateStoreRef.currentSessionIdx.load(std::memory_order_acquire);
            if (stim_hz == stateStoreRef.saved_sessions[idx].freq_left_hz)  stim_state = "left";
            else if (stim_hz == stateStoreRef.saved_sessions[idx].freq_right_hz) stim_state = "right";
            else stim_state = "other";
        }

        auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - run_log_t0).count();
        
        csv_run_log << tick_count << "," << now_ms
                    << "," << stim_hz << "," << stim_state
                    << "," << (is_artifact ? 1 : 0) << "," << (was_used ? 1 : 0);
        
        if (was_used) {
            csv_run_log << "," << clf.logits[0] << "," << clf.logits[1] << "," << clf.logits[2]
                        << "," << clf.softmax[0] << "," << clf.softmax[1] << "," << clf.softmax[2]
                        << "," << class_out << "," << ssvep_str(pred_state);
        } else {
            csv_run_log << ",,,,,,"  // blank logits/softmax
                        << ",-1,unknown";
        }

        csv_run_log << "," << stable_ct << "," << NUM_REQ_STABLE_WINDOWS_FOR_ACTUATION
                    << "," << (act_fired ? 1 : 0) << "," << (act_fired ? ssvep_str(act_dir) : "")
                    << "," << run_mode_bad_window_count << "," << run_mode_clean_window_count << "\n";
    };

    auto handle_finalize_if_requested = [&]() {
        // quick check flag (locked)
        bool do_finalize = false;
        {
            std::lock_guard<std::mutex> lock(stateStoreRef.mtx_finalize_request);
            do_finalize = stateStoreRef.finalize_requested;
            if (do_finalize) stateStoreRef.finalize_requested = false;
        } // unlock mtx on exit
        if (!do_finalize) return;

        // Always finalize when requested
        LOG_ALWAYS("finalize detected");

        // Close/flush files
        if (win_opened)   { csv_win.flush();   csv_win.close();   win_opened = false; }
        if (chunk_opened) { csv_chunk.flush(); csv_chunk.close(); chunk_opened = false; }

        // remove __IN_PROGRESS from session titles given we've completed calib successfully
        std::string data_dir, model_dir, subject_id, session_id;
        {
            std::lock_guard<std::mutex> lock(stateStoreRef.currentSessionInfo.mtx_);
            data_dir   = stateStoreRef.currentSessionInfo.g_active_data_path;
            model_dir  = stateStoreRef.currentSessionInfo.g_active_model_path;
            session_id = stateStoreRef.currentSessionInfo.g_active_session_id;
            subject_id = stateStoreRef.currentSessionInfo.g_active_subject_id;
        }

        const std::string base_id = sesspaths::strip_in_progress_suffix(session_id);
        std::error_code ec;
        fs::path old_data  = fs::path(data_dir);
        fs::path old_model = fs::path(model_dir);
        fs::path new_data  = old_data.parent_path()  / base_id;
        fs::path new_model = old_model.parent_path() / base_id;

        if (sesspaths::is_in_progress_session_id(session_id)) {
            fs::rename(old_data, new_data, ec);
            if (ec) SESS_LOG("finalize: ERROR data rename: " << ec.message());
            ec.clear();

            fs::rename(old_model, new_model, ec);
            if (ec) SESS_LOG("finalize: ERROR model rename: " << ec.message());

            // Update StateStore paths/ids so training uses the final (non-suffixed) dirs
            {
                std::lock_guard<std::mutex> lock(stateStoreRef.currentSessionInfo.mtx_);
                stateStoreRef.currentSessionInfo.g_active_session_id = base_id;
                stateStoreRef.currentSessionInfo.g_active_data_path  = new_data.string();
                stateStoreRef.currentSessionInfo.g_active_model_path = new_model.string();
            }
        }
        // After creating new dirs
        sesspaths::prune_old_sessions_for_subject(new_data / subject_id, 3);
        // TODO: PRUNE MODELS IF/WHEN TRAINING FAILS !

        // Signal to training manager: data is ready (to launch training thread)
        {
            std::lock_guard<std::mutex> lock(stateStoreRef.mtx_train_job_request);
            stateStoreRef.train_job_requested = true;
        }
        LOG_ALWAYS("notify");
        stateStoreRef.cv_train_job_request.notify_one();
    };
    
    UIState_E currState  = UIState_None;
    UIState_E prevState = UIState_None;
    TestFreq_E currLabel  = TestFreq_None;
    TestFreq_E prevLabel = TestFreq_None;
    bool firstCalibWindowBuilt = false;
    bool firstRunWindowBuilt = false;
    bool* firstWinCheckPtr = &firstRunWindowBuilt;

    while(!g_stop.load(std::memory_order_relaxed)){
        // if currently actuating -> skip everything
        if(stateStoreRef.g_is_currently_actuating.load(std::memory_order_acquire)){
            continue;
        }
        
        // 1) ========= before we slide/build new window: assess artifacts + read snapshot of ui_state and freq ============ 
        // check if we need to stop writing calib data and start training thread
        handle_finalize_if_requested();

        // Keep active session paths fresh (no-op if unchanged)
        refresh_active_session_paths();

        currState = stateStoreRef.g_ui_state.load(std::memory_order_acquire);
        currLabel = stateStoreRef.g_freq_hz_e.load(std::memory_order_acquire);
        if((currState == UIState_Instructions || currState == UIState_Home || currState == UIState_None)){
            // pop but don't build window 
            // need to pop bcuz need to prevent buffer overflow 
            // TODO: clean up implementation to always pull/pop and then save window logic to end
            if(!rb.pop(&temp)) break;
            continue; //back to top while loop
        }

        // decide which sliding window we're using right now...
        // & BUILD FIRST WINDOW - do we need this??
        // based on state machine -> only possible to start calib from the no ssvep state, so this is the rising edge we want to detect for making a calib window instead of run
        if(currState == UIState_NoSSVEP_Test && prevState != UIState_NoSSVEP_Test){
            window_calib.stash_len = 0; // flush stale stash
            std::vector<float> tmp(window_calib.winLen); // max amount we might need to drain is winLen
            window_calib.sliding_window.drain(tmp.data()); // as well as any partial refill
            windowPtr = &window_calib;
            firstCalibWindowBuilt = false; // reset
            firstWinCheckPtr = &firstCalibWindowBuilt;
        }
        // falling edge to go back to default run mode window is when the currState enters pendingTraining (just finished calib)
        else if (currState == UIState_Pending_Training && prevState != UIState_Pending_Training){
            window_run.stash_len = 0; // flush stale stash
            std::vector<float> tmp(window_run.winLen);
            window_run.sliding_window.drain(tmp.data()); // as well as any partial refill
            windowPtr = &window_run;
            firstRunWindowBuilt = false; // reset
            firstWinCheckPtr = &firstRunWindowBuilt;
        } 

        // run_mode_onnx_ready must be reset to false on run mode exit
        if(prevState == UIState_Active_Run && currState != UIState_Active_Run){
            run_mode_onnx_ready = false;
            run_log_opened = false; // redo logging each run sess
        }

        // save this as prev state to check after window is built to make sure UI state hasn't changed in between
        prevState = currState;
        prevLabel = currLabel;
        
        // 2) ============================= build the new window =============================
        float discard; // first pop
        if(*firstWinCheckPtr){
            size_t to_pop = std::min(windowPtr->winHop, windowPtr->sliding_window.get_count()); // guard against truncated windows
            for(size_t k=0; k<to_pop; k++){
                windowPtr->sliding_window.pop(&discard); 
            }
        }

        while(windowPtr->sliding_window.get_count()<windowPtr->winLen){ // now push
            UIState_E intState = stateStoreRef.g_ui_state.load(std::memory_order_acquire);
            TestFreq_E intLabel = stateStoreRef.g_freq_hz_e.load(std::memory_order_acquire);
            if((intState != prevState) || (intLabel != prevLabel)){
                windowPtr->stash_len = 0; // discard stale overflow

                break; // change in UI; not a good window
            }
			std::size_t amnt_left_to_add = windowPtr->winLen - windowPtr->sliding_window.get_count(); // in samples
			// if there is previous 'len' in stash, we should take it and decrement len
            if (windowPtr->stash_len > 0) {
                // take full amnt_left_to_add from stash if it's available, otherwise take window.stash_len
                const std::size_t take = (windowPtr->stash_len > amnt_left_to_add) ? amnt_left_to_add : windowPtr->stash_len;
                for (std::size_t i = 0; i < take; ++i){
                    windowPtr->sliding_window.push(windowPtr->stash[i]);
				}
                // move leftover stash to front of array for next round
                if (take < windowPtr->stash_len) {
                    std::memmove(windowPtr->stash.data(), // start of stash array (dest)
                                 windowPtr->stash.data() + take, // ptr to where remaining data starts (src)
                                 (windowPtr->stash_len - take) * sizeof(float)); // number of bytes to move
                }
                windowPtr->stash_len -= take; // how much we lost for next time; will never be <0

                continue; // go check while-condition again
            }
            // stash is empty
            if(windowPtr->stash_len != 0){
                LOG_ALWAYS("There's an issue with sliding window stash.");
                break;
            }
			if(!rb.pop(&temp)){
				break;
			} else {
				// pop successful -> push into sliding window
				if(amnt_left_to_add >= NUM_SAMPLES_CHUNK){
                    for(std::size_t j=0;j<NUM_SAMPLES_CHUNK;j++){
                        windowPtr->sliding_window.push(temp.data[j]);
                    } // goes back to check while for next chunk
				}
				else {
					// take what we need and stash the rest for next window
					for(std::size_t j=0;j<NUM_SAMPLES_CHUNK;j++){
						if(j<amnt_left_to_add){
							windowPtr->sliding_window.push(temp.data[j]);
						} else {
							windowPtr->stash[j-amnt_left_to_add]=temp.data[j];
                            windowPtr->stash_len++; // increasing slots to add from stash for next time
						}
					}
				}
			}
		}

        // 3) once window is full, read ui_state/freq again and decide if window is "valid" to emit based on comparison with initial ui_state and freq
        // -> if the two snapshots disagree, ui changed mid-window: drop this window 
        currState = stateStoreRef.g_ui_state.load(std::memory_order_acquire);
        currLabel = stateStoreRef.g_freq_hz_e.load(std::memory_order_acquire);
        if((currState != prevState) || (currLabel != prevLabel)){
            // changed halfway through -> not a valid window for processing
            windowPtr->decision = SSVEP_Unknown;
            windowPtr->has_label = false;
            continue; // go build next window
        }

        // verified it's an ok window
        if(!(*firstWinCheckPtr)){
            *firstWinCheckPtr = true;
        }
        ++tick_count;
        windowPtr->tick=tick_count;

        // reset before looking at state store vals
        windowPtr->isTrimmed = false;
        windowPtr->has_label = false;
        windowPtr->testFreq = TestFreq_None;

        // always check artifacts and flag bad windows
        SignalQualityAnalyzer.check_artifact_and_flag_window(*windowPtr);

        if(currState == UIState_Active_Calib || currState == UIState_NoSSVEP_Test) {
            if (!ensure_csv_open_window()) {
                continue; // must be open for logging
            }
            if (!ensure_settings_meta_written()) {
                LOG_ALWAYS("WARN: could not write settings_meta.json");
            }

            ++tick_count_per_session; // these are the ticks we log in the calib data file
            
            int n_ch_local = stateStoreRef.g_n_eeg_channels.load(std::memory_order_acquire);
            if (n_ch_local <= 0 || n_ch_local > NUM_CH_CHUNK) n_ch_local = NUM_CH_CHUNK;
            
            // trim window ends for training data (GUARD)
            windowPtr->trimmed_window.clear();
            windowPtr->sliding_window.get_trimmed_snapshot(windowPtr->trimmed_window,
                TRIM_SCANS_EACH_SIDE * n_ch_local,
                TRIM_SCANS_EACH_SIDE * n_ch_local);
            windowPtr->isTrimmed = true;
            const std::size_t expected_trimmed = windowPtr->winLen - (2 * TRIM_SCANS_EACH_SIDE * (std::size_t)n_ch_local);
            if (windowPtr->trimmed_window.size() != expected_trimmed) {
                LOG_ALWAYS("TRIMDBG: trimmed size mismatch: got=" << windowPtr->trimmed_window.size()
                    << " expected=" << expected_trimmed
                    << " winLen=" << windowPtr->winLen
                    << " n_ch=" << n_ch_local
                    << " tick_per_session=" << tick_count_per_session
                    << " ui_state=" << (int)currState
                );
            }

            // we should be attaching a label to our windows for calibration data
            windowPtr->testFreq = currLabel;
            windowPtr->has_label = (currLabel != TestFreq_None);
            // Log trimmed window (only if has label)
            if(windowPtr->has_label){
                log_window_snapshot(*windowPtr, currState, tick_count_per_session, /*use_trimmed=*/true);
            }
        }
        
        else if(currState == UIState_Active_Run){
            // if actuation stop must be acked -> ack and reset all counters/preds to be safe
            if(!stateStoreRef.g_consumer_ack_actuation_stop.load(std::memory_order_acquire)){
                stateStoreRef.g_consumer_ack_actuation_stop.store(true, std::memory_order_release);
                // resets
            }

            // if we have a new sess, we need the onnx models to be reloaded
            // check our curr model path from statestore
            std::string curr_expected_model_path = "";
            int currSessIdx = 0;
            {
                std::lock_guard<std::mutex> saved_sess_lock(stateStoreRef.saved_sessions_mutex);
                currSessIdx = stateStoreRef.currentSessionIdx.load(std::memory_order_acquire);
                curr_expected_model_path = stateStoreRef.saved_sessions[currSessIdx].model_dir;
            }

            // prepare onnx if not already ready -> complete handshake w stim controller & model loading once per run mode entry
            if(!run_mode_onnx_ready){
                // in case stim controller is slower -> wait for it to set g_onnx_session_is_reloading to -1
                reload_timer_guard.start_timer(std::chrono::milliseconds{ 2000 });
                while(stateStoreRef.g_onnx_session_is_reloading.load(std::memory_order_acquire) != -1){
                    // timer guard
                    if(reload_timer_guard.check_timer_expired()){
                        LOG_ALWAYS("Timing Issue with g_onnx_session_is_reloading: Not being set to -1 deterministically from StimController, or not propagating to Consumer Thread.");
                        break;
                    }
                    continue; // should be rly fast from stim controller, so spin-wait is ok
                }
                reload_timer_guard.stop_timer();
            

                if(ONNX_RT.get_curr_onnx_model_path() != curr_expected_model_path && (stateStoreRef.g_onnx_session_is_reloading.load(std::memory_order_acquire)==-1)){
                    // NEED RELOAD, which can be time consuming so we also store status var in statestore
                    stateStoreRef.g_onnx_session_is_reloading.store(1, std::memory_order_release);
                    std::string model_arch = stateStoreRef.saved_sessions[currSessIdx].model_arch;
                    ONNX_RT.init_onnx_model(curr_expected_model_path, TrainArchStringToEnum(model_arch));
                    // reload complete
                    stateStoreRef.g_onnx_session_is_reloading.store(0, std::memory_order_release);
                } else {
                    stateStoreRef.g_onnx_session_is_reloading.store(0, std::memory_order_release);
                }
                run_mode_onnx_ready = true;
            }

            // popup saying 'signal is bad, too many artifactual windows. run hardware checks' when too many bad windows detected in a certain time frame, then reset
            if(run_mode_bad_window_timer.check_timer_expired()){
                // expired -> see if we should throw popup based on bad window counts in the 9s timeout period
                if((run_mode_clean_window_count > 0) && (double(run_mode_bad_window_count) / double(run_mode_clean_window_count) >= 0.25)) { // require 4:1 good:bad ratio
                    // DO POPUP
                    stateStoreRef.g_ui_popup.store(UIPopup_TooManyBadWindowsInRun, std::memory_order_release);
                }
                // reset for next round
                run_mode_bad_window_count = 0;
                run_mode_clean_window_count = 0;
            }
            
            int stim_hz = -1; // default for non-fake acq
#ifdef ACQ_BACKEND_FAKE
            stim_hz = stateStoreRef.g_freq_hz.load(std::memory_order_acquire);
#endif
            if(windowPtr->isArtifactualWindow){
                if(!run_mode_bad_window_timer.is_started()){
                    run_mode_bad_window_timer.start_timer(std::chrono::milliseconds{9000});
                }
                run_mode_bad_window_count++;
                // reset counter for debounce (TODO: DECIDE IF WANT TO KEEP OR REMOVE FOR MORE AGGRESSIVENESS IN PRED)
                num_windows_stable_prediction = 0; // reset
                last_win_prediction = SSVEP_Unknown;
                ClassifyResult_s dummy_clf{};
                log_run_classifier_window(stim_hz, true, false, dummy_clf, -1, SSVEP_Unknown, 0, false, SSVEP_Unknown);
                continue; // don't use this window
            } else {
                // clean window
                if(run_mode_bad_window_timer.is_started()){
                    // add to within-timer clean window count for comparison
                    run_mode_clean_window_count++;
                }
                currResults = ONNX_RT.classify_window(*windowPtr);
                window_class = currResults.final_class;
                if (window_class == -1) {
                    bad_win_from_onnx_ct++;
                }
                // mapping from output logits->freqs
                // hz_a -> 0, hz_b -> 1, hz_rest -> 2
                // Convention: left = freq a, right = freq b
                // logit 0 = LEFT, logit 1 = RIGHT, logit 2 = NONE
                curr_win_prediction = PythonClassToSSVEPState(window_class);
                if(curr_win_prediction==SSVEP_Unknown){
                    num_windows_unknown++;
                }
                
                if(curr_win_prediction==last_win_prediction){
                    // building twds new prediction
                    num_windows_stable_prediction++;
                }
                else if(curr_win_prediction!=last_win_prediction){
                    num_windows_stable_prediction = 0; // reset
                }
                // reset
                last_win_prediction = curr_win_prediction;
                // debounce for minimum windows before we publish new cmd to actuation
                // should have consistent windows for about 2 seconds to register user really looking at target
                // in that span, allow A FEW SSVEP_Nones or SSVEP_Unknowns in case of bad windows - before calling off 
                // check if debounce period has passed
                bool actuation_fired = false;
                SSVEPState_E act_dir = SSVEP_Unknown;
                if(num_windows_stable_prediction > NUM_REQ_STABLE_WINDOWS_FOR_ACTUATION){
                    // passed debounce
                    // request actuation if it's an actuating state
                    if(curr_win_prediction == SSVEP_Left || curr_win_prediction == SSVEP_Right){
                        actuation_fired = true;
                        act_dir = curr_win_prediction;
                        std::lock_guard<std::mutex> lock_actuation(stateStoreRef.mtx_actuation_request);
                        stateStoreRef.actuation_requested = true;
                        stateStoreRef.actuation_direction = curr_win_prediction;
                        stateStoreRef.actuation_request.notify_one(); // notify actuator thread
                        // reset num_windows_stable_prediction
                        num_windows_stable_prediction = 0;
                    }
                }
                
                log_run_classifier_window(stim_hz, false, true, currResults, window_class, curr_win_prediction, 
                                   num_windows_stable_prediction, actuation_fired, act_dir);
            } 
        }
	}
    // exiting due to producer exiting means we need to close window rb
    windowPtr->sliding_window.close();
    rb.close();
    if (chunk_opened)   { csv_chunk.flush(); csv_chunk.close(); }
    if (win_opened)     { csv_win.flush();   csv_win.close();   }
    if (run_log_opened) { csv_run_log.flush(); csv_run_log.close(); }
}
catch (const std::exception& e) {
        LOG_ALWAYS("consumer: FATAL unhandled exception: " << e.what());
        rb.close();
        g_stop.store(true, std::memory_order_relaxed);
    }
catch (...) {
        LOG_ALWAYS("consumer: FATAL unknown exception");
        rb.close();
        g_stop.store(true, std::memory_order_relaxed);
    }
}

void stimulus_thread_fn(StateStore_s& stateStoreRef){
	// runs protocols in calib mode (handle timing & keep track of state in g_record)
	// should toggle g_record on stimulus switch
	// in calib mode producer should wait for g_record to toggle? or it can just always check state for data ya thats better 0s and 1s in known fixed order...
    try {
        LOG_ALWAYS("stim: start");
        StimulusController_C stimController(&stateStoreRef);
        stimController.runUIStateMachine();
        LOG_ALWAYS("stim: exit");
    }
    catch (const std::system_error& e) {
        LOG_ALWAYS("stim: FATAL std::system_error: " << e.what()
            << " | code=" << e.code().value()
            << " | category=" << e.code().category().name()
            << " | message=" << e.code().message());
        // then your shutdown path...
    }
    catch (const std::exception& e) {
        LOG_ALWAYS("stim: FATAL std::exception: " << e.what());
    }
    catch (...) {
        LOG_ALWAYS("stim: FATAL unknown exception");
        g_stop.store(true, std::memory_order_relaxed);
    }
}

void http_thread_fn(HttpServer_C& http){
    logger::tlabel = "http";
    try {
        LOG_ALWAYS("http: listen thread start");
        http.http_listen_for_poll_requests();   // blocks here
        LOG_ALWAYS("http: listen thread exit");
    }
    catch (const std::exception& e) {
        LOG_ALWAYS("http: FATAL unhandled exception: " << e.what());
        g_stop.store(true, std::memory_order_relaxed);
    }
    catch (...) {
        LOG_ALWAYS("http: FATAL unknown exception");
        g_stop.store(true, std::memory_order_relaxed);
    }
}

/* THE SCRIPT MUST OUTPUT
(1) ONNX MODELS
(2) BEST TWO FREQUENCIES TO USE 
--> Script must output to <model_dir>/train_result.json
*/
void training_manager_thread_fn(StateStore_s& stateStoreRef){
    auto write_fail_to_statestore = [&stateStoreRef](std::string fail_reason, 
        std::vector<TrainingIssue_s> issues = {})
    {
        stateStoreRef.g_ui_event.store(UIStateEvent_TrainingFailed);
        std::unique_lock<std::mutex> popup_lock(stateStoreRef.train_fail_mtx);
        stateStoreRef.train_fail_msg = fail_reason;
        stateStoreRef.currentSessionInfo.g_isModelReady.store(false, std::memory_order_release);
        stateStoreRef.train_fail_issues = issues; // clears old issues if empty now
    };
    // build manual 'hash keys' to make sure we don't dupe
    auto make_key = [&](const DataInsufficiency_s& d) {
        return d.stage + "|" + d.message + "|" +
               "|" + d.metric + "|" +
               std::to_string(d.required) + "|" + std::to_string(d.actual) + "|" +
               (d.frequency_hz ? std::to_string(d.frequency_hz) : "null");
    };
    // node here should be the 'issue' object
    // self-passing lambda for zero-overhead recursive lambda
    auto collect_data_insufficiency = [make_key](auto self, const nlohmann::json& node, // current subtree of JSON
        std::vector<DataInsufficiency_s>& out, // shared output vec (accumulates results)
        std::unordered_set<std::string>& seen // enforce no repeats in set published to statestore
        ) -> void 
    {
        if(node.is_object()){
            // if this object contains data_insufficiency, pass it
            auto it = node.find("data_insufficiency");
            if(it!= node.end() && it->is_object()){
                const nlohmann::json& di = *it; // returns where we have data_insuff
                DataInsufficiency_s datainsuff;
                datainsuff.stage = node.value("stage","");
                datainsuff.message = node.value("message","");
                datainsuff.metric = node.value("metric","");
                datainsuff.required = node.value("required",0);
                datainsuff.actual = node.value("actual",0);
                datainsuff.frequency_hz = node.value("frequency_hz",0);

                // make sure we don't have two of same issue
                std::string key = make_key(datainsuff);
                if(seen.insert(key).second){ // second arg of seen.insert returns true if inserted, false if alr existed
                    out.push_back(std::move(datainsuff)); // if inserted, transfer ownership to out
                }
            }
            // recurse into all fields
            for(const auto& [k,v]: node.items()){ // for k,v in the items..
                self(self, v, out, seen); // SELF-PASSING RECURSION: v becomes new node (value or contents assoc w/ key) -> without having to pre-define the fxn name explicitly (invalid bcuz it's auto rtn type...)
            }
            return;
        }
        if(node.is_array()){
            for(const auto& v : node){ 
                self(self, v, out, seen);
            }
            return;
        }
    };
    auto collect_training_issues = [](const nlohmann::json& j, 
        std::vector<TrainingIssue_s>& training_issues) -> void 
    {
        if (j.contains("issues") && j["issues"].is_array()) {
            for (const auto& issue_obj : j["issues"]) {
                TrainingIssue_s issue;
                issue.stage = issue_obj.value("stage", "");
                issue.message = issue_obj.value("message", "");
                // Parse details if present
                if (issue_obj.contains("details") && issue_obj["details"].is_object()) {
                    const auto& det = issue_obj["details"];
                    // Candidate frequencies
                    if (det.contains("cand_freqs") && det["cand_freqs"].is_array()) {
                        for (const auto& freq : det["cand_freqs"]) {
                            if (freq.is_number()) {
                                issue.details_cand_freqs.push_back(freq.get<int>());
                            }
                        }
                    }
                    issue.details_n_pairs = det.value("n_pairs", 0);
                    issue.details_skip_count = det.value("skip_count", 0);
                }
                training_issues.push_back(issue);
            }
        }
    };

    logger::tlabel = "training manager";
    namespace fs = std::filesystem;

    fs::path projectRoot = sesspaths::find_project_root();
    if (fs::exists(projectRoot / "CapstoneProject") && fs::is_directory(projectRoot / "CapstoneProject")) {
        projectRoot /= "CapstoneProject";
    }
    // Script lives at: <CapstoneProject>/model train/python/train_svm.py
    fs::path scriptPath = projectRoot / "model train" / "python" / "train_ssvep.py";

    std::error_code ec;
    scriptPath = fs::weakly_canonical(scriptPath, ec); // normalize path (non-throwing)
    LOG_ALWAYS("trainmgr: projectRoot=" << projectRoot.string());
    LOG_ALWAYS("trainmgr: scriptPath=" << scriptPath.string()
              << " (exists=" << (fs::exists(scriptPath) ? "Y" : "N") << ")"
              << " (ec=" << (ec ? ec.message() : "ok") << ")");
    if (!fs::exists(scriptPath)) {
        LOG_ALWAYS("WARN: training script not found at " << scriptPath.string()
                  << " (training will fail until path is fixed)");
    }

    while(!g_stop.load(std::memory_order_acquire)){
        // wait for training request
        std::unique_lock<std::mutex> lock(stateStoreRef.mtx_train_job_request); // locked
        // mtx gets unlocked (thread sleeps) until notify_one is fired from stim controller & train job is req
        // (avoids busy-wait)
        stateStoreRef.cv_train_job_request.wait(lock, [&stateStoreRef]{ 
            return (stateStoreRef.train_job_requested == true || g_stop.load(std::memory_order_acquire)); // this thread waits for one of these to be true
        });

        if(g_stop.load(std::memory_order_acquire)){
            // exit cleanly
            break;
        }

        // CONSUME EVENT SLOT = set flag back to false while holding mtx
        // reset flag for next time
        stateStoreRef.train_job_requested = false;
        // unlock mtx with std::unique_lock's 'unlock' function
        lock.unlock(); // unlock for heavy work

        // fwd declare vars used by both code paths (regardless of SKIP_TRAINING)
        std::string subject_id;
        std::string session_id;
#if !SKIP_TRAINING
        // what happens when it wakes up:
        // (1) Snapshot session info (paths/ids)
        std::string data_dir, model_dir;
        {
            std::lock_guard<std::mutex> sLk(stateStoreRef.currentSessionInfo.mtx_);
            data_dir   = stateStoreRef.currentSessionInfo.g_active_data_path;
            model_dir  = stateStoreRef.currentSessionInfo.g_active_model_path;
            subject_id = stateStoreRef.currentSessionInfo.g_active_subject_id;
            session_id = stateStoreRef.currentSessionInfo.g_active_session_id;
            // Mark "not ready" while training
            stateStoreRef.currentSessionInfo.g_isModelReady.store(false, std::memory_order_release);
        }
        // poll current settings to pass to Python
        SettingTrainArch_E train_arch = stateStoreRef.settings.train_arch_setting.load(std::memory_order_acquire);
        SettingCalibData_E calib_data = stateStoreRef.settings.calib_data_setting.load(std::memory_order_acquire);
        SettingHparam_E hparam = stateStoreRef.settings.hparam_setting.load(std::memory_order_acquire);
        std::string arch_str  = TrainArchEnumToString(train_arch);
        std::string cdata_str = CalibDataEnumToString(calib_data);
        std::string hparam_str = HParamEnumToString(hparam);
        LOG_ALWAYS("Training settings snapshot: train_arch=" << TrainArchEnumToString(train_arch)
          << ", calib_data=" << CalibDataEnumToString(calib_data)
          << ", hparam=" << HParamEnumToString(hparam));

        // (2) Validate inputs (don’t launch if missing)
        if (data_dir.empty() || model_dir.empty() || subject_id.empty() || session_id.empty() || arch_str == "Unknown" || cdata_str == "Unknown" || hparam_str == "Unknown") {
            write_fail_to_statestore("Backend Issue: Training request from consumer missing session info (had to skip).");
            continue;
        }

        // Ensure model directory exists (script writes outputs here)
        {
            std::error_code ec;
            fs::create_directories(fs::path(model_dir), ec);
            if (ec) {
                write_fail_to_statestore("Backend Issue: could not create model_dir= " + model_dir);
                continue;
            }
        }
        
        // (3) Launch training script (should block here)
        std::stringstream ss;
        ss << "python "
               << "\"" << scriptPath.string() << "\""
               << " --data \""               << data_dir   << "\""
               << " --model \""              << model_dir  << "\""
               << " --arch \""               << arch_str   << "\""
               << " --calibsetting \""       << cdata_str  << "\""
               << " --tunehparams \""        << hparam_str << "\""
               << " --zscorenormalization \""<< "ON"       << "\"";

        const std::string cmd = ss.str();
        LOG_ALWAYS("Launching training: " << cmd);
        int rc = std::system(cmd.c_str());
#else
        constexpr const char* kPretrainedPath = SKIP_TRAINING_MODEL_PATH;
        fs::path model_dir = projectRoot.parent_path() / "models" / kPretrainedPath;
#endif

        // (4) parse json result to update ui, update statestore
        std::string body = "";
#if !SKIP_TRAINING
        fs::path path_to_train_res = fs::path(model_dir) / "train_result.json";
#else
        fs::path path_to_train_res = model_dir / "train_result.json";
#endif
        int best_freq_left_hz;
        int best_freq_right_hz;
        int freq_left_hz_e = TestFreq_None;
        int freq_right_hz_e = TestFreq_None;
        bool train_ok = false;
        bool onnx_ok = false;
        bool cv_ok = false;
        bool final_holdout_ok = false;
        double final_holdout_acc = 0.0;
        double final_train_acc = 0.0;
        std::vector<DataInsufficiency_s> insuff{};
        std::unordered_set<std::string> seen;
        std::vector<TrainingIssue_s> issues{};
        try {
            JSON::read_file_to_string(path_to_train_res, body);
            JSON::extract_json_int(body,    "best_freq_left_hz",   best_freq_left_hz);
            JSON::extract_json_int(body,    "best_freq_right_hz",  best_freq_right_hz);
            JSON::extract_json_int(body,    "best_freq_left_e",    freq_left_hz_e);
            JSON::extract_json_int(body,    "best_freq_right_e",   freq_right_hz_e);
            JSON::extract_json_bool(body,   "train_ok",            train_ok);
            JSON::extract_json_bool(body,   "cv_ok",               cv_ok);
            JSON::extract_json_bool(body,   "onnx_ok",             onnx_ok);
            JSON::extract_json_bool(body,   "final_holdout_ok",    final_holdout_ok);
            JSON::extract_json_double(body, "final_holdout_acc",   final_holdout_acc);
            JSON::extract_json_double(body, "final_train_acc",     final_train_acc);
            nlohmann::json j = nlohmann::json::parse(body);
            collect_data_insufficiency(collect_data_insufficiency, j, insuff, seen);
            collect_training_issues(j, issues);
        } catch(const std::exception& e) {
            write_fail_to_statestore("Failed to parse training results: " + std::string(e.what()), {});
            continue;
        }
        // TODO: switch msg in ui based on freq and metric (e.g. not enough windows per trial -> do more unique trials)
        
        if(train_ok == false){
            write_fail_to_statestore("Python training module failed.", issues);
            continue;
        }
        if(onnx_ok == false){
            write_fail_to_statestore("Python training module failed at ONNX export stage.", issues);
            continue;
        }
        if(cv_ok == false){
            write_fail_to_statestore("Python training module failed at pairwise cross-validation stage.", issues);
            continue;
        }
        if(final_holdout_ok == false){
            write_fail_to_statestore("Python training module faild at holdout (final model split) stage.", issues);
            continue;
        }
#if !SKIP_TRAINING
        if (rc!=0) {
            write_fail_to_statestore("Unknown error occured during Python training", issues);
            continue;
        }
#endif

        // signal to stim controller that model is ready if we made it past all these checks
        {
            std::lock_guard<std::mutex> lock3(stateStoreRef.mtx_model_ready);
            stateStoreRef.model_just_ready = true;
        }
#if !SKIP_TRAINING
        // Add to saved sessions list so UI can pick it later
        StateStore_s::SavedSession_s s;
        s.subject   = subject_id;
        s.session   = session_id;
        s.id        = subject_id + "_" + session_id;
        s.label     = session_id;
        s.model_dir = model_dir;
        s.model_arch = arch_str;
        s.data_insuff = insuff;
        s.freq_left_hz = best_freq_left_hz;
        s.freq_right_hz = best_freq_right_hz;
        s.freq_left_hz_e = static_cast<TestFreq_E>(freq_left_hz_e);
        s.freq_right_hz_e = static_cast<TestFreq_E>(freq_right_hz_e);
        s.final_holdout_acc = final_holdout_acc;
        s.final_train_acc = final_train_acc;
#else
        StateStore_s::SavedSession_s s;
        s.subject   = "DEBUG_SESSION";
        s.session   = "DEBUG_SESSION";
        s.id        = "DEBUG_SESSION";
        s.label     = "DEBUG_SESSION";
        s.model_dir = model_dir.string();
        s.model_arch = "";
        s.data_insuff = insuff;
        s.freq_left_hz = best_freq_left_hz;
        s.freq_right_hz = best_freq_right_hz;
        s.freq_left_hz_e = static_cast<TestFreq_E>(freq_left_hz_e);
        s.freq_right_hz_e = static_cast<TestFreq_E>(freq_right_hz_e);
        s.final_holdout_acc = final_holdout_acc;
        s.final_train_acc = final_train_acc;
#endif
        int lastIdx = 0;
        {
            // add saved session: blocks until mutex is available for acquiring
            std::unique_lock<std::mutex> lock(stateStoreRef.saved_sessions_mutex);
            stateStoreRef.saved_sessions.push_back(s);
            // set idx to it
            lastIdx = static_cast<int>(stateStoreRef.saved_sessions.size() - 1);
        }
        stateStoreRef.currentSessionIdx.store(lastIdx, std::memory_order_release);
        stateStoreRef.currentSessionInfo.g_isModelReady.store(true, std::memory_order_release);
        LOG_ALWAYS("Training SUCCESS.");

        // TODO: CREATE/WRITE TO JSON.META SO WE REMEMBER SESSIONS BTWN BROWSER ON/OFF
        // ALSO CREATE/WRITE-APPEND to global static json that keeps track of all saved sess btwn browser on/off
        // must be written atomically bcuz other threads can access/interrupt ->mutex in statestore or atomics?
        // this json will exist in top lvel of model_dir
        // 2 mechanisms of protection:
        // 1) global_session_list_json_mtx to prevent concurrent reads/writes
        // 2) name tmp initially and only rename to proper once full file has been written (in case of crashes, power offs, etc)

#if !SKIP_TRAINING
        nlohmann::json j_sess;
        j_sess["sess_idx"] = lastIdx;
        j_sess["subject"] = subject_id;
        j_sess["session_id"] = subject_id + "_" + session_id;
        j_sess["model_arch"] = arch_str;
        j_sess["model_holdout_acc"] = final_holdout_acc;
        j_sess["freq_left_hz"] = best_freq_left_hz;
        j_sess["freq_right_hz"] = best_freq_right_hz;
        j_sess["data_insufficiency"] = (insuff.empty()) ? false : true; 
        std::string meta_path_str = model_dir + "/meta.json";
        std::filesystem::path meta_path = std::filesystem::path(meta_path_str);
        // grab individual sessions mtx while writing meta.json (easier to access than trainresult.json)
        {
            // there should be an onnx assoc w/ this now in same model dir
            std::lock_guard<std::mutex> lock_sess_again(stateStoreRef.saved_sessions_mutex);
            // write out j_sess to model_dir/"meta.json"
            try {
                JSON::write_json_atomic(meta_path, j_sess);
                LOG_ALWAYS("Wrote meta.json to " << meta_path.string());
            }
            catch (const std::exception& e){
                LOG_ALWAYS("Failed writing meta.json: " << e.what());
                continue;
            }
        }

        // STRUCTURE FOR ALL SESS JSON: 
        /* {"meta_paths": []}
        */
        nlohmann::json j_all_sess;
        // make all sess path
        std::filesystem::path all_sess_dir = model_dir;
        for(int i = 0; i < 2; i++){
            // should be 2 parents up to be in models/
            auto p = all_sess_dir.parent_path();
            if(p == all_sess_dir || p.empty()) break; // highest parent
            all_sess_dir = p;
        }
        std::filesystem::path j_all_sess_path = all_sess_dir / "all_sessions.json";

        // metas will be stored as relative paths to meta.json from models/
        fs::path meta_path_abs = fs::path(model_dir) / "meta.json";
        std::error_code ec;
        fs::path meta_path_rel = fs::relative(meta_path_abs, all_sess_dir, ec);
        if(ec){
            meta_path_rel = meta_path_abs; // store as abs as fallback
        }
        // grab global mtx while writing full list json
        // save as 
        {
            std::lock_guard<std::mutex> disk_lock(stateStoreRef.global_session_list_json_mtx);
            // overwrite file if exists w/ current + new sess, else create new one
            nlohmann::json j = JSON::read_json_if_exists(j_all_sess_path);
            if(!j.is_object()) j = nlohmann::json::object();
            if (!j.contains("meta_paths") || !j["meta_paths"].is_array()){
                j["meta_paths"] = nlohmann::json::array(); // desired structure must be satisfied
            }
            j["meta_paths"].push_back(meta_path_rel);

            try {
                JSON::write_json_atomic(j_all_sess_path, j);
            } catch (const std::exception& e){
                LOG_ALWAYS("Failed writing session index: " << e.what());
            }
        }
        
#endif
    }
}

void actuation_controller_thread_fn(StateStore_s& stateStoreRef){
    // TODO: turn their Python code into C++ here
    // implement cv to notify from consumer when ssvep signal is detected
    // send torque commands as needed 
    // do this by implementing custom driver for servos that exposes api to turn one way or the other
    // ^^w guards
    // OR could call their python script directly w the appropriate 'turn_left' or 'turn_right' args...,
    // but this would introduce a lot of latency overheads & is def not optimal soln
    ServoDriver_C ServoDriver;

    while(!g_stop.load(std::memory_order_acquire)){
        // (1) continuously wait for cv request from consumer thread -> then wake up
        std::unique_lock<std::mutex> actuation_lock(stateStoreRef.mtx_actuation_request); // acquire lock momentarily
        stateStoreRef.actuation_request.wait(actuation_lock, [&]{return stateStoreRef.actuation_requested;}); // go to sleep until notified -> then reacquire lock and lambda fxn[&] to check predicate (arg2)
        SSVEPState_E requested_direction = stateStoreRef.actuation_direction;
        
        // (2) Implement requested_direction cycle using ServoDriver API

        // sleep (fixed rate frequency)
    }

}

int main() {
    LOG_ALWAYS("start (VERBOSE=" << logger::verbose() << ")");

    // Shared singletons/objects
    RingBuffer_C<bufferChunk_S> ringBuf(ACQ_RING_BUFFER_CAPACITY);
    StateStore_s stateStore;
    HttpServer_C server(stateStore, 7777);
    server.http_start_server();

    // init stateStore defaults if nothing else is set
    for (int i = 0; i < NUM_CH_CHUNK; i++) {
        stateStore.eeg_channel_labels[i] = "Ch" + std::to_string(i + 1);
        stateStore.eeg_channel_enabled[i] = true;
    }

    // interrupt caused by SIGINT -> 'handle_singint' acts like ISR (callback handle)
    std::signal(SIGINT, handle_sigint);

    // START THREADS. We pass the ring buffer by reference (std::ref) becauase each thread needs the actual shared 'ringBuf' instance, not just a copy...
    // This builds a new thread that starts executing immediately, running producer_thread_rn in parallel with the main thread (same for cons)
    std::thread prod(producer_thread_fn,std::ref(ringBuf), std::ref(stateStore));
    std::thread cons(consumer_thread_fn,std::ref(ringBuf), std::ref(stateStore));
    std::thread http(http_thread_fn, std::ref(server));
    std::thread stim(stimulus_thread_fn, std::ref(stateStore));
    std::thread train(training_manager_thread_fn, std::ref(stateStore));
    std::thread actuate(actuation_controller_thread_fn, std::ref(stateStore));

    // Poll the atomic flag g_stop; keep sleep tiny so Ctrl-C feels instant
    while(g_stop.load(std::memory_order_acquire) == 0){
        std::this_thread::sleep_for(std::chrono::milliseconds{30});
    }

    // on system shutdown:
    // ctrl+c called...

    // is this sufficient to handle cv closing idk??
    stateStore.cv_train_job_request.notify_all();
    stateStore.cv_finalize_request.notify_all();

    ringBuf.close();
    server.http_close_server();
    prod.join();
    cons.join(); // close "join" individual threads
    http.join();
    stim.join();
    train.join();
    actuate.join();
    return 0; 
}