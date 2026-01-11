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
#include "utils/SessionPaths.hpp"

#ifdef USE_EEG_FILTERS
#include "utils/Filters.hpp"
#endif

#ifdef ACQ_BACKEND_FAKE
#include "acq/FakeAcquisition.h"
#endif

constexpr bool TEST_MODE = 1;

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

try{
    SignalQualityAnalyzer_C SignalQualityAnalyzer(&stateStoreRef);

    sliding_window_t window; // should acquire the data for 1 window with that many pops n then increment by hop... 
    bufferChunk_S temp; // placeholder

    namespace fs = std::filesystem;

    // Two independent CSV files, one at window level, one at chunk level
    std::ofstream csv_chunk;   // eeg_chunk_data.csv
    std::ofstream csv_win;     // eeg_windows.csv
    bool chunk_opened = false;
    bool win_opened   = false;
    size_t rows_written_chunk = 0;
    size_t rows_written_win   = 0;

    // Track which session these files belong to so we can reopen when session changes
    std::string active_session_id;
    std::string active_data_dir;
    
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

        LOG_ALWAYS("consumer: switched logging session to "
                   << "session_id=" << active_session_id
                   << " data_dir=" << active_data_dir);

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
        if (!ensure_csv_open_window()) return;

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

        if ((rows_written_win % 5000) == 0) csv_win.flush();
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

	// build first window
	while(window.sliding_window.get_count()<window.winLen){
		// sc
		if(!rb.pop(&temp)){ // internally wait here (pop cmd is blocking)
			break;
		} 
		else { 
			// this will fit fine because winLen is a multiple of number of scans per channel = 32
			// pop sucessful -> push into sliding window
			for(int i = 0; i<NUM_SAMPLES_CHUNK;i++){
				window.sliding_window.push(temp.data[i]);
			}
		}
	}
    
    UIState_E currState  = UIState_None;
    UIState_E prevState = UIState_None;
    TestFreq_E currLabel  = TestFreq_None;
    TestFreq_E prevLabel = TestFreq_None;

    while(!g_stop.load(std::memory_order_relaxed)){
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
        // save this as prev state to check after window is built to make sure UI state hasn't changed in between
        prevState = currState;
        prevLabel = currLabel;
        
        // 2) ============================= build the new window =============================
        float discard; // first pop
        for(size_t k=0;k<window.winHop;k++){
            window.sliding_window.pop(&discard); 
        }

        while(window.sliding_window.get_count()<window.winLen){ // now push
            UIState_E intState = stateStoreRef.g_ui_state.load(std::memory_order_acquire);
            TestFreq_E intLabel = stateStoreRef.g_freq_hz_e.load(std::memory_order_acquire);
            if((intState != prevState) || (intLabel != prevLabel)){
                break; // change in UI; not a good window
            }
			std::size_t amnt_left_to_add = window.winLen - window.sliding_window.get_count(); // in samples
			// if there is previous 'len' in stash, we should take it and decrement len
            if (window.stash_len > 0) {
                // take full amnt_left_to_add from stash if it's available, otherwise take window.stash_len
                const std::size_t take = (window.stash_len > amnt_left_to_add) ? amnt_left_to_add : window.stash_len;
                for (std::size_t i = 0; i < take; ++i){
                    window.sliding_window.push(window.stash[i]);
				}
                // move leftover stash to front of array for next round
                if (take < window.stash_len) {
                    std::memmove(window.stash.data(), // start of stash array (dest)
                                 window.stash.data() + take, // ptr to where remaining data starts (src)
                                 (window.stash_len - take) * sizeof(float)); // number of bytes to move
                }
                window.stash_len -= take; // how much we lost for next time; will never be <0

                continue; // go check while-condition again
            }
            // stash is empty
            if(window.stash_len != 0){
                LOG_ALWAYS("There's an issue with sliding window stash.");
                break;
            }
			if(!rb.pop(&temp)){
				break;
			} else {
				// pop successful -> push into sliding window
				if(amnt_left_to_add >= NUM_SAMPLES_CHUNK){
                    for(std::size_t j=0;j<NUM_SAMPLES_CHUNK;j++){
                        window.sliding_window.push(temp.data[j]);
                    } // goes back to check while for next chunk
				}
				else {
					// take what we need and stash the rest for next window
					for(std::size_t j=0;j<NUM_SAMPLES_CHUNK;j++){
						if(j<amnt_left_to_add){
							window.sliding_window.push(temp.data[j]);
						} else {
							window.stash[j-amnt_left_to_add]=temp.data[j];
                            window.stash_len++; // increasing slots to add from stash for next time
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
            window.decision = SSVEP_Unknown;
            window.has_label = false;
            continue; // go build next window
        }

        // verified it's an ok window
        ++tick_count;
        window.tick=tick_count;

        // reset before looking at state store vals
        window.isTrimmed = false;
        window.has_label = false;
        window.testFreq = TestFreq_None;

        // always check artifacts and flag bad windows
        SignalQualityAnalyzer.check_artifact_and_flag_window(window);

        if(currState == UIState_Active_Calib || currState == UIState_NoSSVEP_Test) {
            if (!ensure_csv_open_window()) {
                continue; // must be open for logging
            }

            ++tick_count_per_session; // these are the ticks we log in the calib data file
            
            int n_ch_local = stateStoreRef.g_n_eeg_channels.load(std::memory_order_acquire);
            if (n_ch_local <= 0 || n_ch_local > NUM_CH_CHUNK) n_ch_local = NUM_CH_CHUNK;
            
            // trim window ends for training data (GUARD)
            window.trimmed_window.clear();
            window.sliding_window.get_trimmed_snapshot(window.trimmed_window,
                40 * n_ch_local, 40 * n_ch_local);
            window.isTrimmed = true;

            // we should be attaching a label to our windows for calibration data
            window.testFreq = currLabel;
            window.has_label = (currLabel != TestFreq_None);
            // Log trimmed window (only if has label)
            if(window.has_label){
                log_window_snapshot(window, currState, tick_count_per_session, /*use_trimmed=*/true);
            }
        }
        
        else if(currState == UIState_Active_Run){
            // run ftr extraction + classifier pipeline to get decision
            // TODO WHEN READY: add ftr vector + make decision here for run mode
            
            // TODO: NEEDS TESTING IN RUN MODE (BCUZ WE HAVENT IMPLEMENTED THIS MODE YET)

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

            if(window.isArtifactualWindow){
                if(!run_mode_bad_window_timer.is_started()){
                    run_mode_bad_window_timer.start_timer(std::chrono::milliseconds{9000});
                }
                run_mode_bad_window_count++;
                continue; // don't use this window
            } else {
                // clean window
                if(run_mode_bad_window_timer.is_started()){
                    // add to within-timer clean window count for comparison
                    run_mode_clean_window_count++;
                }
            }
        }
        
	}
    // exiting due to producer exiting means we need to close window rb
    window.sliding_window.close();
    rb.close();
    if (chunk_opened) { csv_chunk.flush(); csv_chunk.close(); }
    if (win_opened)   { csv_win.flush();   csv_win.close();   }
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

/* HADEEL, THE SCRIPT MUST OUTPUT
(1) ONNX MODELS
(2) BEST TWO FREQUENCIES TO USE (HIGHEST SNR FOR THIS PERSON)
--> Script must output to <model_dir>/train_result.json
*/
void training_manager_thread_fn(StateStore_s& stateStoreRef){
    logger::tlabel = "training manager";
    namespace fs = std::filesystem;
    fs::path projectRoot = sesspaths::find_project_root();
    if (fs::exists(projectRoot / "CapstoneProject") && fs::is_directory(projectRoot / "CapstoneProject")) {
        projectRoot /= "CapstoneProject";
    }
    // Script lives at: <CapstoneProject>/model train/python/train_svm.py
    fs::path scriptPath = projectRoot / "model train" / "python" / "train_svm.py";

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

        // what happens when it wakes up:
        // (1) Snapshot session info (paths/ids)
        std::string data_dir, model_dir, subject_id, session_id;
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
        std::string arch_str  = TrainArchEnumToString(train_arch);
        std::string cdata_str = CalibDataEnumToString(calib_data);
        LOG_ALWAYS("Training settings snapshot: train_arch=" << TrainArchEnumToString(train_arch)
          << ", calib_data=" << CalibDataEnumToString(calib_data));

        // (2) Validate inputs (don’t launch if missing)
        if (data_dir.empty() || model_dir.empty() || subject_id.empty() || session_id.empty() || arch_str == "Unknown" || cdata_str == "Unknown") {
            LOG_ALWAYS("Training request missing session info; skipping.");
            continue;
        }

        // Ensure model directory exists (script writes outputs here)
        {
            std::error_code ec;
            fs::create_directories(fs::path(model_dir), ec);
            if (ec) {
                LOG_ALWAYS("ERROR: could not create model_dir=" << model_dir
                          << " (" << ec.message() << ")");
                continue;
            }
        }
        
        // (3) Launch training script (should block here)
        std::stringstream ss;
        
        // TODO: MUST MATCH PYTHON TRAINING SCRIPT PATH AND ARGS
        ss << "python "
               << "\"" << scriptPath.string() << "\""
               << " --data \""     << data_dir   << "\""
               << " --model \""    << model_dir  << "\""
               << " --subject \""  << subject_id << "\""
               << " --session \""  << session_id << "\""
               << " --arch \""     << arch_str   << "\""
               << " --calibsetting \""<< cdata_str  << "\"";

        const std::string cmd = ss.str();
        /* std::system executes the cmd string using host shell
        -> it BLOCKS "this" background thread until the Python script finishes
        -> rc is the exit code of the cmd
        */
        LOG_ALWAYS("Launching training: " << cmd);
        int rc = std::system(cmd.c_str());

        // TODO: parse json result to write best freqs to state store


        //(4) Publich result to state store
        if (rc == 0) {
            // signal to stim controller that model is ready
            {
                std::lock_guard<std::mutex> lock3(stateStoreRef.mtx_model_ready);
                stateStoreRef.model_just_ready = true;
            }

            stateStoreRef.currentSessionInfo.g_isModelReady.store(true, std::memory_order_release);

            // Add to saved sessions list so UI can pick it later
            StateStore_s::SavedSession_s s;
            s.subject   = subject_id;
            s.session   = session_id;
            s.id        = subject_id + "_" + session_id;
            s.label     = session_id; // TODO: make better
            s.model_dir = model_dir;
            // TODO: PARSE JSON AND SET THE FREQS PROPERLY
            // temporary for now:
            s.freq_left_hz = 10;
            s.freq_right_hz = 12;
            s.freq_left_hz_e = TestFreq_10_Hz;
            s.freq_right_hz_e = TestFreq_12_Hz;

            int lastIdx = 0;
            {
                // add saved session: blocks until mutex is available for acquiring
                std::unique_lock<std::mutex> lock(stateStoreRef.saved_sessions_mutex);
                stateStoreRef.saved_sessions.push_back(s);
                // set idx to it
                lastIdx = static_cast<int>(stateStoreRef.saved_sessions.size() - 1);
            }

            stateStoreRef.currentSessionIdx.store(lastIdx, std::memory_order_release);
            LOG_ALWAYS("Training SUCCESS.");

        } else {
            stateStoreRef.currentSessionInfo.g_isModelReady.store(false, std::memory_order_release);
            LOG_ALWAYS("Training job failed (rc=" << rc << ")");
            // TODO: FAULT HANDLING... TELL STIM CONTROLLER WERE FAULTED AND RETURN TO HOME WITH POPUP
            stateStoreRef.g_ui_event.store(UIStateEvent_TrainingFailed);
        }
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
    return 0; 
}