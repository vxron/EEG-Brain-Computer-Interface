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
    size_t tick_count = 0;
    size_t run_mode_bad_windows = 0;
    size_t run_mode_bad_window_count = 0;
    size_t run_mode_clean_window_count = 0;
    SW_Timer_C run_mode_bad_window_timer;

try{
    SignalQualityAnalyzer_C SignalQualityAnalyzer(&stateStoreRef);

    sliding_window_t window; // should acquire the data for 1 window with that many pops n then increment by hop... 
    bufferChunk_S temp; // placeholder

    // init csv: only need to produce csv in calibration mode for model training (training dataset)
    std::ofstream csv;
    bool csv_opened = false;
    size_t rows_written = 0;
    // lambda helper for ensuring csv is open when trying to log
    
    // TODO -> FIX LOGGING TO NOT HARDCODE NUM_CH_CHUNK
    auto ensure_csv_open = [&]() {
        if (csv_opened) return true;
        csv.open("eeg_calib_data.csv", std::ios::out | std::ios::trunc);
        if (!csv.is_open()) {
            LOG_ALWAYS("ERROR: failed to open eeg_calib_data.csv");
            return false;
        }
        csv << "chunk_tick,sample_idx";
        for (std::size_t ch = 0; ch < NUM_CH_CHUNK; ++ch) {
            csv << ",eeg" << (ch + 1);
        }
        csv << ",testfreq_e,testfreq_hz\n";
        csv_opened = true;
        return true;
    };
    // ADD LOGGER AT TRIMMED WINDOW LEVEL FOR CALIB!
    auto ensure_csv_open_window = [&]() {
        if (csv_opened) return true;

        csv.open("eeg_windows.csv", std::ios::out | std::ios::trunc);
        if (!csv.is_open()) {
            LOG_ALWAYS("ERROR: failed to open eeg_windows.csv");
            return false;
        }

        int n_ch_local = stateStoreRef.g_n_eeg_channels.load(std::memory_order_acquire);
        if (n_ch_local <= 0 || n_ch_local > NUM_CH_CHUNK) n_ch_local = NUM_CH_CHUNK;

        csv << "window_idx,ui_state,is_trimmed,is_bad,sample_idx";
        for (int ch = 0; ch < n_ch_local; ++ch) csv << ",eeg" << (ch + 1);
        csv << ",testfreq_e,testfreq_hz\n";

        csv_opened = true;
        LOG_ALWAYS("opened eeg_windows.csv");
        return true;
    };

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
            csv << window_idx
                << "," << static_cast<int>(uiState)
                << "," << (use_trimmed && w.isTrimmed ? 1 : 0)
                << "," << (w.isArtifactualWindow ? 1 : 0)
                << "," << s;

            const std::size_t base = s * static_cast<std::size_t>(n_ch_local);
            for (int ch = 0; ch < n_ch_local; ++ch) {
                csv << "," << buf[base + static_cast<std::size_t>(ch)];
            }

            csv << "," << tf_e << "," << tf_hz << "\n";
            ++rows_written;
        }

        if ((rows_written % 5000) == 0) csv.flush();
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

        if(currState == UIState_Active_Calib ) {
            int n_ch_local = stateStoreRef.g_n_eeg_channels.load(std::memory_order_acquire);
            if (n_ch_local <= 0 || n_ch_local > NUM_CH_CHUNK) n_ch_local = NUM_CH_CHUNK;
            
            SignalQualityAnalyzer.check_artifact_and_flag_window(window);

            // trim window ends if its calibration mode (GUARD)
            window.trimmed_window.clear();
            window.sliding_window.get_trimmed_snapshot(window.trimmed_window,
                40 * n_ch_local, 40 * n_ch_local);
            window.isTrimmed = true;

            // we should be attaching a label to our windows for calibration data
            window.testFreq = currLabel;
            window.has_label = (currLabel != TestFreq_None);
            // Log trimmed window (only if has label)
            if(window.has_label){
                log_window_snapshot(window, currState, window.tick, /*use_trimmed=*/true);
            }
        }

        else if(currState == UIState_Hardware_Checks) {
            // MOVE THIS OUT -> WE SHOULD ALWAYS BE CHECKING THIS MAYBE??
            SignalQualityAnalyzer.check_artifact_and_flag_window(window);
            // Log EVERY window from HW checks; use snapshot (untrimmed)
            // testFreq will be None -> tf_hz becomes -1
            log_window_snapshot(window, currState, window.tick, /*use_trimmed=*/false);
        }
        
        else if(currState == UIState_Active_Run){
            // run ftr extraction + classifier pipeline to get decision
            // TODO WHEN READY: add ftr vector + make decision here for run mode
            
            // TODO: NEEDS TESTING IN RUN MODE (BCUZ WE HAVENT IMPLEMENTED THIS MODE YET)
            SignalQualityAnalyzer.check_artifact_and_flag_window(window);
            // popup saying 'signal is bad, too many artifactual windows. run hardware checks' when too many bad windows detected in a certain time frame, then reset
            if(run_mode_bad_window_timer.check_timer_expired()){
                // expired -> see if we should throw popup based on bad window counts in the 9s timeout period
                if(run_mode_bad_window_count / run_mode_clean_window_count >= 0.25) { // require 4:1 good:bad ratio
                    // DO POPUP
                    stateStoreRef.g_ui_popup.store(UIPopup_TooManyBadWindowsInRun, std::memory_order_release);
                }
                // reset for next round
                run_mode_bad_window_count = 0;
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
    if(csv_opened){ // calib sess
        csv.flush();
        csv.close();
    }
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
    catch (const std::exception& e) {
        LOG_ALWAYS("stim: FATAL unhandled exception: " << e.what());
        g_stop.store(true, std::memory_order_relaxed);
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

    // Poll the atomic flag g_stop; keep sleep tiny so Ctrl-C feels instant
    while(g_stop.load(std::memory_order_acquire) == 0){
        std::this_thread::sleep_for(std::chrono::milliseconds{30});
    }

    // on system shutdown:
    // ctrl+c called...
    ringBuf.close();
    server.http_close_server();
    prod.join();
    cons.join(); // close "join" individual threads
    http.join();
    stim.join();
    return 0; 
}