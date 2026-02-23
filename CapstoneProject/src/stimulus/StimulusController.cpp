#include "StimulusController.hpp"
#include <thread>
#include "../utils/Logger.hpp"
#include "../utils/SessionPaths.hpp"
#include "utils/json.hpp"

struct state_transition{
    UIState_E from;
    UIStateEvent_E event;
    UIState_E to;
};

static const state_transition state_transition_table[] = {
    // from                           event                                     to
    {UIState_None,             UIStateEvent_ConnectionSuccessful,           UIState_Home},
 
    {UIState_Home,             UIStateEvent_UserPushesStartCalib,           UIState_Calib_Options},
    {UIState_Calib_Options,    UIStateEvent_UserPushesStartCalibFromOptions,UIState_NoSSVEP_Test},
    {UIState_Home,             UIStateEvent_UserPushesStartRun,             UIState_Run_Options},
    {UIState_Home,             UIStateEvent_UserPushesHardwareChecks,       UIState_Hardware_Checks},
    {UIState_Home,             UIStateEvent_UserPushesSettings,             UIState_Settings},
     
    {UIState_Instructions,     UIStateEvent_StimControllerTimeout,          UIState_Active_Calib},
    {UIState_Active_Calib,     UIStateEvent_StimControllerTimeout,          UIState_NoSSVEP_Test},
    {UIState_NoSSVEP_Test,     UIStateEvent_StimControllerTimeout,          UIState_Instructions},
    {UIState_Active_Calib,     UIStateEvent_StimControllerTimeoutEndCalib,  UIState_Pending_Training},
    {UIState_Pending_Training, UIStateEvent_ModelReady,                     UIState_Home},
    {UIState_Pending_Training, UIStateEvent_TrainingFailed,                 UIState_Home},
    
    {UIState_Active_Calib,     UIStateEvent_UserPushesExit,                 UIState_Home},
    {UIState_Calib_Options,    UIStateEvent_UserPushesExit,                 UIState_Home},
    {UIState_Instructions,     UIStateEvent_UserPushesExit,                 UIState_Home},
    {UIState_Active_Run,       UIStateEvent_UserPushesExit,                 UIState_Home},
    {UIState_Saved_Sessions,   UIStateEvent_UserPushesExit,                 UIState_Home},
    {UIState_Run_Options,      UIStateEvent_UserPushesExit,                 UIState_Home},
    {UIState_Hardware_Checks,  UIStateEvent_UserPushesExit,                 UIState_Home},
    {UIState_Pending_Training, UIStateEvent_UserPushesExit,                 UIState_Home},
    {UIState_Settings,         UIStateEvent_UserPushesExit,                 UIState_Home},
    {UIState_NoSSVEP_Test,     UIStateEvent_UserPushesExit,                 UIState_Home},
    {UIState_Paused,           UIStateEvent_UserPushesExit,                 UIState_Home},
    
    {UIState_Active_Calib,     UIStateEvent_UserPushesPause,                UIState_Paused},  
    {UIState_Instructions,     UIStateEvent_UserPushesPause,                UIState_Paused},
    {UIState_NoSSVEP_Test,     UIStateEvent_UserPushesPause,                UIState_Paused},
    {UIState_Active_Run,       UIStateEvent_UserPushesPause,                UIState_Paused},
    // return states from state_paused must be specially handled in the detect_event function since they are dynamic
 
    {UIState_Run_Options,      UIStateEvent_UserPushesSessions,             UIState_Saved_Sessions},
    {UIState_Saved_Sessions,   UIStateEvent_UserSelectsSession,             UIState_Home},
    {UIState_Saved_Sessions,   UIStateEvent_UserSelectsNewSession,          UIState_Calib_Options},
    {UIState_Saved_Sessions,   UIStateEvent_UserPushesStartRun,             UIState_Run_Options},
    {UIState_Saved_Sessions,   UIStateEvent_UserPushesHardwareChecks,       UIState_Hardware_Checks},
    {UIState_Saved_Sessions,   UIStateEvent_UserPushesStartCalib,           UIState_Calib_Options},
    {UIState_Saved_Sessions,   UIStateEvent_UserPushesSettings,             UIState_Settings},
    {UIState_Run_Options,      UIStateEvent_UserPushesStartDefault,         UIState_Active_Run},
    {UIState_Run_Options,      UIStateEvent_UserPushesHardwareChecks,       UIState_Hardware_Checks},
    {UIState_Run_Options,      UIStateEvent_UserPushesStartCalib,           UIState_Calib_Options},
    {UIState_Run_Options,      UIStateEvent_UserPushesSettings,             UIState_Settings},

    // Non full screen pages where buttons remain visible (therefore transitions can happen)
    {UIState_Calib_Options,    UIStateEvent_UserPushesStartRun,             UIState_Run_Options},
    {UIState_Settings,         UIStateEvent_UserPushesStartRun,             UIState_Run_Options},
    {UIState_Calib_Options,    UIStateEvent_UserPushesSettings,             UIState_Settings},
    {UIState_Settings,         UIStateEvent_UserPushesStartCalib,           UIState_Calib_Options},
    {UIState_Calib_Options,    UIStateEvent_UserPushesHardwareChecks,       UIState_Hardware_Checks},
    {UIState_Settings,         UIStateEvent_UserPushesHardwareChecks,       UIState_Hardware_Checks},
};
// ^todo: add popup if switching: r u sure u want to exit???


StimulusController_C::StimulusController_C(StateStore_s* stateStoreRef) : state_(UIState_None), stateStoreRef_(stateStoreRef) {
    // default protocol
    trainingProtocol_.activeBlockDuration_s = 11;
    trainingProtocol_.displayInPairs = false;
    trainingProtocol_.freqsToTest = {TestFreq_8_Hz, TestFreq_11_Hz, TestFreq_14_Hz, TestFreq_17_Hz, TestFreq_20_Hz, 
                                     TestFreq_8_Hz, TestFreq_11_Hz, TestFreq_14_Hz, TestFreq_17_Hz, TestFreq_20_Hz}; // Two times
    trainingProtocol_.numActiveBlocks = trainingProtocol_.freqsToTest.size();
    trainingProtocol_.restDuration_s = 8;
    trainingProtocol_.noSSVEPDuration_s = 10; // interleaved after each testfreq 
    activeBlockQueue_ = trainingProtocol_.freqsToTest;
    activeBlockDur_ms_ = std::chrono::milliseconds{
    trainingProtocol_.activeBlockDuration_s * 1000 };
    restBlockDur_ms_ = std::chrono::milliseconds{
    trainingProtocol_.restDuration_s * 1000 };
    noSSVEPBlockDur_ms_ = std::chrono::milliseconds{trainingProtocol_.noSSVEPDuration_s * 1000};    
}

void StimulusController_C::rebuild_protocol_from_settings() {
  int n = stateStoreRef_->settings.selected_freqs_n.load(std::memory_order_acquire);
  n = std::clamp(n, 1, 6);
  int total_cycles = stateStoreRef_->settings.num_times_cycle_repeats.load(std::memory_order_acquire);
  int duration_active = stateStoreRef_->settings.duration_active_s.load(std::memory_order_acquire);
  int duration_rest = stateStoreRef_->settings.duration_rest_s.load(std::memory_order_acquire);
  int duration_none = stateStoreRef_->settings.duration_none_s.load(std::memory_order_acquire);

  trainingProtocol_.restDuration_s = duration_rest;
  trainingProtocol_.activeBlockDuration_s = duration_active;
  trainingProtocol_.noSSVEPDuration_s = duration_none;
  activeBlockDur_ms_  = std::chrono::milliseconds{ trainingProtocol_.activeBlockDuration_s * 1000 };
  restBlockDur_ms_    = std::chrono::milliseconds{ trainingProtocol_.restDuration_s * 1000 };
  noSSVEPBlockDur_ms_ = std::chrono::milliseconds{ trainingProtocol_.noSSVEPDuration_s * 1000 };

  std::unique_lock<std::mutex> lock(stateStoreRef_->settings.selected_freq_array_mtx);
  std::vector<TestFreq_E> pool;
  pool.reserve(n);
  for (int i=0;i<n;i++){
    auto e = stateStoreRef_->settings.selected_freqs_e[i];
    if (e != TestFreq_None) pool.push_back(e);
  }
  lock.unlock();
  if (pool.empty()) {
    return; // don't change default protocol from constructor
  };

  trainingProtocol_.freqsToTest.clear();
  // repeat series twice for better SNR per class
  for (int rep=0; rep<total_cycles; ++rep){
    trainingProtocol_.freqsToTest.insert(trainingProtocol_.freqsToTest.end(),
                                         pool.begin(), pool.end());
  }

  trainingProtocol_.numActiveBlocks = (int)trainingProtocol_.freqsToTest.size();
  activeBlockQueue_ = trainingProtocol_.freqsToTest;
}

std::chrono::milliseconds StimulusController_C::getCurrentBlockTime() const {
    if (currentWindowTimer_.is_started() == false) {
        auto time = std::chrono::milliseconds{0};
        return time;
    }
    auto time = currentWindowTimer_.get_timer_value_ms();
    return time;
}

void StimulusController_C::onStateEnter(UIState_E prevState, UIState_E newState, UIStateEvent_E ev){
    // placeholders for state store variables
    int currSeq = 0;
    int currId = 0;
    int freq = 0;
    TestFreq_E freqToTest = TestFreq_None;
    // first read seq atomically then increment (common to all state enters)
    currSeq = stateStoreRef_->g_ui_seq.load(std::memory_order_acquire);
    stateStoreRef_->g_ui_seq.store(currSeq + 1, std::memory_order_release);
    switch (newState) {
        case UIState_Active_Run: {
            // publish state first so consumer can enter reload block
            stateStoreRef_->g_ui_state.store(UIState_Active_Run, std::memory_order_release);
            stateStoreRef_->g_is_calib.store(false, std::memory_order_release);

            // may take some time since onnx can be huge
            guardAgainstInfLoopTimer_.start_timer(std::chrono::milliseconds{ 10000 });
            stateStoreRef_->g_onnx_session_is_reloading.store(-1, std::memory_order_release);
            while(!(stateStoreRef_->g_onnx_session_is_reloading.load(std::memory_order_acquire)==0)){
                // wait for any reload to be done (0)
                // timer guard to avoid inf loops
                if(guardAgainstInfLoopTimer_.check_timer_expired()){
                    LOG_ALWAYS("Timing Issue with g_onnx_session_is_reloading: Not being set to -1 deterministically from StimController, or not propagating to Consumer Thread.");
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds{5}); // sleep-wait to save CPU, since ONNX load takes 100s of ms anyway
            }
            guardAgainstInfLoopTimer_.stop_timer();

            // if it's in fake acq mode, we're gonna be publishing g_freq_hz for usage
            // need to 1) collect frequency pool, 2) setup timer/first frequency
            // nossvep (4s) -> ssvep1 
#ifdef ACQ_BACKEND_FAKE
            bool isDemoModeOn = stateStoreRef_->settings.demo_mode.load(std::memory_order_acquire);
            if(isDemoModeOn) {
                // must start demo acq (cv-notify)
                std::lock_guard<std::mutex> lock_mtx(stateStoreRef_->mtx_streaming_request);
                stateStoreRef_->streaming_requested = true;
                stateStoreRef_->test_mode_arg = 0; // run mode
                stateStoreRef_->streaming_request.notify_one(); // notifies producer
            }
            else {
                emulatedFreqsForFakeAcq_.push_back(-1); // -1 is no_ssvep
                // grab current models' frequencies
                {
                    std::lock_guard<std::mutex> mtx_lock(stateStoreRef_->saved_sessions_mutex);
                    int currIdx = stateStoreRef_->currentSessionIdx.load(std::memory_order_acquire);
                    emulatedFreqsForFakeAcq_.push_back(stateStoreRef_->saved_sessions[currIdx].freq_left_hz);
                    emulatedFreqsForFakeAcq_.push_back(stateStoreRef_->saved_sessions[currIdx].freq_right_hz);
                }
                // build sequence & startit
                fakeAcq_buildSeqAndShuffle();
                fakeAcq_advanceToNextSSVEP();
            }
           
#endif
            break;
        }
        
        case UIState_Home: {
            stateStoreRef_->g_ui_state.store(UIState_Home, std::memory_order_release);
            stateStoreRef_->g_is_calib.store(false, std::memory_order_release);
            // reset block/freq for clean home:
            stateStoreRef_->g_block_id.store(0, std::memory_order_release);
            stateStoreRef_->g_freq_hz.store(0, std::memory_order_release);
            stateStoreRef_->g_freq_hz_e.store(TestFreq_None, std::memory_order_release);
            // reset hardware page
            std::lock_guard<std::mutex> lock(stateStoreRef_->signal_stats_mtx);
            stateStoreRef_->SignalStats = SignalStats_s{}; //reset 

            // only transition from pending training is to home, so this covers all cases
            if(ev == UIStateEvent_TrainingFailed){
                // want to show popup saying training failed
                stateStoreRef_->g_ui_popup.store(UIPopup_TrainJobFailed);
            }

            // reset any timers that got weird/paused/etc
            if(currentWindowTimer_.is_paused() || currentWindowTimer_.is_started()) { currentWindowTimer_.stop_timer(); }

            if(prevState == UIState_None){
                // STARTUP CONDITIONS
                if (!stateStoreRef_->g_sessions_loaded_from_disk.exchange(true)){
                    // load all sessions
                    const bool ok = sesspaths::load_saved_sessions_from_disk(stateStoreRef_);
                    if (!ok){
                        LOG_ALWAYS("No saved sessions loaded");
                    }
                    else {
                        stateStoreRef_->currentSessionInfo.g_isModelReady.store(true, std::memory_order_release); // we have a model ready
                    }
                }
            }

            if(prevState == UIState_Saved_Sessions){
                // just switched sessions -> intrinsic guards mean the model must be ready.
                stateStoreRef_->currentSessionInfo.g_isModelReady.store(true, std::memory_order_release);
            }

            break;
        }
        
        case UIState_Active_Calib: {
            // stim window
            stateStoreRef_->g_ui_state.store(UIState_Active_Calib, std::memory_order_release);
            // first read block_id atomically then increment
            currId = stateStoreRef_->g_block_id.load(std::memory_order_acquire);
            stateStoreRef_->g_block_id.store(currId + 1,std::memory_order_release);
            // freqs
            freqToTest = activeBlockQueue_[activeQueueIdx_];
            stateStoreRef_->g_freq_hz_e.store(freqToTest, std::memory_order_release);
            // use helper
            freq =  TestFreqEnumToInt(freqToTest);
            stateStoreRef_->g_freq_hz.store(freq,std::memory_order_release);
            // iscalib helper
            stateStoreRef_->g_is_calib.store(true,std::memory_order_release);
            
            // increment queue idx so we move to next test freq on next block
            activeQueueIdx_++;

            // start timer
            currentWindowTimer_.start_timer(activeBlockDur_ms_);
            break;
        }

        case UIState_Calib_Options: {
            // RESET MEMBERS FOR NEW SESS
            end_calib_timeout_emitted_ = false;
            activeQueueIdx_ = 0;
            rebuild_protocol_from_settings();
            stateStoreRef_->g_ui_state.store(UIState_Calib_Options, std::memory_order_release);
            break;
        }

        case UIState_Instructions: {

            // stim window
            stateStoreRef_->g_ui_state.store(UIState_Instructions, std::memory_order_release);
            // instruction windows still get freq info for next active block cuz UI will tell user what freq they'll be seeing next
            freqToTest = activeBlockQueue_[activeQueueIdx_];
            freq =  TestFreqEnumToInt(freqToTest);
            
            // check next planned frequency is an int divisor of refresh
            int refresh = stateStoreRef_->g_refresh_hz.load(std::memory_order_acquire);
            int result = checkStimFreqIsIntDivisorOfRefresh(true, freq); 
            // if bad result
            if(result == -1 && has_divisor_6_to_20(refresh)){
                // we just log a warning but use the frequency anyway
                LOG_ALWAYS("SC: freq=" << freq << " Hz not optimal for " 
                   << refresh << " Hz refresh, but proceeding anyway");
            }
    
            // storing
            stateStoreRef_->g_freq_hz_e.store(freqToTest, std::memory_order_release);
            stateStoreRef_->g_freq_hz.store(freq, std::memory_order_release);
            LOG_ALWAYS("SC: stored a freq=" << static_cast<int>(freq));

            // start timer
            currentWindowTimer_.start_timer(restBlockDur_ms_);
            break;
        }

        case UIState_NoSSVEP_Test: {
            stateStoreRef_->g_ui_state.store(UIState_NoSSVEP_Test, std::memory_order_release);

            // storing
            stateStoreRef_->g_freq_hz_e.store(TestFreq_NoSSVEP, std::memory_order_release);
            stateStoreRef_->g_freq_hz.store(-1, std::memory_order_release);
            LOG_ALWAYS("SC: stored NoSSVEP label testfreq_e=" << (int)TestFreq_NoSSVEP << " testfreq_hz=-1");

            // no ssvep state is entered from Calib_Options or Saved_Sessions
            // need to create new session if we're just entering calib for the first time
            // TODO: delete csv log after if it doesn't include full calib session
            bool isCalib = stateStoreRef_->g_is_calib.load(std::memory_order_acquire);
            if(!isCalib){
                // first time entry

                // if from calib_options
                if(prevState == UIState_Calib_Options){
                    if(pending_epilepsy_ == EpilepsyRisk_YesButHighFreqOk) {
                        // need to adapt training protocol
                        trainingProtocol_.freqsToTest = {TestFreq_20_Hz, TestFreq_25_Hz, TestFreq_30_Hz, TestFreq_35_Hz};
                        activeBlockQueue_ = trainingProtocol_.freqsToTest;
                        trainingProtocol_.numActiveBlocks = trainingProtocol_.freqsToTest.size();
                        activeQueueIdx_ = 0;
                    } 
                }

#ifdef ACQ_BACKEND_FAKE
                bool isDemoModeOn = stateStoreRef_->settings.demo_mode.load(std::memory_order_acquire);
                if(isDemoModeOn) {
                    pending_subject_name_ = "DEMO";
                    pending_epilepsy_ = EpilepsyRisk_No;
                    std::lock_guard<std::mutex> lock_proto(stateStoreRef_->mtx_streaming_request);
                    stateStoreRef_->training_proto = trainingProtocol_;
                    stateStoreRef_->test_mode_arg = 1; // calib
                    stateStoreRef_->streaming_requested = true;
                    stateStoreRef_->streaming_request.notify_one(); // producer
                }
#endif 
                // new session publishing
                SessionPaths SessionPath;
                try {
                    SessionPath = sesspaths::create_session(pending_subject_name_);
                    // publish to stateStore...
                } catch (const std::exception& e) {
                    LOG_ALWAYS("SC: create_session failed: " << e.what());
                    // TODO: transition back to Home by injecting an event or setting state
                    // (don’t set g_stop)
                    return;
                }

                LOG_ALWAYS("SC: create_session used subject_name=" << pending_subject_name_);
            
                // lock again to write everything to state store; PUBLISH!
                // the new subject's model isn't ready yet
                {
                    std::lock_guard<std::mutex> lock(stateStoreRef_->currentSessionInfo.mtx_);
                    stateStoreRef_->currentSessionInfo.g_isModelReady.store(false, std::memory_order_release);
                    stateStoreRef_->currentSessionInfo.g_active_model_path = SessionPath.model_session_dir.string();
                    stateStoreRef_->currentSessionInfo.g_active_data_path = SessionPath.data_session_dir.string();
                    stateStoreRef_->currentSessionInfo.g_active_subject_id = SessionPath.subject_id;
                    stateStoreRef_->currentSessionInfo.g_active_session_id = SessionPath.session_id;
                    stateStoreRef_->currentSessionInfo.g_epilepsy_risk = pending_epilepsy_;
                }

                // clear pending entries we just used to create session
                pending_subject_name_.clear();
                pending_epilepsy_ = EpilepsyRisk_Unknown;
            }
            stateStoreRef_->g_is_calib.store(true,std::memory_order_release);

            // start timer
            currentWindowTimer_.start_timer(noSSVEPBlockDur_ms_);
            break;

        }

        case UIState_Paused: {
            stateStoreRef_->g_ui_state.store(UIState_Paused, std::memory_order_release);
            if(currentWindowTimer_.is_started()){
                currentWindowTimer_.pause_timer();
            }
            pausedFromState_ = prevState;
            break;
        }

        case UIState_Run_Options: {
            stateStoreRef_->g_ui_state.store(UIState_Run_Options, std::memory_order_release);
            stateStoreRef_->g_is_calib.store(false, std::memory_order_release);
            stateStoreRef_->g_block_id.store(0, std::memory_order_release);
            stateStoreRef_->g_freq_hz.store(0, std::memory_order_release);
            stateStoreRef_->g_freq_hz_e.store(TestFreq_None, std::memory_order_release);
            break;
        }

        case UIState_Saved_Sessions: {
            // TODO : LOAD ALL THE METAS FROM DISK (CAN DO FROM FRONTEND THO)
            stateStoreRef_->g_ui_state.store(UIState_Saved_Sessions, std::memory_order_release);
            stateStoreRef_->g_is_calib.store(false, std::memory_order_release);
            stateStoreRef_->g_block_id.store(0, std::memory_order_release);
            stateStoreRef_->g_freq_hz.store(0, std::memory_order_release);
            stateStoreRef_->g_freq_hz_e.store(TestFreq_None, std::memory_order_release);
            break;
        }

        case UIState_Hardware_Checks: {
            stateStoreRef_->g_ui_state.store(UIState_Hardware_Checks, std::memory_order_release);
            stateStoreRef_->g_is_calib.store(false, std::memory_order_release);
            stateStoreRef_->g_block_id.store(0, std::memory_order_release);
            stateStoreRef_->g_freq_hz.store(0, std::memory_order_release);
            stateStoreRef_->g_freq_hz_e.store(TestFreq_None, std::memory_order_release);
            break;
        }

        case UIState_Settings: {
            stateStoreRef_->g_ui_state.store(UIState_Settings, std::memory_order_release);
            stateStoreRef_->g_is_calib.store(false, std::memory_order_release);
            stateStoreRef_->g_block_id.store(0, std::memory_order_release);
            stateStoreRef_->g_freq_hz.store(0, std::memory_order_release);
            stateStoreRef_->g_freq_hz_e.store(TestFreq_None, std::memory_order_release);
            break;
        }

        case UIState_Pending_Training: {
            // start to display loading bar with "training completing, this may take several minutes..." until model ready event is detected
            stateStoreRef_->g_ui_state.store(UIState_Pending_Training, std::memory_order_release);
            stateStoreRef_->g_is_calib.store(false, std::memory_order_release);
            stateStoreRef_->g_block_id.store(0, std::memory_order_release);
            stateStoreRef_->g_freq_hz.store(0, std::memory_order_release);
            stateStoreRef_->g_freq_hz_e.store(TestFreq_None, std::memory_order_release);
            break;
        }

        case UIState_None: {
            // “offline” / not connected / shut down
            stateStoreRef_->g_ui_state.store(UIState_None, std::memory_order_release);
            stateStoreRef_->g_ui_popup.store(UIPopup_None, std::memory_order_release);
            stateStoreRef_->g_is_calib.store(false, std::memory_order_release);
            break;
        }
        
        default: {
            break;
        }
    }
}

void StimulusController_C::onStateExit(UIState_E state, UIStateEvent_E ev){
#ifdef ACQ_BACKEND_FAKE
    bool isDemoModeOn = stateStoreRef_->settings.demo_mode.load(std::memory_order_acquire);
#endif
    switch(state){
        case UIState_Active_Calib:
        case UIState_NoSSVEP_Test:
        case UIState_Instructions:
            if(ev != UIStateEvent_UserPushesPause){
                currentWindowTimer_.stop_timer();
                // clear g_freq_hz for fake acq
#ifdef ACQ_BACKEND_FAKE
                stateStoreRef_->g_freq_hz.store(-1, std::memory_order_release);
#endif
            } 
            if(ev == UIStateEvent_StimControllerTimeoutEndCalib){
                // calib over... need to save csv in consumer thread (finalize training data)
                {
                    LOG_ALWAYS("SC: SETTING FINALIZE REQUESTED — state=" << (int)state_ 
                    << " event=" << (int)ev);  
                    // scope for locking & changing bool flag (mtx unlocked again at end of scope)
                    std::lock_guard<std::mutex> lock(stateStoreRef_->mtx_finalize_request);
                    stateStoreRef_->finalize_requested = true;
                }
                stateStoreRef_->cv_finalize_request.notify_one();   
            }
            if(ev == UIStateEvent_UserPushesExit) {
                // calib incomplete... delete session (if still __IN_PROGRESS)
                SessionPaths sp; // temp object
                {
                    std::lock_guard<std::mutex> lock(stateStoreRef_->currentSessionInfo.mtx_);
                    sp.subject_id        = stateStoreRef_->currentSessionInfo.g_active_subject_id;
                    sp.session_id        = stateStoreRef_->currentSessionInfo.g_active_session_id;
                    sp.data_session_dir  = std::filesystem::path(stateStoreRef_->currentSessionInfo.g_active_data_path);
                    sp.model_session_dir = std::filesystem::path(stateStoreRef_->currentSessionInfo.g_active_model_path);
                }
                sesspaths::delete_session_dirs_if_in_progress(sp);

                // clear active session info so UI doesn't show stale sessions
                {
                    std::lock_guard<std::mutex> lock(stateStoreRef_->currentSessionInfo.mtx_);
                    stateStoreRef_->currentSessionInfo.g_active_session_id.clear();
                    stateStoreRef_->currentSessionInfo.g_active_data_path.clear();
                    stateStoreRef_->currentSessionInfo.g_active_model_path.clear();
                }
            }

#ifdef ACQ_BACKEND_FAKE
            if(isDemoModeOn && ev != UIStateEvent_StimControllerTimeout){
                // in any case except regular flow -> we'll want to notify acqDemoDriver to stop streaming
                std::lock_guard<std::mutex> lock_streamer(stateStoreRef_->mtx_streaming_request);
                stateStoreRef_->streaming_requested = false;
                stateStoreRef_->streaming_request.notify_one(); // notify producer
            }
#endif

            break;

        case UIState_Paused: {
            if(currentWindowTimer_.is_paused()){
                currentWindowTimer_.unpause_timer();
            }

            // TODO: not sure if this is needed, may be handled fine in onstateenter for run/calib modes
#ifdef ACQ_BACKEND_FAKE
            bool prevCalib = (prevState_ == (UIState_Active_Calib || prevState_ == UIState_Instructions || prevState_ == UIState_NoSSVEP_Test));
            bool prevRun = prevState_ == UIState_Active_Run;
            if((prevCalib || prevRun) && isDemoModeOn){
                std::lock_guard<std::mutex> lock_streamer(stateStoreRef_->mtx_streaming_request);
                stateStoreRef_->streaming_requested = true;
                if(prevRun) { stateStoreRef_->test_mode_arg = 0; }
                else { stateStoreRef_->test_mode_arg = 1; }
                stateStoreRef_->streaming_request.notify_one(); // notify producer
            }
#endif
            
            break;
        }

        case UIState_Active_Run: {
            // stop timer, clear g_freq_hz to no ssvep
#ifdef ACQ_BACKEND_FAKE
            if (ev != UIStateEvent_UserPushesPause){
                stateStoreRef_->g_freq_hz.store(-1, std::memory_order_release);
                fakeAcqRunModeTimer_.stop_timer();
                emulatedFreqsForFakeAcq_.clear();
                fakeAcqSeqIdx_ = 0;
            }
            else if(isDemoModeOn){
                // notify acqDemoDriver to stop streaming
                std::lock_guard<std::mutex> lock_streamer(stateStoreRef_->mtx_streaming_request);
                stateStoreRef_->streaming_requested = false;
                stateStoreRef_->streaming_request.notify_one(); // notify producer
            }
#endif
            break;
        }
        
        default:
            break;

    }
}

void StimulusController_C::processEvent(UIStateEvent_E ev){
    // special dynamic proessing for resume from pause because it depends on previous state
    if(ev == UIStateEvent_UserPushesResume && state_ == UIState_Paused){
        onStateExit(state_, ev);
        state_ = pausedFromState_; // next state is whatever the prev state was
        prevState_ = UIState_Paused;
        // dont run onstateenter cuz we don't want the side effects; just publish state to ui
        stateStoreRef_->g_ui_state.store(state_, std::memory_order_release);

        // if the state was previously run mode, need to tell cons that we've entered it for operational g_onnx_session_is_reloading pathway
        // signal -1 from stim controller
        if(state_ == UIState_Active_Run){
            stateStoreRef_->g_onnx_session_is_reloading.store(-1, std::memory_order_release);
        }
        return;
    }

    std::size_t table_size = sizeof(state_transition_table)/sizeof(state_transition);
	for(std::size_t i=0; i<table_size; i++){
        const auto& t = state_transition_table[i];
		if(state_ == t.from && ev == t.event) {
			// match found
            LOG_ALWAYS("SC: TRANSITION " << (int)state_ << " --(" << (int)ev << ")-> " << (int)t.to);
			onStateExit(state_, ev);
            prevState_ = state_;
			state_ = t.to;
			onStateEnter(prevState_, state_, ev);
            return;
		}
	}
    LOG_ALWAYS("SC: NO TRANSITION for state=" << (int)state_ << " event=" << (int)ev);
    return;
}

std::optional<UIStateEvent_E> StimulusController_C::detectEvent(){
    // the following are in order of priority 

    // todo: add check for ui toggles sim mode event
    // (1) read UI event sent in by POST: consume event & write it's now None
    UIStateEvent_E currEvent = stateStoreRef_->g_ui_event.exchange(UIStateEvent_None, std::memory_order_acq_rel);
    if(currEvent != UIStateEvent_None){
        LOG_ALWAYS("SC: detected UI event=" << static_cast<int>(currEvent));

        // special case where user is trying to press a btn that they shouldn't be allowed yet
        // want to rtn event w 'invalid' tag
        if(currEvent == UIStateEvent_UserPushesStartRun){
            std::lock_guard<std::mutex> lock(stateStoreRef_->saved_sessions_mutex);
            size_t existingSessions = stateStoreRef_->saved_sessions.size();
            LOG_ALWAYS("SC: UserPushesStartRun, existingSessions=" << existingSessions);
            if(existingSessions <= 1) { // 1 for default
                stateStoreRef_->g_ui_popup.store(UIPopup_MustCalibBeforeRun, std::memory_order_release);
                return std::nullopt; // swallow event; no transition
            }
        }

        if(currEvent == UIStateEvent_UserPushesStartCalibFromOptions){
            // (ie trying to submit name + epilepsy status)
            bool shouldStartCalib = true;
            // (1) consume form inputs
            {
                std::lock_guard<std::mutex> lock(stateStoreRef_->calib_options_mtx);
                pending_epilepsy_ = stateStoreRef_->pending_epilepsy;
                pending_subject_name_ = stateStoreRef_->pending_subject_name;
            }

            // (2a) clean & check if inputs are bad
            while (!pending_subject_name_.empty() && std::isspace((unsigned char)pending_subject_name_.back())) pending_subject_name_.pop_back();
            while (!pending_subject_name_.empty() && std::isspace((unsigned char)pending_subject_name_.front())) pending_subject_name_.erase(pending_subject_name_.begin());
            if((pending_epilepsy_ == EpilepsyRisk_Unknown) || (pending_subject_name_.length() < 3)){
                shouldStartCalib = false;
            }

            // (2b) check if it matches any name in previously stored sessions
            // if it matches, we should say "found existing calibration models for <username>. are you sure you want to restart?" with popup
            bool exists = false;
            {
                std::lock_guard<std::mutex> lock(stateStoreRef_->saved_sessions_mutex);
                for (const auto& s : stateStoreRef_->saved_sessions) { // iterate over saved sessions
                    if (s.subject == pending_subject_name_) {
                        exists = true;
                    }
                }
            }
            if(exists){
                // ask for confirm instead of immediately overwriting
                awaiting_calib_overwrite_confirm_ = true;
                stateStoreRef_->g_ui_popup.store(UIPopup_ConfirmOverwriteCalib, std::memory_order_release);
                return std::nullopt; // swallow until user confirms
            }

            // (2c) check if high frequency popup is now waiting
            if(pending_epilepsy_ == EpilepsyRisk_YesButHighFreqOk) {
                awaiting_highfreq_confirm_ = true;
                stateStoreRef_->g_ui_popup.store(UIPopup_ConfirmHighFreqOk, std::memory_order_release);
                return std::nullopt; // swallow until user presses ok on popup
            }

            // (3) start calib if (2a) and (2b) (happens automatically w state transition)
            // otherwise, swallow transition
            if(!shouldStartCalib){
                stateStoreRef_->g_ui_popup.store(UIPopup_InvalidCalibOptions, std::memory_order_release);
                return std::nullopt; // swallow event; no transition
            }

            awaiting_calib_overwrite_confirm_ = false;
            awaiting_highfreq_confirm_ = false;
            
            // right before return - clear statestore for next calib options entry
            {
                std::lock_guard<std::mutex> lock_final(stateStoreRef_->calib_options_mtx);
                stateStoreRef_->pending_subject_name.clear();
                stateStoreRef_->pending_epilepsy = EpilepsyRisk_Unknown;
            }

            return currEvent;
            
        }

        // repoll current popup val to make sure we don't break on stale awaiting_* vals
        auto popup = stateStoreRef_->g_ui_popup.load(std::memory_order_acquire);

        if(currEvent == UIStateEvent_UserCancelsPopup){
            // always hide popup on user cancel
            stateStoreRef_->g_ui_popup.store(UIPopup_None, std::memory_order_release);
            
            // handling special cases
            if(popup == UIPopup_ConfirmOverwriteCalib && awaiting_calib_overwrite_confirm_){
                // popup in question is for 'session name already exists' detected
                // cancels in this context means don't transition to calib from calib_options
                awaiting_calib_overwrite_confirm_ = false; 
            }
            return std::nullopt; // swallow transition 
        }

        if(currEvent == UIStateEvent_UserAcksPopup){
            // always clear popup on user ack or cancel, regardless of transition
            stateStoreRef_->g_ui_popup.store(UIPopup_None, std::memory_order_release);
            
            // handling special cases
            if(popup == UIPopup_ConfirmOverwriteCalib && awaiting_calib_overwrite_confirm_){
                // clear flag
                awaiting_calib_overwrite_confirm_ = false;
                // proceed into Instructions exactly like the original submit would have
                return UIStateEvent_UserPushesStartCalibFromOptions; // corresponding to state transition row
            }
            else if (popup == UIPopup_ConfirmHighFreqOk && awaiting_highfreq_confirm_){
                awaiting_highfreq_confirm_ = false;
                return UIStateEvent_UserPushesStartCalibFromOptions;
            }
            else {
                // no transition needed
                return std::nullopt;
            }
        }

        return currEvent;
    }

    // responsible for detecting some INTERNAL events:
    // (2) check if window timer is exceeded and we've reached the end of a training bout
    // only emit end in active calib 
    if ((state_ == UIState_Active_Calib) && 
        (activeQueueIdx_ >= trainingProtocol_.numActiveBlocks) && 
        (currentWindowTimer_.check_timer_expired()) &&
        (!end_calib_timeout_emitted_))
    {
        end_calib_timeout_emitted_ = true; // rising edge trigger
        currentWindowTimer_.stop_timer();
        LOG_ALWAYS("SC: returning internal event=" << (int)UIStateEvent_StimControllerTimeoutEndCalib
          << " state=" << (int)state_
          << " idx=" << activeQueueIdx_
          << " num=" << trainingProtocol_.numActiveBlocks
          << " timer_expired=" << currentWindowTimer_.check_timer_expired());
        return UIStateEvent_StimControllerTimeoutEndCalib;
    }
    // (3) check window timer exceeded
    if(currentWindowTimer_.check_timer_expired())
    {
        return UIStateEvent_StimControllerTimeout;
    }
    // (4) check if refresh rate has been written to if were in NONE state
    // read atomically
    int refresh_val = stateStoreRef_->g_refresh_hz.load(std::memory_order_acquire);
    if (state_==UIState_None && refresh_val > 0)
    {
        return UIStateEvent_ConnectionSuccessful;
    }
    // (5) check if training is done (model ready)
    if(state_ == UIState_Pending_Training){
        bool ready = false;
        // poll condition var
        {
            std::lock_guard<std::mutex> lock4(stateStoreRef_->mtx_model_ready);
            if(stateStoreRef_->model_just_ready){
                ready = true;
                stateStoreRef_->model_just_ready = false;
            }
        }
        if(ready){ return UIStateEvent_ModelReady; }
        return std::nullopt; // if not ready
    }

    return std::nullopt;  // no event this iteration
}

int StimulusController_C::checkStimFreqIsIntDivisorOfRefresh(bool isCalib, int desiredTestFreq){
    int flag = 0;
    // only call from instructions in calib mode
    if (state_ != UIState_Instructions && state_ != UIState_Run_Options) {
        return -1; // not a safe time/state to adjust or validate
    }
    int refresh = stateStoreRef_->g_refresh_hz.load(std::memory_order_acquire);
    if (refresh <= 0 || desiredTestFreq <= 0) return -1;
    while(refresh % desiredTestFreq != 0){
        // not an int divisor of refresh; increase until we find
        desiredTestFreq ++;
        flag = 1;
    }
    if (!isCalib){
        return desiredTestFreq; // in run mode return correct freq
    }
    else if(isCalib && flag == 1) {
        // the original freq is not what an int divisor of the refresh
        return -1;
    } 
    else {
        return 0; // original freq is int divisor of refresh, all g
    }
   
}

// helper to see if our refresh rate is simply just cooked and we must accept non int divisors :,)
bool StimulusController_C::has_divisor_6_to_20(int n) {
    if (n == 0) return true;  // everything divides 0

    for (int d = 6; d <= 20; ++d) {
        if (d != 0 && n % d == 0) {
            return true;     // found a divisor in [6, 20]
        }
    }
    return false;            // no divisors in that range
}

#ifdef ACQ_BACKEND_FAKE
void StimulusController_C::fakeAcq_buildSeqAndShuffle() {
    fakeAcqShuffledSeq_.clear();

    // emulatedFreqsForFakeAcq_ = { -1(REST), leftHz, rightHz }
    // gather all the freqs w appropriate reps
    for(int i = 0; i<static_cast<int>(emulatedFreqsForFakeAcq_.size()); i++){
        int hz = emulatedFreqsForFakeAcq_[i];
        int reps = (hz == -1) ? FAKE_NO_SSVEP_REPS : FAKE_ACTIVE_REPS;
        for(int r = 0; r < reps; r++){
            fakeAcqShuffledSeq_.push_back(hz);
        }
    }
    // Fisher-Yates shuffle
    for(int i = static_cast<int>(fakeAcqShuffledSeq_.size() - 1); i>0; i--){
        std::uniform_int_distribution<int> pick(0, i);
        std::swap(fakeAcqShuffledSeq_[i], fakeAcqShuffledSeq_[pick(fakeAcqRng_)]);
    }
    // init idx
    fakeAcqSeqIdx_ = 0;
}

void StimulusController_C::fakeAcq_advanceToNextSSVEP() {
    // Reshuffle when pool exhausted
    if(fakeAcqSeqIdx_ > (fakeAcqShuffledSeq_.size() - 1)) {
        fakeAcq_buildSeqAndShuffle();
    }

    int nextHz = fakeAcqShuffledSeq_[fakeAcqSeqIdx_];
    fakeAcqSeqIdx_++;
    auto dur = fakeAcq_getDurationForHz(nextHz);

    // publish
    stateStoreRef_->g_freq_hz.store(nextHz, std::memory_order_release);
    // reset timer for this iter
    if(fakeAcqRunModeTimer_.is_started()){
        fakeAcqRunModeTimer_.stop_timer();
    }
    fakeAcqRunModeTimer_.start_timer(dur);
}

std::chrono::milliseconds StimulusController_C::fakeAcq_getDurationForHz(int hz){
    if(hz == -1){
        // REST
        std::uniform_int_distribution<int> d(FAKE_REST_MIN_MS, FAKE_REST_MAX_MS);
        return std::chrono::milliseconds{ d(fakeAcqRng_) };
    }
    else {
        // active (left or right)
        std::uniform_int_distribution<int> d(FAKE_ACTIVE_MIN_MS, FAKE_ACTIVE_MAX_MS);
        return std::chrono::milliseconds{ d(fakeAcqRng_) };
    }
}
#endif

void StimulusController_C::runUIStateMachine(){
    logger::tlabel = "StimulusController";
    LOG_ALWAYS("SC: starting in state=" << static_cast<int>(state_));
    // Optional: publish initial state
    //onStateEnter(UIState_None, state_);

    // is_stopped_ lets us cleanly exit loop operation
    while(!is_stopped_){

        // special processing for fake eeg emulator in run mode: need to sim random assortment of SSVEP freqs from active model (the two being used) + no ssvep states
        // this gets published in g_freq_hz, which is what the emulator uses to create its ssvep responses
#ifdef ACQ_BACKEND_FAKE
        if(state_ == UIState_Active_Run){
            if (fakeAcqRunModeTimer_.check_timer_expired()){
                // move to next stim
                fakeAcq_advanceToNextSSVEP();
            }
        }
#endif
        // detect internal events that happened since last loop (polling)
        // external (browser) events will use event-based handling
        std::optional<UIStateEvent_E> ev = detectEvent();
        if(ev.has_value()){
            LOG_ALWAYS("SC: event " << static_cast<int>(*ev)
                     << " in state " << static_cast<int>(state_));
            processEvent(ev.value());
            LOG_ALWAYS("SC: now in state " << static_cast<int>(state_));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}

void StimulusController_C::stopStateMachine(){
    // clean exit
    is_stopped_ = true;
}