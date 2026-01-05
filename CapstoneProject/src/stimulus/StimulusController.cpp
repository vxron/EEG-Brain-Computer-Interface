#include "StimulusController.hpp"
#include <thread>
#include "../utils/Logger.hpp"
#include "../utils/SessionPaths.hpp"

struct state_transition{
    UIState_E from;
    UIStateEvent_E event;
    UIState_E to;
};

static const state_transition state_transition_table[] = {
    // from                           event                                     to
    {UIState_None,             UIStateEvent_ConnectionSuccessful,           UIState_Home},
 
    {UIState_Home,             UIStateEvent_UserPushesStartCalib,           UIState_Calib_Options},
    {UIState_Calib_Options,    UIStateEvent_UserPushesStartCalibFromOptions,UIState_Instructions},
    {UIState_Home,             UIStateEvent_UserPushesStartRun,             UIState_Run_Options},
    {UIState_Home,             UIStateEvent_UserPushesHardwareChecks,       UIState_Hardware_Checks},
     
    {UIState_Active_Calib,     UIStateEvent_StimControllerTimeout,          UIState_Instructions},
    {UIState_Active_Calib,     UIStateEvent_StimControllerTimeoutEndCalib,  UIState_Pending_Training},
    {UIState_Pending_Training, UIStateEvent_ModelReady,                     UIState_Home},
    {UIState_Instructions,     UIStateEvent_StimControllerTimeout,          UIState_Active_Calib},
       
    {UIState_Active_Calib,     UIStateEvent_UserPushesExit,                 UIState_Home},
    {UIState_Calib_Options,    UIStateEvent_UserPushesExit,                 UIState_Home},
    {UIState_Instructions,     UIStateEvent_UserPushesExit,                 UIState_Home},
    {UIState_Active_Run,       UIStateEvent_UserPushesExit,                 UIState_Home},
    {UIState_Saved_Sessions,   UIStateEvent_UserPushesExit,                 UIState_Home},
    {UIState_Run_Options,      UIStateEvent_UserPushesExit,                 UIState_Home},
    {UIState_Hardware_Checks,  UIStateEvent_UserPushesExit,                 UIState_Home},
    {UIState_Pending_Training, UIStateEvent_UserPushesExit,                 UIState_Home},
 
    {UIState_Run_Options,      UIStateEvent_UserPushesSessions,             UIState_Saved_Sessions},
    {UIState_Saved_Sessions,   UIStateEvent_UserSelectsSession,             UIState_Active_Run},
    {UIState_Saved_Sessions,   UIStateEvent_UserSelectsNewSession,          UIState_Instructions},
    {UIState_Saved_Sessions,   UIStateEvent_UserPushesStartRun,             UIState_Run_Options},
    {UIState_Run_Options,      UIStateEvent_UserPushesStartDefault,         UIState_Active_Run},
    
};
// ^todo: add popup if switching: r u sure u want to exit???


StimulusController_C::StimulusController_C(StateStore_s* stateStoreRef, std::optional<trainingProto_S> trainingProtocol) : state_(UIState_None), stateStoreRef_(stateStoreRef) {
    if (trainingProtocol.has_value()){
        trainingProto_S trainingProtocol_value = trainingProtocol.value();
        trainingProtocol_ = trainingProtocol_value; 
    }
    else {
        // default protocol
        trainingProtocol_.activeBlockDuration_s = 15;
        trainingProtocol_.displayInPairs = false;
        trainingProtocol_.freqsToTest = {TestFreq_8_Hz, TestFreq_9_Hz, TestFreq_10_Hz, TestFreq_11_Hz, TestFreq_12_Hz};
        trainingProtocol_.numActiveBlocks = trainingProtocol_.freqsToTest.size();
        trainingProtocol_.restDuration_s = 10;
    }
     activeBlockQueue_ = trainingProtocol_.freqsToTest;
     activeBlockDur_ms_ = std::chrono::milliseconds{
     trainingProtocol_.activeBlockDuration_s * 1000 };
     restBlockDur_ms_ = std::chrono::milliseconds{
     trainingProtocol_.restDuration_s * 1000 };     

}

std::chrono::milliseconds StimulusController_C::getCurrentBlockTime() const {
    if (currentWindowTimer_.is_started() == false) {
        auto time = std::chrono::milliseconds{0};
        return time;
    }
    auto time = currentWindowTimer_.get_timer_value_ms();
    return time;
}

void StimulusController_C::onStateEnter(UIState_E prevState, UIState_E newState){
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
            stateStoreRef_->g_ui_state.store(UIState_Active_Run, std::memory_order_release);
            stateStoreRef_->g_is_calib.store(false, std::memory_order_release);
            // TODO: make these per person based on saved sessions (see struct in types.h)
            //stateStoreRef_->g_freq_left_hz.store(10, std::memory_order_release);
            //stateStoreRef_->g_freq_right_hz.store(12, std::memory_order_release);
            //stateStoreRef_->g_freq_left_hz_e.store(TestFreq_10_Hz, std::memory_order_release);
            //stateStoreRef_->g_freq_right_hz_e.store(TestFreq_12_Hz, std::memory_order_release);
            break;
        }
        
        case UIState_Home: {
            stateStoreRef_->g_ui_state.store(UIState_Home, std::memory_order_release);
            stateStoreRef_->g_is_calib.store(false, std::memory_order_release);
            // reset block/freq for clean home:
            stateStoreRef_->g_block_id.store(0, std::memory_order_release);
            stateStoreRef_->g_freq_hz.store(0, std::memory_order_release);
            stateStoreRef_->g_freq_hz_e.store(TestFreq_None, std::memory_order_release);
            //stateStoreRef_->g_ui_popup.store(UIPopup_None);
            // reset hardware page
            std::lock_guard<std::mutex> lock(stateStoreRef_->signal_stats_mtx);
            stateStoreRef_->SignalStats = SignalStats_s{}; //reset 
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
            while(result == -1 && has_divisor_6_to_20(refresh)){
                LOG_ALWAYS("SC: dropped testcase=" << static_cast<int>(freq));
                // reasonably drop since we have other divisors
                activeQueueIdx_++;
                // guard against boundless incrementing
                if ( (activeQueueIdx_ >= (int)activeBlockQueue_.size()) || (activeQueueIdx_ >= trainingProtocol_.numActiveBlocks) ) break;
                freqToTest = activeBlockQueue_[activeQueueIdx_];
                freq =  TestFreqEnumToInt(freqToTest);
                result = checkStimFreqIsIntDivisorOfRefresh(true, freq);
            } // otherwise accept "bad" test freq vis a vis refresh
    
            // storing
            stateStoreRef_->g_freq_hz_e.store(freqToTest, std::memory_order_release);
            stateStoreRef_->g_freq_hz.store(freq, std::memory_order_release);
            LOG_ALWAYS("SC: stored a freq=" << static_cast<int>(freq));

            // instructions state is entered from Calib_Options or Saved_Sessions
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
            currentWindowTimer_.start_timer(restBlockDur_ms_);
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
            stateStoreRef_->g_ui_popup.store(UIPopup_None);
            stateStoreRef_->g_is_calib.store(false, std::memory_order_release);
            break;
        }
        
        default: {
            break;
        }
    }
}

void StimulusController_C::onStateExit(UIState_E state, UIStateEvent_E ev){
    switch(state){
        case UIState_Active_Calib:
        case UIState_Instructions:
            currentWindowTimer_.stop_timer();
            if(ev == UIStateEvent_StimControllerTimeoutEndCalib){
                // calib over... need to save csv in consumer thread (finalize training data)
                {
                    LOG_ALWAYS("reached inside final event");
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
            // TODO: any fault cases
            break;
    
        case UIState_Active_Run:
        // idk yet whether or not we want to be clearing here !
            //stateStoreRef_->g_freq_left_hz_e.store(TestFreq_None, std::memory_order_release);
            //stateStoreRef_->g_freq_left_hz.store(0, std::memory_order_release);
            //stateStoreRef_->g_freq_right_hz.store(0, std::memory_order_release);
            //stateStoreRef_->g_freq_right_hz_e.store(TestFreq_None, std::memory_order_release);
            break;
        default:
            break;

    }
}

void StimulusController_C::processEvent(UIStateEvent_E ev){
    std::size_t table_size = sizeof(state_transition_table)/sizeof(state_transition);
	for(std::size_t i=0; i<table_size; i++){
        const auto& t = state_transition_table[i];
		if(state_ == t.from && ev == t.event) {
			// match found
            LOG_ALWAYS("SC: TRANSITION " << (int)state_ << " --(" << (int)ev << ")-> " << (int)t.to);
			onStateExit(state_, ev);
            prevState_ = state_;
			state_ = t.to;
			onStateEnter(prevState_, state_);
            break;
		}
	}
    LOG_ALWAYS("SC: NO TRANSITION for state=" << (int)state_ << " event=" << (int)ev);
    return;
}

std::optional<UIStateEvent_E> StimulusController_C::detectEvent(){
    // the following are in order of priority 
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

        if(currEvent == UIStateEvent_UserCancelsPopup && awaiting_calib_overwrite_confirm_){
            // popup in question is for 'session name already exists' detected
            // cancels means don't transition to calib
            awaiting_calib_overwrite_confirm_ = false; 
            return std::nullopt; // swallow transition from calib options -> calib 
        }

        if(currEvent == UIStateEvent_UserAcksPopup && awaiting_calib_overwrite_confirm_){
            // clear flag
            awaiting_calib_overwrite_confirm_ = false;
            // proceed into Instructions exactly like the original submit would have
            LOG_ALWAYS("SC: popup ack -> remap to StartCalibFromOptions (awaiting_highfreq_confirm_)");
            return UIStateEvent_UserPushesStartCalibFromOptions; // corresponding to state transition row
        }

        // check other popup on calib options page
        if(currEvent == UIStateEvent_UserAcksPopup && awaiting_highfreq_confirm_){
            awaiting_highfreq_confirm_ = false;
            LOG_ALWAYS("SC: popup ack -> remap to StartCalibFromOptions (awaiting_highfreq_confirm_)");
            return UIStateEvent_UserPushesStartCalibFromOptions;
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

// if it's calib -> return -1 if the freq doesn't match and just skip that freq in training protocol. 
int StimulusController_C::checkStimFreqIsIntDivisorOfRefresh(bool isCalib, int desiredTestFreq){
    int flag = 0;
    // require state to be transitional (don't want to be checking in middle of active mode) but connection must be established
    if(state_ == UIState_Active_Calib || state_ == UIState_Active_Run || state_ == UIState_None){
        return -1;
    }
    int refresh = stateStoreRef_->g_refresh_hz.load(std::memory_order_acquire);
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


void StimulusController_C::runUIStateMachine(){
    logger::tlabel = "StimulusController";
    LOG_ALWAYS("SC: starting in state=" << static_cast<int>(state_));
    // Optional: publish initial state
    onStateEnter(UIState_None, state_);

    // is_stopped_ lets us cleanly exit loop operation
    while(!is_stopped_){
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

        // might not even need this
        switch(state_){
            case UIState_None:
                // waiting for g_refresh_hz to get set
                break;
            case UIState_Active_Calib:
                break;
            case UIState_Instructions:
                break;
            case UIState_Active_Run:
                break;
            case UIState_Home:
                break;
            default:
                break;

        }
    }
}

void StimulusController_C::stopStateMachine(){
    // clean exit
    is_stopped_ = true;
}