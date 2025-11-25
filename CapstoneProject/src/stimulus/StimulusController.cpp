#include "StimulusController.hpp"
#include <thread>
#include "../utils/Logger.hpp"
#include "../classifier/LaunchTrainingJob.hpp"

static struct state_transition{
    UIState_E from;
    UIStateEvent_E event;
    UIState_E to;
};

static const state_transition state_transition_table[] = {
    // from                           event                                     to
    {UIState_None,            UIStateEvent_ConnectionSuccessful,           UIState_Home},

    {UIState_Home,            UIStateEvent_UserPushesStartCalib,           UIState_Instructions},
    {UIState_Home,            UIStateEvent_UserPushesStartRun,             UIState_Run_Options},
    {UIState_Home,            UIStateEvent_UserPushesHardwareChecks,       UIState_Hardware_Checks},
    
    {UIState_Active_Calib,    UIStateEvent_StimControllerTimeout,          UIState_Instructions},
    {UIState_Active_Calib,    UIStateEvent_StimControllerTimeoutEndCalib,  UIState_Home},
    {UIState_Instructions,    UIStateEvent_StimControllerTimeout,          UIState_Active_Calib},
      
    {UIState_Active_Calib,    UIStateEvent_UserPushesExit,                 UIState_Home},
    {UIState_Instructions,    UIStateEvent_UserPushesExit,                 UIState_Home},
    {UIState_Active_Run,      UIStateEvent_UserPushesExit,                 UIState_Home},
    {UIState_Saved_Sessions,  UIStateEvent_UserPushesExit,                 UIState_Home},
    {UIState_Run_Options,     UIStateEvent_UserPushesExit,                 UIState_Home},
    {UIState_Hardware_Checks, UIStateEvent_UserPushesExit,                 UIState_Home},

    {UIState_Run_Options,     UIStateEvent_UserPushesSessions,             UIState_Saved_Sessions},
    {UIState_Saved_Sessions,  UIStateEvent_UserSelectsSession,             UIState_Active_Run},
    {UIState_Saved_Sessions,  UIStateEvent_UserSelectsNewSession,          UIState_Instructions},
    {UIState_Saved_Sessions,  UIStateEvent_UserPushesStartRun,             UIState_Run_Options},
    {UIState_Run_Options,     UIStateEvent_UserPushesStartDefault,         UIState_Active_Run},
    
};
// ^todo: add popup if switching btwn run <-> calib: r u sure u want to exit???


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
    switch(newState){
        case UIState_Active_Run: {
            stateStoreRef_->g_ui_state.store(UIState_Active_Run, std::memory_order_release);
            stateStoreRef_->g_is_calib.store(false, std::memory_order_release);
            // TODO: make these per person based on saved sessions (see struct in types.h)
            stateStoreRef_->g_freq_left_hz.store(10, std::memory_order_release);
            stateStoreRef_->g_freq_right_hz.store(12, std::memory_order_release);
            stateStoreRef_->g_freq_left_hz_e.store(TestFreq_10_Hz, std::memory_order_release);
            stateStoreRef_->g_freq_right_hz_e.store(TestFreq_12_Hz, std::memory_order_release);
            break;
        }
        
        case UIState_Home: {
            stateStoreRef_->g_ui_state.store(UIState_Home, std::memory_order_release);
            stateStoreRef_->g_is_calib.store(false, std::memory_order_release);
            // reset block/freq for clean home:
            stateStoreRef_->g_block_id.store(0, std::memory_order_release);
            stateStoreRef_->g_freq_hz.store(0, std::memory_order_release);
            stateStoreRef_->g_freq_hz_e.store(TestFreq_None, std::memory_order_release);
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
            stateStoreRef_->g_freq_hz_e.store(freqToTest, std::memory_order_acquire);
            // use helper
            freq =  TestFreqEnumToInt(freqToTest);
            stateStoreRef_->g_freq_hz.store(freq,std::memory_order_acquire);
            // iscalib helper
            stateStoreRef_->g_is_calib.store(true,std::memory_order_release);

            // increment queue idx so we move to next test freq on next block
            activeQueueIdx_++;

            // start timer
            currentWindowTimer_.start_timer(activeBlockDur_ms_);
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
                freqToTest = activeBlockQueue_[activeQueueIdx_];
                freq =  TestFreqEnumToInt(freqToTest);
                result = checkStimFreqIsIntDivisorOfRefresh(true, freq);
            } // otherwise accept "bad" test freq vis a vis refresh
    
            // storing
            stateStoreRef_->g_freq_hz_e.store(freqToTest, std::memory_order_acquire);
            stateStoreRef_->g_freq_hz.store(freq, std::memory_order_acquire);
            LOG_ALWAYS("SC: stored a freq=" << static_cast<int>(freq));

            // iscalib helper
            stateStoreRef_->g_is_calib.store(true,std::memory_order_release);

            // start timer
            currentWindowTimer_.start_timer(restBlockDur_ms_);
            break;
        }

        case UIState_None: {
            // “offline” / not connected / shut down
            stateStoreRef_->g_ui_state.store(UIState_None, std::memory_order_release);
            stateStoreRef_->g_is_calib.store(false, std::memory_order_release);
            break;
        }
        
        default:
            break;
    }
}

void StimulusController_C::onStateExit(UIState_E state, UIStateEvent_E ev){
    switch(state){
        case UIState_Active_Calib:
        case UIState_Instructions:
            currentWindowTimer_.stop_timer();
            if(ev == UIStateEvent_StimControllerTimeoutEndCalib){
                // calib over... need to save csv and trigger python training script
                TrainingJob_C job(); // need info from somewhere (TODO -> saved sessions in state store)
            }
            break;
        
        case UIState_Active_Run:
        // idk yet whether or not we want to be clearing here !
            stateStoreRef_->g_freq_left_hz_e.store(TestFreq_None, std::memory_order_release);
            stateStoreRef_->g_freq_left_hz.store(0, std::memory_order_release);
            stateStoreRef_->g_freq_right_hz.store(0, std::memory_order_release);
            stateStoreRef_->g_freq_right_hz_e.store(TestFreq_None, std::memory_order_release);
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
			onStateExit(state_, ev);
            prevState_ = state_;
			state_ = t.to;
			onStateEnter(prevState_, state_);
            break;
		}
	}
    return;
}

std::optional<UIStateEvent_E> StimulusController_C::detectEvent(){
    // the following are in order of priority 
    // (1) UI events sent in by POST (EXTERNAL): user presses exit, user presses start calib, user presses start run
    UIStateEvent_E currEvent = stateStoreRef_->g_ui_event.load(std::memory_order_acquire);
    if(currEvent != UIStateEvent_None){
        LOG_ALWAYS("SC: detected UI event=" << static_cast<int>(currEvent));
        // a new event hasn't been detected yet !
        // reset g_ui_event to NONE for next event
        stateStoreRef_->g_ui_event.store(UIStateEvent_None, std::memory_order_release);
        // return
        return currEvent;
    }

    // responsible for detecting three INTERNAL events:
    // (2) check if window timer is exceeded and we've reached the end of a training bout
    if ((activeQueueIdx_ >= trainingProtocol_.numActiveBlocks) && 
        (currentWindowTimer_.check_timer_expired()))
    {
        LOG_ALWAYS("SC: detected end event=" << static_cast<int>(currEvent));
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