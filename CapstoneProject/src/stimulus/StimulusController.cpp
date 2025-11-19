#include "StimulusController.hpp"
#include <thread>

static struct state_transition{
    UIState_E from;
    UIStateEvent_E event;
    UIState_E to;
};

static const state_transition state_transition_table[] = {
    // from                           event                                     to
    {UIState_Active_Calib,  UIStateEvent_StimControllerTimeout,          UIState_Instructions},
    {UIState_Active_Calib,  UIStateEvent_StimControllerTimeoutEndCalib,  UIState_Home},
    {UIState_Instructions,  UIStateEvent_StimControllerTimeout,          UIState_Active_Calib},
    {UIState_None,          UIStateEvent_ConnectionSuccessful,           UIState_Home},
    {UIState_Home,          UIStateEvent_UserPushesStartCalib,           UIState_Instructions},
    {UIState_Home,          UIStateEvent_UserPushesStartRun,             UIState_Active_Run},
    {UIState_Home,          UIStateEvent_UserPushesExit,                 UIState_None},
    {UIState_Active_Calib,  UIStateEvent_UserPushesExit,                 UIState_Home},
    {UIState_Instructions,  UIStateEvent_UserPushesExit,                 UIState_Home},
    {UIState_Active_Run,    UIStateEvent_UserPushesExit,                 UIState_Home},
};

// process_inputs to determine event

// 

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
        trainingProtocol_.restDuration_s = 15;
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
    switch(newState){
        case UIState_Active_Calib:
            // first read seq atomically then increment
            int currSeq = stateStoreRef_->g_ui_seq.load(std::memory_order_acquire);
            stateStoreRef_->g_ui_seq.store(currSeq + 1,std::memory_order_release);
            // stim window
            stateStoreRef_->g_ui_state.store(UIState_Active_Calib, std::memory_order_release);
            // first read block_id atomically then increment
            int currId = stateStoreRef_->g_block_id.load(std::memory_order_acquire);
            stateStoreRef_->g_block_id.store(currId + 1,std::memory_order_release);
            // freqs
            TestFreq_E freqToTest = activeBlockQueue_[activeQueueIdx_];
            stateStoreRef_->g_freq_hz_e.store(freqToTest, std::memory_order_acquire);
            // use helper
            int freq =  TestFreqEnumToInt(freqToTest);
            stateStoreRef_->g_freq_hz.store(freq,std::memory_order_acquire);
            // iscalib helper
            stateStoreRef_->g_is_calib.store(true,std::memory_order_release);

            // increment queue idx so we move to next test freq on next block
            activeQueueIdx_++;

            // start timer
            currentWindowTimer_.start_timer(activeBlockDur_ms_);
            break;

        case UIState_Instructions:
            // first read seq atomically then increment
            int currSeq = stateStoreRef_->g_ui_seq.load(std::memory_order_acquire);
            stateStoreRef_->g_ui_seq.store(currSeq + 1,std::memory_order_release);
            // stim window
            stateStoreRef_->g_ui_state.store(UIState_Instructions, std::memory_order_release);
            // instruction windows still get freq info for next active block cuz UI will tell user what freq they'll be seeing next
            TestFreq_E freqToTest = activeBlockQueue_[activeQueueIdx_];
            stateStoreRef_->g_freq_hz_e.store(freqToTest, std::memory_order_acquire);
            int freq =  TestFreqEnumToInt(freqToTest);
            stateStoreRef_->g_freq_hz.store(freq,std::memory_order_acquire);
            // iscalib helper
            stateStoreRef_->g_is_calib.store(true,std::memory_order_release);

            // start timer
            currentWindowTimer_.start_timer(restBlockDur_ms_);
            break;
        default:
            break;
    }
}

void StimulusController_C::onStateExit(UIState_E state){
    switch(state){
        case UIState_Active_Calib:
        case UIState_Instructions:
            currentWindowTimer_.stop_timer();
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
			onStateExit(state_);
            prevState_ = state_;
			state_ = t.to;
			onStateEnter(prevState_, state_);
            break;
		}
	}
    return;
}

// LATER NEED TO ADD EVENT BASED EVENT DETECTION FOR USER EVENTS
std::optional<UIStateEvent_E> StimulusController_C::detectEvent(){
    // responsible for detecting three internal events
    // (1) check if window timer is exceeded and we've reached the end of a training bout
    if ((activeQueueIdx_ >= trainingProtocol_.numActiveBlocks) && 
        (currentWindowTimer_.check_timer_expired()))
    {
        return UIStateEvent_StimControllerTimeoutEndCalib;
    }
    // (2) check window timer exceeded
    if(currentWindowTimer_.check_timer_expired())
    {
        return UIStateEvent_StimControllerTimeout;
    }
    // (3) check if refresh rate has been written to if were in NONE state
    // read atomically
    int refresh_val = stateStoreRef_->g_refresh_hz.load(std::memory_order_acquire);
    if (state_==UIState_None && refresh_val > 0)
    {
        return UIStateEvent_ConnectionSuccessful;
    }

    return std::nullopt;  // no event this iteration
}

void StimulusController_C::runUIStateMachine(){
    while(1){
        // detect internal events that happened since last loop (polling)
        // external (browser) events will use event-based handling
        std::optional<UIStateEvent_E> ev = detectEvent();
        if(ev.has_value()){
            processEvent(ev.value());
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