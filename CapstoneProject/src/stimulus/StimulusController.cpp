#include "StimulusController.hpp"

static struct state_transition{
    UIState_E from;
    UIStateEvent_E event;
    UIState_E to;
};

static const state_transition state_transition_table[] = {
    {UIState_Active_Calib,  UIStateEvent_StimControllerTimeout,          UIState_Instructions},
    {UIState_Active_Calib,  UIStateEvent_StimControllerTimeoutEndCalib,  UIState_Home},
    {UIState_Instructions,  UIStateEvent_StimControllerTimeout,          UIState_Active_Calib},
    {UIState_Instructions,  UIStateEvent_StimControllerTimeoutEndCalib,  UIState_Home},
    {UIState_None,          UIStateEvent_ConnectionSuccessful,           UIState_Home},
    {UIState_Home,          UIStateEvent_UserPushesStartCalib,           UIState_Instructions},
    {UIState_Home,          UIStateEvent_UserPushesStartRun,             UIState_Active_Run},
    {UIState_Home,          UIStateEvent_UserPushesExit,                 UIState_None},
    {UIState_Active_Calib,  UIStateEvent_UserPushesExit,                 UIState_Home},
    {UIState_Instructions,  UIStateEvent_UserPushesExit,                 UIState_Home},
    {UIState_Active_Run,    UIStateEvent_UserPushesExit,                 UIState_Home},
}

// process_inputs to determine event

// 

StimulusController_C::runActiveCalibWindow(){

}

StimulusController_C::runInstructionsCalibWindow(){

}

StimulusController_C::runModeWindow(){

}

// process_event to determine next state using state transition table

StimulusController_C::runUIStateMachine(){
    while(1){
        // (1) poll for events (process inputs) ** INTERRUPT
        // (2) process event
        // (3) get next state
        // (4) enter next state operation
    }
}