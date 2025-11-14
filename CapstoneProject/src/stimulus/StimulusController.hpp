/*
STIMULUS CONTROLLER : writer
- builds the training schedule (STATE MACHINE - which phase we're in; home, instructions, calib block, run)
- every ~100ms, updates atomics in the shared StateStore and publishes a new 'seq' to server so HTML can see it & compare against lastSeq
- launches the HTML once at start, sets a neutral end state on finish
*/

#pragma once
#include "../utils/Types.h"
#include "../utils/SWTimer.hpp"

// SINGLETON
class StimulusController_C{
public:
    UIState_E getUIState() const {return state_;};
    void runUIStateMachine();
private:
    UIState_E state_;
    UIState_E nextState_;
    SW_Timer_C currentBlockTimer;
    
    void runHomeWindow();
    void runActiveCalibWindow();
    void runInstructionsCalibWindow();
    void runModeWindow();
    void transitionBetweenWindows(); // maybe this should be handled on UI client side
    void manageTrainingProtocol(); // uses currentBlockTimer to control calib blocks, trigger timeout events for state machine
    void process_inputs();
    void process_events();
}