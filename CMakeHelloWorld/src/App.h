/*
==============================================================================
	File: App.h
	Desc: Owns FSM and state entry/exit methods through user requests over GUI.
==============================================================================
*/
#include "utils/Types.h"   // common types
#include <memory>         // std::unique_ptr>
#include <vector>
#include "utils/ModelStore.h" // PatientModel_C
#include "utils/TimingManager.h" // TimingManager_C

class App_C {
public:
	AppState_E get_current_state() const { return currState_; };
	AppState_E get_next_state() const { return nextState_; };
	bool throw_error(const std::string& errMsg);
	bool start_calibrate_mode();
	bool start_run_mode(std::unique_ptr < const PatientModel_C > ptModel); // pass model object  
	bool go_idle();
	bool shutdown();
private:
	// Timing Manager
	TimingManager_C timingManager_;

	AppState_E nextState_;
	AppState_E currState_;
	AppState_E onStateEntry(AppState_E currentState, AppState_E nextState);
	AppState_E onStateExit(AppState_E currentState, AppState_E nextState);
}; // App_C


