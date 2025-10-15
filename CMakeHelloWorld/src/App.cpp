// State Machine

#include "App.h"
#include <spdlog/spdlog.h>   // main spdlog API (info/warn/error, set_pattern, set_level)

bool App_C::throw_error(const std::string& errMsg) {
	spdlog::error("Error: {}", errMsg);
	currState_ = appState_Error;
	// idk what we can do here to try and recover...
	if (nextState_ != appState_Shutdown) {
		nextState_ = appState_Error;
	}
	return true;
}

bool App_C::start_calibrate_mode() {
	if (currState_ != appState_Idle) {
		spdlog::error("Cannot start calibration from current state.");
		nextState_ = appState_Idle; // i think try to recover by going to idle? have to set this up tho
		return false;
	}
	// need to get trainingProto from GUI settings
	timingManager_.run_training_protocol(trainingProto_S{});
	currState_ = appState_Calibrate;
	
}
bool App_C::start_run_mode(std::unique_ptr<const PatientModel_C> ptModel) {
	if (currState_ != appState_Idle) {
		spdlog::error("Cannot start run from current state.");
		nextState_ = appState_Idle; // i think try to recover by going to idle? have to set this up tho
		return false;
	}
	currState_ = appState_Run;
	// start using the model for prediction
}
bool App_C::go_idle(){
}

bool App_C::shutdown() {

}

// STATE TRANSITIONS
// calibrate --> idle
// run --> idle
// idle --> run
// idle --> calibrate
// any --> error
// any --> shutdown
// error --> idle (RECOVERY)
// error --> shutdown (after timeout)

