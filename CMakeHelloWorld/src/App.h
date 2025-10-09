/*
==============================================================================
	File: App.h
	Desc: Owns FSM and state entry/exit methods.
==============================================================================
*/
#include "utils/Types.h"   // common types

class App_C {
public:
	AppState_E globalAppState;
	void startCalibrate();
	void endCalibrate();
	void startRunMode(std::unique_ptr<const PatientModel_C> ptModel); // pass model object  
	void endRunMode();
	void shutdown();
private:
	AppState_E nextState;
	AppState_E currState;
}; // App_C