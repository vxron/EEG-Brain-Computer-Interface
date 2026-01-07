/*
==============================================================================
	File: Types.h
	Desc: Common type definitions between modules.
	This header is used by:
  - Acquisition (producer): creates a Chunk and pushes it to the SPSC queue.
  - Decoder (consumer): pops a Chunk and appends samples into its window.

==============================================================================
*/

#pragma once
#include <cstdint>
#include <vector>
#include <array>
#include <cstddef>
#include <string>
#include <chrono>
#include "RingBuffer.hpp"
#include <deque>
#include "../classifier/ONNXClassifier.hpp"
#include <filesystem>
namespace fs = std::filesystem;

// _T for type
// Use steady clock for time measurements (monotonic, not affected by system clock changes)
using clock_T = std::chrono::steady_clock;
using ms_T = std::chrono::milliseconds;
using time_point_T = std::chrono::time_point<clock_T>;
//using time_point_ms_T = std::chrono::time_point<clock_T, ms_T>;
// convert to ms later so we don't bake in truncation errors for now

/* START CONFIGS */

// CHUNKING POLICY
inline constexpr std::size_t NUM_CH_CHUNK = 8; // Unicorn EEG has 8 channels (EEG1...EEG8)
inline constexpr std::size_t NUM_SCANS_CHUNK = 32; // ~128ms latency @ 250Hz
inline constexpr std::size_t NUM_SAMPLES_CHUNK = NUM_CH_CHUNK * NUM_SCANS_CHUNK;

/* END CONFIGS */

/* START ENUMS */

enum SSVEPState_E {
	SSVEP_Left,
	SSVEP_Right,
	SSVEP_None,
	SSVEP_Unknown,
}; // SSVEPState_E

enum TestFreq_E {
	TestFreq_None, // 0
	TestFreq_8_Hz, // 1
	TestFreq_9_Hz, // 2
	TestFreq_10_Hz, // 3
	TestFreq_11_Hz, // 4
	TestFreq_12_Hz, // 5
	TestFreq_20_Hz, // 6
	TestFreq_25_Hz, // 7
	TestFreq_30_Hz, // 8
	TestFreq_35_Hz, // 9
};

enum ActuatorState_E {
	ActuatorState_Fwd,
	ActuatorState_Bcwd,
	ActuatorState_None,
};

enum UIState_E {
	UIState_Active_Run, // 0 selects most recent session as default
	UIState_Active_Calib, // 1
	UIState_Instructions, // 2 also for calib
	UIState_Home, // 3
	UIState_Saved_Sessions, // 4 from run mode -> can press another button to select saved session 
	UIState_Run_Options, // 5
	UIState_Hardware_Checks, // 6
	UIState_Calib_Options, // 7
	UIState_Pending_Training, // 8 when we're waiting for training to complete after calib
	UIState_Settings, // 9
	UIState_None, // 10
};

enum EpilepsyRisk_E {
	EpilepsyRisk_No,
	EpilepsyRisk_YesButHighFreqOk,
	EpilepsyRisk_Yes,
	EpilepsyRisk_Unknown,
};

enum UIStateEvent_E {
	UIStateEvent_StimControllerTimeout, // switch btwn instructions/active during calib
	UIStateEvent_StimControllerTimeoutEndCalib, // 1
	UIStateEvent_UserPushesStartRun, // 2
	UIStateEvent_UserPushesStartRunInvalid, // 3
	UIStateEvent_UserPushesStartCalib, // 4
	UIStateEvent_LostConnection, // 5
	UIStateEvent_UserPushesExit, // 6
	UIStateEvent_ConnectionSuccessful, // 7
	UIStateEvent_UserPushesSessions, // 8
	UIStateEvent_UserSelectsSession, // 9
	UIStateEvent_UserSelectsNewSession, // 10
	UIStateEvent_UserPushesStartDefault, // 11
	UIStateEvent_UserPushesHardwareChecks, // 12
	UIStateEvent_UserPushesStartCalibFromOptions, // 13
	UIStateEvent_UserCancelsPopup, // 14
	UIStateEvent_UserAcksPopup, // 15
	UIStateEvent_ModelReady, // 16
	UIStateEvent_TrainingFailed, // 17
	UIStateEvent_UserPushesSettings, // 18
	UIStateEvent_UserSavesSettings, // 19
	UIStateEvent_None, // 20
};

enum UIPopup_E {
	UIPopup_None, // 0
	UIPopup_MustCalibBeforeRun, // 1
	UIPopup_ModelFailedToLoad, // 2
	UIPopup_TooManyBadWindowsInRun, // 3
	UIPopup_InvalidCalibOptions, // 4
	UIPopup_ConfirmOverwriteCalib, // 5
	UIPopup_ConfirmHighFreqOk, // 6
	UIPopup_TrainJobFailed, // 7
};

// SETTINGS PAGE ENUMS
enum SettingStimShape_E {
	// TODO IF TIME;
	StimShape_Circle,
	StimShape_Square,
	StimShape_Arrow,
};

enum SettingCalibData_E {
    CalibData_MostRecentOnly,
    CalibData_UsePastUpTo3
};

enum SettingTrainArch_E {
	TrainArch_CNN,
	TrainArch_SVM,
	TrainArch_RNN, 
};

enum SettingFreqRange_E {
	// TODO IF TIME;
};


/* END ENUMS */



/* START HELPERS */

inline int TestFreqEnumToInt(TestFreq_E enumVal){
	switch (enumVal) {
		case TestFreq_8_Hz:
			return 8;
		case TestFreq_10_Hz:
			return 10;
		case TestFreq_11_Hz:
			return 11;
		case TestFreq_12_Hz:
			return 12;
		case TestFreq_9_Hz:
			return 9;
		case TestFreq_20_Hz:
			return 20;
		case TestFreq_25_Hz:
			return 25;
		case TestFreq_30_Hz:
			return 30;
		case TestFreq_35_Hz:
			return 35;
		case TestFreq_None:
			return 0;
		default:
			return 0;
	}
}

inline std::string TrainArchEnumToString(SettingTrainArch_E e){
	switch (e) {
        case TrainArch_SVM: return "SVM";
        case TrainArch_CNN: return "CNN";
        default:            return "Unknown";
    }
}

inline std::string CalibDataEnumToString(SettingCalibData_E e){
	switch (e) {
        case CalibData_MostRecentOnly: return "most_recent_only";
        case CalibData_UsePastUpTo3:   return "all_sessions";
        default:                       return "Unknown";
    }
}

/* END HELPERS */

/* START STRUCTS */
/* Generic EEG sample type that will fill ring buffers: comprised of one scan of the number of enabled channels */
/*Possibly delete...*/
struct eeg_sample_t {
	std::vector<float> per_channel_values{}; // value for each of the 8 channels **could consider making this an array
	uint32_t tick;                           // monotonic sample index
	bool active_label;                       // obtained from stimulus global state 
};

/*
* bufferChunk_s: replacing eeg_sample_t
* A short duration, fixed-size, array of samples from the EEG device, grouped by time into "scans".
* ONE scan = ONE sample from EVERY enabled channel at a given time.
*/
struct bufferChunk_S {
	uint64_t tick = 0;                           // monotic sequence number (0,1,2,3...) assigned by producer so consumers can detect dropped chunks
	double epoch_ms = 0.0;                       // timestamp of first scan in chunk
	std::size_t numCh = NUM_CH_CHUNK; 		     // number of enabled channels
	std::size_t numScans = NUM_SCANS_CHUNK;      // number of scans (time steps) in this chunk (32)
	std::array<float, NUM_SAMPLES_CHUNK> data{}; // interleaved samples: [ch0s0, ch1s0, ch2s0, ..., chN-1s0, ch0s1, ch1s1, ..., chN-1sM-1]
	bool active_label;                           // obtained from stimulus global state
}; // bufferChunk_S

struct trainingProto_S {
	std::size_t numActiveBlocks; // number of blocks in this trial (1..N), assumes same number of L/R
	std::size_t activeBlockDuration_s; // duration of each active block in s
	std::size_t restDuration_s;  // rest duration between blocks in s
	bool displayInPairs; // whether or not the stimuli under test should be displayed in pairs or alone for 1 window
	std::deque<TestFreq_E> freqsToTest;  // TODO: make this stimulus_s instead of testFreq_E

}; // trainingProto_S

// STIMULUS OBJECT FACTORY
struct color_s {
    uint8_t r;
    uint8_t g;
    uint8_t b;

    color_s(uint8_t r_, uint8_t g_, uint8_t b_)
        : r(r_), g(g_), b(b_) {}
};

struct stimulus_s {

	stimulus_s(int freq, TestFreq_E freq_e, color_s color, SettingStimShape_E shape)
		: freq_hz_(freq), freq_hz_e_(freq_e), color_(color), shape_(shape) {}

	int freq_hz_;
	TestFreq_E freq_hz_e_;
	color_s color_{255, 255, 255};  // default white; use Color struct above
	SettingStimShape_E shape_;
};

// individual person run mode setup that gets formed after calibration
struct sessionConfigs_S {
	stimulus_s left_stimulus;
	stimulus_s right_stimulus;
};

/* SIGNAL STATS */
struct Stats_s {
	// value init all arrays to 0
    std::array<float, NUM_CH_CHUNK> mean_uv{};
    std::array<float, NUM_CH_CHUNK> std_uv{};
    std::array<float, NUM_CH_CHUNK> rms_uv{};
    std::array<float, NUM_CH_CHUNK> mad_uv{};
    std::array<float, NUM_CH_CHUNK> max_abs_uv{};
    std::array<float, NUM_CH_CHUNK> max_step_uv{};
    std::array<float, NUM_CH_CHUNK> kurt{};
    std::array<float, NUM_CH_CHUNK> entropy{};
	bool isBad = false;
};

// Running statistic measures of signals (rolling 45s)
// AFTER bandpass + CAR + artifact rejection
struct SignalStats_s {
	Stats_s rollingStats{};
    float current_bad_win_rate = 0.0; // last 45 s
    float overall_bad_win_rate = 0.0; // since start
    size_t num_win_in_rolling = 0;
};

/* SESSION INFO */
// - Consumer logging uses data_session_dir.
// - Training outputs use model_session_dir.
// - Both share the same subject_id + session_id.
struct SessionPaths {
    fs::path project_root;      // .../CapstoneProject
    std::string subject_id;     // "hadeel" or "person3"
    std::string session_id;     // "2025-12-22_14-31-08"
    fs::path data_session_dir;  // .../data/<subject>/<session>/
    fs::path model_session_dir; // .../models/<subject>/<session>/
};

/* END STRUCTS */