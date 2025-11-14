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

// unicorn sampling rate of 250 Hz means 1 scan is about 4ms (or, 32 scans per getData() call is about 128ms)
// for 1.2s windows -> we need 300 individual scans with hop 38 (every 0.152s)
inline constexpr std::size_t WINDOW_SCANS         = NUM_SCANS_CHUNK*10;     // 320 samples @250Hz (sampling period 4ms), this is 1.28s
inline constexpr std::size_t WINDOW_HOP_SCANS     = 40;      // every 0.16s (~87% overlap)  

/* END CONFIGS */

/* START ENUMS */

// use enum CLASS to avoid name collisions & strongly typed (no implicit conversion to/from int)
enum AppState_E {
	AppState_Calibrate,
	AppState_Idle,
	AppState_Run,
	AppState_Shutdown,
	AppState_Error
}; // AppMode_E

enum SSVEPState_E {
	SSVEP_Left,
	SSVEP_Right,
	SSVEP_None,
	SSVEP_Unknown,
}; // SSVEPState_E

enum TestFreq_E {
	TestFreq_None,
	TestFreq_8_Hz,
	TestFreq_9_Hz,
	TestFreq_10_Hz,
	TestFreq_11_Hz,
	TestFreq_12_Hz
};

enum ActuatorState_E {
	ActuatorState_Fwd,
	ActuatorState_Bcwd,
	ActuatorState_None,
};

enum UIState_E {
	UIState_Active_Run,
	UIState_Active_Calib,
	UIState_Instructions, // also for calib
	UIState_Home,
	UIState_None,
};

enum UIStateEvent_E {
	UIStateEvent_StimControllerTimeout, // switch btwn instructions/active during calib
	UIStateEvent_StimControllerTimeoutEndCalib,
	UIStateEvent_UserPushesStartRun,
	UIStateEvent_UserPushesStartCalib,
	UIStateEvent_LostConnection,
	UIStateEvent_UserPushesExit,
	UIStateEvent_ConnectionSuccessful,
};

enum ProtocolPattern_E {
	Pattern_Alternating,  // L R L R L R ...
	Pattern_Blocked,      // L L L ... R R R ... (half time left, half time right)
	Pattern_Random,       // random sequence of L and R
}; // ProtocolPattern_E

enum BitOperation_E {
	BitOp_Toggle,
	BitOp_Set,
	BitOp_Clear,
	BitOp_Read
}; // BitOperation_E

/* END ENUMS */

/* START STRUCTS */


/* Generic EEG sample type that will fill ring buffers: comprised of one scan of the number of enabled channels */
/*Possibly delete...*/
struct eeg_sample_t {
	std::vector<float> per_channel_values{}; // value for each of the 8 channels **could consider making this an array
	uint32_t tick;                           // monotonic sample index
#if CALIB_MODE
	bool active_label;                       // obtained from stimulus global state 
#endif 
};

/*
* bufferChunk_s: replacing eeg_sample_t
* A short duration, fixed-size, array of samples from the EEG device, grouped by time into "scans".
* ONE scan = ONE sample from EVERY enabled channel at a given time.
*/
struct bufferChunk_S {
	uint64_t tick = 0;                            // monotic sequence number (0,1,2,3...) assigned by producer so consumers can detect dropped chunks
	double epoch_ms = 0.0;                       // timestamp of first scan in chunk
	std::size_t numCh = NUM_CH_CHUNK; 		     // number of enabled channels
	std::size_t numScans = NUM_SCANS_CHUNK;      // number of scans (time steps) in this chunk (32)
	std::array<float, NUM_SAMPLES_CHUNK> data{}; // interleaved samples: [ch0s0, ch1s0, ch2s0, ..., chN-1s0, ch0s1, ch1s1, ..., chN-1sM-1]
#if CALIB_MODE
	bool active_label;                       // obtained from stimulus global state 
#endif 
}; // bufferChunk_S

struct sliding_window_t {
    size_t const winLen = WINDOW_SCANS*NUM_CH_CHUNK; // period of about 200*8ms = 1.6s
    size_t const winHop = WINDOW_HOP_SCANS*NUM_CH_CHUNK; // amount to jump for next window
    RingBuffer_C<float> sliding_window{WINDOW_SCANS*NUM_CH_CHUNK}; // major interleaved samples ; take by iterating over buffer chunks in ring buffer
	std::array<float, NUM_SAMPLES_CHUNK> stash{}; // overflow storage
	std::size_t stash_len = 0; // how many floats in stash are valid
	std::size_t tick = 0;
	// then each % somthn in this sliding_window RingBuf will correspond to a particular channel ! (can export to csv)
};

// send to stimulus module directly from timing manager
struct trainingProto_S {
	std::size_t numActiveBlocks; // number of blocks in this trial (1..N), assumes same number of L/R
	std::size_t activeBlockDuration_s; // duration of each active block in s
	std::size_t restDuration_s;  // rest duration between blocks in s
	ProtocolPattern_E pattern; // alternating, blocked, random
}; // trainingProto_S

struct LabelSource_S {
	time_point_T blockStartTime; // timestamp for when new instruction block is sent to display thread
	time_point_T blockEndTime;   // timestamp for when display completes the block and sends ack
	SSVEPState_E label = SSVEP_None;         // right, left, none (if no ssvep task, e.g. idle, setup, instructions, rest)
	uint32_t blockId = 0;       // number of blocks seen so far in the protocol (seq 0,1, 2... unless missing data)
	//TrainingBlocks_E blockType = TrainingBlock_None; // type of block (rest, instructions, active)
};






/* END STRUCTS */