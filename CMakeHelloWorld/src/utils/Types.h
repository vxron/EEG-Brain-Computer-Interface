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

// Tunables for chunking policy
constexpr std::size_t NUM_CH_CHUNK = 8; // Unicorn EEG has 8 channels (EEG1...EEG8)
constexpr std::size_t NUM_SCANS_CHUNK = 32; // ~128ms @ 250Hz
constexpr std::size_t NUM_SAMPLES_CHUNK = NUM_CH_CHUNK * NUM_SCANS_CHUNK;

/* START ENUMS */

// use enum CLASS to avoid name collisions & strongly typed (no implicit conversion to/from int)
enum AppState_E {
	appState_Calibrate,
	appState_Idle,
	appState_Run,
	appState_Shutdown,
	appState_Error
}; // AppMode_E

enum SSVEPState_E {
	SSVEP_Left,
	SSVEP_Right,
	SSVEP_None,
	SSVEP_Unknown, // e.g., classifier score below threshold
}; // SSVEPState_E

enum ProtocolPattern_E {
	Pattern_Alternating,  // L R L R L R ...
	Pattern_Blocked,      // L L L ... R R R ... (half time left, half time right)
	Pattern_Random,       // random sequence of L and R
}; // ProtocolPattern_E

/* END ENUMS */

/* START STRUCTS */
/*
* bufferChunk_s:
* A short duration, fixed-size, array of samples from the EEG device, grouped by time into "scans".
* ONE scan = ONE sample from EVERY enabled channel at a given time.
*/
struct bufferChunk_S {
	uint64_t seq = 0;                            // monotic sequence number (0,1,2,3...) assigned by producer so consumers can detect dropped chunks
	uint64_t idx0 = 0;                           // absolute index of the first sample in this chunk (sample 0 of the entire stream is 0)
	double epoch_ms = 0.0;                       // timestamp of first scan in chunk
	std::size_t numCh = NUM_CH_CHUNK; 		     // number of enabled channels
	std::size_t numScans = NUM_SCANS_CHUNK;      // number of scans (time steps) in this chunk 
	std::array<float, NUM_SAMPLES_CHUNK> data{}; // interleaved samples: [ch0s0, ch1s0, ch2s0, ..., chN-1s0, ch0s1, ch1s1, ..., chN-1sM-1]
}; // bufferChunk_S

struct WindowMeta_S {
	std::uint64_t tStart_ms{}; // window time start
	std::uint64_t tEnd_ms{}; // window time end
	// helps us check for drift, gaps
	std::uint64_t idxStart{};   // first absolute sample index in stream
	std::uint64_t idxEnd{};   // last absolute sample index
};

struct EmittedWindow_S {
	// Major-interleaved samples, ready for feature extractor
	std::vector<float> samples_major;
	WindowMeta_S meta;
	// Present only in Calibrate; empty in Run
	SSVEPState_E label;
};

struct trainingProto_S {
	std::size_t numActiveBlocks; // number of blocks in this trial (1..N), assumes same number of L/R
	std::size_t activeBlockDuration_s; // duration of each active block in s
	std::size_t restDuration_s;  // rest duration between blocks in s
	ProtocolPattern_E pattern; // alternating, blocked, random
}; // trainingProto_S

/* END STRUCTS */