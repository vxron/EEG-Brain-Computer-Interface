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

/*
* bufferChunk_s:
* A short duration, fixed-size, array of samples from the EEG device, grouped by time into "scans".
* ONE scan = ONE sample from EVERY enabled channel at a given time.
*/
struct bufferChunk_S {
	uint64_t seq = 0;                            // monotic sequence number (0,1,2,3...) assigned by producer so consumers can detect dropped chunks
	double t0 = 0.0;                             // timestamp of first scan in chunk (uses std::chrono::steady_clock::now())
	std::size_t numCh = NUM_CH_CHUNK; 		     // number of enabled channels
	std::size_t numScans = NUM_SCANS_CHUNK;      // number of scans (time steps) in this chunk 
	std::array<float, NUM_SAMPLES_CHUNK> data{}; // interleaved samples: [ch0s0, ch1s0, ch2s0, ..., chN-1s0, ch0s1, ch1s1, ..., chN-1sM-1]
}; // bufferChunk_S


/*
* public Class for per channel vectors that will be used for feature extraction
*/
struct dataVecPerCh_S {
	std::size_t chIdx; // e.g., 0, 1, etc
	std::string chLabel; // e.g., EEG0, EEG1, etc
	std::vector<float> dataVecPerCh; 
}; // dataVecPerCh_S