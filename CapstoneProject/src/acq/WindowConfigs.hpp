#pragma once
#include "../utils/Types.h"
#include "../utils/RingBuffer.hpp"

// unicorn sampling rate of 250 Hz means 1 scan is about 4ms (or, 32 scans per getData() call is about 128ms)
inline constexpr std::size_t WINDOW_SCANS         = NUM_SCANS_CHUNK*20;     // 640 samples @250Hz (sampling period 4ms), this is 2.56s
inline constexpr std::size_t WINDOW_HOP_SCANS     = 80;      // every 0.32s (87.5% overlap) 

struct sliding_window_t {
    size_t const winLen = WINDOW_SCANS*NUM_CH_CHUNK;
    size_t const winHop = WINDOW_HOP_SCANS*NUM_CH_CHUNK; // amount to jump for next window
    
	std::size_t tick = 0; // contains number of bufferchunk samples in window
	
	RingBuffer_C<float> sliding_window{WINDOW_SCANS*NUM_CH_CHUNK}; // major interleaved samples ; take by iterating over buffer chunks in ring buffer
	
	std::array<float, NUM_SAMPLES_CHUNK> stash{}; // overflow storage
	std::size_t stash_len = 0; // how many floats in stash are valid

	// Window quality score to detect artifacts
	bool isArtifactualWindow = 0;
	
	// labelling attributes (calib mode)
	bool has_label = false; 
	TestFreq_E testFreq = TestFreq_None; // from statestore
	TestFreq_E testFreq_other = TestFreq_None; // if we have two stims at a time on the screen... (isPaired = true)

	// associated feature vector/classification output for run mode
	SSVEPState_E decision = SSVEP_None;
};