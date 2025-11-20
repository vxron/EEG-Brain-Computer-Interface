#pragma once
#include "../utils/Types.h"
#include "../utils/RingBuffer.hpp"

// unicorn sampling rate of 250 Hz means 1 scan is about 4ms (or, 32 scans per getData() call is about 128ms)
// for 1.2s windows -> we need 300 individual scans with hop 38 (every 0.152s)
inline constexpr std::size_t WINDOW_SCANS         = NUM_SCANS_CHUNK*10;     // 320 samples @250Hz (sampling period 4ms), this is 1.28s
inline constexpr std::size_t WINDOW_HOP_SCANS     = 40;      // every 0.16s (~87% overlap) 

struct sliding_window_t {
    size_t const winLen = WINDOW_SCANS*NUM_CH_CHUNK; // period of about 200*8ms = 1.6s
    size_t const winHop = WINDOW_HOP_SCANS*NUM_CH_CHUNK; // amount to jump for next window
    std::size_t tick = 0; // contains number of bufferchunk samples in window
	
	RingBuffer_C<float> sliding_window{WINDOW_SCANS*NUM_CH_CHUNK}; // major interleaved samples ; take by iterating over buffer chunks in ring buffer
	
	std::array<float, NUM_SAMPLES_CHUNK> stash{}; // overflow storage
	std::size_t stash_len = 0; // how many floats in stash are valid
	
	// labelling attributes (calib mode)
	bool has_label = false; 
	TestFreq_E testFreq = TestFreq_None; // from statestore
	TestFreq_E testFreq_other = TestFreq_None; // if we have two stims at a time on the screen... (isPaired = true)

	// associated feature vector/classification output for run mode
	SSVEPState_E decision = SSVEP_None;
};