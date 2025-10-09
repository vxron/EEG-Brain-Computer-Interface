/*
==============================================================================
	File: SlidingWindow.h
	Desc: Maintains a rolling, fixed-length, multi-channel EEG window with a configurable
	hop (step). Consumes full chunks from acquisition and emits feature processing/extraction-ready 
	windows at hop boundaries.
	Threading Model: 
	- Called only from the consumer (i.e., decoder thread).
	- No internal locks. Do not call from multiple threads.
	Inputs:
	- fs (Hz), nCh (channels), window length (seconds), hop length (seconds)
	Outputs:
	- full window for downstream preproc/features
==============================================================================
*/

#pragma once
#include <cstddef>   // std::size_t
#include <vector>    // std::vector
#include "../utils/Types.h"   // bufferChunk_S
#include <cstdint>   // uint64_t

// Sliding window class
class SlidingWindow_C {
public:
	std::uint64_t windowLength_ms = 2560; // 20 128ms chunks
	std::uint64_t hop_ms = 640; // 5 128ms chunks

	// Constructor needs time window length & hop
	explicit SlidingWindow_C(std::uint64_t windowLength_ms, std::uint64_t hop_ms);

	// This queue should not be copyable: delete copy constructor/assignment operator
	SlidingWindow_C(const SlidingWindow_C&) = delete;
	SlidingWindow_C& operator=(const SlidingWindow_C&) = delete;
	// This queue can be moveable (RAII): default move constructor/assignment operator
	SlidingWindow_C(SlidingWindow_C&&) = default;
	SlidingWindow_C& operator=(SlidingWindow_C&&) = default;
	~SlidingWindow_C() = default;

	// Method to push a new chunk into the sliding window
	bool push_chunk_to_window(const bufferChunk_S chunk, std::vector<float>* dest); // pass by const ref to avoid copy, non-const because we modify internal state
	void close(); // close the queue

private:
	// don't need to call this from any other module/class
	// emits chunk in the form of a vector of floats to processing unit 
	// dest is pointer to float vector that should get formed in process module
	bool emit_vectors_for_processing(std::vector<float>* dest); //should it be const? dont think so cuz its modifying data at dest?

	EmittedWindow build_payload_for_current_window_();

	std::vector<bufferChunk_S> buffer_; // the ring buffer storage; preallocate upon construction

	bool closed_ = false;

	std::size_t tailIdx_ = 0; // index of tail (where next chunk should be pushed to in circular buffer)
	std::size_t headIdx_ = 0; // index of head (where next chunk should be read from during emission)
	std::size_t windowCapacity_; // total number of chunks needed in buffer for "full" state
	std::size_t hopSizeInChunks_; // number of chunks to hop/step on each emit
	std::size_t chunkCount_ = 0; // number of chunks currently in buffer
	std::size_t hopCountdown_; // hop counter to keep track of emit time

}; // SlidingWindow_C