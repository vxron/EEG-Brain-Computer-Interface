#include <iostream> // std::cerr and std::endl
#include "SlidingWindow.h"
#include <spdlog/spdlog.h>   // main spdlog API (info/warn/error, set_pattern, set_level)

SlidingWindow_C::SlidingWindow_C(std::uint64_t windowLengthIn_ms, std::uint64_t hopIn_ms) : windowLength_ms(windowLengthIn_ms), hop_ms(hopIn_ms) {
	
	// import chunk period from Types.h constants
	constexpr std::uint64_t FS_HZ = 250; // Hz
	// return error if there's a remainder
	if( (1000 * NUM_SCANS_CHUNK) % FS_HZ != 0 ) {
		throw std::runtime_error("[ERROR] SlidingWindow_C: The chunk period (in ms) must be an integer value. Change constants in Types.h");
	}
	const std::uint64_t chunkPeriod_ms = (1000 * NUM_SCANS_CHUNK) / FS_HZ; // in ms, e.g. 128ms @ 250Hz, integer math ok here

	while (windowLength_ms % chunkPeriod_ms != 0) {
		// adjust window length to be multiple of chunk period
		windowLength_ms = ((windowLength_ms / chunkPeriod_ms) + 1) * chunkPeriod_ms; // round up to next multiple
		// log a warning
		std::cerr << "[WARNING] SlidingWindow_C: windowLength_ms adjusted to " << windowLength_ms << " ms to be a multiple of chunk period (" << chunkPeriod_ms << " ms)." << std::endl;
	}
	while (hop_ms % chunkPeriod_ms != 0) {
		hop_ms = ((hop_ms / chunkPeriod_ms) + 1) * chunkPeriod_ms; // round up to next multiple
		// log a warning
		std::cerr << "[WARNING] SlidingWindow_C: hop_ms adjusted to " << hop_ms << " ms to be a multiple of chunk period (" << chunkPeriod_ms << " ms)." << std::endl;
	}

	// need to define windowCapacity_ here to preallocate buffer_ storage so we don't reallocate on every push
	windowCapacity_ = static_cast<std::size_t>(windowLength_ms) / static_cast<std::size_t>(chunkPeriod_ms);
	hopSizeInChunks_ = static_cast<std::size_t>(hop_ms) / static_cast<std::size_t>(chunkPeriod_ms);

	buffer_.resize(windowCapacity_); // pre-allocate storage

	hopCountdown_ = hopSizeInChunks_; // must initialize before first decrement (timer)
}

bool SlidingWindow_C::push_chunk_to_window(const bufferChunk_S chunk, std::vector<float>* dest) {
	if (closed_) {
		return false;
	}

	// copy chunk into buffer at tailIdx_
	buffer_[tailIdx_] = std::move(chunk); // does this need to use std::move instead of just direct assignment?
	tailIdx_++;
	hopCountdown_--;

	if (chunkCount_ < windowCapacity_) {
		// still filling buffer, not ready
		chunkCount_++;
	}
	else {
		// buffer is full --> increment head to continue reflecting oldest value in queue
		headIdx_++;
		if (headIdx_ == windowCapacity_) {
			headIdx_ = 0; // wrap around
		}
		// check to see if hop countdown has expired, indicating time to emit
		if (hopCountdown_ == 0) {
			// only emit if buffer is full
			if (chunkCount_ == windowCapacity_) {
				emit_vectors_for_processing(dest);
				hopCountdown_ = hopSizeInChunks_; // reset
			}
			else {
				hopCountdown_ = 0; // keep at 0 in case its full next iter
			}
		}
	}
	if (tailIdx_ == windowCapacity_) {
		tailIdx_ = 0; // wrap around
	}

	return true;

}

void SlidingWindow_C::close() {
	closed_ = true;
}

// input is pointer to vector of dataVecPerCh_S objects 
bool SlidingWindow_C::emit_vectors_for_processing(std::vector<float>* dest) {
	// emit snapshot of current vector (cumulative chunks) for processing
	// format: chunks
	// each chunk contains N samples ch1.. ch8 
	// start popping from head
	
	if (!dest) return false;
	// ensure dest has enough space; if not, allocate it
	if (dest->size() != NUM_SAMPLES_CHUNK * windowCapacity_) {
		spdlog::warn("Suspicious: Resized float vector in SlidingWindow.cpp");
		dest->resize(NUM_SAMPLES_CHUNK * windowCapacity_);
	}

	
	std::size_t chunkIdx = headIdx_;
	for (std::size_t i = 0; i < windowCapacity_; i++) {
		bufferChunk_S oldestChunk = buffer_[chunkIdx];
		chunkIdx++;
		// wrap around
		if (chunkIdx == windowCapacity_) {
			chunkIdx = 0;
		}

		int offset = 0;
		int N = 0;
		// iterate over chunk and push floats to vector in CHANNEL-MAJOR LAYOUT [ch0 samples] [ch1 samples]... [chN-1 samples]
		for (int j = 0; j < NUM_SAMPLES_CHUNK; j++) {
			
			if (offset < NUM_CH_CHUNK) {
				offset++;
			}
			else {
				// new sample started, increment sample number
				N++;
				offset = 0; // reset
			}

			(*dest)[i*NUM_SAMPLES_CHUNK + offset+N*NUM_CH_CHUNK] = oldestChunk.data[j];
			
		}
	}

	return true;

}



