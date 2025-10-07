#include <iostream> // std::cerr and std::endl
#include "SlidingWindow.h"

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
}

bool SlidingWindow_C::push_chunk_to_window(const bufferChunk_S& chunk, std::vector<dataVecPerCh_S>* dest) {
	if (closed_) {
		return false;
	}

	std::size_t hopCountdown = hopSizeInChunks_;

	// copy chunk into buffer at tailIdx_
	buffer_[tailIdx_] = chunk; // does this need to use std::move instead of just direct assignment?
	tailIdx_++;
	hopCountdown--;

	if (chunkCount_ < windowCapacity_) {
		// still filling buffer, not ready
		chunkCount_++;
	}
	else {
		// buffer is full --> increment head to continue reflecting oldest value in queue
		headIdx_++;
		// check to see if hop countdown has expired, indicating time to emit
		if (hopCountdown <= 0) {
			emit_vectors_for_processing(dest);
			hopCountdown = hopSizeInChunks_; // reset
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
bool SlidingWindow_C::emit_vectors_for_processing(std::vector<dataVecPerCh_S>* dest) {
	// emit snapshot of current vector (cumulative chunks) for processing
	// format: chunks
	// each chunk contains N samples ch1.. ch8 
	// start popping from head
	
	std::size_t chunkIdx = headIdx_;
	for (std::size_t i = 0; i < windowCapacity_; i++) {
		bufferChunk_S oldestChunk = buffer_[chunkIdx];
		chunkIdx++;
		// wrap around
		if (chunkIdx == windowCapacity_) {
			chunkIdx = 0;
		}

		int k = 0;
		// iterate over chunk and push floats to per channel vectors
		for (int j = 0; j < NUM_SAMPLES_CHUNK; j++) {
			
			// 0 should go to ch0, up to num of channels, then repeat
			if (k < NUM_CH_CHUNK) {
				// write to channel
				dest[k]->dataVecPerCh.push_back(oldestChunk.data[j]); // is it arrow or point idk to access data vector of dataVecPerCh_S object??
				k++;
			}
			else {
				// new sample started, write to 0th
				k = 0;
				dest[k]->dataVecPerCh.push_back(oldestChunk.data[j]);
				k = 1; // increment for next iter
			}
			
		}
	}

}



