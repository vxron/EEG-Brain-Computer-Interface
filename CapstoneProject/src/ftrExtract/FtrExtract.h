#pragma once
#include "SlidingWindow.h"
#include <vector>

class FtrExtract_C {
public:
	// vector of floats writable to from SlidingWindow module
	std::vector<float> dataVector_;
private:
}; // FtrExtract_C


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
			(*dest)[k].dataVecPerCh.push_back(oldestChunk.data[j]); // is it arrow or point idk to access data vector of dataVecPerCh_S object??
			k++;
		}
		else {
			// new sample started, write to 0th
			k = 0;
			(*dest)[k].dataVecPerCh.push_back(oldestChunk.data[j]);
			k = 1; // increment for next iter
		}

	}
}