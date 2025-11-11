#include "TimingManager.h"
#include "stimulus/Stim.h"

constexpr auto GUARD_MS = 2500; // guard in ms to avoid transition periods

// have main call set_and_get_base_time() when protocol starts
TimingManager_C::TimingManager_C(){
	baseTime_ = clock_T::now();
}

void TimingManager_C::run_training_protocol(const trainingProto_S& trainingProto) {
	started_ = true;
	Stim_C stimulus;
	stimulus.run_training_protocol(trainingProto);
	// needs to open another thread so this thread can continue doing other stuff
}

SSVEPState_E TimingManager_C::get_label_for_window(const time_point_T& winStartTime, const time_point_T& winEndTime) {
	// Get most updated (recent) batch of labels from Stim.h
	Stim_C stimulus;
	// dest should get all of them, not just the new ones since last query
	stimulus.drain_labels_to_tm(indexedLabels_);

	// for the first one we want to start from headIdx_ = 0
	// for subsequent ones, we want to start from where we left off (headIdx_)

	if(headIdx_ >= indexedLabels_.size()) {
		// no new labels to process
		return SSVEP_None;
	}
	std::size_t i;
	for (i = headIdx_; i < indexedLabels_.size(); i++) {
		// find first entry where window start time is > block start time
		// and window end time is < block end time
		// with a margin of error (guard) to avoid transition periods

		auto frontMargin = winStartTime - indexedLabels_[i].blockStartTime;
		auto frontMarginMs = std::chrono::duration_cast<ms_T>(frontMargin).count();
		auto backMargin = indexedLabels_[i].blockEndTime - winEndTime;
		auto backMarginMs = std::chrono::duration_cast<ms_T>(backMargin).count();

		
		if (winStartTime > indexedLabels_[i].blockEndTime) {
			// window starts after this block ends, move to next block
			// we can also increment headIdx_ here since we know this block is useless for future queries
			headIdx_++;
			continue;
		}

		// now we know window is in this block or overlaps with it
		else if (frontMarginMs > GUARD_MS && backMarginMs > GUARD_MS) {
			// found a valid label for this window
			return indexedLabels_[i].label;
		}

		else if (i == indexedLabels_.size() - 1) {
			// reached end of indexedLabels_ without finding a valid label
			return SSVEP_None;
		}

		else {
			// window overlaps with block but is within guard period, return None
			return SSVEP_None;
		}
	}

	
}

