#include "Stim.h"

bool Stim_C::run_training_protocol(const trainingProto_S& trainingProto) {
	blockOpen_ = false;
	indexedLabels_.clear();
	indexedLabels_.shrink_to_fit();

	for (int i = 0; i < trainingProto.numActiveBlocks; i++) {
		// every active block will have an associated preparotory (rest) block with instructions for the next active one
		// tell stim to display instructions for next block
		std::string instrLeft = "Get ready to focus on the LEFT flicker for " + std::to_string(trainingProto.activeBlockDuration_s) + " seconds.";
		std::string instrRight = "Get ready to focus on the RIGHT flicker for " + std::to_string(trainingProto.activeBlockDuration_s) + " seconds.";
		std::string instrRest = "Rest for " + std::to_string(trainingProto.restDuration_s) + " seconds.";

		// do we want to wait for user to be ready here?
		if (i != 0) {
			start_new_block(TrainingBlock_Instructions, instrRest);
		}
		
		// tell stim to display arrows/active block
		switch (trainingProto.pattern) {
			case ProtocolPattern_Alternating:
				if (i % 2 == 0) {
					start_new_block(TrainingBlock_Instructions, instrLeft);
					close_current_block(); // close instructions block when user is ready (or after fixed time?)
					start_new_block(TrainingBlock_ActiveLeft);
				}
				else {
					start_new_block(TrainingBlock_Instructions, instrRight);
					close_current_block(); // close instructions block when user is ready (or after fixed time?)
					start_new_block(TrainingBlock_ActiveRight);
				}
				break;
			case ProtocolPattern_Blocked:
				if (i < trainingProto.numActiveBlocks / 2) {
					start_new_block(TrainingBlock_Instructions, instrLeft);
					close_current_block(); 
					start_new_block(TrainingBlock_ActiveLeft);
				}
				else {
					start_new_block(TrainingBlock_Instructions, instrRight);
					close_current_block();
					start_new_block(TrainingBlock_ActiveRight);
				}
				break;
			case ProtocolPattern_Random:
				if (rand() % 2 == 0) {
					start_new_block(TrainingBlock_Instructions, instrLeft);
					close_current_block();
					start_new_block(TrainingBlock_ActiveLeft);
				}
				else {
					start_new_block(TrainingBlock_Instructions, instrRight);
					close_current_block();
					start_new_block(TrainingBlock_ActiveRight);
				}
				break;
			default:
				// should never happen
				start_new_block(TrainingBlock_ActiveLeft);
				break;
		}

		close_current_block();

	}



	// tell stimulus to change frequency

}

// when stimulus reads those new blocks in indexedlabels_, should drain, i.e. reset vector
// replaces current indexedLabels vector in TM with most updated version from Stim
bool Stim_C::drain_labels_to_tm(std::vector<LabelSource_S>& dest) {
	std::copy(indexedLabels_.begin(), indexedLabels_.end(), dest.end());
	// wipe indexedLabels since we now have them stored safely in TimingManager
	indexedLabels_.clear();
	indexedLabels_.shrink_to_fit(); // should i do this or since were gonna reallocate anyways on next query, it is useless or harmful?
	return true;
}

void Stim_C::start_new_block(TrainingBlocks_E blockType, std::optional<std::string> text) {
	// If a block is already open, close it
	if (blockOpen_) {
		close_current_block();
	}
	blockOpen_ = true;
	currentOpenBlock_.blockType = blockType;

	switch (blockType) {
		case TrainingBlock_ActiveLeft:
			currentOpenBlock_.label = SSVEP_Left;
			// switch window type to arrows and leave for duration of active block
			break;
		case TrainingBlock_ActiveRight:
			currentOpenBlock_.label = SSVEP_Right;
			// switch window type to arrows and leave for duration of active block
			break;
		case TrainingBlock_Instructions:
			currentOpenBlock_.label = SSVEP_None;
			// switch window type to instructions and leave for duration of rest block
			// display text if provided
			break;
		case TrainingBlock_None:
			currentOpenBlock_.label = SSVEP_None;
			break;

	}

	currentOpenBlock_.blockStartTime = clock_T::now();

}

void Stim_C::close_current_block() {
	if (!blockOpen_) {
		return; // nothing to close
	}
	currentOpenBlock_.blockEndTime = clock_T::now();
	indexedLabels_.push_back(currentOpenBlock_);
	blockOpen_ = false;
}

