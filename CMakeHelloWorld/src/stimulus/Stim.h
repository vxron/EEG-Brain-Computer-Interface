/*
==============================================================================
	File: Stim.h
	Desc:
==============================================================================
*/

#pragma once
#include <cstdint>
#include <string>
#include "../utils/Types.h"
#include <vector>
#include <optional>


class Stim_C {
public:
	// Gets called from TM
	bool run_training_protocol(const trainingProto_S& trainingProto);
	bool drain_labels_to_tm(std::vector<LabelSource_S>& dest);
private:
	// comes from app settings ideally; for now we can change in code
	struct stimConfig_S {
		float leftFreq_Hz = 10.0f;  // default 10Hz
		float rightFreq_Hz = 15.0f; // default 15Hz
		float brightness = 1.0f;    // [0.0..1.0]
		std::string windowName = "SSVEP Stimulus"; // name of the stimulus window
		int windowPosX = 100;       // position of stimulus window on screen (x)
		int windowPosY = 100;       // position of stimulus window on screen (y)
		int windowWidth = 800;      // width of stimulus window in pixels
		int windowHeight = 600;     // height of stimulus window in pixels
	}; // stimConfig_S
	
	bool blockOpen_ = false; // true if a calibration block is currently open (waiting for close)

	// the following would get called from run_training_protocol()
	void start_new_block(TrainingBlocks_E blockType, std::optional<std::string> text); // prompts stim with new display instruction, updates new labelSource_S instance currentOpenBlock_
	void close_current_block(); // closes current block by writing end time to currentOpenBlock_ and pushing it to indexedLabels_

	LabelSource_S currentOpenBlock_; // current open block that gets updated when stimulus changes, closed when next block opens


	std::vector<LabelSource_S> indexedLabels_; // stores all labels from the current training protocol
};