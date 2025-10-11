/*
==============================================================================
	File: TimingManager.h
	Desc: Owns time reference and protocol (training) scheduling.
	* drives stimulus changes in calibration mode
	* we can decide if we want it to drive stimulus changes in run mode (probably not)
	* USE REAL TIME (NOT INDICES)
==============================================================================
*/

#include <cstdint>
#include "Types.h"
#include <vector>
#include <chrono>

// _T for type
// Use steady clock for time measurements (monotonic, not affected by system clock changes)
using clock_T = std::chrono::steady_clock;
using ms_T = std::chrono::milliseconds;
using time_point_ms_T = std::chrono::time_point<clock_T, ms_T>;

class TimingManager_C {
public:
	// set/get base time reference (called at protocol initiation), return time_point in ms
	time_point_ms_T set_and_get_base_time();/*
		baseTime_ = steady_clock::now();
		return baseTime_;
	}*/
	//returns tBase;
	
	// Lifecycle
	bool is_started() const { return started_;  }; // true after start_stream() until reset/stop
	// the following gets called from App 
	void run_training_protocol(const trainingProto_S& trainingProto);
	// Disallow copying
	TimingManager_C(const TimingManager_C&) = delete;
	TimingManager_C& operator=(const TimingManager_C&) = delete;

	// when stimulus acks
	void on_stimulus_ack(); // called by stimulus thread when stimulus changes. body should set timestamp tend
	
	// LOOKUP METHOD
	// guard will get implemented in this function --> what sliding window queries on every emit
	SSVEPState_E get_label_for_window(const double& winStartTime, const double& winEndTime) const; // returns label for window if valid, else SSVEP_None

private:
	// LABEL SOURCE
	// should it be a vector of (time, label) or (idx, label)???? whats better??
	// INTERNAL INTERVAL RECORD: stores the intervals of labels in terms of indices (steady, guard already applied)
	// gets used in get_label_for_window() to answer queries
	struct LabelSource_S {
		std::size_t blockStartTime; // timestamp for when new instruction block is sent to display thread
		std::size_t blockEndTime;   // timestamp for when display completes the block and sends ack
		SSVEPState_E label;         // right, left, none (if no ssvep task, e.g. idle, setup, instructions, rest)
		uint32_t blockId = 0;       // number of blocks seen so far in the protocol (seq 0,1, 2... unless missing data)
	};

	bool started_ = false;
	bool block_open_ = false; // true if a calibration block is currently open (waiting for close)

	// the following would get called from run_training_protocol()
	void start_new_block(); // prompts stim with new display instruction, updates new labelSource_S instance currentOpenBlock_
	
	// the following would get called from on_stimulus_ack()
	void close_current_block(); // closes current block by writing end time to currentOpenBlock_ and pushing it to indexedLabels_
	void end_calibration_protocol(); // resets internal counters/state

	std::vector < LabelSource_S > indexedLabels_; // Stores all labels from the current training protocol
	std::size_t headIdx_;                         // where we start read from in next query from sliding window

	LabelSource_S currentOpenBlock_; // current open block that gets updated when stimulus changes, closed when next block opens

	time_point_ms_T baseTime_;

	/*
	std::size_t allowedEndIdxForEmit_ = 0; // guard, will use one of the conversion functions for time to index 
	std::size_t allowedStartIdxForEmit_ = 0; // guard
	// then SW will wait until start_idx > next_allowed_start_idx and run until end idx is allowed_start_idx_for_emit
	*/

	/*
	float idx_to_t_event(uint32_t idx); // converts idx (eeg sample indices) to stimulus events
	uint32_t t_event_to_idx(float event_time_ms); // convert stimulus schedules to index spans
	*/
	
	
	// Stores the block timeline from TM to answer queries like "is window [i,j) valid and what its label?"
		// how do they get the label? from STIMULUS TO TM --> when there's a stimulus event (change)
	/*
	* assume from these timestapmedlabels_ the TM must have a func to calc allowed end/start idx based on the guard and stuff ? and it does this everytime it receives a new event_t from the stimulus?
	*/
}; // TimingManager_C

