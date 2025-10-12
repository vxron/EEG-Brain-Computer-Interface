/*
==============================================================================
	File: TimingManager.h
	Desc: Owns time reference and protocol (training) scheduling.
	* drives stimulus changes in calibration mode
	* we can decide if we want it to drive stimulus changes in run mode (probably not)
	* USE REAL TIME (NOT INDICES)
==============================================================================
*/
#pragma once
#include <cstdint>
#include "Types.h"
#include <vector>
#include <chrono>

class TimingManager_C {
public:
	// set/get base time reference (called at protocol initiation), return time_point in ms
	time_point_T get_base_time() const { return baseTime_; };
	
	// Lifecycle
	bool is_started() const { return started_;  }; // true after start_stream() until reset/stop
	// the following gets called from App 
	void run_training_protocol(const trainingProto_S& trainingProto);
	
	TimingManager_C();
	~TimingManager_C() = default;
	// Disallow copying
	TimingManager_C(const TimingManager_C&) = delete;
	TimingManager_C& operator=(const TimingManager_C&) = delete;
	// Allow moving
	TimingManager_C(TimingManager_C&&) = default;
	TimingManager_C& operator=(TimingManager_C&&) = default;

	// when stimulus acks
	//void on_stimulus_ack(); // called by stimulus thread when stimulus changes . body should set timestamp tend
	
	// LOOKUP METHOD
	// guard will get implemented in this function --> what sliding window queries on every emit
	// only time TM should drain events from the stim's indexedLabels into its own indexedLabels_
	SSVEPState_E get_label_for_window(const time_point_T& winStartTime, const time_point_T& winEndTime); // returns label for window if valid, else SSVEP_None

private:
	// LABEL SOURCE
	// should it be a vector of (time, label) or (idx, label)???? whats better??
	// INTERNAL INTERVAL RECORD: stores the intervals of labels in terms of indices (steady, guard already applied)
	// gets used in get_label_for_window() to answer queries
	
	bool started_ = false;
	void end_calibration_protocol(); // resets internal counters/state
	void drain_events_to_indexed_labels(); // drains events from stimulus to indexedLabels_ (called when stim window makes query)
	// can replace from start (reset head, tail) since sliding window has presumably already seen those labels from last query and time has moved on

	std::vector < LabelSource_S > indexedLabels_; // Stores all labels from the current training protocol
	std::size_t headIdx_ = 0;                     // where we start read from in next query from sliding window
	std::size_t tailIdx_ = 0;                     // where we write new events to in indexedLabels_

	time_point_T baseTime_;

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

