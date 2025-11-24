/*
==============================================================================
	File: FakeAcquisition.h
	Desc: Internally, this module generates a continuous EEG sample stream at 250Hz, 
	mimicking the Unicorn EEG headset. The data is published as fixed size-N chunks,
	mimicking the UNICORN_GetData() API call.
	The UNICORN_GetData() returns digitized, scaled EEG values in microvolts, so
	this is the unit used here, representing typical values.

	This header is used by:
  - Acquisition: reads in chunks and pushes them to the ring buffer.

	Layout contract for dest buffer (matches bufferChunk_S):
  - Time-major interleaved scans:
	  idx = scan * nCh + ch
  - Units: microvolts (uV)

==============================================================================
*/

#pragma once
#include <cstddef>   // std::size_t
#include <cstdint>   // uint32_t
#include "../utils/Types.h" // gives NUM_CH_CHUNK, NUM_SCANS_CHUNK
#include <random>    // std::mt19937, std::normal_distribution 
#include "IAcqProvider.h" // IAcqProvider_S

class FakeAcquisition_C : public IAcqProvider_S {
public:
	// Configs Object (must be declared before in-class usage/declarations)
	// NOTE: Uses NUM_CH_CHUNK and NUM_SCANS_CHUNK from Types.h for layout
	// TODO: knobs for signal/noise variation
	// TODO: on/off for 2nd harmonic
	// TODO: enable/disable real-time sleeping/timing jitter
	struct stimConfigs_S {
		double ssvepAmplitude_uV = 20.0;
		double noiseSigma_uV = 5.0;
		double leftStimFreq = 15.0; // Hz
		double rightStimFreq = 10.0; // Hz
	}; // stimConfigs_S
	
	// Constructors/Destructors
	explicit FakeAcquisition_C(const stimConfigs_S &configs);
	// should not be able to copy: delete copy constructor/assignment operator
	FakeAcquisition_C(const FakeAcquisition_C&) = delete;
	FakeAcquisition_C& operator=(const FakeAcquisition_C&) = delete;
	// Default move constructor/assignment operator, destructor
	FakeAcquisition_C(FakeAcquisition_C&&) = default;
	FakeAcquisition_C& operator=(FakeAcquisition_C&&) = default;
	~FakeAcquisition_C() = default;

	// Lifecycle methods to match IAcqProvider_S interface
	bool unicorn_init() override { return true; } // nothing to init
	bool unicorn_start_acq() override { return true; }
	bool unicorn_stop_and_close() override {return true; } // nothing to cleanup

	bool getData(std::size_t const numberOfScans, float* dest) override; // mirrors Unicorn C API GetData()
	void setActiveStimulus(double fStimHz); // sets the active stimulus frequency (0 = none)

private:
	const double fs = 250.0;
	double phaseOffset_ = 0; // persistent variable to keep track of phase offset between synthesize_data_stream function calls for continuity
	void synthesize_data_stream(float* dest, std::size_t numberOfScans); // used by mock_GetData
	std::size_t sampleCount_ = 0; // running sample counter to stamp each chunk's starting sample index
	double activeStimulusHz_ = 0.0; // current active stimulus frequency (0 = none)
	stimConfigs_S configs_{}; // default initiliazer _{}

	// Noise generator state
	std::mt19937 rng_;
	std::normal_distribution<double> noiseNorm_{ 0.0, 1.0 }; // std=1; scaled later
}; // FakeAcquisition_C