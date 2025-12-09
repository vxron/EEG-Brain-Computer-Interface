/*
==============================================================================
	File: FakeAcquisition.h
	Desc: Internally, this module generates a continuous EEG sample stream at 250Hz, 
	mimicking the Unicorn EEG headset. The data is published as fixed size-N chunks,
	mimicking the UNICORN_GetData() API call.
	The UNICORN_GetData() returns digitized, scaled EEG values in microvolts, so
	this is the unit used here, representing typical values.

	Layout contract for dest buffer (matches bufferChunk_S):
  - Time-major interleaved scans:
	  idx = scan * nCh + ch
  - Units: microvolts (uV)

  NOTE: This sims calib mode only (one specific frequency we're testing based on 
  state store, see producer thread in main)

  NOTE: Not designed to be used across multiple threads (no atomics)
  (Should be used in 'producer' only, otherwise race conds could arise)

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
	struct waveComponent_S {
		double freqHz   = 0.0; // 0 = off, or use enabled flag
		double amp_uV   = 0.0;
    	bool   enabled  = false;
	};
	struct stimConfigs_S {

		double ssvepAmplitude_uV = 20.0;
		double noiseSigma_uV = 5.0;

		// Background components (freq, amp (uV), enabled)
		waveComponent_S dcDrift   { 0.1,  3.0,  false };  // "drift" at 0.1 Hz
    	waveComponent_S alpha     { 10.0, 4.0,  false };  // 8–12 Hz band
    	waveComponent_S beta      { 20.0, 3.0,  false };  // 12–30 Hz band
    	waveComponent_S lineNoise { 60.0, 5.0,  false };  // 60hz noise

		bool occasionalArtifactsEnabled = 0;

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

	int getNumChannels() const override {
        return numChannels_;
    }

    void getChannelLabels(std::vector<std::string>& out) const override {
        out = channelLabels_;
    }

private:
	const double fs = 250.0;
	stimConfigs_S configs_{}; // default initiliazer _{}
	
	double phaseOffset_ = 0; // persistent variable to keep track of phase offset between synthesize_data_stream function calls for continuity
	std::size_t sampleCount_ = 0; // running sample counter to stamp each chunk's starting sample index
	double activeStimulusHz_ = 0.0; // current active stimulus frequency (0 = none)

	// Random sources for noise gen
	std::mt19937 rng_;
	std::normal_distribution<double> noiseNorm_{ 0.0, 1.0 }; // std=1; scaled later
	std::uniform_real_distribution<double> uni01_{0.0, 1.0};

	// Phase offsets for the superimposed signals
	double driftPhase_   = 0.0;
	double alphaPhase_   = 0.0;
	double betaPhase_   = 0.0;
	double linePhase_    = 0.0;

	// Artifact generation (blinks, muscle motions, etc)
	std::size_t artifactSamplesLeft_   = 0;
	std::size_t samplesToNextArtifact_ = 0;

	// Helpers
	void synthesize_data_stream(float* dest, std::size_t numberOfScans); // used by mock_GetData

	inline double background_noise_signal(double noise_uV){
		// per channel Gaussian noise
		return noise_uV * noiseNorm_(rng_);
	}
	inline double stimulus_signal(double sigAmp_uV, double phase){
		return sigAmp_uV * std::sin(phase);
	}

	// channel configs
	int numChannels_;
    std::vector<std::string> channelLabels_;

};