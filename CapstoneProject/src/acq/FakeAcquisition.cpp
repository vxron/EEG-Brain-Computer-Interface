#include "FakeAcquisition.h"
#include <cmath>
#include <numbers>
#include <algorithm>

static constexpr double kTwoPi = std::numbers::pi * 2.0;

FakeAcquisition_C::FakeAcquisition_C(const stimConfigs_S &configs) : configs_(configs), rng_(0xC0FFEEu), activeStimulusHz_(configs.rightStimFreq) {
	// nothing
}

/*
* Random but plausible EEG data generator
* - TODO: Slow drift low frequency noise ~3uV @ 0.1Hz
* - TODO: Artifact (large spike) every ~5 seconds at 100uV
* - TODO: Add per-channel amplitude variation in SSVEP signal
*/
void FakeAcquisition_C::synthesize_data_stream(float* dest, std::size_t numberOfScans) {
	// Local cache
	const double activeFreq = activeStimulusHz_;
	const double sigAmp_uV = configs_.ssvepAmplitude_uV;
	const double noise_uV = configs_.noiseSigma_uV;

	// Per sample phase increments to move sine wave along consistently per sample without using real time
	const double dt = 1 / fs;
	double dphi_f = kTwoPi * activeFreq * dt; // angular dist = angular speed(2pif) x time

	// Loop
	for (std::size_t i = 0; i < numberOfScans; i++) {

		// Compute sinusoid once per scan - Same signal for all channels for now
		const double sigVal = sigAmp_uV * std::sin(phaseOffset_);
		
		for (std::size_t j = 0; j < NUM_CH_CHUNK; j++) {
			// Additive Gaussian noise (per-channel)
			const double noiseCh = noise_uV * noiseNorm_(rng_);
			// Noise + signal
			const double netVal = sigVal + noiseCh;
			dest[NUM_CH_CHUNK * i + j] = static_cast<float>(netVal); // cast to float to mimick Unicorn output
		}
		
		phaseOffset_ += dphi_f;

		// wrap around if next iter will give phaseIdx*dphi_f more than 2pi
		if (phaseOffset_ >= kTwoPi) {
			// start back by overshot amount for continuity
			phaseOffset_ = phaseOffset_ - kTwoPi;
		}
		sampleCount_++;
	}
}

void FakeAcquisition_C::setActiveStimulus(double fStimHz) {
	// prevent dysfunctional stimulus frequencies
	if (fStimHz < 0) {
		// clamp
		fStimHz = 0;
	}
	if (fStimHz > fs / 2) {
		// clamp to satisfy Nyquist
		fStimHz = fs / 2;
	}
	activeStimulusHz_ = fStimHz;
}

bool FakeAcquisition_C::getData(std::size_t const numberOfScans, float* dest) {
	// validate arguments
	if (dest == NULL || numberOfScans <= 0) {
		return 0;
	}
	// work in std::size_t for math, upcast uint32_t --> size_t (64 bits) is safe (no truncation)
	const std::size_t requiredLen = numberOfScans * NUM_CH_CHUNK;
	synthesize_data_stream(dest, numberOfScans);
	return 1;
}



