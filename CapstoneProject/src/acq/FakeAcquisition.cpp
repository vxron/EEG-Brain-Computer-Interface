#include "FakeAcquisition.h"
#include <cmath>
#include <numbers>
#include <algorithm>

static constexpr double kTwoPi = std::numbers::pi * 2.0;

FakeAcquisition_C::FakeAcquisition_C(const stimConfigs_S &configs) : configs_(configs), rng_(0xC0FFEEu), activeStimulusHz_(0.0) {
	// nothing (start in "no stim" mode)

	if (configs_.occasionalArtifactsEnabled){
		// Schedule first artifact 3–7 seconds from "now"
    	const double firstBlinkDelaySec = 3.0 + 4.0 * uni01_(rng_);
    	samplesToNextArtifact_ = static_cast<std::size_t>(firstBlinkDelaySec * fs);
	}
}

void FakeAcquisition_C::synthesize_data_stream(float* dest, std::size_t numberOfScans) {
	// Local cache
	const double activeFreq = activeStimulusHz_; // can have frequencies change instantly for any given chunk
	const double sigAmp_uV = configs_.ssvepAmplitude_uV;
	const double noise_uV = configs_.noiseSigma_uV;

	// Per sample phase increments to move sine wave along consistently per sample without using real time
	const double dt = 1 / fs;

	const bool stimEnabled = (activeStimulusHz_ > 0.0); // adding sin for ssvep?
	 // What extras are enabled?
    const bool enableDrift      = configs_.dcDrift.enabled;
    const bool enableArtifacts  = configs_.occasionalArtifactsEnabled;
    const bool enableLine       = configs_.lineNoise.enabled;
    const bool enableAlpha      = configs_.alpha.enabled;
    const bool enableBeta       = configs_.beta.enabled;

	// phase increments (harmless even if not used) -> how we get freq of wave
	const double dphi_ssvep = kTwoPi * activeFreq * dt; // angular dist = angular speed(2pif) x time
	const double dphi_drift = kTwoPi * configs_.dcDrift.freqHz * dt;
    const double dphi_alpha = kTwoPi * configs_.alpha.freqHz * dt;
    const double dphi_beta  = kTwoPi * configs_.beta.freqHz  * dt;
    const double dphi_line  = kTwoPi * configs_.lineNoise.freqHz * dt;

	// Helper: background signal (drift + rhythms + line + optional artifacts)
    auto sampleBackground = [&]() -> double {
        double bg = 0.0;

        // (1) DC drift
        if (enableDrift) {
            bg += configs_.dcDrift.amp_uV * std::sin(driftPhase_);
            driftPhase_ += dphi_drift;
			// wrap around if next iter will give phaseIdx*dphi_ssvep more than 2pi
			// and start back by overshot amount for continuity
            if (driftPhase_ >= kTwoPi) driftPhase_ = driftPhase_ - kTwoPi;
        }

        // (2) Alpha rhythm
        if (enableAlpha) {
            bg += configs_.alpha.amp_uV * std::sin(alphaPhase_);
            alphaPhase_ += dphi_alpha;
            if (alphaPhase_ >= kTwoPi) alphaPhase_ = driftPhase_ - kTwoPi;
        }

        // (3) Beta rhythm
        if (enableBeta) {
            bg += configs_.beta.amp_uV * std::sin(betaPhase_);
            betaPhase_ += dphi_beta;
            if (betaPhase_ >= kTwoPi) betaPhase_ = driftPhase_ - kTwoPi;
        }

        // (4) Line noise
        if (enableLine) {
            bg += configs_.lineNoise.amp_uV * std::sin(linePhase_);
            linePhase_ += dphi_line;
            if (linePhase_ >= kTwoPi) linePhase_ = driftPhase_ - kTwoPi;
        }

        // (5) Occasional artifacts (blinks, motion)
        if (enableArtifacts) {
            if (artifactSamplesLeft_ > 0) {
                // Rectangular artifact pulse ~80 µV
                bg += 80.0;
                artifactSamplesLeft_--;
            } else {
                if (samplesToNextArtifact_ == 0) {
                    // New artifact ~100 ms long
                    artifactSamplesLeft_ = static_cast<std::size_t>(0.1 * fs);
                    // Next artifact in 3–7 seconds
                    const double delaySec = 3.0 + 4.0 * uni01_(rng_);
                    samplesToNextArtifact_ = static_cast<std::size_t>(delaySec * fs);
                } else {
                    samplesToNextArtifact_--;
                }
            }
        }
        return bg;
    };

	// Helper: SSVEP component (0 if no active stim)
    auto sampleSsvEp = [&]() -> double {
        if (!stimEnabled) return 0.0; 
        const double sigVal = stimulus_signal(sigAmp_uV, phaseOffset_);
        phaseOffset_ += dphi_ssvep;
        if (phaseOffset_ >= kTwoPi) phaseOffset_ -= kTwoPi;
        return sigVal; // return sinusoidal magnitude based on phase
    };

	// MAIN LOOP
	for (std::size_t i = 0; i < numberOfScans; i++) {

		const double bgVal    = sampleBackground();
        const double ssvepVal = sampleSsvEp();   // may be 0 if no stim
		
		for (std::size_t j = 0; j < NUM_CH_CHUNK; j++) {
			// Additive white Gaussian noise (per-channel) for realistic signal
			const double noiseVal = background_noise_signal(noise_uV);
			// Noise + ssvep signal + background
			const double netVal = noiseVal + bgVal + ssvepVal;
			dest[NUM_CH_CHUNK * i + j] = static_cast<float>(netVal); // cast to float to mimick Unicorn output
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



