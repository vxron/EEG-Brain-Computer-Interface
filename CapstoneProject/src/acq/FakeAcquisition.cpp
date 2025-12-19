#include "FakeAcquisition.h"
#include <cmath>
#include <numbers>
#include <algorithm>

static constexpr double kTwoPi = std::numbers::pi * 2.0;

FakeAcquisition_C::FakeAcquisition_C(const stimConfigs_S &configs) : configs_(configs), rng_(0xC0FFEEu), activeStimulusHz_(0.0), numChannels_(NUM_CH_CHUNK) {
	// nothing (start in "no stim" mode)

	if (configs_.occasionalArtifactsEnabled){
		// Schedule first artifact 3–7 seconds from "now"
    	const double firstBlinkDelaySec = 3.0 + 4.0 * uni01_(rng_);
    	samplesToNextArtifact_ = static_cast<std::size_t>(firstBlinkDelaySec * fs);
	}

    // Just number them in fake acq...
    channelLabels_.resize(numChannels_);
    for (int i = 0; i < numChannels_; ++i) {
        channelLabels_[i] = "Ch" + std::to_string(i + 1);
    }

    // per channel variability
    for (int ch = 0; ch < numChannels_; ++ch) {
        // small generic gain variation
        chGain_[ch] = randu(0.9, 1.1);

        // per-channel noise sigma (0.8x..1.2x)
        chNoiseSigma_[ch] = configs_.noiseSigma_uV * randu(0.8, 1.2);

        // random phases
        chSsvEpPhase_[ch] = randu(0.0, kTwoPi);
        chAlphaPhase_[ch] = randu(0.0, kTwoPi);
        chBetaPhase_[ch]  = randu(0.0, kTwoPi);
        chLinePhase_[ch]  = randu(0.0, kTwoPi);

        // per-channel background gains
        chAlphaGain_[ch] = randu(0.7, 1.3);
        chBetaGain_[ch]  = randu(0.7, 1.3);
        chLineGain_[ch]  = randu(0.7, 1.3);

        // SSVEP is spatially weighted: “occipital-ish” channels stronger.
        // just weight last 2 channels higher.
        const bool occipitalish = (ch >= (numChannels_ - 2));
        chSsvEpGain_[ch] = occipitalish ? randu(1.0, 1.6) : randu(0.2, 0.6);
    }
}

void FakeAcquisition_C::maybe_start_artifact() {
    if (!configs_.occasionalArtifactsEnabled) return;

    // already running
    if (artSamplesLeft_ > 0) return;

    // countdown until next artifact
    if (samplesToNextArtifact_ > 0) {
        samplesToNextArtifact_--;
        return;
    }

    // pick artifact type
    const double r = uni01_(rng_);
    if (r < 0.70) {
        // ----------------- BLINK -----------------
        artType_ = ArtifactType::Blink;
        blinkTotalSamples_ = static_cast<std::size_t>(0.20 * fs); // 200 ms
        blinkProgress_ = 0;
        blinkAmp_uV_ = randu(60.0, 140.0); // decent blink amplitude
        artSamplesLeft_ = blinkTotalSamples_;
    } else {
        // -------------- ELECTRODE POP -------------
        artType_ = ArtifactType::ElectrodePop;
        popChannel_ = static_cast<int>(randu(0.0, (double)numChannels_));
        if (popChannel_ >= numChannels_) popChannel_ = numChannels_ - 1;

        popLevel_uV_ = randu(120.0, 350.0) * (uni01_(rng_) < 0.5 ? -1.0 : 1.0); // big DC step
        popDecay_ = randu(0.992, 0.998); // slower/faster recovery
        artSamplesLeft_ = static_cast<std::size_t>(randu(0.30, 0.80) * fs); // 300–800ms
    }

    // schedule next artifact in 3–7 seconds
    const double delaySec = 3.0 + 4.0 * uni01_(rng_);
    samplesToNextArtifact_ = static_cast<std::size_t>(delaySec * fs);
}

double FakeAcquisition_C::artifact_value_for_channel(std::size_t ch) {
    if (artSamplesLeft_ == 0 || artType_ == ArtifactType::None) return 0.0;

    if (artType_ == ArtifactType::Blink) {
        // half-sine bump: A * sin(pi*t/T), t in [0,T]
        const double T = (blinkTotalSamples_ > 0) ? (double)blinkTotalSamples_ : 1.0;
        const double t = (double)blinkProgress_;
        const double x = std::sin(std::numbers::pi * (t / T));
        const double bump = blinkAmp_uV_ * x;

        // apply strongest to “frontal-ish” channels: first 2 channels stronger
        double scale = 0.2;
        if (ch == 0) scale = 1.0;
        else if (ch == 1) scale = 0.7;
        else if (ch == 2) scale = 0.4;

        return scale * bump;
    }

    if (artType_ == ArtifactType::ElectrodePop) {
        // DC step + exponential decay on one channel
        if ((int)ch == popChannel_) {
            return popLevel_uV_;
        }
        return 0.0;
    }

    return 0.0;
}

void FakeAcquisition_C::synthesize_data_stream(float* dest, std::size_t numberOfScans) {
	// Local cache
	const double activeFreq = activeStimulusHz_; // can have frequencies change instantly for any given chunk
	const double sigAmp_uV = configs_.ssvepAmplitude_uV;
	const double noise_uV = configs_.noiseSigma_uV;

	// Per sample phase increments to move sine wave along consistently per sample without using real time
	const double dt = 1.0 / fs;

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

	double attnModPhase = 0.0;
    const double dphi_attn = kTwoPi * 0.15 * dt; // 0.15 Hz slow modulation

    for (std::size_t i = 0; i < numberOfScans; i++) {

        // advance artifact scheduling
        if (enableArtifacts) maybe_start_artifact();

        // global drift phase
        double drift = 0.0;
        if (enableDrift) {
            drift = configs_.dcDrift.amp_uV * std::sin(driftPhase_);
            driftPhase_ += dphi_drift;
            if (driftPhase_ >= kTwoPi) driftPhase_ -= kTwoPi;
        }

        // compute attention modulation scalar (0.9..1.1) -> for more realistic SSVEP mod by attn
        const double attn = 1.0 + 0.10 * std::sin(attnModPhase);
        attnModPhase += dphi_attn;
        if (attnModPhase >= kTwoPi) attnModPhase -= kTwoPi;

        // ========================= per-channel composition =========================
        for (std::size_t ch = 0; ch < (std::size_t)numChannels_; ch++) {

            double bg = 0.0;

            // drift applies to all (scaled a tiny bit by channel gain)
            bg += drift * chGain_[ch];

            // alpha (per-channel phase + gain)
            if (enableAlpha) {
                bg += (configs_.alpha.amp_uV * chAlphaGain_[ch]) * std::sin(chAlphaPhase_[ch]);
                chAlphaPhase_[ch] += dphi_alpha;
                if (chAlphaPhase_[ch] >= kTwoPi) chAlphaPhase_[ch] -= kTwoPi; // NEW (per-channel)
            }

            // beta (per-channel phase + gain)
            if (enableBeta) {
                bg += (configs_.beta.amp_uV * chBetaGain_[ch]) * std::sin(chBetaPhase_[ch]);
                chBetaPhase_[ch] += dphi_beta;
                if (chBetaPhase_[ch] >= kTwoPi) chBetaPhase_[ch] -= kTwoPi; // NEW
            }

            // line noise (per-channel phase + gain)
            if (enableLine) {
                bg += (configs_.lineNoise.amp_uV * chLineGain_[ch]) * std::sin(chLinePhase_[ch]);
                chLinePhase_[ch] += dphi_line;
                if (chLinePhase_[ch] >= kTwoPi) chLinePhase_[ch] -= kTwoPi; // NEW
            }

            // SSVEP (per-channel gain + per-channel phase) with harmonic (2f)
            double ssvep = 0.0;
            if (stimEnabled) {
                const double A = sigAmp_uV * chSsvEpGain_[ch] * attn;

                // fundamental
                ssvep += A * std::sin(chSsvEpPhase_[ch]);

                // harmonic 2f at 0.35 amplitude
                ssvep += 0.35 * A * std::sin(2.0 * chSsvEpPhase_[ch]);

                // advance phase
                chSsvEpPhase_[ch] += dphi_ssvep;
                if (chSsvEpPhase_[ch] >= kTwoPi) chSsvEpPhase_[ch] -= kTwoPi;
            }

            // artifact (blink/pop)
            const double art = artifact_value_for_channel(ch);

            // noise (per-channel sigma)
            const double noiseVal = background_noise_signal(chNoiseSigma_[ch]);

            // net
            const double netVal = bg + ssvep + art + noiseVal;

            dest[NUM_CH_CHUNK * i + ch] = static_cast<float>(netVal);
        }

        // advance artifact state counters once per scan (not per channel)
        if (enableArtifacts && artSamplesLeft_ > 0) {
            artSamplesLeft_--;

            if (artType_ == ArtifactType::Blink) {
                blinkProgress_++;
                if (blinkProgress_ > blinkTotalSamples_) blinkProgress_ = blinkTotalSamples_;
            }
            if (artType_ == ArtifactType::ElectrodePop) {
                // decay pop each scan
                popLevel_uV_ *= popDecay_;
            }

            if (artSamplesLeft_ == 0) {
                artType_ = ArtifactType::None;
            }
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



