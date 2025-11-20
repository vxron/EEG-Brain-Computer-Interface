#pragma once
#include <vector>
#include "../utils/Types.h"
#include "ONNXClassifier.hpp"
#include "../acq/UnicornCheck.h"
#include "../acq/WindowConfigs.hpp"

enum class FeatureKind_E {
    // TODO
    // Error catching
    Unknown,
};

struct FeatureCache_S {
    // Individual channel datastreams cached once per window
    std::vector<float> ch1;
    std::vector<float> ch2;
    std::vector<float> ch3;
    std::vector<float> ch4;
    std::vector<float> ch5;
    std::vector<float> ch6;
    std::vector<float> ch7;
    std::vector<float> ch8;
    std::size_t fs = UNICORN_SAMPLING_RATE_HZ;

    // TODO: Cache for expensive features so we dont reconstruct objects every window (fourier/AR)
    bool mag_computed_this_window = false;
    std::vector<float> mag;
    bool psd_computed_this_window = false;
    std::vector<float> freq;
    std::vector<float> power;
};

class FeatureVector_C {
public:
    FeatureVector_C();
    explicit FeatureVector_C(const OnnxConfigs_S& cfgs);
    // to set configs after default construction:
    void setConfigs(const OnnxConfigs_S& cfgs);

    std::vector<float> write_feature_vector(sliding_window_t window);
private:
    float compute_one_feature(FeatureKind_E ftrKind);
    void extract_individual_channel_vectors(sliding_window_t window, std::vector<float>* ch1, std::vector<float>* ch2, std::vector<float>* ch3, std::vector<float>* ch4,std::vector<float>* ch5,std::vector<float>* ch6,std::vector<float>* ch7,std::vector<float>* ch8);

    const OnnxConfigs_S& cfgs_; // ref to classifier meta
    FeatureCache_S cache_;
    std::vector<FeatureKind_E> ops_; // list of operating enum ftr kinds we must get for these cfgs (init on construction)
};