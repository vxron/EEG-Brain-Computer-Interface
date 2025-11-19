#pragma once
#include <string>
#include <vector>

// Struct holding configs from ONNX JSON (Python training)
struct OnnxConfigs_S {
    std::vector<std::string> feat_names; // global order for full feature vector
    
    // labels to map back to class from classifier output
    int SSVEP_left_id;
    int SSVEP_right_id;
    int SSVEP_none_id;
};

