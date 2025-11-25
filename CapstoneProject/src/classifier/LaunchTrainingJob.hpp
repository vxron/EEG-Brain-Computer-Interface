/*
-> Bridge between Python and C++ for:
- Python model training after calibration
- ONNX export to C++ after training
-> Spawns a worker thread to run Python concurrently
*/

#pragma once
#include <string>
#include <filesystem>
#include "../stimulus/StateStore.hpp"

class TrainingJob_C {
public:
    TrainingJob_C(std::string subject_id, std::string session_id, std::filesystem::path data_dir, std::filesystem::path model_dir);
    void launch_training_job(StateStore_s& stateStore);
private:
    // these should all be known at time of job launching (obj construction)
    std::string subject_id_;
    std::string session_id_;
    std::filesystem::path data_dir_; // for csv (training set)
    std::filesystem::path model_dir_; // for onnx export
};

