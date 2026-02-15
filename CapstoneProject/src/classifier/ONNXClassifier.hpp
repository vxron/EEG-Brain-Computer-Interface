#pragma once
#include <string>
#include <vector>
#include "../utils/Types.h"
#include "../acq/WindowConfigs.hpp"
#include <array>
#include <filesystem>
#include <algorithm>
#include <thread>
#include <optional>
#include <numeric> 
#include <stdexcept>
#include <cmath> // std::exp(value)
#include <onnxruntime_cxx_api.h>

// consumer can use currentSessionIdx CHANGE to detect when it must reload model
// ^^i.e. call init_onnx_model...
// Inference selection = saved_sessions[currentSessionIdx].model_dir
constexpr std::size_t RESERVED_THREADS = 7; // number of threads in app + margin of
constexpr int32_t CNN_EXPECTED_C = NUM_CH_CHUNK; // num channels
constexpr int32_t CNN_EXPECTED_T = WINDOW_SCANS; // num time samples in window
constexpr ONNXTensorElementDataType CNN_EXPECTED_INPUT_DATA_TYPE = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
constexpr std::array<int64_t, 4> CNN_EXPECTED_SHAPE = {1, 1, CNN_EXPECTED_C, CNN_EXPECTED_T};
constexpr float REQ_CONFIDENCE_TO_PUBLISH = 0.75; // required confidence in active prediction to publish to consumer (else rtn unknown -> acts like no-op)
// TODO: debounce window in consumer thread
// - req sustained prediction for actuation before beginning actuation
// - then freeze new predictions until actuation is complete

// Window shape reference: row major TIME interleaved -> this is exactly [1, T, C]
// [s0ch0, s0ch1, s0ch2, s0ch3, s0ch4, s0ch5, s0ch6, s0ch7, s1ch0, s1ch1, s1ch2, s1ch3, s1ch4, s1ch5, s1ch6, s1ch7... sNch0, sNch1, sNch2, sNch3, sNch4, sNch5, sNch6, sNch7...]
// alternative: [ch0s0, ch0s1, ch0s2, ch0s3, ch0s4.. ; ch1s0, ch1s1...] 
struct LoadedModel_s {
    std::string model_path; 
    SettingTrainArch_E model_arch;

    // ORT objects (model-specific) -> store as ptr bcuz its not default-constructible
    std::unique_ptr<Ort::Session> sessionPtr;
    
    // IO name caching (for io channel names returned by ORT)
    // duration = model lifetime
    // so we know which tensor to feed input into and which to extract as output
    // intermediate tensors are not exposed
    // onnx rt session.Run() expects const char** (array of c-style str pointers) <- ptrs to char* that point to input & output names
    std::vector<const char*> input_window_tensor_name;
    std::vector<const char*> output_logits_tensor_name;
    // string wrappers in order to SAVE STR DATA from name='GetInputNameAllocated' & avoid dangling pointers when 'name' is destroyed...
    // since char* is a ptr, meaning once it's freed, it points to garbage...
    std::vector<std::string> input_window_tensor_name_saved_str;
    std::vector<std::string> output_logits_tensor_name_saved_str;

    std::vector<float> input_window_f32; // preallocated buffer for tensor input

    // IO Contract
    // expected datatype for tensors
    std::array<int64_t, CNN_EXPECTED_SHAPE.size()> input_shape; // TODO HADEEL: if svm input shape is longer, must preallocate that size instead.
    bool input_tensor_order_is_1ct = true; // true for [1,C,T], false for [1,T,C]
    // artifact cleaning happens at window-level, so we should always have same-size windows.
    int32_t expected_C;
    int32_t expected_T;
    ONNXTensorElementDataType expected_dtype;
};
class ONNX_RT_C {
public:
    // default construct
    ONNX_RT_C(); // should setup ORT environment (logging)

    // each time we enter new session in consumer
    // read input count/output count, names, type/shape & cache everything 
    // load if diff, no-op if same
    bool init_onnx_model(std::string model_dir, SettingTrainArch_E model_arch);
    
    int classify_window(const sliding_window_t& window); // return class idx w/ guard if probability is too low... -> consumer handles the rest

    // allow checking of currently loaded model path
    std::string get_curr_onnx_model_path() const;
private:
    // global refs we can reuse safely throughout diff sessions
    Ort::Env env_; // save ref to env (once per app turn-on)
    Ort::SessionOptions session_options_; // ref to session options
    Ort::AllocatorWithDefaultOptions allocator_; // for name allocations
    Ort::MemoryInfo mem_info_; // CPU tensor memory usage info

    std::string model_path_ = ""; // save ref to current model path
    
    // (ORT can allocate the rest of them??)
    bool load_onnx_on_new_session(std::string model_path, SettingTrainArch_E model_arch);
    bool verify_requirements(const sliding_window_t& window); // make sure window satisfies input tensor requirements
    // cached pointer here or somewhere else??
    std::array<float, 3> run_inference(const sliding_window_t& window); // get logits
    
    // PTR TO CACHED MODEL
    std::unique_ptr<LoadedModel_s> loaded_model_cache_; // inits to nullptr

    // get final class from model logits output (w guards for low probability results)
    int apply_softmax_to_publish_final_class(std::array<float,3> &logits);
};