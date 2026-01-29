#include "ONNXClassifier.hpp"
#include <iostream>
#include <array>

    // create ONNX env (once per app)
    // arena allocation: pre-allocate large mem pool, then sub-allocate (no constant malloc/free, reduces heap fragmentation); default CPU mem
ONNX_RT_C::ONNX_RT_C() : env_(ORT_LOGGING_LEVEL_WARNING, "SSVEP_BCI"), mem_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) 
{
    // session options - fine in body bcuz it has default constructor
    Ort::SessionOptions session_options;
    // optimization levels: want aggressive
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    // allow memory optimizations
    session_options.EnableMemPattern();
    session_options.EnableCpuMemArena(); // for OrtArenaAllocator, pre-allocated block of heap rather than fragmented mallocs
    // use maximal number of threads for heavy cnn or ftr extraction ops
    uint32_t total_cpu_available_threads = std::thread::hardware_concurrency();
    session_options.SetIntraOpNumThreads(total_cpu_available_threads - RESERVED_THREADS);
    session_options_ = std::move(session_options);
}

bool ONNX_RT_C::init_onnx_model(std::string model_dir, SettingTrainArch_E model_arch){
    /*
    - Switch active model when the selected session changes
    - Load once; use for all inference requests
    */
   // convert model_dir to model path
   std::string model_path = model_dir + "/ssvep_model.onnx";
    if((model_path != model_path_) || (loaded_model_cache_ == nullptr))
    {
        try{
            // if model changed or the cache hasn't been made yet...
            // new model must be loaded
            load_onnx_on_new_session(model_path, model_arch);
            model_path_ = model_path;
        } catch (const std::exception& e) {
            std::cout << "Error: " << e.what() << "\n";
            return false;
        }
    } else {
        // use cached model, same model path
    }
    return true;
}

bool ONNX_RT_C::verify_requirements(const sliding_window_t& window){
    // check shape against expected for current model
    if(loaded_model_cache_ == nullptr){
        return false;
    }
    else if((loaded_model_cache_->expected_C*loaded_model_cache_->expected_T) != window.sliding_window.get_count()){
        // unexpected window shape
        // skip this window
        return false;
    }
    else {
        return true;
    }
}

int ONNX_RT_C::classify_window(const sliding_window_t& window){
    bool isOk = verify_requirements(window);
    if(isOk){
        std::array<float, 3> logits = run_inference(window);
        return apply_softmax_to_publish_final_class(logits);
    }
    else {
        return -1; // will skip this window -> perform no action
        // TODO: from consumer, check for too many ssvep_unknown cases
    }
}

int ONNX_RT_C::apply_softmax_to_publish_final_class(std::array<float,3> &logits){
    // for each logit x_i: softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)
    // improve for numerical stability (avoid exp overflow for large logits) -> subtract off largest logit
    // softmax(x_i) = exp(x_i-max(x)) / sum(exp(x_j-max(x)) for all j)
    std::array<float,3> softmax_res;
    float denom = 0;
    auto max_logit_it = std::max_element(logits.begin(),logits.end()); // get iterator
    float max_logit = *max_logit_it; // deref to get value
    for(int i=0; i<3; i++){
        float val = std::exp(logits[i]-max_logit);
        denom += val;
    }
    if(denom==0){
        throw std::runtime_error("[ONNX] divide by 0 error during softmax calculation");
    }
    for(int i=0; i<3;i++){
        softmax_res[i] = std::exp(logits[i]-max_logit)/denom;
    }
    
    // get max softmax now
    auto max_softmax_it = std::max_element(softmax_res.begin(), softmax_res.end());
    float max_softmax = *max_softmax_it; //deref
    // get index of max element - returns idx (distance) btwn 2 iterators
    int idx_max = std::distance(softmax_res.begin(), max_softmax_it);
    if(max_softmax > REQ_CONFIDENCE_TO_PUBLISH){
        return idx_max;
    }
    else {
        return -1; // should map to unknown ssvep in classify_window
    }
}

std::array<float, 3> ONNX_RT_C::run_inference(const sliding_window_t& window) {
    /*
    - pack window into correct input tensor layout
    - call ORT Session.Run
    - return logits
    */
    std::array<float, 3> logits;
    // 1) convert/pack window samples into input_window_f32_ in model's expected layout
    if(loaded_model_cache_->input_tensor_order_is_1ct){
        // expect [1,C,T] -> this is CHANNEL interleaved
        // need to flip C,T (transpose)
        std::vector<float> tmp;
        window.sliding_window.get_data_snapshot(tmp); // original
        int j=0;
        for(int ch=0;ch<loaded_model_cache_->expected_C;ch++)
            // collect all samples w one ch at a time to form input window word
            for(int s=0;s<loaded_model_cache_->expected_T;s+=loaded_model_cache_->expected_C){
                loaded_model_cache_->input_window_f32[j]=tmp[ch+s];
                j++;
            }
    }
    else{
        // expect [1,T,C] -> this is TIME interleaved (matches our window layout)
        // place data arr into the cached buffer
        window.sliding_window.get_data_snapshot(loaded_model_cache_->input_window_f32);
    }

    // 2) create ORT input tensor
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info_.GetConst(),                            // const ptr to OrtMemoryInfo -> where tensor mem is allocated (set at construction time)
        loaded_model_cache_->input_window_f32.data(),    // ptr to input data container
        loaded_model_cache_->input_window_f32.size(),    // length of input data
        loaded_model_cache_->input_shape.data(),         // ptr to shape array
        3                                                // length of shape array (#dims)
    );
    // build array of input tensors (will be size 1 since we have 1 input) for .run method
    // use std::move bcuz Ort:value is non-copyable... transfer ownership
    std::vector<Ort::Value> input_tensor_vec;
    input_tensor_vec.push_back(std::move(input_tensor));

    // 3) call session.run & capture returned outputs
    std::vector<Ort::Value> output_tensors = loaded_model_cache_->session.Run(
        Ort::RunOptions{nullptr},                              // keep null
        loaded_model_cache_->input_window_tensor_name.data(),  // array of input tensor names (will be size 1)
        input_tensor_vec.data(),                               // array of input tensors (Ort::Value) (will be size 1)
        1,                                                     // number of inputs
        loaded_model_cache_->output_logits_tensor_name.data(), // array of output tensor names (will be size 1)
        1                                                      // number of outputs
    );

    // 4) read output data and return
    float* out_ptr = output_tensors[0].GetTensorMutableData<float>(); // direct access to onnx runtime memory (internal data buffer)
    // validate against expected shape
    auto shape_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = shape_info.GetShape();
    if(!(shape[0]==1 && shape[1]==3) && !(shape[0]==3 && shape[1]==1)) {
        // incorrect output shape
        throw std::runtime_error("[ONNX] Expected 3 output classes in .onnx model shape");
    }

    logits[0]=*out_ptr;
    logits[1]=*(++out_ptr); // preinc ptr, then read (idx 1)
    logits[2]=*(++out_ptr); // preinc ptr, then read (idx 2)
    return logits;
}

bool ONNX_RT_C::load_onnx_on_new_session(std::string model_path, SettingTrainArch_E model_arch){
    /*
    - create ORT session
    - cache everything that will be needed for per-window inference
    */
    loaded_model_cache_->model_arch = model_arch;
    loaded_model_cache_->model_path = model_path;
    
    // TODO: need to map ssvepstate_e (left vs right vs none) to class output logits & save to local cache
    // 1) create session (loads model) -> stateless inference engine
    // onnx env api needs ptr to wstring_t (wstring_t*) for windows, c-str (const char*) for linux/mac
#ifdef _WIN32
    std::wstring wmodel_path(model_path.begin(),model_path.end());
    loaded_model_cache_->session = Ort::Session(env_, wmodel_path.c_str(), session_options_);
#else
    loaded_model_cache_->session = Ort::Session(env_, model_path.c_str(), session_options_);
#endif

    // 2) verify io counts are 1 (expected 1 tensor count for each)
    if((loaded_model_cache_->session.GetInputCount() != 1) || (loaded_model_cache_->session.GetOutputCount() != 1)) {
        throw std::runtime_error("[ONNX] ONNX Runtime error: Expected 1 input tensor, 1 output tensor. Got more from exported .onnx file.");
        return false;
    }

    // 3) get/allocate & cache io names
    auto in_name = loaded_model_cache_->session.GetInputNameAllocated(0, allocator_); // returns smart pointer for c-style string (char*) to where memory gets allocated for name
    auto out_name = loaded_model_cache_->session.GetOutputNameAllocated(0, allocator_);
    // .get returns raw pointer (char*) without transfering memory ownership (from 'name'), then std::string turns the char* into a string automatically for safe-keeping
    loaded_model_cache_->input_window_tensor_name_saved_str.push_back(std::string(in_name.get()));
    loaded_model_cache_->output_logits_tensor_name_saved_str.push_back(std::string(out_name.get()));
    // save const char versions for feeding to session.Run... c-style ptrs to the safe strings we've defined so when we go out of scope, we don't lose 'name' and 'out_name'
    loaded_model_cache_->input_window_tensor_name.push_back(loaded_model_cache_->input_window_tensor_name_saved_str[0].c_str());
    loaded_model_cache_->output_logits_tensor_name.push_back(loaded_model_cache_->output_logits_tensor_name_saved_str[0].c_str());

    // 4) read/verify input type + shape (for CNN: expect float32, either [1,C,T] or [1,T,C])
    auto typeInfo = loaded_model_cache_->session.GetInputTypeInfo(0); // type info for input 0 (only one)
    auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType dtype = tensorInfo.GetElementType();
    std::vector<int64_t> shape = tensorInfo.GetShape();

    switch(model_arch){
        case TrainArch_CNN: {
            loaded_model_cache_->expected_C = CNN_EXPECTED_C;
            loaded_model_cache_->expected_T = CNN_EXPECTED_T;
            loaded_model_cache_->expected_dtype = CNN_EXPECTED_INPUT_DATA_TYPE;
            break;
        }
        case TrainArch_SVM: {
            break; // hadeel todo
        }
        default: {
            break; // never reach here
        }

    }
    if(shape.size() != 3){
        // TODO: graceful error handling (don't do runtime error... just signal in statestore & then do ui popup n go back to home, disable runmode w this model)
        throw std::runtime_error("[ONNX] Expected input shape with 3 elements [1,C,T], got "+std::to_string(shape.size()));
    }
    if(shape[0] != 1){
        throw std::runtime_error("[ONNX] Expected input shape's first element to be 1 for 2D tensor, got "+std::to_string(shape[0]));
    }
    // try and find the one that equals the expected C and T 
    if(shape[1]==loaded_model_cache_->expected_C && shape[2]==loaded_model_cache_->expected_T){
        // shape [1,C,T]
        loaded_model_cache_->input_tensor_order_is_1ct = true;
        std::copy(shape.begin(), shape.end(), loaded_model_cache_->input_shape.begin());
    } 
    else if (shape[2]==loaded_model_cache_->expected_C && shape[1]==loaded_model_cache_->expected_T){
        // shape [1,T,C]
        loaded_model_cache_->input_tensor_order_is_1ct = false;
        std::copy(shape.begin(), shape.end(), loaded_model_cache_->input_shape.begin());
    }
    else {
        // shape not okoaded_model_tmp.
        throw std::runtime_error("[ONNX] Expected input shape to be [1,C,T] or [1,T,C], where C=channels, T=window length, got "+std::to_string(shape[1])+" by "+std::to_string(shape[2]));
    }
    if(loaded_model_cache_->expected_dtype != dtype){
        throw std::runtime_error("[ONNX] Expected input tensor data type to be float32");
    }

    // 5) read output shape for number of classes and make sure its 3
    auto typeInfo_out = loaded_model_cache_->session.GetOutputTypeInfo(0);
    auto tensorInfo_out = typeInfo_out.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape_out = tensorInfo_out.GetShape();
    if(!(shape_out[0]==1 && shape_out[1]==3) && !(shape_out[0]==3 && shape_out[1]==1)) {
        // incorrect output shape
        throw std::runtime_error("[ONNX] Expected 3 output classes in .onnx model shape");
    }

    // 6) allocate input tensor vector with TxC data slots (our data is stored major-interleaved)
    loaded_model_cache_->input_window_f32.resize(loaded_model_cache_->expected_C*loaded_model_cache_->expected_T);
    
    return true;
}