#include "LaunchTrainingJob.hpp"
#include <thread>
#include <cstdlib>
#include <sstream>
#include "../utils/Logger.hpp"

// TODO: IMPLEMENT SOMTHN ON UI TO SHOW 'WAITING FOR TRAINING'
// since it could take a while :,)

TrainingJob_C::TrainingJob_C(std::string subject_id, std::string session_id, std::filesystem::path data_dir, std::filesystem::path model_dir) : 
    subject_id_(subject_id), session_id_(session_id), data_dir_(data_dir), model_dir_(model_dir) 
{}

void TrainingJob_C::launch_training_job(StateStore_s& stateStore){
    logger::tlabel = "launchTrainingJob_C";
    const std::filesystem::path scriptPath =
        R"(C:\Users\fsdma\capstone\capstone\CapstoneProject\model train\python\train_svm.py)";
        // ^ RODO: FIX THIS SO ITS NOT LOCAL TO MY PC
    
    // Spawn worker thread
    std::thread([&stateStore, this, scriptPath]() {
        std::stringstream ss;
        // MUST MATCH PYTHON TRAINING SCRIPT PATH AND ARGS (HADEEL)
        ss << "python "
               << "\"" << scriptPath.string() << "\""
               << " --data \""     << data_dir_.string()   << "\""
               << " --model \""    << model_dir_.string()  << "\""
               << " --subject \""  << subject_id_          << "\""
               << " --session \""  << session_id_          << "\"";

        const std::string cmd = ss.str();
        /* std::system executes the cmd string using host shell
        -> it BLOCKS "this" background thread until the Python script finishes
        -> rc is the exit code of the cmd
        */
        int rc = std::system(cmd.c_str());

        if(rc == 0){
            // success - update state store
            stateStore.currentSessionInfo.g_isModelReady.store(true, std::memory_order_release);
            stateStore.currentSessionInfo.set_active_model_path(model_dir_.string());
            stateStore.currentSessionInfo.set_active_subject_id(subject_id_);
        }
        else {
            // failed
            stateStore.currentSessionInfo.g_isModelReady.store(false, std::memory_order_release);
            LOG_ALWAYS("Training job failed to launch.");
        }
    }).detach(); 
    /* 
    -> "detaches" thread from std::thread object so that it can continue running independently...
    -> when its function returns -> os automatically cleans it up
    -> so main doesn't have to wait for training to finish (which could take a while...)
    */
}
