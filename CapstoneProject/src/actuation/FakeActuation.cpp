#include "FakeActuation.hpp"
#include <iostream>
#include <thread>

FakeActuation_C::FakeActuation_C() : busy_(false){
};

void FakeActuation_C::sendCommand(const std::string& cmd){
    if (busy_) return;
    busy_ = true;
    std::cout << "[Mock Actuator] Executing: " << cmd << std::endl;

    std::thread([this](){
        std::this_thread::sleep_for(std::chrono::seconds(2)); //simulate 2s page turn
        std::cout << "[Mock Actuation] Done\n";
        busy_ = false;
    }).detach();
};

bool FakeActuation_C::isBusy() const {
    return busy_;
};


