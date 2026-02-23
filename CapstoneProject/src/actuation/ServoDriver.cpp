#include "ServoDriver.h"
#include <iostream> 
#include <chrono>

// Constructor/Destructor
ServoDriver_C::ServoDriver_C() : 
    motion_state_(IDLE), 
    last_step_time_(0), 
    clip1_pos_(180), 
    clip2_pos_(180), 
    flip_pos_(180), 
    slider_pos_(0) {
}

ServoDriver_C::~ServoDriver_C() {
    // TODO: implement
}

// Initialization
bool ServoDriver_C::init() {
    motion_state_ = IDLE;
    last_step_time_ = 0;
    return true;
}

// Zero Position

// Move Stepper

// Fwd Page Turn

// Bwd Page Turn

// Update

// Advance State 

// Check Motion Complete



