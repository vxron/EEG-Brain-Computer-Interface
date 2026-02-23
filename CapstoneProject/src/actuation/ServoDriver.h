#pragma once
#include <cstdint>

// Map motor IDs exposed by Servo API to enum so we can easily assign cmds to apropriate motor
enum MotorId_E {
    CLIP_LEFT,
    CLIP_RIGHT,
    SLIDER,
    FLIPPER,
    // etc
};

// Tracks what motion driver is currently doing
enum MotionState_E {
    IDLE,
    OPEN_CLIPS,
    CLOSE_CLIPS,
    SLIDER_FORWARD,
    SLIDER_BACK,
    FLIP_UP,
    FLIP_DOWN,
    COMPLETE //Motion finished
};

// struct TorqueCmd_S {
//     float pending_torque_cmd_ = 0.0;
//     float last_published_torque_cmd_ = 0.0;
//     MotorId_E dest_motor;
//     //.. anything that should be aggregated w the numerical cmd
// };

class ServoDriver_C {
public:
    // Default Constructor/Destructor
    ServoDriver_C();
    ~ServoDriver_C();

    // Disable copy & move constructors / assignment operators
    // (Should only have one instance of driver)
	ServoDriver_C(const ServoDriver_C&) = delete; // copy
    ServoDriver_C(ServoDriver_C&&) = delete; // move

    ServoDriver_C& operator=(const ServoDriver_C&) = delete;
    ServoDriver_C& operator=(ServoDriver_C&&) = delete;

    // API consumer thread will use
    bool init(); // Initialize servos and pins
    bool execute_prev_page_turn();
    bool execute_fwd_page_turn();

    // Repeatedly called by thread (non-blocking timing)
    void update(); // moves state machine forward without blocking
    bool motion_complete() const; // thread checks this to issue next command

private:
    // Move to next state in sequence  (open clips,..., complete)
    void advance_state();

    // Internal states
    MotionState_E motion_state_;
    unsigned long last_step_time_; // Non-blocking timing, so update() only moves servos when enough time has passed
  //  bool validate_torque_cmd_before_publishing();
    int clip1_pos_, clip2_pos_, flip_pos_, slider_pos_;
    
};