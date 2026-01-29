#pragma once

// Map motor IDs exposed by Servo API to enum so we can easily assign cmds to apropriate motor
enum MotorId_E {
    SERVOMOTOR_1,
    SERVOMOTOR_2,
    SERVOMOTOR_3,
    SERVOMOTOR_4,
    // etc
};

struct TorqueCmd_S {
    float pending_torque_cmd_ = 0.0;
    float last_published_torque_cmd_ = 0.0;
    MotorId_E dest_motor;
    //.. anything that should be aggregated w the numerical cmd
};

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
    bool init_all_motors();
    bool execute_prev_page_turn();
    bool execute_fwd_page_turn();
private:
    bool validate_torque_cmd_before_publishing();
    
};