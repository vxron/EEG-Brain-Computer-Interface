#pragma once

// Serial interface to the Arduino page turner
// Sends single character command over USB (or Bluetooth) COM port
// Reads DONE ack from Arduino for closed-loop fdbk

// Commands sent to Arduino: 
//     'F' = fwd page flip
//     'B' = bwd page flip
//     'O' = open clips
//     'C' = close clips/zero pos

// Architecture: A background reader thread inside ServoDriver continuously reads
// the serial poer. When DONE is received, 'busy' flag is cleared atomically.
// isBusy() reflects real hardware state (no blind timers).

// Non-blocking mechanism: sendChar() writes 1 byte and returns in ms.
// isBusy() is an atomic load (returns instantly).
// Reader thread runs independently and never blocks the caller.

#include <cstdint>
#include <chrono>
#include <mutex>
#include <string>
#include <atomic>
#include <thread>
#include "../utils/SWTimer.hpp"
#ifndef MOCK_HARDWARE
  #define WIN32_LEAN_AND_MEAN   // strips rarely-used Windows headers, keeps compile fast
  #include <windows.h>
#endif

// // Map motor IDs exposed by Servo API to enum so we can easily assign cmds to apropriate motor
// enum MotorId_E {
//     CLIP_LEFT,
//     CLIP_RIGHT,
//     SLIDER,
//     FLIPPER,
//     // etc
// };

// Tracks what motion driver is currently doing
// enum MotionState_E {
//     IDLE,
//     OPEN_CLIPS,
//     CLOSE_CLIPS,
//     SLIDER_FORWARD,
//     SLIDER_BACK,
//     FLIP_UP,
//     FLIP_DOWN,
//     COMPLETE //Motion finished
// };

// struct TorqueCmd_S {
//     float pending_torque_cmd_ = 0.0;
//     float last_published_torque_cmd_ = 0.0;
//     MotorId_E dest_motor;
//     //.. anything that should be aggregated w the numerical cmd
// };

class ServoDriver_C {
public:
    // Default Constructor/Destructor
    explicit ServoDriver_C(const std::string& portName = "COM3", unsigned int baudRate = 9600);
    ~ServoDriver_C();

    // Disable copy & move constructors / assignment operators
    // (Should only have one instance of driver)
	ServoDriver_C(const ServoDriver_C&) = delete; // copy
    ServoDriver_C(ServoDriver_C&&) = delete; // move

    ServoDriver_C& operator=(const ServoDriver_C&) = delete;
    ServoDriver_C& operator=(ServoDriver_C&&) = delete;
 
    // Commands
    void flipForward();
    void flipBackward();
    void openClips();
    void zeroPosition();
    
    // States
    bool isConnected() const; // True if COM port opened successfully
    bool isBusy() const; // True while Arduino is executing command

    // Returns how long we expect the current command to still be running.
    // Mock: exact (based on mockBusyUntil).
    // Real: estimate based on per-command constant, counts down from send time.
    //       Returns 0 once Arduino sends DONE (busy cleared) or estimate expires (from consumer sanity check).
    std::chrono::milliseconds estimatedBusyRemaining() const;

private:
 
    // Send one byte char, set busy flag
    bool sendChar(char command);
    // Background reader thread, watches serial port for DONE
    void readerThreadFn();

    bool openPort(const std::string& portName, unsigned int baudRate);
    void closePort();
    void log(const std::string& msg) const;

    // atomic so these bools can be read/written from multiple threads
    // without a mutex
    std::atomic<bool> connected  {false};
    std::atomic<bool> busy       {false};
    std::atomic<bool> stopReader {false};

    // background thread that watches for DONE from arduino
    std::thread readerThread;
    // protects WriteFile() so only one thread can write to serial port at a time
    std::mutex writeMutex;

    // handle pointer
    #ifndef MOCK_HARDWARE
        void* m_hSerial = nullptr;
    #endif

    // FOR MOCK ACTUATION, timeout estimation
    using Clock = std::chrono::steady_clock;
    using TimePoint = std::chrono::time_point<Clock>;
    TimePoint mockBusyUntil {};
    static constexpr int MOCK_BUSY_FLIP_FWD_MS   = 3500;
    static constexpr int MOCK_BUSY_FLIP_BWD_MS  = 3500;
    static constexpr int MOCK_BUSY_OPEN_CLIPS_MS =  500;
    static constexpr int MOCK_BUSY_ZERO_MS       = 1000;

    // time estimates for handling actuator timeout
    // conservative upper bounds (actual DONE should arrive earlier)
    // consumer deadline will be these nums + a small buffer (~ms)
    static constexpr int REAL_BUSY_FLIP_FWD_MS   = 7000; // TODO: measure once hardware is ready
    static constexpr int REAL_BUSY_FLIP_BWD_MS   = 7000;
    static constexpr int REAL_BUSY_OPEN_CLIPS_MS =  800;
    static constexpr int REAL_BUSY_ZERO_MS       = 1500;
    // Tracks estimated duration for current real-hardware command (started in sendChar(), expires at the estimate)
    SW_Timer_C realBusyEstimateTimer_;
    int realBusyEstimate_ms_ = 0; // set per command before starting timer
};