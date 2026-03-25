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
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
#endif

// ─── Arduino connection/health status ────────────────────────────────────────
// Populated by ServoDriver's reader thread as STATUS: lines arrive from Arduino.
// Read by UI/HTTP and consumer thread for diagnostics.
//
// Thread safety: all atomics are read/written lock-free.
//                last_status_msg uses arduino_status_mtx.
 
enum class ArduinoMode_E {
    Unknown,
    Idle,
    Calibration,
    Function
};

struct ArduinoStatus_s {
    // Connection lifecycle
    std::atomic<bool> port_opened       {false}; // COM port opened by ServoDriver
    std::atomic<bool> handshake_ok      {false}; // READY received + P sent + CONNECTED received
    std::atomic<bool> connected         {false}; // same as handshake_ok but stays true across cmds
    // Machine health 
    std::atomic<bool> calibrated        {false}; // bookWidth > 0 on Arduino
    std::atomic<int>  book_width        {0};      // last reported bookWidth
    std::atomic<bool> estop_active      {false};
    std::atomic<ArduinoMode_E> mode     {ArduinoMode_E::Unknown};
 
    // Command tracking
    std::atomic<int>  heartbeats_sent   {0};
    std::atomic<int>  heartbeats_acked  {0};
 
    using Clock = std::chrono::steady_clock;
    // last time ANY byte arrived from Arduino (reader thread sets this)
    std::atomic<long long> last_rx_ts_ms {0}; // ms since epoch, steady_clock
 
    // Last human-readable status line
    mutable std::mutex arduino_status_mtx;
    std::string last_status_msg; // latest STATUS: line, or last non-DBG line
 
    void set_last_status(const std::string& s) {
        std::lock_guard<std::mutex> lk(arduino_status_mtx);
        last_status_msg = s;
    }
    std::string get_last_status() const {
        std::lock_guard<std::mutex> lk(arduino_status_mtx);
        return last_status_msg;
    }
    
    // PUBLIC HELPERS FOR ARDUINOSTATUS_S
    // ms since last byte from Arduino (for heartbeat watchdog)
    long long ms_since_last_rx() const {
        auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            Clock::now().time_since_epoch()).count();
        long long last = last_rx_ts_ms.load(std::memory_order_acquire);
        if (last == 0) return -1; // never received anything
        return now_ms - last;
    }
    static const char* mode_str(ArduinoMode_E m) {
        switch(m) {
            case ArduinoMode_E::Idle:        return "IDLE";
            case ArduinoMode_E::Calibration: return "CALIB";
            case ArduinoMode_E::Function:    return "FUNCTION";
            default:                         return "UNKNOWN";
        }
    }
};

// ─── END Arduino connection/health status ────────────────────────────────────────

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

    // Live health/status (updated by reader thread!)
    // *fields r atomic/mtx-protected for thread safety
    ArduinoStatus_s arduinoStatus;

private:
 
    // Send one byte char, set busy flag
    bool sendChar(char command);
    
    // Background reader threads to talk to arduino, watches serial port for DONE
    void readerThreadFn();
    void heartbeatThreadFn();
    std::thread readerThread; // background thread that watches for DONE from arduino
    std::thread heartbeatThread;
    // protects WriteFile() so only one thread can write to serial port at a time
    std::mutex writeMutex;

    // Serial port lifecycle
    bool openPort(const std::string& portName, unsigned int baudRate);
    void closePort();
    bool waitForReady(std::chrono::milliseconds timeout);
    void setBlockingTimeouts(bool blocking);

    void parseStatusLine(const std::string& line);
    void log(const std::string& msg) const;

    // Atomics across threads
    std::atomic<bool> connected  {false};
    std::atomic<bool> busy       {false};
    std::atomic<bool> stopReader {false};
    std::atomic<bool> stopHeartbeat {false};

    // handle COM
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

    // HEARTBEAT CONFIG
    // Send 'H' every N ms; warn if no reply for > TIMEOUT ms
    static constexpr int HEARTBEAT_INTERVAL_MS = 3000;  // 3s between pings
    static constexpr int HEARTBEAT_TIMEOUT_MS  = 10000; // warn after 10s silence

    bool handshakeOk_ = false;

};