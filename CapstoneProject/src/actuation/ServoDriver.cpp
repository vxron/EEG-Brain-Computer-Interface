#include "ServoDriver.h"
#include <iostream> 
#include <chrono>
#include "../utils/Logger.hpp"
#include <sstream>
#include <iomanip>
#include <ctime>

// ------------- Lil timestamp helper for status checking ------------
static std::string sd_timestamp() {
    auto now   = std::chrono::system_clock::now();
    auto now_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  now.time_since_epoch()) % 1000;
    char buf[32];
    std::strftime(buf, sizeof(buf), "%H:%M:%S", std::localtime(&now_t));
    std::ostringstream ss;
    ss << buf << "." << std::setw(3) << std::setfill('0') << ms.count();
    return ss.str();
}
 
void ServoDriver_C::log(const std::string& msg) const {
    LOG_ALWAYS("[ServoDriver][" << sd_timestamp() << "] " << msg);
}

void ServoDriver_C::emitLine(const std::string& line, bool fromArduino) {
    // write to serial log in StateStore (both real and mock paths)
    if (stateStoreRef_) {
        auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        std::lock_guard<std::mutex> lk(stateStoreRef_->hw_serial_log_mtx);
        stateStoreRef_->hw_serial_log.push_back({line, ts, fromArduino});
        if (stateStoreRef_->hw_serial_log.size() > StateStore_s::HW_LOG_CAPACITY)
            stateStoreRef_->hw_serial_log.pop_front();
    }

    if (!fromArduino) return; // TX lines just get logged, nothing else

    // update last-rx timestamp
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        Clock::now().time_since_epoch()).count();
    arduinoStatus.last_rx_ts_ms.store(now_ms, std::memory_order_release);

    // same dispatch for both real and mock
    if (line == "DONE") {
        busy.store(false, std::memory_order_release);
#ifdef MOCK_HARDWARE
        mockBusy_.store(false, std::memory_order_release);
#endif
        log("DONE received");
    } else if (line == "HB") {
        arduinoStatus.heartbeats_acked.fetch_add(1, std::memory_order_relaxed);
    } else if (line == "NOCALIB") {
        busy.store(false, std::memory_order_release);
#ifdef MOCK_HARDWARE
        mockBusy_.store(false, std::memory_order_release);
#endif
        log("NOCALIB received");
    } else if (line == "ESTOP") {
        busy.store(false, std::memory_order_release);
#ifdef MOCK_HARDWARE
        mockBusy_.store(false, std::memory_order_release);
#endif
        arduinoStatus.estop_active.store(true);
        log("ESTOP received");
    } else if (line.rfind("STATUS:", 0) == 0) {
        parseStatusLine(line.substr(7));
    } else if (line.rfind("DBG:", 0) == 0) {
        log("[Arduino DBG] " + line.substr(4));
    } else {
        log("[Arduino] " + line);
    }
}

// ---------- Constructor/Destructor ----------
ServoDriver_C::ServoDriver_C(const std::string& portName, unsigned int baudRate, StateStore_s* ss) : 
    stateStoreRef_(ss) {
#ifdef MOCK_HARDWARE
    arduinoStatus.port_opened.store(true);
    connected.store(true);
    log("MOCK MODE - no COM port opened. Commands logged, busy auto-clears.");
    // Simulate Arduino boot sequence — exact same STATUS lines real firmware sends
    emitLine("STATUS:BOOT",         true);
    emitLine("READY",               true);
    emitLine("STATUS:CONNECTED",    true);
    arduinoStatus.handshake_ok.store(true);
    arduinoStatus.connected.store(true);
    emitLine("STATUS:CALIBRATED=0", true);
    emitLine("STATUS:BOOKWIDTH=0",  true);
    emitLine("STATUS:ESTOP=0",      true);
    emitLine("STATUS:MODE=IDLE",    true);
    // Auto-calibrate since there's no physical button in mock mode
    mockThread_ = std::thread(&ServoDriver_C::mockRunCalibration, this);
#else 
    log("Opening port " + portName + " @ " + std::to_string(baudRate) + " baud...");
    if (!openPort(portName, baudRate)){
        log("ERROR: failed to open " + portName + " — running without servo");
        // connected stays false, all sendChar calls will early-return
        return;
    }
    arduinoStatus.port_opened.store(true);
    log("Connected to "+portName+" @ "+std::to_string(baudRate)+"baud.");

    if (!handshakeOk_) {
        log("WARN: running without confirmed handshake");
    } else {
        log("Handshake confirmed — Arduino reported CONNECTED.");
        arduinoStatus.handshake_ok.store(true);
        arduinoStatus.connected.store(true);
    }
    
    // start background reader thread
    stopReader.store(false);
    readerThread = std::thread(&ServoDriver_C::readerThreadFn, this);
    log("Actuation reader thread started.");
    // Start heartbeat thread
    stopHeartbeat.store(false);
    heartbeatThread = std::thread(&ServoDriver_C::heartbeatThreadFn, this);
    log("Heartbeat thread started (interval=" + std::to_string(HEARTBEAT_INTERVAL_MS) + "ms).");
#endif
}

ServoDriver_C::~ServoDriver_C() {
    log("Destructor stopping threads...");
#ifdef MOCK_HARDWARE
    mockStop_.store(true);
    if (mockThread_.joinable()) mockThread_.join();
#endif
    stopHeartbeat.store(true);
    stopReader.store(true);
    if (heartbeatThread.joinable()) heartbeatThread.join();
    if (readerThread.joinable())    readerThread.join();
    closePort();
}

// ---------- Public Commands ----------
// Each command sets the real estimate BEFORE calling sendChar,
// so the timer starts with the right duration

void ServoDriver_C::flipForward(){
    log("flipForward() -> 'F'");
#ifdef MOCK_HARDWARE
    mockBusyUntil = Clock::now() + std::chrono::milliseconds(MOCK_BUSY_FLIP_FWD_MS);
#else
    realBusyEstimate_ms_ = REAL_BUSY_FLIP_FWD_MS;
#endif
    sendChar('F');
}

void ServoDriver_C::flipBackward(){
    log("flipBackward() -> 'B'");
#ifdef MOCK_HARDWARE
    mockBusyUntil = Clock::now() + std::chrono::milliseconds(MOCK_BUSY_FLIP_BWD_MS);
#else
    realBusyEstimate_ms_ = REAL_BUSY_FLIP_BWD_MS;
#endif
    sendChar('B');
}

void ServoDriver_C::openClips(){
    log("openClips() -> 'O'");
#ifdef MOCK_HARDWARE
    mockBusyUntil = Clock::now() + std::chrono::milliseconds(MOCK_BUSY_OPEN_CLIPS_MS);
#else
    realBusyEstimate_ms_ = REAL_BUSY_OPEN_CLIPS_MS;
#endif
    sendChar('O');
}

void ServoDriver_C::zeroPosition(){
    log("zeroPosition() -> 'C'");
#ifdef MOCK_HARDWARE
    mockBusyUntil = Clock::now() + std::chrono::milliseconds(MOCK_BUSY_ZERO_MS);
#else
    realBusyEstimate_ms_ = REAL_BUSY_ZERO_MS;
#endif
    sendChar('C');
}

// ---------- States ----------
bool ServoDriver_C::isBusy() const {
#ifdef MOCK_HARDWARE
    return mockBusy_.load(std::memory_order_acquire);
#else
    return busy.load(std::memory_order_acquire);
#endif
}

bool ServoDriver_C::isConnected() const {
    return connected.load(std::memory_order_acquire);
}

std::chrono::milliseconds ServoDriver_C::estimatedBusyRemaining() const {
#ifdef MOCK_HARDWARE
    auto now = Clock::now();
    if (now >= mockBusyUntil) return std::chrono::milliseconds{0};
    return std::chrono::duration_cast<std::chrono::milliseconds>(mockBusyUntil - now);
#else
    if (!busy.load(std::memory_order_acquire))   return std::chrono::milliseconds{0};
    if (!realBusyEstimateTimer_.is_started())    return std::chrono::milliseconds{0};
    auto elapsed = realBusyEstimateTimer_.get_timer_value_ms();
    auto total   = std::chrono::milliseconds{realBusyEstimate_ms_};
    if (elapsed >= total) return std::chrono::milliseconds{0};
    return total - elapsed;
#endif
}

// ---------- sendChar (writes 1 byte, sets busy flag) ----------
bool ServoDriver_C::sendChar(char cmd){
#ifdef MOCK_HARDWARE
    emitLine(std::string("TX:'") + cmd + "'", false); // log the outgoing byte
    mockBusy_.store(true, std::memory_order_release);
    busy.store(true, std::memory_order_release);
    if (mockThread_.joinable()) mockThread_.join();
    if      (cmd == 'F' || cmd == 'f')
        mockThread_ = std::thread(&ServoDriver_C::mockRunFlip, this, true);
    else if (cmd == 'B' || cmd == 'b')
        mockThread_ = std::thread(&ServoDriver_C::mockRunFlip, this, false);
    // O and C don't need a thread — they just clear busy after the timeout
    // (no STATUS protocol for clips/zero in Arduino firmware)
    else {
        // brief sleep then auto-clear, same as real firmware DONE
        std::thread([this, cmd](){
            std::this_thread::sleep_for(std::chrono::milliseconds{
                cmd == 'O' ? MOCK_BUSY_OPEN_CLIPS_MS : MOCK_BUSY_ZERO_MS});
            emitLine("DONE", true);
        }).detach();
    }
    return true;
#else
    std::lock_guard<std::mutex> lock(writeMutex); // grab the mutex
    if(!connected.load() || m_hSerial == nullptr){  // if port isnt open, dont try to write 
        log("sendChar called but port is not open.");
        return false;
    }
    HANDLE h = static_cast<HANDLE>(m_hSerial);
    DWORD bytesWritten = 0;
    // WriteFile is the Windows function that sends the byte down the COM port 
    //      Arguments:
    //          h = which port
    //          &cmd = what to send (gives WriteFile a ptr to the char)
    //          1 = how many bytes to send
    //          &bytesWrittedn = Windows fills this in with how many bytes actually got sent
    //          nullptr = ignore this, used for async mode which we arent using
    BOOL ok = WriteFile(h, &cmd, 1, &bytesWritten, nullptr);
    if(!ok || bytesWritten != 1){
        log(std::string("WriteFile failed for cmd '") + cmd + "'.");
        return false;
    }
    // Send confirmed — start estimate timer and set busy
    realBusyEstimateTimer_.start_timer(std::chrono::milliseconds{realBusyEstimate_ms_});
    busy.store(true, std::memory_order_release);
    return true;
#endif
}

// ---------- readerThreadFn (bg thread that reads serial for DONE) ----------
void ServoDriver_C::readerThreadFn(){
#ifdef MOCK_HARDWARE
    return; // no reader thread needed, mock uses emitLine directly
#else
    log("Reader thread running.");
    std::string lineBuffer;
    while (!stopReader.load(std::memory_order_acquire)){
        if(!connected.load() || m_hSerial == nullptr){
            std::this_thread::sleep_for(std::chrono::milliseconds(10));        
            continue;
        }
        HANDLE h = static_cast<HANDLE>(m_hSerial);
        char byte = 0;
        DWORD bytesRead = 0;
        // ReadFile reads 1 byte from COM port into byte
        BOOL ok = ReadFile(h, &byte, 1, &bytesRead, nullptr);
        if(!ok || bytesRead==0){
            std::this_thread::sleep_for(std::chrono::milliseconds(1)); // continuously check (non-blocking)
            continue;
        }
        // Update last-rx timestamp
        auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            Clock::now().time_since_epoch()).count();
        arduinoStatus.last_rx_ts_ms.store(now_ms, std::memory_order_release);
        
        if(byte == '\n'){
            // Line complete, check if it's DONE
            // Arduino send "DONE\r\n" one char at a time, so collect chars into lineBuffer until we hit \n
            if (!lineBuffer.empty() && lineBuffer.back() == '\r')
                lineBuffer.pop_back();
            const std::string line = lineBuffer;
            lineBuffer.clear();
            if (line.empty()) continue;
            emitLine(line, true); // this single call handles log + arduinoStatus + busy
        } else {
            lineBuffer += byte;
        }
    }
    log("Reader thread exited.");
#endif
}

// PARSESTATUSLINE Called from emitLine for both paths for every line starting with "STATUS:" from Arduino to monitor health
void ServoDriver_C::parseStatusLine(const std::string& line) {
    log("[Arduino STATUS] " + line);
    arduinoStatus.set_last_status(line);

    if (line == "BOOT") {
        connected.store(false, std::memory_order_release);
        arduinoStatus.connected.store(false, std::memory_order_release);
        arduinoStatus.handshake_ok.store(false, std::memory_order_release);
        arduinoStatus.calibrated.store(false);
        arduinoStatus.mode.store(ArduinoMode_E::Unknown);
        arduinoStatus.estop_active.store(false);
        busy.store(false, std::memory_order_release);
        log("WARN: Arduino BOOT detected!");
    } else if (line == "CONNECTED") {
        connected.store(true, std::memory_order_release);
        arduinoStatus.handshake_ok.store(true, std::memory_order_release);
        arduinoStatus.connected.store(true, std::memory_order_release);
    } else if (line == "CALIB_START") {
        arduinoStatus.calibrated.store(false);
        arduinoStatus.mode.store(ArduinoMode_E::Calibration);
    } else if (line.rfind("CALIB_RIGHT=", 0) == 0) {
        try { log("Calib right edge = " + line.substr(12) + " steps"); } catch (...) {}
    } else if (line.rfind("CALIB_LEFT=", 0) == 0) {
        try { log("Calib left edge = " + line.substr(11) + " steps"); } catch (...) {}
    } else if (line.rfind("CALIB_DONE", 0) == 0) {
        arduinoStatus.calibrated.store(true);
        auto pos = line.find("bookWidth=");
        if (pos != std::string::npos) {
            try {
                int bw = std::stoi(line.substr(pos + 10));
                arduinoStatus.book_width.store(bw);
#ifdef MOCK_HARDWARE
                mockState_.bookWidth = bw;
                mockState_.calibrated = true;
#endif
                log("Calibration complete. bookWidth=" + std::to_string(bw));
            } catch (...) {}
        }
    } else if (line == "ESTOP_ON") {
        arduinoStatus.estop_active.store(true);
#ifdef MOCK_HARDWARE
        mockState_.estopActive = true;
#endif
        busy.store(false, std::memory_order_release);
        log("WARN: Arduino E-STOP activated!");
    } else if (line == "ESTOP_OFF") {
        arduinoStatus.estop_active.store(false);
#ifdef MOCK_HARDWARE
        mockState_.estopActive = false;
#endif
        log("Arduino E-STOP cleared.");
    } else if (line == "CMD_F") {
        log("Arduino acknowledged CMD_F");
    } else if (line == "CMD_B") {
        log("Arduino acknowledged CMD_B");
    } else if (line.rfind("MODE=", 0) == 0) {
        std::string m = line.substr(5);
        if      (m == "IDLE")     arduinoStatus.mode.store(ArduinoMode_E::Idle);
        else if (m == "CALIB")    arduinoStatus.mode.store(ArduinoMode_E::Calibration);
        else if (m == "FUNCTION") arduinoStatus.mode.store(ArduinoMode_E::Function);
        log("Arduino mode -> " + m);
    } else if (line.rfind("CALIBRATED=", 0) == 0) {
        arduinoStatus.calibrated.store(line.substr(11) == "1");
    } else if (line.rfind("BOOKWIDTH=", 0) == 0) {
        try {
            int bw = std::stoi(line.substr(10));
            arduinoStatus.book_width.store(bw);
            arduinoStatus.calibrated.store(bw > 0);
        } catch (...) {}
    } else if (line.rfind("ESTOP=", 0) == 0) {
        arduinoStatus.estop_active.store(line.substr(6) == "1");
    } else if (line.rfind("CONNECTED=", 0) == 0) {
        arduinoStatus.connected.store(line.substr(10) == "1");
    }
}

// ----------- Heartbeat Thread Fxn ---------------------
void ServoDriver_C::heartbeatThreadFn() {
#ifdef MOCK_HARDWARE
    return;
#else
    log("Heartbeat thread running (every " + std::to_string(HEARTBEAT_INTERVAL_MS) + "ms).");
    while (!stopHeartbeat.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds{HEARTBEAT_INTERVAL_MS});
        if (stopHeartbeat.load(std::memory_order_acquire)) break;
        
        // Only send heartbeat when not busy (avoid polluting mid-command serial stream)
        if (!busy.load(std::memory_order_acquire) && connected.load(std::memory_order_acquire)) {
            {
                std::lock_guard<std::mutex> lk(writeMutex);
                if (m_hSerial) {
                    char hb = 'H';
                    DWORD w = 0;
                    WriteFile(static_cast<HANDLE>(m_hSerial), &hb, 1, &w, nullptr);
                    arduinoStatus.heartbeats_sent.fetch_add(1, std::memory_order_relaxed);
                }
            }
            // Check if heartbeat was acked recently (liveness check)
            // note this timeout is acc based on any RX (not just heartbeat ack)
            long long ms_silent = arduinoStatus.ms_since_last_rx();
            if (ms_silent > HEARTBEAT_TIMEOUT_MS) {
                const bool was_connected = connected.exchange(false, std::memory_order_acq_rel);
                // we were conn and now were not (so its likely not a bootup delay or anything..) not safe to actuate (connection loss) mia arduino <3
                connected.store(false, std::memory_order_release);
                arduinoStatus.connected.store(false, std::memory_order_release);
                arduinoStatus.handshake_ok.store(false, std::memory_order_release);
                busy.store(false, std::memory_order_release); // avoid getting stuck waiting for a DONE that will never come 
                if (was_connected) {
                    log("ERROR: Arduino connection lost");
                }
                arduinoStatus.set_last_status("HEARTBEAT_TIMEOUT");
            }
        }
    }
    log("Heartbeat thread exited.");
#endif
}

// ---------- Open and Close Win32 Serial Port ----------
bool ServoDriver_C::openPort(const std::string& portName, unsigned int baudRate){
#ifdef MOCK_HARDWARE
    (void)portName;
    (void)baudRate;
    return true;
#else
    std::string fullPort = "\\\\.\\" + portName;
    HANDLE h = CreateFileA( // Windows function that opens COM port
        fullPort.c_str(),
        GENERIC_READ | GENERIC_WRITE,
        0,
        nullptr,
        OPEN_EXISTING, //only open if port exists
        FILE_ATTRIBUTE_NORMAL,
        nullptr
    );

    if (h == INVALID_HANDLE_VALUE) {
        log("CreateFileA failed on " + fullPort + " — WinErr=" + std::to_string(GetLastError()));
        return false;
    }

    DCB dcb = {};
    dcb.DCBlength = sizeof(DCB);
    if (!GetCommState(h, &dcb)) { CloseHandle(h); return false; }
 
    dcb.BaudRate = baudRate;
    dcb.ByteSize = 8;
    dcb.StopBits = ONESTOPBIT;
    dcb.Parity   = NOPARITY;
 
    if (!SetCommState(h, &dcb)) { CloseHandle(h); return false; }
 
    m_hSerial = static_cast<void*>(h);
    connected.store(true);
    // handshake phase; needs blocking reads
    setBlockingTimeouts(true);
    PurgeComm(h, PURGE_RXCLEAR | PURGE_TXCLEAR);
    log("Waiting 1500ms for Arduino reset...");
    std::this_thread::sleep_for(std::chrono::milliseconds{1500});
    handshakeOk_ = waitForReady(std::chrono::milliseconds{6000});

    // reader thread phase; needs non-blocking polls
    setBlockingTimeouts(false);
    if (!handshakeOk_) {
        log("WARN: no READY from Arduino within 5s — port open but Arduino state unknown");
    } else {
        log("Handshake OK — Arduino confirmed ready");
        arduinoStatus.handshake_ok.store(true);
        arduinoStatus.connected.store(true);
    }
    return true;
    #endif
}

#ifndef MOCK_HARDWARE
bool ServoDriver_C::waitForReady(std::chrono::milliseconds timeout) {
    auto deadline = Clock::now() + timeout;
    std::string lineBuf;
    HANDLE h = static_cast<HANDLE>(m_hSerial);
    log("waitForReady: listening for 'READY' beacon from Arduino...");
    while (Clock::now() < deadline) {
        char byte = 0; DWORD bytesRead = 0;
        BOOL ok = ReadFile(h, &byte, 1, &bytesRead, nullptr);
        if (!ok || bytesRead == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds{5});
            continue;
        }
        if (byte == '\n') {
            if (!lineBuf.empty() && lineBuf.back() == '\r') lineBuf.pop_back();
            emitLine(lineBuf, true); // log it
            if (lineBuf == "READY") {
                DWORD written = 0; char ping = 'P';
                WriteFile(h, &ping, 1, &written, nullptr);
                emitLine("TX:'P'", false);
                log("Sent 'P' ping. Waiting for STATUS:CONNECTED...");
                std::string lineBuf2;
                auto ackDeadline = Clock::now() + std::chrono::milliseconds{2000};
                while (Clock::now() < ackDeadline) {
                    char b2 = 0; DWORD r2 = 0;
                    BOOL ok2 = ReadFile(h, &b2, 1, &r2, nullptr);
                    if (!ok2 || r2 == 0) { std::this_thread::sleep_for(std::chrono::milliseconds{2}); continue; }
                    if (b2 == '\n') {
                        if (!lineBuf2.empty() && lineBuf2.back() == '\r') lineBuf2.pop_back();
                        emitLine(lineBuf2, true);
                        if (lineBuf2.rfind("STATUS:CONNECTED", 0) == 0) {
                            log("Handshake complete.");
                            return true;
                        }
                        if (lineBuf2.rfind("STATUS:CALIBRATED=", 0) == 0)
                            arduinoStatus.calibrated.store(lineBuf2.substr(18) == "1");
                        if (lineBuf2.rfind("STATUS:BOOKWIDTH=", 0) == 0)
                            try { arduinoStatus.book_width.store(std::stoi(lineBuf2.substr(17))); } catch (...) {}
                        if (lineBuf2.rfind("STATUS:ESTOP=", 0) == 0)
                            arduinoStatus.estop_active.store(lineBuf2.substr(13) == "1");
                        lineBuf2.clear();
                    } else { lineBuf2 += b2; }
                }
                log("WARN: No STATUS:CONNECTED received after P.");
                return true;
            }
            lineBuf.clear();
        } else { lineBuf += byte; }
    }
    log("waitForReady: timed out.");
    return false;
}
#else
bool ServoDriver_C::waitForReady(std::chrono::milliseconds) {
    return true;
}
#endif

void ServoDriver_C::closePort() {
#ifndef MOCK_HARDWARE
    if (m_hSerial != nullptr) {
        CloseHandle(static_cast<HANDLE>(m_hSerial));
        m_hSerial = nullptr;
        connected.store(false);
        arduinoStatus.connected.store(false);
    }
#endif
}

void ServoDriver_C::setBlockingTimeouts(bool blocking) {
#ifndef MOCK_HARDWARE
    HANDLE h = static_cast<HANDLE>(m_hSerial);
    COMMTIMEOUTS t = {};
    if (blocking) {
        t.ReadIntervalTimeout         = 0;
        t.ReadTotalTimeoutMultiplier  = 0;
        t.ReadTotalTimeoutConstant    = 100;   // block up to 100ms per ReadFile
        t.WriteTotalTimeoutConstant   = 2000;
    } else {
        t.ReadIntervalTimeout         = MAXDWORD; // return immediately if no data
        t.ReadTotalTimeoutMultiplier  = 0;
        t.ReadTotalTimeoutConstant    = 0;
        t.WriteTotalTimeoutConstant   = 2000;
    }
    SetCommTimeouts(h, &t);
#endif
}

// MOCK IMPLEMENTATION
#ifdef MOCK_HARDWARE
void ServoDriver_C::mockRunFlip(bool forward) {
    auto sleepMs = [&](int ms) {
        auto end = Clock::now() + std::chrono::milliseconds(ms);
        while (Clock::now() < end && !mockStop_.load())
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
    };

    emitLine(forward ? "STATUS:CMD_F" : "STATUS:CMD_B", true);

    // Reject conditions — same as real firmware
    if (mockState_.estopActive) {
        emitLine("ESTOP", true);
        return; // busy cleared by emitLine
    }
    if (mockState_.injectNocalib || !mockState_.calibrated) {
        mockState_.injectNocalib = false; // one-shot
        emitLine("NOCALIB", true);
        return;
    }

    emitLine("STATUS:MODE=FUNCTION", true);
    sleepMs(mockState_.flipDuration_ms / 2);

    // Mid-flip estop injection
    if (mockState_.injectEstopMid) {
        mockState_.injectEstopMid = false;
        mockState_.estopActive = true;
        emitLine("STATUS:ESTOP_ON", true);
        emitLine("!!! ALL MOTION STOPPED !!!", true);
        return;
    }

    sleepMs(mockState_.flipDuration_ms / 2);
    emitLine("STATUS:MODE=IDLE", true);
    emitLine("DONE", true); // emitLine clears busy when it sees DONE
}

void ServoDriver_C::mockRunCalibration() {
    auto sleepMs = [&](int ms) {
        auto end = Clock::now() + std::chrono::milliseconds(ms);
        while (Clock::now() < end && !mockStop_.load())
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
    };

    emitLine("STATUS:CALIB_START",    true);
    emitLine("STATUS:MODE=CALIB",     true);
    sleepMs(400);
    emitLine("STATUS:CALIB_RIGHT=320", true);
    sleepMs(500);
    emitLine("STATUS:CALIB_LEFT=960",  true);
    sleepMs(300);
    mockState_.bookWidth  = 640;
    mockState_.calibrated = true;
    emitLine("STATUS:CALIB_DONE bookWidth=640", true);
    emitLine("STATUS:MODE=IDLE",      true);
}



#endif