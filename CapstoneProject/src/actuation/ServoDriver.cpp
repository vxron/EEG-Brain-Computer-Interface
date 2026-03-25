#include "ServoDriver.h"
#include <iostream> 
#include <chrono>
#include "../utils/Logger.hpp"
#include <sstream>

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

// ---------- Constructor/Destructor ----------
ServoDriver_C::ServoDriver_C(const std::string& portName, unsigned int baudRate){
#ifdef MOCK_HARDWARE
    arduinoStatus.port_opened.store(true);
    arduinoStatus.handshake_ok.store(true);
    arduinoStatus.connected.store(true);
    connected.store(true);
    log("MOCK MODE - no COM port opened. Commands logged, busy auto-clears.");
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
bool ServoDriver_C::isBusy() const{
#ifdef MOCK_HARDWARE
    return Clock::now() < mockBusyUntil; //busy until estimated timeout elapses
#else
    return busy.load(std::memory_order_acquire); // busy until reader thread clears flag on DONE
#endif
}

bool ServoDriver_C::isConnected() const{
    return connected.load(std::memory_order_acquire);
}

// ---------- sendChar (writes 1 byte, sets busy flag) ----------
bool ServoDriver_C::sendChar(char cmd){
#ifdef MOCK_HARDWARE
    std::cout << "[" << sd_timestamp() << "] [ServoDriver] [MOCK TX]" << cmd << "\n";
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
    return; // no reader thread needed, isBusy() uses mockBusyUntil
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
            // Since Arduino Serial.println sends \r\n, strip trailing \r if present
            if(!lineBuffer.empty() && lineBuffer.back() == '\r'){
                lineBuffer.pop_back();
            }
            
            const std::string line = lineBuffer;
            lineBuffer.clear();
            
            if (line.empty()) continue;
            
            if(line == "DONE"){
                busy.store(false, std::memory_order_release);
                log("ACK received - Arduino DONE. Ready for next command.");
            }
            
            else if(line == "HB"){
                arduinoStatus.heartbeats_acked.fetch_add(1, std::memory_order_relaxed);
                // don't log every heartbeat at ALWAYS level
                // but track the timestamp update (done above)
 
            } else if (line == "NOCALIB") {
                // Arduino rejected command because bookWidth == 0
                busy.store(false, std::memory_order_release); // unblock consumer
                log("ERROR: Arduino NOCALIB — command rejected (bookWidth==0). "
                    "Consumer unblocked. calibrated=" +
                    std::string(arduinoStatus.calibrated.load() ? "true" : "false"));
 
            } else if (line == "ESTOP") {
                busy.store(false, std::memory_order_release);
                arduinoStatus.estop_active.store(true);
                log("ERROR: Arduino ESTOP — command rejected, e-stop active. Consumer unblocked.");
 
            } else if (line.rfind("STATUS:", 0) == 0) {
                parseStatusLine(line.substr(7));
 
            } else if (line.rfind("DBG:", 0) == 0) {
                // Verbose Arduino debug line
                log("[Arduino DBG] " + line.substr(4));
 
            } else {
                // Any other line from Arduino (legacy prints etc.)
                log("[Arduino] " + line);
            }
        } else {
            lineBuffer += byte;
        }
    }
    log("Reader thread exited.");
#endif
}

// PARSESTATUSLINE Called from readerThreadFn for every line starting with "STATUS:"
void ServoDriver_C::parseStatusLine(const std::string& line) {
    // line is the part AFTER "STATUS:"
    log("[Arduino STATUS] " + line);
    arduinoStatus.set_last_status(line);
 
    if (line == "BOOT") {
        // Arduino just reset; mark as not calibrated, mode unknown
        arduinoStatus.calibrated.store(false);
        arduinoStatus.mode.store(ArduinoMode_E::Unknown);
        arduinoStatus.estop_active.store(false);
        log("WARN: Arduino BOOT detected — it may have reset unexpectedly!");
 
    } else if (line == "CONNECTED") {
        arduinoStatus.handshake_ok.store(true);
        arduinoStatus.connected.store(true);
        log("Arduino confirmed connected.");
 
    } else if (line == "CALIB_START") {
        arduinoStatus.calibrated.store(false);
        arduinoStatus.mode.store(ArduinoMode_E::Calibration);
 
    } else if (line.rfind("CALIB_RIGHT=", 0) == 0) {
        // CALIB_RIGHT=<steps>
        try {
            int v = std::stoi(line.substr(12));
            log("Calib right edge = " + std::to_string(v) + " steps");
        } catch (...) {}
 
    } else if (line.rfind("CALIB_LEFT=", 0) == 0) {
        try {
            int v = std::stoi(line.substr(11));
            log("Calib left edge = " + std::to_string(v) + " steps");
        } catch (...) {}
 
    } else if (line.rfind("CALIB_DONE", 0) == 0) {
        arduinoStatus.calibrated.store(true);
        // try to parse bookWidth from "CALIB_DONE bookWidth=<n>"
        auto pos = line.find("bookWidth=");
        if (pos != std::string::npos) {
            try {
                int bw = std::stoi(line.substr(pos + 10));
                arduinoStatus.book_width.store(bw);
                log("Calibration complete. bookWidth=" + std::to_string(bw));
            } catch (...) {}
        }
 
    } else if (line == "ESTOP_ON") {
        arduinoStatus.estop_active.store(true);
        busy.store(false, std::memory_order_release); // unblock consumer bc no DONE is coming
        log("WARN: Arduino E-STOP activated!");
 
    } else if (line == "ESTOP_OFF") {
        arduinoStatus.estop_active.store(false);
        log("Arduino E-STOP cleared.");
 
    } else if (line == "CMD_F") {
        log("Arduino acknowledged CMD_F (forward flip)");
 
    } else if (line == "CMD_B") {
        log("Arduino acknowledged CMD_B (backward flip)");
 
    } else if (line.rfind("MODE=", 0) == 0) {
        std::string modeStr = line.substr(5);
        if      (modeStr == "IDLE")     arduinoStatus.mode.store(ArduinoMode_E::Idle);
        else if (modeStr == "CALIB")    arduinoStatus.mode.store(ArduinoMode_E::Calibration);
        else if (modeStr == "FUNCTION") arduinoStatus.mode.store(ArduinoMode_E::Function);
        log("Arduino mode -> " + modeStr);
 
    } else if (line.rfind("STEPPOS=", 0) == 0) {
        try {
            int pos = std::stoi(line.substr(8));
        } catch (...) {}
 
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
            long long ms_silent = arduinoStatus.ms_since_last_rx();
            if (ms_silent > HEARTBEAT_TIMEOUT_MS) {
                log("WARN: No data from Arduino for " + std::to_string(ms_silent)
                    + "ms — Arduino may be unresponsive or frozen!");
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

bool ServoDriver_C::waitForReady(std::chrono::milliseconds timeout) {
    auto deadline = Clock::now() + timeout;
    std::string lineBuf;
    HANDLE h = static_cast<HANDLE>(m_hSerial);
    log("waitForReady: listening for 'READY' beacon from Arduino...");
 
    while (Clock::now() < deadline) {
        char  byte = 0;
        DWORD bytesRead = 0;
        BOOL  ok = ReadFile(h, &byte, 1, &bytesRead, nullptr);
        if (!ok || bytesRead == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds{5});
            continue;
        }
 
        if (byte == '\n') {
            if (!lineBuf.empty() && lineBuf.back() == '\r') lineBuf.pop_back();
            log("Arduino says: [" + lineBuf + "]");
 
            if (lineBuf == "READY") {
                // Send ping
                DWORD written = 0;
                char  ping    = 'P';
                WriteFile(h, &ping, 1, &written, nullptr);
                log("Sent 'P' ping. Waiting for STATUS:CONNECTED...");
 
                // Now wait for STATUS:CONNECTED confirmation
                // (Arduino sends this immediately after receiving P)
                std::string lineBuf2;
                auto ackDeadline = Clock::now() + std::chrono::milliseconds{2000};
                while (Clock::now() < ackDeadline) {
                    char b2 = 0; DWORD r2 = 0;
                    BOOL ok2 = ReadFile(h, &b2, 1, &r2, nullptr);
                    if (!ok2 || r2 == 0) { std::this_thread::sleep_for(std::chrono::milliseconds{2}); continue; }
                    if (b2 == '\n') {
                        if (!lineBuf2.empty() && lineBuf2.back() == '\r') lineBuf2.pop_back();
                        log("Arduino says: [" + lineBuf2 + "]");
                        if (lineBuf2.rfind("STATUS:CONNECTED", 0) == 0) {
                            log("Handshake complete — STATUS:CONNECTED received.");
                            return true;
                        }
                        // Also accept the full status dump lines that follow CONNECTED
                        // (Arduino calls emitFullStatus() right after sending STATUS:CONNECTED)
                        if (lineBuf2.rfind("STATUS:CALIBRATED=", 0) == 0) {
                            arduinoStatus.calibrated.store(lineBuf2.substr(18) == "1");
                        }
                        if (lineBuf2.rfind("STATUS:BOOKWIDTH=", 0) == 0) {
                            try { arduinoStatus.book_width.store(std::stoi(lineBuf2.substr(17))); } catch (...) {}
                        }
                        if (lineBuf2.rfind("STATUS:ESTOP=", 0) == 0) {
                            arduinoStatus.estop_active.store(lineBuf2.substr(13) == "1");
                        }
                        lineBuf2.clear();
                    } else {
                        lineBuf2 += b2;
                    }
                }
                log("WARN: No STATUS:CONNECTED received after P. Treating as partial handshake.");
                return true; // We at least got READY + sent P, so mostly OK
            }
            lineBuf.clear();
        } else {
            lineBuf += byte;
        }
    }
    log("waitForReady: timed out waiting for READY ping from Arduino.");
    return false;
}

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

std::chrono::milliseconds ServoDriver_C::estimatedBusyRemaining() const {
#ifdef MOCK_HARDWARE
    auto now = Clock::now();
    if(now >= mockBusyUntil) return std::chrono::milliseconds{0};
    return std::chrono::duration_cast<std::chrono::milliseconds>(mockBusyUntil - now);
#else
    // If Arduino already sent DONE, busy is false
    if(!busy.load(std::memory_order_acquire)) return std::chrono::milliseconds{0};

    if(!realBusyEstimateTimer_.is_started()) return std::chrono::milliseconds{0};

    // Remaining = estimate_ms - elapsed
    auto elapsed = realBusyEstimateTimer_.get_timer_value_ms();
    auto total   = std::chrono::milliseconds{realBusyEstimate_ms_};
    if(elapsed >= total) return std::chrono::milliseconds{0};
    return total - elapsed;
#endif
}

void ServoDriver_C::setBlockingTimeouts(bool blocking) {
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
}