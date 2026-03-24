#include "ServoDriver.h"
#include <iostream> 
#include <chrono>
#include "../utils/Logger.hpp"

// ---------- Constructor/Destructor ----------
ServoDriver_C::ServoDriver_C(const std::string& portName, unsigned int baudRate){
#ifdef MOCK_HARDWARE
    connected.store(true);
    log("MOCK MODE - no COM port opened. Commands logged, busy auto-clears.");
#else 
    if (!openPort(portName, baudRate)){
        LOG_ALWAYS("ServoDriver: failed to open " << portName << ", running without servo");
        // connected stays false, all sendChar calls will early-return
        return;
    }
    log("Connected to "+portName+" @ "+std::to_string(baudRate)+"baud.");

    // start background reader thread
    stopReader.store(false);
    readerThread = std::thread(&ServoDriver_C::readerThreadFn, this);
    log("Reader thread started.");
#endif
}

ServoDriver_C::~ServoDriver_C() {
    stopReader.store(true);
    if(readerThread.joinable()){ // checks if thread is running
        readerThread.join(); // waits for thread to finish before destroying object
    }
    closePort();
}

// ---------- Logging ----------
static std::string sd_timestamp(){
    auto now = std::chrono::system_clock::now();
    auto now_t = std::chrono::system_clock::to_time_t(now);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%H:%M:%S", std::localtime(&now_t));
    return std::string(buf);
}

void ServoDriver_C::log(const std::string& msg) const{
    std::cout << "[" << sd_timestamp() << "] [ServoDriver] " << msg << "\n";
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
        if(byte == '\n'){
            // Line complete, check if it's DONE
            // Arduino send "DONE\r\n" one char at a time, so collect chars into lineBuffer until we hit \n
            // Since Arduino Serial.println sends \r\n, strip trailing \r if present
            if(!lineBuffer.empty() && lineBuffer.back() == '\r'){
                lineBuffer.pop_back();
            }
            if(lineBuffer == "DONE"){
                busy.store(false, std::memory_order_release);
                log("ACK received - Arduino DONE. Ready for next command.");
            }
            else if(!lineBuffer.empty()){
                log("[Arduino] " + lineBuffer);
            }
            lineBuffer.clear();
        } else {
            lineBuffer += byte;
        }
    }
    log("Reader thread stopped.");
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
    if (h == INVALID_HANDLE_VALUE) return false;

    DCB dcb = {};
    dcb.DCBlength = sizeof(DCB);
    if (!GetCommState(h, &dcb)) { CloseHandle(h); return false; }
 
    dcb.BaudRate = baudRate;
    dcb.ByteSize = 8;
    dcb.StopBits = ONESTOPBIT;
    dcb.Parity   = NOPARITY;
 
    if (!SetCommState(h, &dcb)) { CloseHandle(h); return false; }
 
    // Read timeout: return immediately if no data (reader thread polls at 1ms)
    // Write timeout: 2 seconds
    COMMTIMEOUTS timeouts             = {};
    timeouts.ReadIntervalTimeout      = MAXDWORD;
    timeouts.ReadTotalTimeoutConstant = 0;
    timeouts.ReadTotalTimeoutMultiplier = 0;
    timeouts.WriteTotalTimeoutConstant  = 2000;
    timeouts.WriteTotalTimeoutMultiplier = 0;
 
    if (!SetCommTimeouts(h, &timeouts)) { CloseHandle(h); return false; }
 
    m_hSerial = static_cast<void*>(h);
    connected.store(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(2000)); // let Arduino boot
    return true;
#endif
}

void ServoDriver_C::closePort() {
#ifndef MOCK_HARDWARE
    if (m_hSerial != nullptr) {
        CloseHandle(static_cast<HANDLE>(m_hSerial));
        m_hSerial = nullptr;
        connected.store(false);
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