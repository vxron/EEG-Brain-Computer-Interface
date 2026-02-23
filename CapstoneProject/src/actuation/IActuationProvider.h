#pragma once
#include <string>

class IActuationProvider_C {
public:
    virtual ~IActuationProvider_C() = default; // virtual destructor for proper cleanup of derived classes
    virtual void sendCommand(const std::string& cmd) = 0;
    virtual bool isBusy() const = 0;
};