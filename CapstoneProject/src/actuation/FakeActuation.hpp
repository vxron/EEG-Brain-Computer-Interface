#pragma once
#include <string>
#include "IActuationProvider.h"
#include <atomic>

class FakeActuation_C : public IActuationProvider_C {
public:
    FakeActuation_C();
    void sendCommand(const std::string& cmd) override;
    bool isBusy() const override;
private:
    std::atomic<bool> busy_;
};
