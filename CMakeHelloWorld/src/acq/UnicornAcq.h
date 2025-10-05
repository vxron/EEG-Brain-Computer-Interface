/*
* Real-device “getData” unit

Files: src/acq/UnicornAcq.h, src/acq/UnicornAcq.cpp

Goal: expose a function with the exact same signature as your fake’s getData.

What to include / provide (no code)

A getData(numberOfScans, dest, destLen) function that:

Validates destLen >= numberOfScans * NUM_CH_CHUNK.

Calls UNICORN_GetData(handle, (uint32_t)numberOfScans, dest).

Returns true/false per SDK return code.

A pair of lifecycle functions:

openDevice() — opens/configures Unicorn and stores a static or internal handle.

closeDevice() — shuts down/cleans up.

Keep Unicorn headers only in this pair of files. Everything else stays SDK-agnostic.
*/