/*
STIMULUS CONTROLLER : writer
- builds the training schedule
- every ~100ms, updates atomics in the shared StateStore and publishes a new 'seq' to server so HTML can see it & compare against lastSeq
- launches the HTML once at start, sets a neutral end state on finish
*/