C++ program for EEG-Based Brain-Computer-Interface (BCI) Using SSVEP and State Vector Machine

main --> CMakeHelloWorld.h/.cpp
[acq]
--> EEG data ingest from Unicorn Black EEG headset (using their C API)
--> "Fake" data stream file for unit testing
[decode]
--> Calibrate mode (training protocol): extract tuned decision model for new user with labelled (R/L) SSVEP inputs (SVM lives here)
--> Run mode: decode SSVEPs vs no action states with extracted model
[process]
--> Condition raw EEG data for frequencies of interest, artifact reduction
--> Feature extraction for SSVEP decoding (SVM model)
[stimulus]
--> Flashing L/R arrows of 2 different frequencies (on its own thread)
--> Calibrate and run modes are separate, controllable from terminal (calibrate must be accompanied with protocol instructions)
[utils]
--> Files containing common classes, structs, enums used by different modules
