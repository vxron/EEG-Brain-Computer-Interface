# EEG-Based Brain-Computer Interface (BCI) Using SSVEP and Support Vector Machine (SVM)

This C++ program will implement a Steady-State Visually Evoked Potential (SSVEP)-based Brain-Computer Interface (BCI) pipeline using EEG data acquired from the Unicorn Black headset.  
The system will support both calibration (training) and run modes, decoding left/right flicker responses into control commands via a trained SVM.

---

## Project Structure

- **main/**
  - `CMakeHelloWorld.cpp`, `CMakeHelloWorld.h`
  - Entry point and initialization

- **acq/**
  - EEG data acquisition via Unicorn Black C API
  - "Fake" data stream generator for unit testing

- **decode/**
  - Calibration mode:
    - Guides the user through a training protocol
    - Extracts an SVM model using labeled (Left/Right) trials
  - Run mode:
    - Uses trained SVM to classify live EEG into Left / Right / No Action

- **process/**
  - Preprocessing (filtering, artifact reduction)
  - Feature extraction (SSVEP frequency-domain features for SVM)

- **stimulus/**
  - Displays flickering Left/Right arrows at distinct frequencies
  - Runs in its own thread for precise visual timing
  - Supports both Calibration and Run modes (controlled via terminal)

- **utils/**
  - Shared utilities: common classes, structs, and enums across modules

## Workflow

1. **Calibration**
   - Launch calibration mode
   - Present flickering arrows
   - Collect labeled EEG samples
   - Train the SVM classifier

2. **Run**
   - Start real-time decoding
   - Classify user intent (Left, Right, or No Action)
   - Output control signals for downstream actuation

