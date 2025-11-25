cmake_minimum_required(VERSION 3.22)

project(CapstoneProject LANGUAGES CXX)

# Require C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Show we are in the top-level file and whether a toolchain is set
message(STATUS ">>> Top-level CMakeLists is running")
message(STATUS ">>> CMAKE_TOOLCHAIN_FILE='${CMAKE_TOOLCHAIN_FILE}'")

option(USE_FAKE_ACQ "Use fake acquisition backend" OFF)
option(CALIBRATION_MODE "Calibrate Mode to Train Model" ON)
option(USE_EEG_FILTERS "Use EEG filters for preprocessing" ON)

# Build the subdir that defines the executable
add_subdirectory(CapstoneProject)