# Create the executable
add_executable(CMakeHelloWorld
  src/CMakeHelloWorld.cpp
  src/CMakeHelloWorld.h
  src/utils/Types.h
  src/utils/RingBuffer.h
  src/acq/FakeAcquisition.h
  src/acq/FakeAcquisition.cpp
 "src/process/SlidingWindow.h" "src/process/SlidingWindow.cpp" "src/acq/RunAcq.cpp" "src/acq/RunAcq.h" "src/acq/IAcqProvider.h" "src/acq/UnicornAcq.h")

# Link spdlog to the exe
target_link_libraries(CMakeHelloWorld PRIVATE spdlog::spdlog)

target_include_directories(CMakeHelloWorld PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Help VS IntelliSense see spdlog headers
target_include_directories(CMakeHelloWorld PRIVATE
  "$<TARGET_PROPERTY:spdlog::spdlog,INTERFACE_INCLUDE_DIRECTORIES>"
)


# Enforce C++20 for this target (explicit)
set_property(TARGET CMakeHelloWorld PROPERTY CXX_STANDARD 20)


# TODO: Add tests and install targets if needed.

# Make VS treat it as C++ for IntelliSense, not a build input
set_source_files_properties(src/utils/RingBuffer.tpp
  PROPERTIES
    LANGUAGE CXX
    HEADER_FILE_ONLY TRUE
)
