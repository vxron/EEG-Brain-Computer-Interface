# Create the executable
add_executable(CMakeHelloWorld
  src/CMakeHelloWorld.cpp
  src/CMakeHelloWorld.h

  # Utils / types
  src/utils/Types.h
  src/utils/RingBuffer.h

  # Acquisition (fake always present)
  src/acq/IAcqProvider.h
  src/acq/FakeAcquisition.h
  src/acq/FakeAcquisition.cpp

  # RunAcq / processing
  src/acq/RunAcq.h
  
  src/process/SlidingWindow.h
  src/process/SlidingWindow.cpp
   "src/process/FtrExtract.h" "src/process/FtrExtract.cpp")

# Choose backend at build time (option defined in the ROOT CMakeLists.txt)
if(USE_FAKE_ACQ)
  message(STATUS "Building with FAKE acquisition backend")
  target_compile_definitions(CMakeHelloWorld PRIVATE ACQ_BACKEND_FAKE)
else()
  message(STATUS "Building with REAL Unicorn acquisition backend")
  # Only add Unicorn provider sources in REAL builds:
  target_sources(CMakeHelloWorld PRIVATE
    src/acq/UnicornAcq.h
  )
  # And link the SDK here when ready, e.g.:
  # target_include_directories(CMakeHelloWorld PRIVATE ${UNICORN_SDK_INCLUDE_DIR})
  # target_link_libraries(CMakeHelloWorld PRIVATE ${UNICORN_SDK_LIBS})
endif()

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

# Make VS treat it as C++ for IntelliSense, not a build input
set_source_files_properties(src/utils/RingBuffer.tpp
  PROPERTIES
    LANGUAGE CXX
    HEADER_FILE_ONLY TRUE
)
