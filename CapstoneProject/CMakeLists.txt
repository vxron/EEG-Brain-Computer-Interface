# Unicorn SDK paths (relative to this CapstoneProject/ folder)
set(UNICORN_ROOT     "${CMAKE_CURRENT_SOURCE_DIR}/unicorn")
set(UNICORN_INCLUDE  "${UNICORN_ROOT}/include")
set(UNICORN_LIB_DIR  "${UNICORN_ROOT}/lib/x64")
set(UNICORN_BIN_DIR  "${UNICORN_ROOT}/bin/x64")

# Create an imported target for the import lib
add_library(unicorn::unicorn STATIC IMPORTED)
set_target_properties(unicorn::unicorn PROPERTIES
  IMPORTED_LOCATION             "${UNICORN_LIB_DIR}/Unicorn.lib"
  INTERFACE_INCLUDE_DIRECTORIES "${UNICORN_INCLUDE}"
)

add_executable(CapstoneProject
  src/CapstoneProject.cpp
  src/acq/FakeAcquisition.cpp
  src/utils/Logger.cpp
)

# expose headers to IDEs (no compilation)
target_sources(CapstoneProject
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    FILES
      src/CapstoneProject.h
      src/utils/Types.h
      src/utils/RingBuffer.hpp
      src/utils/RingBuffer.tpp   # fine; it’s header-only
      src/utils/ModelStore.h
      src/acq/IAcqProvider.h
      src/acq/FakeAcquisition.h
)

# ==================== UI UNIT TESTS ==========================
# Simple HTTP server self-test (does not depend on Unicorn)
add_executable(HttpServerSelfTest
  unit_tests/HttpServerSelfTest.cpp
  src/utils/Logger.cpp
  src/stimulus/HttpServer.cpp
)
target_include_directories(HttpServerSelfTest PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# UI State machine self-test
add_executable(UIStateMachineSelfTest
  unit_tests/UIStateMachineSelfTest.cpp
  src/stimulus/HttpServer.cpp
  src/stimulus/StimulusController.cpp
  src/utils/Logger.cpp
)
target_include_directories(UIStateMachineSelfTest PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
set_property(TARGET UIStateMachineSelfTest PROPERTY CXX_STANDARD 20)
# ==========================================================

# Choose backend at build time (option defined in the ROOT CMakeLists.txt)
if(USE_FAKE_ACQ)
  message(STATUS "Building with FAKE acquisition backend")
  target_compile_definitions(CapstoneProject PRIVATE ACQ_BACKEND_FAKE)
else()
  add_executable(UnicornSelfTest
  unit_tests/UnicornSelfTest.cpp
  )
  target_sources(UnicornSelfTest PRIVATE
  src/utils/Logger.cpp
  )
  target_include_directories(UnicornSelfTest PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  message(STATUS "Building with REAL Unicorn acquisition backend")
  target_compile_definitions(CapstoneProject PRIVATE ACQ_BACKEND_UNICORN)
  target_sources(CapstoneProject PRIVATE
    src/acq/UnicornDriver.cpp
  )
  # Link to the Unicorn import lib
  target_link_libraries(CapstoneProject PRIVATE unicorn::unicorn)
  target_link_libraries(UnicornSelfTest PRIVATE unicorn::unicorn)

  # Copy Unicorn.dll next to the built .exe (so it runs without PATH edits)
  add_custom_command(TARGET CapstoneProject POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${UNICORN_BIN_DIR}/Unicorn.dll"
            "$<TARGET_FILE_DIR:CapstoneProject>/Unicorn.dll")
  
  add_custom_command(TARGET UnicornSelfTest POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${UNICORN_BIN_DIR}/Unicorn.dll"
          "$<TARGET_FILE_DIR:UnicornSelfTest>/Unicorn.dll")
endif()

if(CALIBRATION_MODE)
  message(STATUS "Building for Calibration mode")
  target_compile_definitions(CapstoneProject PRIVATE CALIB_MODE)
else()
  message(STATUS "Building for Run mode")
  target_compile_definitions(CapstoneProject PRIVATE RUN_MODE)
endif()

target_include_directories(CapstoneProject PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Enforce C++20
set_property(TARGET CapstoneProject PROPERTY CXX_STANDARD 20)

# Treat tpp as header-only
set_source_files_properties(src/utils/RingBuffer.tpp
  PROPERTIES LANGUAGE CXX HEADER_FILE_ONLY TRUE)
