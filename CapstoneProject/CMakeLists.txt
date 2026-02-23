# ================== Unicorn SDK paths (REAL backend only) ==================
set(UNICORN_ROOT     "${CMAKE_CURRENT_SOURCE_DIR}/unicorn")
set(UNICORN_INCLUDE  "${UNICORN_ROOT}/include")
set(UNICORN_LIB_DIR  "${UNICORN_ROOT}/lib/x64")
set(UNICORN_BIN_DIR  "${UNICORN_ROOT}/bin/x64")

# Imported Unicorn library (safe even if not used when USE_FAKE_ACQ=ON)
add_library(unicorn::unicorn STATIC IMPORTED)
set_target_properties(unicorn::unicorn PROPERTIES
  IMPORTED_LOCATION             "${UNICORN_LIB_DIR}/Unicorn.lib"
  INTERFACE_INCLUDE_DIRECTORIES "${UNICORN_INCLUDE}"
)

# ========================== MAIN EXEC =========================
add_executable(CapstoneProject
  src/CapstoneProject.cpp
  src/utils/Logger.cpp
  src/stimulus/HttpServer.cpp
  src/stimulus/StimulusController.cpp
  src/utils/SignalQualityAnalyzer.cpp
  src/utils/SessionPaths.cpp
  src/classifier/ONNXClassifier.cpp
  src/actuation/ServoDriver.cpp
  src/actuation/FakeActuation.cpp
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
      src/acq/IAcqProvider.h
      src/acq/FakeAcquisition.h
      src/acq/UnicornDriver.h
      src/utils/Filters.hpp
      src/classifier/ONNXClassifier.hpp
      src/actuation/ServoDriver.h
      src/actuation/FakeActuation.hpp
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

# ==================== ACQ BACKEND SELECTION ===================
# Choose backend at build time (option defined in the ROOT CMakeLists.txt)
if(USE_FAKE_ACQ)
  message(STATUS "Building with FAKE acquisition backend")
  target_compile_definitions(CapstoneProject PRIVATE ACQ_BACKEND_FAKE)
  target_sources(CapstoneProject PRIVATE
    src/acq/FakeAcquisition.cpp
  )

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

# ================== ONNX Runtime paths ==================
set(ORT_ROOT     "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime")
set(ORT_INCLUDE  "${ORT_ROOT}/include")
set(ORT_LIB_DIR  "${ORT_ROOT}/lib")

# Imported ONNX Runtime library
add_library(onnxruntime::onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime::onnxruntime PROPERTIES
  IMPORTED_IMPLIB             "${ORT_LIB_DIR}/onnxruntime.lib"
  IMPORTED_LOCATION           "${ORT_LIB_DIR}/onnxruntime.dll"
  INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE}"
)

# Link it
target_link_libraries(CapstoneProject PRIVATE onnxruntime::onnxruntime)

# Copy onnxruntime.dll next to the built .exe
add_custom_command(TARGET CapstoneProject POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${ORT_LIB_DIR}/onnxruntime.dll"
          "$<TARGET_FILE_DIR:CapstoneProject>/onnxruntime.dll")

# ======================= EEG FILTERS ===========================
if(USE_EEG_FILTERS)
  message(STATUS "Building with EEG filters")
  target_compile_definitions(CapstoneProject PRIVATE USE_EEG_FILTERS)
  target_sources(CapstoneProject PRIVATE
    src/utils/Filters.cpp
  )
endif()

# ======================= SKIP TRAINING (pretrained path) ===========================
if(NOT "${SKIP_TRAINING}" STREQUAL "")
  message(STATUS "Building with SKIP_TRAINING='${SKIP_TRAINING}'")

  # flag for code paths
  target_compile_definitions(CapstoneProject PRIVATE SKIP_TRAINING=1)

  # expose the path as a *string literal* macro in all of C++ 
  target_compile_definitions(CapstoneProject PRIVATE
    SKIP_TRAINING_MODEL_PATH="${SKIP_TRAINING}"
  )
else()
  target_compile_definitions(CapstoneProject PRIVATE SKIP_TRAINING=0)
endif()

