# ESP-IDF project for SSVEP decoder TCP replay on ESP32-S3.
#
# Required workflow:
#   python scripts/generate_firmware_artifacts.py --backend <name> --fold <n> --artifact-set reference
#   cd firmware/replay_fw
#   idf.py set-target esp32s3
#   idf.py build
cmake_minimum_required(VERSION 3.16)

get_filename_component(FIRMWARE_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(SHARED_COMPONENTS_DIR "${FIRMWARE_ROOT}/components")
if(NOT EXISTS "${SHARED_COMPONENTS_DIR}")
    message(FATAL_ERROR
        "Firmware component directory not found: ${SHARED_COMPONENTS_DIR}")
endif()

set(_DEFAULT_EXPORT_ROOT "${FIRMWARE_ROOT}/generated")
set(EXPORT_ARTIFACT_ROOT "${_DEFAULT_EXPORT_ROOT}" CACHE PATH
    "Staged firmware artifact root (firmware/generated)")

include("${FIRMWARE_ROOT}/cmake/export_artifact_root.cmake")

if(NOT DEFINED SSVEP_IDF_BACKEND OR "${SSVEP_IDF_BACKEND}" STREQUAL "")
    message(FATAL_ERROR "selected_backend.cmake did not define SSVEP_IDF_BACKEND")
endif()

# Staging metadata selects the backend. Optional -DTARGET_BACKEND may override
# only when it matches the staged backend.
if(DEFINED TARGET_BACKEND AND NOT "${TARGET_BACKEND}" STREQUAL "")
    string(TOLOWER "${TARGET_BACKEND}" _CLI_BACKEND)
    string(TOLOWER "${SSVEP_IDF_BACKEND}" _STAGED_BACKEND)
    if(NOT "_${_CLI_BACKEND}" STREQUAL "_${_STAGED_BACKEND}")
        message(FATAL_ERROR
            "TARGET_BACKEND=${TARGET_BACKEND} does not match staged backend "
            "${SSVEP_IDF_BACKEND}. Re-stage or omit -DTARGET_BACKEND.")
    endif()
endif()
set(TARGET_BACKEND "${SSVEP_IDF_BACKEND}" CACHE STRING "Selected replay backend" FORCE)
if(DEFINED SSVEP_SELECTED_TAG AND NOT "${SSVEP_SELECTED_TAG}" STREQUAL "")
    set(TARGET_FOLD_SUBJECT_TAG "${SSVEP_SELECTED_TAG}" CACHE STRING
        "LOSO embed tag from staging" FORCE)
endif()

message(STATUS "replay_fw: TARGET_FOLD_SUBJECT_TAG=${TARGET_FOLD_SUBJECT_TAG}")
message(STATUS "replay_fw: EXPORT_ARTIFACT_ROOT=${EXPORT_ARTIFACT_ROOT}")
message(STATUS "replay_fw: TARGET_BACKEND=${TARGET_BACKEND}")

# Defaults off — selected_backend.cmake + replay_backend_select enable one backend.
option(SMOKE_ENABLE_CCA    "Enable CCA backend"    OFF)
option(SMOKE_ENABLE_FBCCA  "Enable FBCCA backend"  OFF)
option(SMOKE_ENABLE_MSI    "Enable MSI backend"    OFF)
option(SMOKE_ENABLE_TRCA   "Enable TRCA backend"   OFF)
option(SMOKE_ENABLE_EEGNET_F32  "Enable EEGNet float32"      OFF)
option(SMOKE_ENABLE_EEGNET_INT8 "Enable EEGNet int8"         OFF)
option(SMOKE_ENABLE_CCNN_F32    "Enable CCNN float32"        OFF)
option(SMOKE_ENABLE_CCNN_INT8   "Enable CCNN int8"           OFF)
option(SMOKE_ENABLE_CONVCA_F32  "Enable ConvCA float32"      OFF)
option(SMOKE_ENABLE_FBTCNN_INT8 "Enable FBtCNN int8"         OFF)
option(SMOKE_ENABLE_FBTCNN_F32  "Enable FBtCNN float32"      OFF)

include(${CMAKE_CURRENT_LIST_DIR}/../cmake/replay_smoke_enable_dl.cmake)
set(SMOKE_ENABLE_DL "${SMOKE_ENABLE_DL}" CACHE BOOL "Enable DL TFLite backends" FORCE)

set(BACKEND_DEFS "")
macro(add_backend_def FLAG)
    if(${FLAG})
        list(APPEND BACKEND_DEFS "-D${FLAG}=1")
    endif()
endmacro()

add_backend_def(SMOKE_ENABLE_CCA)
add_backend_def(SMOKE_ENABLE_FBCCA)
add_backend_def(SMOKE_ENABLE_MSI)
add_backend_def(SMOKE_ENABLE_TRCA)
add_backend_def(SMOKE_ENABLE_EEGNET_F32)
add_backend_def(SMOKE_ENABLE_EEGNET_INT8)
add_backend_def(SMOKE_ENABLE_CCNN_F32)
add_backend_def(SMOKE_ENABLE_CCNN_INT8)
add_backend_def(SMOKE_ENABLE_CONVCA_F32)
add_backend_def(SMOKE_ENABLE_CONVCA_INT8)
add_backend_def(SMOKE_ENABLE_FBTCNN_INT8)
add_backend_def(SMOKE_ENABLE_FBTCNN_F32)
add_backend_def(SMOKE_ENABLE_DL)

list(APPEND EXTRA_COMPONENT_DIRS "${SHARED_COMPONENTS_DIR}")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(ssvep_replay_fw)

if(BACKEND_DEFS)
    idf_build_set_property(COMPILE_OPTIONS "${BACKEND_DEFS}" APPEND)
endif()
