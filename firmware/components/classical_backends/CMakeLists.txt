# Classical analytical backends. Include dirs come from firmware/generated staging.
get_filename_component(FIRMWARE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

if(CMAKE_BUILD_EARLY_EXPANSION)
    idf_component_register(REQUIRES common preprocess)
    return()
endif()

include("${FIRMWARE_ROOT}/cmake/export_artifact_root.cmake")
include("${FIRMWARE_ROOT}/cmake/replay_backend_select.cmake")

if(NOT DEFINED TARGET_FOLD_SUBJECT_TAG OR "${TARGET_FOLD_SUBJECT_TAG}" STREQUAL "")
    if(DEFINED SSVEP_SELECTED_TAG)
        set(TARGET_FOLD_SUBJECT_TAG "${SSVEP_SELECTED_TAG}")
    else()
        message(FATAL_ERROR "TARGET_FOLD_SUBJECT_TAG / SSVEP_SELECTED_TAG not set")
    endif()
endif()
set(_TAG "${TARGET_FOLD_SUBJECT_TAG}")

set(_SRCS "")
set(_INCS ".")
list(APPEND _INCS "${EXPORT_ARTIFACT_ROOT}/vectors")

set(_CLASSICAL_ENABLED "")
if(SMOKE_ENABLE_CCA)
    list(APPEND _SRCS "cca_forward.c" "cca_smoke.c")
    list(APPEND _INCS "${EXPORT_ARTIFACT_ROOT}/12jfpm_cca_${_TAG}")
    list(APPEND _CLASSICAL_ENABLED "CCA")
endif()
if(SMOKE_ENABLE_FBCCA)
    list(APPEND _SRCS "fbcca_forward.c" "fbcca_smoke.c")
    list(APPEND _INCS "${EXPORT_ARTIFACT_ROOT}/12jfpm_fbcca_${_TAG}")
    list(APPEND _CLASSICAL_ENABLED "FBCCA")
endif()
if(SMOKE_ENABLE_MSI)
    list(APPEND _SRCS "msi_forward.c" "msi_smoke.c")
    list(APPEND _INCS "${EXPORT_ARTIFACT_ROOT}/12jfpm_msi_${_TAG}")
    list(APPEND _CLASSICAL_ENABLED "MSI")
endif()
if(SMOKE_ENABLE_TRCA)
    list(APPEND _SRCS "trca_forward.c" "trca_smoke.c")
    list(APPEND _INCS "${EXPORT_ARTIFACT_ROOT}/12jfpm_trca_${_TAG}")
    list(APPEND _CLASSICAL_ENABLED "TRCA")
endif()

string(JOIN ", " _CLASSICAL_ENABLED_STR ${_CLASSICAL_ENABLED})
if(_CLASSICAL_ENABLED_STR)
    message(STATUS "classical_backends: tag=${_TAG} enabled=${_CLASSICAL_ENABLED_STR}")
else()
    message(STATUS "classical_backends: tag=${_TAG} (no classical backends enabled)")
    # Provide an empty translation unit so the component always has a source.
    list(APPEND _SRCS "classical_stub.c")
endif()

foreach(_inc IN LISTS _INCS)
    if(NOT "_${_inc}" STREQUAL "_." AND NOT IS_DIRECTORY "${_inc}")
        message(FATAL_ERROR
            "classical_backends missing include directory: ${_inc}\n"
            "  Re-stage the selected backend into firmware/generated/")
    endif()
endforeach()

idf_component_register(
    SRCS ${_SRCS}
    INCLUDE_DIRS ${_INCS}
    REQUIRES common preprocess
)
