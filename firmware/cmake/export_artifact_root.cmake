# Resolve EXPORT_ARTIFACT_ROOT to firmware/generated for public builds.
# Requires selected_backend.cmake produced by scripts/generate_firmware_artifacts.py.
get_filename_component(FIRMWARE_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
get_filename_component(REPO_ROOT "${FIRMWARE_ROOT}/.." ABSOLUTE)

if(NOT DEFINED EXPORT_ARTIFACT_ROOT OR "${EXPORT_ARTIFACT_ROOT}" STREQUAL "")
    set(EXPORT_ARTIFACT_ROOT "${FIRMWARE_ROOT}/generated")
endif()

if(NOT IS_DIRECTORY "${EXPORT_ARTIFACT_ROOT}")
    message(FATAL_ERROR
        "EXPORT_ARTIFACT_ROOT is not a directory: ${EXPORT_ARTIFACT_ROOT}\n"
        "  Stage artifacts first:\n"
        "  python scripts/generate_firmware_artifacts.py --backend <name> --fold <n> --artifact-set reference")
endif()

set(_SSVEP_SELECTED "${EXPORT_ARTIFACT_ROOT}/selected_backend.cmake")
if(NOT EXISTS "${_SSVEP_SELECTED}")
    message(FATAL_ERROR
        "Missing staging metadata: ${_SSVEP_SELECTED}\n"
        "  Stage artifacts first:\n"
        "  python scripts/generate_firmware_artifacts.py --backend <name> --fold <n> --artifact-set reference\n"
        "  Do not fall back to a cached backend or fold.")
endif()

include("${_SSVEP_SELECTED}")
message(STATUS "ssvep staging: backend=${SSVEP_SELECTED_BACKEND} fold=${SSVEP_SELECTED_FOLD} subject=${SSVEP_SELECTED_SUBJECT} kind=${SSVEP_BACKEND_KIND}")
