# Early requirements pass runs this file without project-level staging variables.
if(CMAKE_BUILD_EARLY_EXPANSION)
    idf_component_register(
        SRCS
            "replay_main.c"
            "replay_boot_smoke.c"
            "replay_tcp.c"
            "wifi_connect.c"
        INCLUDE_DIRS "."
        REQUIRES
            classical_backends
            dl_backends
            preprocess
            common
            ssvep_dispatch
            resource_profile
            esp_timer
            nvs_flash
            esp_wifi
            esp_netif
            lwip
            heap
    )
    return()
endif()

if(NOT DEFINED TARGET_FOLD_SUBJECT_TAG OR "${TARGET_FOLD_SUBJECT_TAG}" STREQUAL "")
    if(DEFINED SSVEP_SELECTED_TAG AND NOT "${SSVEP_SELECTED_TAG}" STREQUAL "")
        set(TARGET_FOLD_SUBJECT_TAG "${SSVEP_SELECTED_TAG}")
    else()
        message(FATAL_ERROR
            "TARGET_FOLD_SUBJECT_TAG not set. Stage artifacts first:\n"
            "  python scripts/generate_firmware_artifacts.py --backend <name> --fold <n> --artifact-set reference")
    endif()
endif()

set(EMBED_TAG "${TARGET_FOLD_SUBJECT_TAG}")
configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/replay_embed_tag.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/replay_embed_tag.h"
    @ONLY
)

if(NOT DEFINED EXPORT_ARTIFACT_ROOT OR "${EXPORT_ARTIFACT_ROOT}" STREQUAL "")
    message(FATAL_ERROR "EXPORT_ARTIFACT_ROOT not set")
endif()
set(_IDENTITY_C "${EXPORT_ARTIFACT_ROOT}/backend_identity.c")
if(NOT EXISTS "${_IDENTITY_C}")
    message(FATAL_ERROR "Missing staged backend identity source: ${_IDENTITY_C}")
endif()

idf_component_register(
    SRCS
        "replay_main.c"
        "replay_boot_smoke.c"
        "replay_tcp.c"
        "wifi_connect.c"
        "${_IDENTITY_C}"
    INCLUDE_DIRS
        "."
        "${CMAKE_CURRENT_BINARY_DIR}"
        "${EXPORT_ARTIFACT_ROOT}"
    REQUIRES
        classical_backends
        dl_backends
        preprocess
        common
        ssvep_dispatch
        resource_profile
        esp_timer
        nvs_flash
        esp_wifi
        esp_netif
        lwip
        heap
)
