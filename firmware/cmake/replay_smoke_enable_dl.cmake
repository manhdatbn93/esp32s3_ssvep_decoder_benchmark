# Apply -DTARGET_BACKEND=... and derive SMOKE_ENABLE_DL for component CMake passes.
# ESP-IDF configures components before the top-level replay_fw CMakeLists finishes;
# include this from dl_backends/CMakeLists.txt and replay_fw/CMakeLists.txt.
get_filename_component(_REPLAY_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)
include("${_REPLAY_CMAKE_DIR}/replay_backend_select.cmake")

if(SMOKE_ENABLE_EEGNET_F32 OR SMOKE_ENABLE_EEGNET_INT8 OR
   SMOKE_ENABLE_CCNN_F32 OR SMOKE_ENABLE_CCNN_INT8 OR
   SMOKE_ENABLE_CONVCA_F32 OR SMOKE_ENABLE_CONVCA_INT8 OR
   SMOKE_ENABLE_FBTCNN_INT8 OR SMOKE_ENABLE_FBTCNN_F32)
    set(SMOKE_ENABLE_DL ON)
else()
    set(SMOKE_ENABLE_DL OFF)
endif()
