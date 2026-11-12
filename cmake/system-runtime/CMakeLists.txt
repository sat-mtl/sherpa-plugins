# Kept in a separate CMake process: upstream defines an onnxruntime target and
# changes global build options which must not leak into score or other addons.
cmake_minimum_required(VERSION 3.28)
project(sherpa_plugins_system_runtime LANGUAGES C CXX)

if(NOT EXISTS "${SHERPA_SYSTEM_ORT_INCLUDE_DIR}/onnxruntime_cxx_api.h"
   OR NOT EXISTS "${SHERPA_SYSTEM_ORT_LIBRARY}")
  message(FATAL_ERROR "The system ONNX Runtime headers and shared library are required")
endif()

# These are the exact cache entries consumed by upstream cmake/onnxruntime.cmake.
# Supplying both prevents its prebuilt download fallback. Ignore inherited env
# overrides so the runtime is built against the same installation score found.
unset(ENV{SHERPA_ONNXRUNTIME_INCLUDE_DIR})
unset(ENV{SHERPA_ONNXRUNTIME_LIB_DIR})
set(location_onnxruntime_header_dir "${SHERPA_SYSTEM_ORT_INCLUDE_DIR}" CACHE PATH "" FORCE)
set(location_onnxruntime_lib "${SHERPA_SYSTEM_ORT_LIBRARY}" CACHE FILEPATH "" FORCE)
set(SHERPA_ONNX_USE_PRE_INSTALLED_ONNXRUNTIME_IF_AVAILABLE ON CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
# Explicit host CRT settings still win; otherwise match score's dynamic default.
set(SHERPA_ONNX_USE_STATIC_CRT OFF CACHE BOOL "" FORCE)
set(SHERPA_ONNX_ENABLE_C_API ON CACHE BOOL "" FORCE)
set(SHERPA_ONNX_ENABLE_TTS ON CACHE BOOL "" FORCE)
set(SHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION ON CACHE BOOL "" FORCE)
foreach(option PYTHON TESTS BINARY PORTAUDIO WEBSOCKET JNI GPU DIRECTML)
  set(SHERPA_ONNX_ENABLE_${option} OFF CACHE BOOL "" FORCE)
endforeach()
set(SHERPA_ONNX_BUILD_C_API_EXAMPLES OFF CACHE BOOL "" FORCE)

include(FetchContent)
FetchContent_Declare(sherpa_onnx_system
  URL "https://github.com/k2-fsa/sherpa-onnx/archive/a63f82130d91d99a23a43f346104706881b31ff1.tar.gz")
FetchContent_MakeAvailable(sherpa_onnx_system)
add_subdirectory("${SHERPA_PLUGINS_SOURCE_DIR}/phonemizer" phonemizer EXCLUDE_FROM_ALL)

# Upstream dependencies are static; only these two C ABI libraries are dlopened.
# A dedicated component excludes upstream's headers, static archives and any ORT
# install rules. Never copy or package the distro's ONNX Runtime itself.
install(TARGETS sherpa-onnx-c-api sherpa-plugins-phonemizer
  RUNTIME DESTINATION bin COMPONENT SherpaPluginsRuntime
  LIBRARY DESTINATION lib COMPONENT SherpaPluginsRuntime)
