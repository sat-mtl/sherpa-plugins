# Resolve sherpa-onnx for sherpa-plugins.
#
# We dlopen the library at runtime (see src/SherpaLoader.hpp), so COMPILATION
# depends only on the C header, which is VENDORED under
# 3rdparty/sherpa-onnx/include (pinned to a known sherpa version). The prebuilt
# .so/.dll/.dylib is purely a runtime/packaging concern. This module exposes:
#   * sherpa::headers       - INTERFACE target carrying the vendored include dir
#                             (no link! we dlopen)
#   * SHERPA_SUPPORT_FILES   - runtime libraries to bundle beside each standalone
#                             module (avnd_addon_package SUPPORT). Empty unless the
#                             prebuilt libraries are located; not needed to compile.
#
# To locate the runtime libraries (only needed to actually run/package):
#   1. -DSHERPA_ONNX_DIR=/path/to/install   (a tree with lib/), e.g. the result of
#      `cmake --install` on a `-DBUILD_SHARED_LIBS=ON` sherpa build, or
#   2. -DSHERPA_ONNX_FETCH_LIBS=ON to download a prebuilt release archive.
#
# Build the prebuilt library once per platform with the supported toolchain:
#   cmake -S sherpa-onnx -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
#     -DSHERPA_ONNX_ENABLE_C_API=ON -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
#     -DSHERPA_ONNX_ENABLE_BINARY=OFF -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
#     -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF -DSHERPA_ONNX_ENABLE_TESTS=OFF
#   cmake --install build --prefix sherpa-onnx-install
# (PYTHON=OFF is required: BUILD_SHARED_LIBS=ON silently turns it on otherwise.)

include(FetchContent)

set(SHERPA_ONNX_DIR "" CACHE PATH
    "Prebuilt sherpa-onnx install (with lib/). For running/packaging, not compiling.")
set(SHERPA_ONNX_VERSION "1.13.3" CACHE STRING "sherpa-onnx version (matches the vendored header)")
set(SHERPA_ONNX_FETCH_LIBS OFF CACHE BOOL
    "Download a prebuilt sherpa-onnx release to obtain the runtime libraries")

# --- Headers: always the vendored copy. Compilation needs nothing else. ---
add_library(sherpa_headers INTERFACE)
target_include_directories(sherpa_headers INTERFACE
    "${CMAKE_CURRENT_LIST_DIR}/../3rdparty/sherpa-onnx/include")
add_library(sherpa::headers ALIAS sherpa_headers)

# --- Runtime libraries (optional): only needed to run/package the modules. ---
set(_sherpa_root "")
if(SHERPA_ONNX_DIR)
  set(_sherpa_root "${SHERPA_ONNX_DIR}")
elseif(SHERPA_ONNX_FETCH_LIBS)
  # Prebuilt C-API libraries produced by .github/workflows/sherpa-prebuilt.yml and
  # published to this repo's releases under the tag prebuilt-v<version>. Each
  # archive's root is include/ + lib/ (+ bin/ on Windows).
  set(_base
      "https://github.com/sat-mtl/sherpa-plugins/releases/download/prebuilt-v${SHERPA_ONNX_VERSION}")
  if(WIN32)
    set(_triplet "windows-x64")
    set(_ext "zip")
  elseif(APPLE)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
      set(_triplet "macos-arm64")
    else()
      set(_triplet "macos-x64")
    endif()
    set(_ext "tar.gz")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64.*|arm64")
    set(_triplet "linux-arm64")
    set(_ext "tar.gz")
  else()
    set(_triplet "linux-x64")
    set(_ext "tar.gz")
  endif()
  FetchContent_Declare(sherpa_onnx
      URL "${_base}/sherpa-onnx-cabi-${_triplet}.${_ext}")
  FetchContent_MakeAvailable(sherpa_onnx)
  set(_sherpa_root "${sherpa_onnx_SOURCE_DIR}")
endif()

set(SHERPA_SUPPORT_FILES "")
if(_sherpa_root)
  if(APPLE)
    file(GLOB SHERPA_SUPPORT_FILES "${_sherpa_root}/lib/*.dylib")
  elseif(WIN32)
    file(GLOB SHERPA_SUPPORT_FILES
         "${_sherpa_root}/lib/*.dll" "${_sherpa_root}/bin/*.dll")
  else()
    file(GLOB SHERPA_SUPPORT_FILES "${_sherpa_root}/lib/*.so*")
  endif()

  # The dump/standalone introspection executables run at build time and dlopen the
  # library; let them find it in the prebuilt tree. Standalone only -- in a score
  # build the host manages rpath and bundles the libraries itself.
  # include()'d into the top-level scope, so these already land where the
  # CMakeLists that included us can see them -- no PARENT_SCOPE (which would refer
  # above the top level and warn "current scope has no parent").
  if(NOT AVND_ADDON_SCORE)
    list(APPEND CMAKE_BUILD_RPATH   "${_sherpa_root}/lib")
    list(APPEND CMAKE_INSTALL_RPATH "${_sherpa_root}/lib")
  endif()
else()
  message(STATUS
      "sherpa-plugins: no runtime sherpa-onnx libraries located. Objects will "
      "compile but report unavailable at runtime until libsherpa-onnx-c-api is "
      "found next to the module. Set -DSHERPA_ONNX_DIR=... to run/package.")
endif()
