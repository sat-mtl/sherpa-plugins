# Resolve sherpa-onnx for sherpa-plugins.
#
# We dlopen the library at runtime (see src/SherpaLoader.hpp), so COMPILATION
# depends only on the C header, which is VENDORED under
# 3rdparty/sherpa-onnx/include (pinned to a known sherpa version). The prebuilt
# .so/.dll/.dylib is purely a runtime/packaging concern. This module exposes:
#   * sherpa::headers       - INTERFACE target carrying the vendored include dir
#                             (no link! we dlopen)
#   * SHERPA_SUPPORT_FILES   - the sherpa runtime libraries, shipped everywhere
#   * SHERPA_ORT_SUPPORT_FILES - the onnxruntime beside them, shipped into
#                             standalone packages only; a score build binds to the
#                             host's copy (see src/OnnxRuntimeLoader.hpp)
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
set(SHERPA_ONNX_VERSION "1.13.4" CACHE STRING "sherpa-onnx version (matches the vendored header)")
if(AVND_ADDON_SCORE)
  set(_sherpa_fetch_default ON)
else()
  set(_sherpa_fetch_default OFF)
endif()
set(SHERPA_ONNX_FETCH_LIBS ${_sherpa_fetch_default} CACHE BOOL
    "Download a prebuilt sherpa-onnx release to obtain the runtime libraries")

# The onnxruntime the prebuilt libsherpa-onnx-c-api is linked against. Must be the
# release ossia score deploys, per platform, or its library cannot bind to it.
set(SHERPA_ONNXRUNTIME_VERSION "1.27.1" CACHE STRING
    "onnxruntime version the prebuilt sherpa-onnx links (must match ossia score)")
set(SHERPA_ONNXRUNTIME_API_VERSION "27" CACHE STRING
    "ORT_API_VERSION of SHERPA_ONNXRUNTIME_VERSION")

# macOS x64 is stuck on the last onnxruntime release supporting it, so it links a
# different version from every other platform.
set(_sherpa_macos_x64 0)
if(APPLE)
  if(CMAKE_OSX_ARCHITECTURES MATCHES ".*x86.*")
    set(_sherpa_macos_x64 1)
  elseif(NOT CMAKE_OSX_ARCHITECTURES
         AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(_sherpa_macos_x64 1)
  endif()
endif()

set(_sherpa_ort_version "${SHERPA_ONNXRUNTIME_VERSION}")
set(_sherpa_ort_api "${SHERPA_ONNXRUNTIME_API_VERSION}")
if(_sherpa_macos_x64)
  set(_sherpa_ort_version "1.23.2")
  set(_sherpa_ort_api "23")
endif()

# Named after the release-wide onnxruntime; the macos-x64 asset inside is built
# against the version above.
set(SHERPA_ONNX_PREBUILT_TAG
    "prebuilt-v${SHERPA_ONNX_VERSION}-ort${SHERPA_ONNXRUNTIME_VERSION}"
    CACHE STRING "Release tag holding the prebuilt sherpa-onnx C API archives")

# --- Headers: always the vendored copy. Compilation needs nothing else. ---
add_library(sherpa_headers INTERFACE)
target_include_directories(sherpa_headers INTERFACE
    "${CMAKE_CURRENT_LIST_DIR}/../3rdparty/sherpa-onnx/include")
target_compile_definitions(sherpa_headers INTERFACE
    SHERPA_PLUGINS_ORT_API_VERSION=${_sherpa_ort_api}
    SHERPA_PLUGINS_ORT_VERSION="${_sherpa_ort_version}")
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
      "https://github.com/sat-mtl/sherpa-plugins/releases/download/${SHERPA_ONNX_PREBUILT_TAG}")
  if(WIN32)
    set(_triplet "windows-x64")
    set(_ext "zip")
  elseif(APPLE)
    if(_sherpa_macos_x64)
      set(_triplet "macos-x64")
    else()
      set(_triplet "macos-arm64")
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
set(SHERPA_ORT_SUPPORT_FILES "")
if(_sherpa_root)
  if(APPLE)
    file(GLOB _sherpa_all "${_sherpa_root}/lib/*.dylib")
  elseif(WIN32)
    file(GLOB _sherpa_all
         "${_sherpa_root}/lib/*.dll" "${_sherpa_root}/bin/*.dll")
  else()
    file(GLOB _sherpa_all "${_sherpa_root}/lib/*.so*")
  endif()

  foreach(_f IN LISTS _sherpa_all)
    get_filename_component(_n "${_f}" NAME)
    if(_n MATCHES "onnxruntime")
      list(APPEND SHERPA_ORT_SUPPORT_FILES "${_f}")
    else()
      list(APPEND SHERPA_SUPPORT_FILES "${_f}")
    endif()
  endforeach()

  # Keep only the SONAME out of the libonnxruntime.so -> .so.1 -> .so.1.x.y chain,
  # plus its target when it is a link, so a standalone package carries one copy
  # under the name libsherpa-onnx-c-api asks for.
  if(NOT APPLE AND NOT WIN32 AND SHERPA_ORT_SUPPORT_FILES)
    set(_ort_keep "")
    foreach(_f IN LISTS SHERPA_ORT_SUPPORT_FILES)
      get_filename_component(_n "${_f}" NAME)
      if(_n STREQUAL "libonnxruntime.so.1")
        list(APPEND _ort_keep "${_f}")
        if(IS_SYMLINK "${_f}")
          file(REAL_PATH "${_f}" _ort_real)
          list(APPEND _ort_keep "${_ort_real}")
        endif()
      endif()
    endforeach()
    if(_ort_keep)
      set(SHERPA_ORT_SUPPORT_FILES ${_ort_keep})
    endif()
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
