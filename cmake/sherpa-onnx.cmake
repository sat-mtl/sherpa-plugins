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
# Neither is required to CONFIGURE or COMPILE: when the download is unavailable
# (the release for this sherpa/onnxruntime pair has not been published yet) the
# fetch degrades to a warning and the objects simply report unavailable at run
# time. It is an ERROR only when the libraries are actually about to be shipped,
# i.e. SHERPA_ONNX_REQUIRE_LIBS (default: on in a score deployment build).
#
# Build the library yourself, per platform, with the supported toolchain --
# pointing it at the SAME onnxruntime score deploys, or the result cannot bind to
# score's copy (see src/OnnxRuntimeLoader.hpp and the commit that added it):
#   export SHERPA_ONNXRUNTIME_INCLUDE_DIR=<score-build>/_deps/onnxruntime-src/include
#   export SHERPA_ONNXRUNTIME_LIB_DIR=<score-build>/_deps/onnxruntime-src/lib
#   cmake -S sherpa-onnx -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
#     -DSHERPA_ONNX_ENABLE_C_API=ON -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
#     -DSHERPA_ONNX_ENABLE_BINARY=OFF -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
#     -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF -DSHERPA_ONNX_ENABLE_TESTS=OFF
#   cmake --install build --prefix sherpa-onnx-install
# (PYTHON=OFF is required: BUILD_SHARED_LIBS=ON silently turns it on otherwise.)
# Then configure this addon with -DSHERPA_ONNX_DIR=<...>/sherpa-onnx-install.
# Check the result on Linux with
#   readelf -d lib/libsherpa-onnx-c-api.so | grep NEEDED    # libonnxruntime.so.1
#   nm -D --undefined-only lib/libsherpa-onnx-c-api.so | grep Ort  # @VERS_<ort>

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

# A deployment build is the one case where missing runtime libraries are fatal:
# the package would install a plugin whose every object reports unavailable, and
# nothing downstream would notice. Ordinary dev/CI builds keep configuring.
if(SCORE_DEPLOYMENT_BUILD)
  set(_sherpa_require_default ON)
else()
  set(_sherpa_require_default OFF)
endif()
set(SHERPA_ONNX_REQUIRE_LIBS ${_sherpa_require_default} CACHE BOOL
    "Fail configuration when the sherpa-onnx runtime libraries cannot be located")

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

# --- Agreement with score-addon-onnx --------------------------------------------
# score-addon-onnx is the addon that actually brings onnxruntime into score (its
# cmake/onnxruntime.cmake fetches it under the FetchContent name `onnxruntime` and
# installs it), and the one we bind to at run time. Its ONNXRUNTIME_VERSION is a
# directory-scope variable, so a sibling addon cannot read it -- but the package it
# fetched is right there in the build tree, and its VERSION_NUMBER is the version
# that will really be in the process. Cross-check against it: sherpa's imported
# OrtGetApiBase carries the symbol version of the release it was BUILT against
# (OrtGetApiBase@VERS_<x> on Linux), so a host on a different release can leave it
# unable to bind -- the objects then report unavailable, silently, at run time.
# Advisory only, and skipped when score-addon-onnx has not been configured yet
# (addons are configured in directory order, score-addon-onnx before this one).
set(_sherpa_host_ort_file "${CMAKE_BINARY_DIR}/_deps/onnxruntime-src/VERSION_NUMBER")
if(AVND_ADDON_SCORE AND EXISTS "${_sherpa_host_ort_file}")
  file(READ "${_sherpa_host_ort_file}" _sherpa_host_ort)
  string(STRIP "${_sherpa_host_ort}" _sherpa_host_ort)
  if(NOT _sherpa_host_ort STREQUAL _sherpa_ort_version)
    message(WARNING
        "sherpa-plugins: onnxruntime version mismatch. score deploys "
        "${_sherpa_host_ort} (score-addon-onnx), the prebuilt sherpa-onnx here is "
        "built against ${_sherpa_ort_version}. sherpa's library may not bind to "
        "score's copy, in which case every sherpa object reports unavailable. Set "
        "SHERPA_ONNXRUNTIME_VERSION (+ SHERPA_ONNXRUNTIME_API_VERSION) to match "
        "ONNXRUNTIME_VERSION in score-addon-onnx/cmake/onnxruntime.cmake and "
        "publish the matching prebuilt release.")
  endif()
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
  set(_sherpa_url "${_base}/sherpa-onnx-cabi-${_triplet}.${_ext}")

  # Downloaded by hand rather than through FetchContent: FetchContent turns any
  # download failure into a FATAL_ERROR, and this addon is configured as part of
  # score -- a release that has not been published yet (or an offline machine)
  # must not take the entire score configure down with it. The stamp carries the
  # tag so a version bump re-downloads and a reconfigure does not.
  set(_sherpa_dl "${CMAKE_BINARY_DIR}/_deps/sherpa_onnx-prebuilt")
  set(_sherpa_tree "${_sherpa_dl}/${SHERPA_ONNX_PREBUILT_TAG}")
  set(_sherpa_stamp "${_sherpa_tree}/.extracted")
  if(NOT EXISTS "${_sherpa_stamp}")
    set(_sherpa_archive "${_sherpa_dl}/sherpa-onnx-cabi-${_triplet}.${_ext}")
    message(STATUS "sherpa-plugins: downloading ${_sherpa_url}")
    file(DOWNLOAD "${_sherpa_url}" "${_sherpa_archive}"
         STATUS _sherpa_dl_status TLS_VERIFY ON)
    list(GET _sherpa_dl_status 0 _sherpa_dl_code)
    # file(ARCHIVE_EXTRACT) is fatal on a malformed archive, which would put us
    # back to taking score's configure down; treat a zero-byte body (what a failed
    # download leaves behind) as a failed download instead of feeding it in.
    set(_sherpa_dl_size 0)
    if(EXISTS "${_sherpa_archive}")
      file(SIZE "${_sherpa_archive}" _sherpa_dl_size)
    endif()
    if(_sherpa_dl_size EQUAL 0)
      set(_sherpa_dl_code 22)
      set(_sherpa_dl_status "22;\"empty archive\"")
    endif()
    if(_sherpa_dl_code EQUAL 0)
      file(REMOVE_RECURSE "${_sherpa_tree}")
      file(ARCHIVE_EXTRACT INPUT "${_sherpa_archive}"
           DESTINATION "${_sherpa_tree}")
      file(REMOVE "${_sherpa_archive}")
      file(TOUCH "${_sherpa_stamp}")
    else()
      list(GET _sherpa_dl_status 1 _sherpa_dl_message)
      # An empty/partial file is left behind on failure; do not keep it around.
      file(REMOVE "${_sherpa_archive}")
      set(_sherpa_dl_error
          "sherpa-plugins: could not download the prebuilt sherpa-onnx runtime "
          "libraries from ${_sherpa_url} (${_sherpa_dl_message}). The release "
          "'${SHERPA_ONNX_PREBUILT_TAG}' is produced by "
          ".github/workflows/sherpa-prebuilt.yml -- push that tag to publish it. "
          "Meanwhile, build sherpa-onnx ${SHERPA_ONNX_VERSION} against "
          "onnxruntime ${_sherpa_ort_version} and pass -DSHERPA_ONNX_DIR=<install> "
          "(see the header of cmake/sherpa-onnx.cmake).")
      string(JOIN "" _sherpa_dl_error ${_sherpa_dl_error})
      if(SHERPA_ONNX_REQUIRE_LIBS)
        message(FATAL_ERROR "${_sherpa_dl_error}")
      endif()
      message(WARNING
          "${_sherpa_dl_error} Configuring anyway: the plugin builds, but every "
          "sherpa object reports unavailable at run time.")
      set(_sherpa_reported 1)
    endif()
  endif()
  if(EXISTS "${_sherpa_stamp}")
    set(_sherpa_root "${_sherpa_tree}")
    # The published archives are rooted at include/ + lib/ directly, but tolerate
    # one wrapping directory (the shape `tar czf` gives when handed a directory
    # instead of `.`), so a repackaged asset does not silently resolve to nothing.
    if(NOT IS_DIRECTORY "${_sherpa_root}/lib")
      file(GLOB _sherpa_sub LIST_DIRECTORIES true "${_sherpa_tree}/*")
      foreach(_d IN LISTS _sherpa_sub)
        if(IS_DIRECTORY "${_d}/lib")
          set(_sherpa_root "${_d}")
          break()
        endif()
      endforeach()
    endif()
  endif()
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
endif()

# A tree without libsherpa-onnx-c-api in it is as good as no tree at all (a
# mistyped SHERPA_ONNX_DIR, an install built with SHERPA_ONNX_ENABLE_C_API=OFF,
# or a static build): report it the same way rather than silently shipping
# nothing.
if(NOT SHERPA_SUPPORT_FILES)
  set(_sherpa_none
      "sherpa-plugins: no runtime sherpa-onnx libraries located. Objects "
      "compile but report unavailable at run time until libsherpa-onnx-c-api is "
      "found next to the module. Pass -DSHERPA_ONNX_DIR=<install with lib/> (a "
      "BUILD_SHARED_LIBS=ON + SHERPA_ONNX_ENABLE_C_API=ON sherpa build) or "
      "-DSHERPA_ONNX_FETCH_LIBS=ON to run/package.")
  string(JOIN "" _sherpa_none ${_sherpa_none})
  if(SHERPA_ONNX_REQUIRE_LIBS)
    message(FATAL_ERROR "${_sherpa_none}")
  elseif(NOT _sherpa_reported)
    message(STATUS "${_sherpa_none}")
  endif()
endif()
