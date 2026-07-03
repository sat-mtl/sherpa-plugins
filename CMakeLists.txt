cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

# Meaningless on WebAssembly: the objects dlopen a native sherpa-onnx .so at run
# time, which wasm cannot do. Bail out early -- score has a wasm lane that globs
# src/addons/*, and standalone has no wasm back-end. Mirrors score-addon-onnx.
if(EMSCRIPTEN)
  return()
endif()

# C is enabled because the vendored dlopen helper and the sherpa C header are C.
project(sherpa_plugins LANGUAGES C CXX)

# Max/MSP externals link the static CRT (/MT) on Windows; set it for the whole
# addon -- before Avendish and the object libraries are created -- so every
# target shares one runtime (mixing /MT and /MD trips MSVC with LNK2038).
if(MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()
if(WIN32)
  add_compile_definitions(NOMINMAX _USE_MATH_DEFINES)
endif()

# The vendored ossia::dylib_loader throws std::runtime_error when the library is
# missing, and we catch it to set `available = false`; enable exceptions (like
# ossia/score-addon-onnx does) so that try/catch is real and not a terminate().
set(AVND_USES_EXCEPTIONS ON CACHE BOOL "" FORCE)

# Score's external-addon bootstrap (mirrors score-addon-cv): when the addon is
# configured as the TOP-LEVEL project against a score source checkout
# (-DCMAKE_MODULE_PATH=<score>/cmake -DSCORE_SOURCE_DIR=...) or a score SDK
# (-DSCORE_SDK=...) -- the ossia/actions dev and sdk CI tracks -- this module
# sets up the score imported targets and puts score's bundled Avendish on
# CMAKE_PREFIX_PATH (via ScoreAvndHelper), so the find_package(Avendish) below
# resolves to the host's copy and the REAL score plugin is built. In a score
# build tree score_lib_base already exists (skip); standalone the module is
# simply not found (OPTIONAL -> no-op).
if(NOT TARGET score_lib_base)
  include(ScoreExternalAddon OPTIONAL)
endif()

# --- Avendish resolution (the dual-build switch) ---
# In a score build the host ALREADY provides Avendish: the headers come from
# score-plugin-avnd (include_directories on 3rdparty/avendish/include), and the
# avnd_addon_* CMake API from score's bundled 3rdparty/avendish, which
# ScoreAvndHelper puts on CMAKE_PREFIX_PATH. We STILL call find_package(Avendish)
# here -- AvendishConfig no-ops the build when the host provides the commands, and
# re-running it is what sets AVND_ADDON_SCORE (and re-defines the addon macros) in
# THIS directory's scope. avnd_addon_object is a macro that reads AVND_ADDON_SCORE
# from the caller scope, so a sibling addon having loaded Avendish is not enough --
# each addon must find_package it itself or it wrongly takes the standalone path
# (calling avnd_make_dump, which does not exist in a score build).
#
# What we must NEVER do in a score build is FetchContent our own Avendish: a
# fetched copy lands on CMAKE_PREFIX_PATH and shadows the host's, and a
# stale/mismatched shadow is exactly what breaks avnd_addon_init in a score tree.
# So the fetch fallback is gated to standalone builds only.
find_package(Avendish QUIET)
if(NOT Avendish_FOUND
   AND NOT (TARGET score_lib_base OR COMMAND score_common_setup))
  include(FetchContent)
  FetchContent_Declare(
    Avendish
    GIT_REPOSITORY "https://github.com/celtera/avendish"
    GIT_TAG 6d08f1ef040da7cb62973859b62b14159c864ffe)
  FetchContent_Populate(Avendish)
  list(APPEND CMAKE_PREFIX_PATH "${avendish_SOURCE_DIR}")
  find_package(Avendish REQUIRED)
endif()

# Resolves the prebuilt sherpa-onnx: exposes sherpa::headers (include-dir only,
# because we dlopen the .so -- nothing on the link line) and SHERPA_SUPPORT_FILES
# (the runtime libs to bundle next to each standalone module).
include(cmake/sherpa-onnx.cmake)

# General score setup (warning flags, PCH, sanitizer wiring, ...). Only defined
# inside a score build tree; a no-op standalone. Called before avnd_addon_init,
# mirroring score-addon-onnx / score-addon-cv.
if(COMMAND score_common_setup)
  score_common_setup()
endif()

avnd_addon_init(NAME sherpa_plugins)

# Compile the base plugin target at C++23. In a score build avnd folds the object
# TUs into this single plugin target; score already builds at >= C++23, but set it
# explicitly so nothing falls back to the compiler default (C++17) -- mirrors
# score-addon-onnx. The per-object standalone back-ends get C++23 from Avendish,
# and sherpa_plugins_deps carries it (and the includes) via DEPENDENCIES.
target_compile_features(sherpa_plugins PUBLIC cxx_std_23)

# One INTERFACE target shared by every object: our src/ include dir + the sherpa
# headers. Passed via DEPENDENCIES so it reaches whatever compiles each object
# (the score base target, or each standalone back-end executable).
add_library(sherpa_plugins_deps INTERFACE)
target_include_directories(sherpa_plugins_deps
    INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_features(sherpa_plugins_deps INTERFACE cxx_std_23)
target_link_libraries(sherpa_plugins_deps INTERFACE sherpa::headers)

# sherpa objects do audio/data I/O (no textures), so they target the object-category
# back-end set including pd. `dump` is the SDK-free introspection target the
# portability CI builds. flavoured back-ends use the `:` syntax.
set(SHERPA_BACKENDS
    dump ossia pd max standalone python
    touchdesigner:CHOP_MESSAGE godot:NODE)

# NOTE: this MUST be a macro(), not a function(). In a score build
# avnd_addon_object (itself a macro) forwards to avnd_score_plugin_add, which
# accumulates the object into AVND_ADDITIONAL_CLASSES / AVND_CUSTOM_FACTORIES via
# `set(... PARENT_SCOPE)`. A function() wrapper would swallow that PARENT_SCOPE one
# level down (in the wrapper's own scope), so nothing reaches the top-level scope
# where avnd_addon_finalize generates the plugin factory .cpp -- leaving its
# factories() body empty and registering ZERO objects in score. A macro runs in
# the caller's (top-level) scope, so the accumulation propagates correctly. (This
# is the same latent bug present in score-addon-cv's function(add_cv_object).)
macro(sherpa_object c_name class)
  avnd_addon_object(
    BASE sherpa_plugins
    C_NAME ${c_name}
    CLASS ${class}
    NAMESPACE sherpa
    BACKENDS ${SHERPA_BACKENDS}
    DEPENDENCIES sherpa_plugins_deps
    MAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/${class}.hpp"
    SOURCES "src/${class}.hpp" "src/${class}.cpp")
endmacro()

sherpa_object(sherpa_asr               Asr)
sherpa_object(sherpa_asr_offline       OfflineRecognizer)
sherpa_object(sherpa_asr_online        OnlineRecognizer)
sherpa_object(sherpa_vad               Vad)
sherpa_object(sherpa_tts               Tts)
sherpa_object(sherpa_text              TextProcessor)
sherpa_object(sherpa_audio_labels      AudioLabels)
sherpa_object(sherpa_denoiser          Denoiser)
sherpa_object(sherpa_speaker           Speaker)
sherpa_object(sherpa_keyword_spotter   KeywordSpotter)
sherpa_object(sherpa_diarization       Diarization)
sherpa_object(sherpa_source_separation SourceSeparation)

# Carry the shared headers along so IDEs/the archive see them.
target_sources(sherpa_plugins PRIVATE
    src/compat/dylib_loader.hpp
    src/SherpaLoader.hpp
    src/helpers/Handles.hpp
    src/helpers/ModelConfig.hpp
    src/helpers/Options.hpp
    src/helpers/AudioChunker.hpp)

avnd_addon_finalize(NAME sherpa_plugins
    UUID 692ee9eb-6b46-428c-acd8-83528ea9d139 VERSION 1)

# In a score build, sherpa-onnx is NOT a host-provided library (unlike onnxruntime
# for score-addon-onnx, which the score host bundles). So when the runtime libs
# were located (-DSHERPA_ONNX_DIR=<install-with-lib/> or -DSHERPA_ONNX_FETCH_LIBS=ON),
# copy them next to the built plugin: the objects dlopen libsherpa-onnx-c-api from
# their own module folder (get_module_folder()), and that library's $ORIGIN rpath
# then resolves libonnxruntime beside it. Without this the plugin loads but every
# object reports unavailable at runtime ("libsherpa-onnx-c-api.so ... not found").
# Standalone back-ends get the libs via avnd_addon_package below instead.
if(AVND_ADDON_SCORE AND SHERPA_SUPPORT_FILES)
  add_custom_command(TARGET sherpa_plugins POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${SHERPA_SUPPORT_FILES} "$<TARGET_FILE_DIR:sherpa_plugins>"
    COMMAND_EXPAND_LISTS VERBATIM
    COMMENT "sherpa-plugins: copying sherpa-onnx runtime libs next to the plugin")
endif()

# Standalone distributable packages: bundle the prebuilt sherpa shared libs into
# the Max / TouchDesigner / Godot packages so each object dlopens them from beside
# its own module. Ignored inside score (the host bundles them). NAME MUST equal the
# GitHub repo name so the shared CI's "Prefer CMake-built package" step finds
# build/package/<NAME>/.
if(NOT AVND_ADDON_SCORE AND COMMAND avnd_addon_package)
  avnd_addon_package(
    NAME sherpa-plugins
    SOURCE_PATH "${CMAKE_CURRENT_SOURCE_DIR}"
    PACKAGE_ROOT "${CMAKE_BINARY_DIR}/package"
    BACKENDS max touchdesigner godot
    SUPPORT ${SHERPA_SUPPORT_FILES})
endif()

# --- Tests (Catch2) -------------------------------------------------------------
# Structural tests run anywhere; the per-object workflow tests validate real
# ASR/VAD/TTS when the runtime library + a model are available (env vars, see the
# test sources and .github/workflows/tests.yml). Reuses Avendish's avnd_common_setup
# for the halp include/standard, so it's only wired for standalone builds.
option(SHERPA_PLUGINS_TESTS "Build the Catch2 tests" OFF)
if(SHERPA_PLUGINS_TESTS AND COMMAND avnd_common_setup)
  include(CTest)
  if(NOT TARGET Catch2::Catch2WithMain)
    find_package(Catch2 3 QUIET)
  endif()
  if(NOT TARGET Catch2::Catch2WithMain)
    include(FetchContent)
    FetchContent_Declare(Catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG v3.7.1)
    FetchContent_MakeAvailable(Catch2)
  endif()

  # Aggregate target so CI can build every test with one name (no hardcoded list).
  add_custom_target(sherpa_all_tests)

  function(sherpa_test name)
    add_executable(${name} "tests/${name}.cpp")
    avnd_common_setup("" ${name})
    target_link_libraries(${name} PRIVATE Catch2::Catch2WithMain sherpa_plugins_deps)
    add_test(NAME ${name} COMMAND ${name})
    # Label so `ctest -L sherpa_plugins` runs only these, not the per-backend
    # smoke tests (e.g. godot:NODE) that avnd_addon_object also registers.
    set_tests_properties(${name} PROPERTIES LABELS "sherpa_plugins")
    add_dependencies(sherpa_all_tests ${name})
  endfunction()

  sherpa_test(test_structural)
  sherpa_test(test_coverage)
  sherpa_test(test_asr)
  sherpa_test(test_asr_48k)
  sherpa_test(test_roundtrip)
  sherpa_test(test_asr_offline)
  sherpa_test(test_asr_online)
  sherpa_test(test_vad)
  sherpa_test(test_tts)
  sherpa_test(test_text)
  sherpa_test(test_audio_labels)
  sherpa_test(test_denoiser)
  sherpa_test(test_speaker)
  sherpa_test(test_keyword_spotter)
  sherpa_test(test_diarization)
  sherpa_test(test_source_separation)
endif()
