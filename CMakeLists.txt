cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

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

# --- The dual-build switch (identical across all celtera/ossia Avendish addons) ---
# Inside an ossia/score tree, score provides Avendish on CMAKE_PREFIX_PATH and the
# avnd_addon_* calls emit a single score plugin. Standalone, we fetch Avendish and
# the same calls emit one target per back-end. No other code path differs; when a
# branch is genuinely needed, Avendish sets AVND_ADDON_SCORE (1 in score, 0 here).
find_package(Avendish QUIET)
if(NOT Avendish_FOUND)
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

avnd_addon_init(NAME sherpa_plugins)

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

function(sherpa_object c_name class)
  avnd_addon_object(
    BASE sherpa_plugins
    C_NAME ${c_name}
    CLASS ${class}
    NAMESPACE sherpa
    BACKENDS ${SHERPA_BACKENDS}
    DEPENDENCIES sherpa_plugins_deps
    MAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/${class}.hpp"
    SOURCES "src/${class}.hpp" "src/${class}.cpp")
endfunction()

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
    src/helpers/AudioChunker.hpp)

avnd_addon_finalize(NAME sherpa_plugins
    UUID 692ee9eb-6b46-428c-acd8-83528ea9d139 VERSION 1)

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
