# sherpa-plugins phonemizer: piper-phonemize + espeak-ng behind a C ABI, built
# as one self-contained shared library (both are linked in statically).
#
# Built by .github/workflows/sherpa-prebuilt.yml beside sherpa-onnx and shipped in
# the same archive; the plugin dlopens it (src/PhonemizerLoader.hpp). Standalone:
#   cmake -S phonemizer -B build-phonemizer -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-phonemizer --target install
#
# espeak-ng is GPL-3.0-or-later, so is this library. Same espeak-ng and
# piper-phonemize revisions sherpa-onnx built in before removing them
# (k2-fsa/sherpa-onnx#4055).
cmake_minimum_required(VERSION 3.28) # FetchContent_Declare(EXCLUDE_FROM_ALL)
project(sherpa-plugins-phonemizer LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
if(MSVC)
  # IPA constants in piper-phonemize; espeak-ng's own sources too.
  add_compile_options("/utf-8")
endif()

include(FetchContent)

# --- espeak-ng (static) ----------------------------------------------------------
set(BUILD_SHARED_LIBS OFF)
set(BUILD_ESPEAK_NG_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_ESPEAK_NG_EXE OFF CACHE BOOL "" FORCE)
set(USE_ASYNC OFF CACHE BOOL "" FORCE)
set(USE_MBROLA OFF CACHE BOOL "" FORCE)
set(USE_LIBSONIC OFF CACHE BOOL "" FORCE)
set(USE_LIBPCAUDIO OFF CACHE BOOL "" FORCE)
set(USE_KLATT OFF CACHE BOOL "" FORCE)
set(USE_SPEECHPLAYER OFF CACHE BOOL "" FORCE)
set(EXTRA_cmn ON CACHE BOOL "" FORCE)
set(EXTRA_ru ON CACHE BOOL "" FORCE)
FetchContent_Declare(espeak_ng
  URL https://github.com/csukuangfj/espeak-ng/archive/ed530aa113046142eb5115cf2fc9157854d0ffe1.zip
  URL_HASH SHA256=e4e262cbe34f7fe21f91f1ba3397f2728e1f30eafbae7853f2b753a9ed13f0dd
  EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(espeak_ng)
target_include_directories(espeak-ng INTERFACE
  ${espeak_ng_SOURCE_DIR}/src/include
  ${espeak_ng_SOURCE_DIR}/src/ucd-tools/src/include)
if(MSVC)
  target_compile_options(ucd PRIVATE /wd4309)
  target_compile_options(espeak-ng PRIVATE
    /wd4005 /wd4018 /wd4067 /wd4068 /wd4090 /wd4101 /wd4244 /wd4267 /wd4996)
elseif(UNIX AND NOT APPLE)
  target_compile_options(espeak-ng PRIVATE
    -Wno-unused-result -Wno-format-overflow -Wno-format-truncation
    -Wno-uninitialized -Wno-format)
endif()

# --- piper-phonemize: only its espeak frontend, compiled in directly ---------------
# Its own CMakeLists pulls a second espeak-ng through ExternalProject.
FetchContent_Declare(piper_phonemize
  URL https://github.com/csukuangfj/piper-phonemize/archive/f3ff95afc03640bc1399e113e83361192a2fafb4.zip
  URL_HASH SHA256=d9cca4e2bdc7d6dd8dffb96a4668283dbd3f77a9c194a3e530c1e8eba9406a5d
  SOURCE_SUBDIR this-directory-does-not-exist)
FetchContent_MakeAvailable(piper_phonemize)

# --- the library -------------------------------------------------------------------
add_library(sherpa-plugins-phonemizer SHARED
  src/phonemizer.cpp
  ${piper_phonemize_SOURCE_DIR}/src/phonemize.cpp)
target_compile_definitions(sherpa-plugins-phonemizer PRIVATE
  SHERPA_PLUGINS_PHONEMIZER_BUILD
  # piper's PIPERPHONEMIZE_EXPORT is dllexport on Windows: keep its C++ symbols in.
  PIPERPHONEMIZE_EXPORT=)
target_include_directories(sherpa-plugins-phonemizer
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${piper_phonemize_SOURCE_DIR}/src)
target_link_libraries(sherpa-plugins-phonemizer PRIVATE espeak-ng)
# espeak-ng marks its API visibility("default"): export only ours, so another
# espeak-ng in the host process can never bind to (or be shadowed by) this one.
if(APPLE)
  set_target_properties(sherpa-plugins-phonemizer PROPERTIES INSTALL_NAME_DIR "@rpath")
  target_link_options(sherpa-plugins-phonemizer PRIVATE
    "LINKER:-exported_symbol,_SherpaPlugins*")
elseif(UNIX)
  target_link_options(sherpa-plugins-phonemizer PRIVATE "LINKER:--exclude-libs,ALL")
endif()

include(GNUInstallDirs)
install(TARGETS sherpa-plugins-phonemizer
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
