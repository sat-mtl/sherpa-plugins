# Automatic score system-library path. Explicit SHERPA_ONNX_DIR bypasses this
# module completely, including discovery and all source/dependency downloads.
if(CMAKE_VERSION VERSION_LESS 3.28)
  message(FATAL_ERROR "Building sherpa-onnx with system ONNX Runtime requires CMake 3.28 or newer")
endif()

# Reuse score-addon-onnx's cache entries and search conventions. Require a real
# shared library: linking a static ORT would embed a second runtime in score.
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  PATH_SUFFIXES onnxruntime onnxruntime/core/session REQUIRED)
find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime REQUIRED)
file(REAL_PATH "${ONNXRUNTIME_LIBRARY}" _sherpa_system_ort_library)
get_filename_component(_sherpa_system_ort_name "${_sherpa_system_ort_library}" NAME)
if(_sherpa_system_ort_name MATCHES "\\.(a|lib)$" AND NOT WIN32)
  message(FATAL_ERROR "sherpa-plugins requires a shared system ONNX Runtime, not ${ONNXRUNTIME_LIBRARY}")
endif()
# Load the runtime SONAME, not a patch-version filename which distro upgrades
# may replace while preserving the ABI (and without the development symlink).
if(CMAKE_SYSTEM_NAME MATCHES "Linux|FreeBSD|OpenBSD|NetBSD|DragonFly")
  if(NOT CMAKE_READELF)
    find_program(CMAKE_READELF NAMES llvm-readelf readelf REQUIRED)
  endif()
  execute_process(COMMAND "${CMAKE_READELF}" -d -W "${_sherpa_system_ort_library}"
    OUTPUT_VARIABLE _sherpa_system_dynamic
    COMMAND_ERROR_IS_FATAL ANY)
  if(NOT _sherpa_system_dynamic MATCHES "\\(SONAME\\)[^\n]*\\[([^]]+)\\]")
    message(FATAL_ERROR "System ONNX Runtime has no shared-library SONAME")
  endif()
  set(_sherpa_system_ort_name "${CMAKE_MATCH_1}")
endif()
file(STRINGS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_c_api.h" _sherpa_api_line
  REGEX "^#define[ \t]+ORT_API_VERSION[ \t]+[0-9]+")
if(NOT _sherpa_api_line)
  message(FATAL_ERROR "Cannot read ORT_API_VERSION from ${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_c_api.h")
endif()
string(REGEX REPLACE ".*ORT_API_VERSION[ \t]+([0-9]+).*" "\\1"
  _sherpa_ort_api "${_sherpa_api_line}")

# GetVersionString's literal is also present when a distro uses a shortened
# SONAME (Debian's libonnxruntime.so.1.21). Read target bytes, never execute target
# code, so cross-compiles derive the version from their own runtime as well.
file(STRINGS "${_sherpa_system_ort_library}" _sherpa_system_versions
  REGEX "^1\\.${_sherpa_ort_api}\\.[0-9]+([-+][0-9A-Za-z.-]+)?$")
list(REMOVE_DUPLICATES _sherpa_system_versions)
list(LENGTH _sherpa_system_versions _sherpa_system_version_count)
if(NOT _sherpa_system_version_count EQUAL 1)
  message(FATAL_ERROR
    "Cannot determine a unique ONNX Runtime version matching API ${_sherpa_ort_api} from ${_sherpa_system_ort_library}")
endif()
list(GET _sherpa_system_versions 0 _sherpa_ort_version)
message(STATUS "sherpa-plugins: building pinned sherpa-onnx against system ONNX Runtime ${_sherpa_ort_version} (API ${_sherpa_ort_api}): ${_sherpa_system_ort_library}")

include(ExternalProject)
set(_sherpa_system_prefix "${CMAKE_CURRENT_BINARY_DIR}/system-runtime")
set(_sherpa_system_install "${_sherpa_system_prefix}/install")
set(_sherpa_system_cmake_args
  "-DCMAKE_INSTALL_PREFIX:PATH=${_sherpa_system_install}"
  "-DSHERPA_SYSTEM_ORT_INCLUDE_DIR:PATH=${ONNXRUNTIME_INCLUDE_DIR}"
  "-DSHERPA_SYSTEM_ORT_LIBRARY:FILEPATH=${_sherpa_system_ort_library}"
  "-DSHERPA_PLUGINS_SOURCE_DIR:PATH=${CMAKE_CURRENT_LIST_DIR}/..")
# Keep the host toolchain, sysroot and flags, but not score's global target/cache
# namespace. Forward FetchContent source overrides for pre-populated source trees.
get_cmake_property(_sherpa_cmake_variables VARIABLES)
foreach(_var IN LISTS _sherpa_cmake_variables)
  if(_var MATCHES "^FETCHCONTENT_SOURCE_DIR_"
     OR _var MATCHES "^CMAKE_(TOOLCHAIN_FILE|C_COMPILER|CXX_COMPILER|C_COMPILER_TARGET|CXX_COMPILER_TARGET|SYSROOT|FIND_ROOT_PATH|PREFIX_PATH|BUILD_TYPE|OSX_ARCHITECTURES|OSX_DEPLOYMENT_TARGET|OSX_SYSROOT|MSVC_RUNTIME_LIBRARY|POSITION_INDEPENDENT_CODE)$"
     OR _var MATCHES "^CMAKE_(C|CXX|SHARED_LINKER|EXE_LINKER)_FLAGS($|_)")
    string(REPLACE ";" "|" _value "${${_var}}")
    list(APPEND _sherpa_system_cmake_args "-D${_var}:STRING=${_value}")
  endif()
endforeach()

if(WIN32)
  set(_sherpa_system_runtime_dir "${_sherpa_system_install}/bin")
else()
  set(_sherpa_system_runtime_dir "${_sherpa_system_install}/lib")
endif()
set(_sherpa_system_files "")
foreach(_name sherpa-onnx-c-api sherpa-plugins-phonemizer)
  list(APPEND _sherpa_system_files
    "${_sherpa_system_runtime_dir}/${CMAKE_SHARED_LIBRARY_PREFIX}${_name}${CMAKE_SHARED_LIBRARY_SUFFIX}")
endforeach()
ExternalProject_Add(sherpa_plugins_system_runtime
  SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/system-runtime"
  BINARY_DIR "${_sherpa_system_prefix}/build"
  PREFIX "${_sherpa_system_prefix}"
  DOWNLOAD_COMMAND ""
  LIST_SEPARATOR "|"
  CMAKE_ARGS ${_sherpa_system_cmake_args}
  BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR> --config $<CONFIG>
    --target sherpa-onnx-c-api sherpa-plugins-phonemizer
  INSTALL_COMMAND "${CMAKE_COMMAND}" --install <BINARY_DIR> --config $<CONFIG>
    --component SherpaPluginsRuntime
  BUILD_ALWAYS ON
  BUILD_BYPRODUCTS ${_sherpa_system_files})

# Imported targets describe generated files without a configure-time glob. They
# are dependencies, not link dependencies: the plugin still dlopens both C ABIs.
set(SHERPA_SUPPORT_FILES "")
foreach(_name sherpa-onnx-c-api sherpa-plugins-phonemizer)
  add_library(${_name}-runtime SHARED IMPORTED)
  set_target_properties(${_name}-runtime PROPERTIES IMPORTED_LOCATION
    "${_sherpa_system_runtime_dir}/${CMAKE_SHARED_LIBRARY_PREFIX}${_name}${CMAKE_SHARED_LIBRARY_SUFFIX}")
  add_dependencies(${_name}-runtime sherpa_plugins_system_runtime)
  list(APPEND SHERPA_SUPPORT_FILES "$<TARGET_FILE:${_name}-runtime>")
endforeach()
set(SHERPA_ORT_SUPPORT_FILES "")
set(SHERPA_RUNTIME_TARGET sherpa_plugins_system_runtime)
