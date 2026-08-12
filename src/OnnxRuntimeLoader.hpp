#pragma once
// libsherpa-onnx-c-api links onnxruntime. Map it here first, so that when we
// dlopen sherpa its dependency binds to the copy the host already deploys rather
// than to a second one shipped next to the plugin.

#include "compat/dylib_loader.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// The onnxruntime the prebuilt libsherpa-onnx-c-api was linked against; set from
// cmake/sherpa-onnx.cmake.
#if !defined(SHERPA_PLUGINS_ORT_API_VERSION)
#define SHERPA_PLUGINS_ORT_API_VERSION 27
#endif
#if !defined(SHERPA_PLUGINS_ORT_VERSION)
#define SHERPA_PLUGINS_ORT_VERSION "1.27.1"
#endif

#if defined(_WIN32) && !defined(_WIN64)
#define SHERPA_ORT_CALL __stdcall
#else
#define SHERPA_ORT_CALL
#endif

namespace sherpa
{

// First two members of onnxruntime's OrtApiBase, so we need no onnxruntime
// headers. GetApi returns const OrtApi*, opaque to us.
struct ort_api_base
{
  const void*(SHERPA_ORT_CALL* GetApi)(std::uint32_t);
  const char*(SHERPA_ORT_CALL* GetVersionString)();
};

struct OnnxRuntimeLoader
{
  bool available = false;
  std::string resolved_path;
  std::string version;

  static const OnnxRuntimeLoader& instance() noexcept
  {
    static const OnnxRuntimeLoader self;
    return self;
  }

private:
  static std::vector<std::string> names()
  {
#if defined(_WIN32)
    return {"onnxruntime.dll"};
#elif defined(__APPLE__)
    // Microsoft moved the install name to the major-only libonnxruntime.1.dylib
    // in 1.26; before that it was the full version.
    return {
        "libonnxruntime.dylib", "libonnxruntime.1.dylib",
        "libonnxruntime." SHERPA_PLUGINS_ORT_VERSION ".dylib"};
#else
    return {"libonnxruntime.so.1", "libonnxruntime.so"};
#endif
  }

  static std::vector<std::string> folders()
  {
    const std::string mod = ossia::get_module_folder();
    const std::string exe = ossia::get_exe_folder();
#if defined(_WIN32)
    return {
        mod + "/",
        mod + "/../", // score: bin/plugins -> bin
        mod + "/../support/",
        exe + "/",
        exe + "/_deps/onnxruntime-src/lib/",
        exe + "/../_deps/onnxruntime-src/lib/",
    };
#elif defined(__APPLE__)
    return {
        mod + "/",
        mod + "/../",
        mod + "/../../Frameworks/", // score: Contents/MacOS/plugins -> Frameworks
        mod + "/../Frameworks/",
        mod + "/../../../../support/", // Max: <x>.mxo/Contents/MacOS -> support/
        exe + "/../Frameworks/",
        exe + "/",
        exe + "/_deps/onnxruntime-src/lib/",
        exe + "/../_deps/onnxruntime-src/lib/",
    };
#else
    return {
        mod + "/",
        mod + "/../",     // score DEB: lib/score -> lib
        mod + "/../lib/", // score install: plugins -> lib
        mod + "/../support/",
        mod + "/../_deps/onnxruntime-src/lib/", // score build tree
        exe + "/",
        exe + "/lib/",
        exe + "/../lib/",
        exe + "/_deps/onnxruntime-src/lib/",
        exe + "/../_deps/onnxruntime-src/lib/",
    };
#endif
  }

  static bool debug() noexcept
  {
    const char* v = std::getenv("SHERPA_PLUGINS_DEBUG");
    return v && *v && *v != '0';
  }

  static void* open(const char* path, bool noload) noexcept
  {
#if defined(_WIN32)
    if(noload)
      return (void*)GetModuleHandleA(path);
    return (void*)LoadLibraryA(path);
#else
    int flags = RTLD_LAZY | RTLD_LOCAL | RTLD_NODELETE;
    if(noload)
      flags |= RTLD_NOLOAD;
    return dlopen(path, flags);
#endif
  }

  static void* symbol(void* handle, const char* name) noexcept
  {
#if defined(_WIN32)
    return (void*)GetProcAddress((HMODULE)handle, name);
#else
    return dlsym(handle, name);
#endif
  }

  void* m_handle = nullptr; // never closed: process-wide, outlived by sherpa

  OnnxRuntimeLoader() noexcept
  {
    const bool trace = debug();

    // Already mapped: the host may load onnxruntime lazily, so whoever gets
    // there first wins and we must not become the second copy.
    for(const auto& n : names())
    {
      if((m_handle = open(n.c_str(), true)))
      {
        resolved_path = n;
        break;
      }
    }

    if(!m_handle)
    {
      const char* p = std::getenv("SHERPA_ONNXRUNTIME_PATH");
      if(p && *p && (m_handle = open(p, false)))
        resolved_path = p;
    }

    if(!m_handle)
    {
      for(const auto& dir : folders())
      {
        for(const auto& n : names())
        {
          const std::string candidate = dir + n;
          if((m_handle = open(candidate.c_str(), false)))
          {
            resolved_path = candidate;
            break;
          }
        }
        if(m_handle)
          break;
      }
    }

    // System search path last: preferring it over the host's own copy is how you
    // end up with two images.
    if(!m_handle)
    {
      for(const auto& n : names())
      {
        if((m_handle = open(n.c_str(), false)))
        {
          resolved_path = n;
          break;
        }
      }
    }

    if(!m_handle)
    {
      if(trace)
        std::fprintf(stderr, "sherpa-plugins: no onnxruntime found\n");
      return;
    }

    auto get_api_base
        = (const ort_api_base*(SHERPA_ORT_CALL*)())symbol(m_handle, "OrtGetApiBase");
    if(!get_api_base)
    {
      std::fprintf(
          stderr, "sherpa-plugins: %s exports no OrtGetApiBase\n",
          resolved_path.c_str());
      return;
    }

    const ort_api_base* base = get_api_base();
    if(!base || !base->GetApi)
    {
      std::fprintf(
          stderr, "sherpa-plugins: %s returned no OrtApiBase\n", resolved_path.c_str());
      return;
    }

    if(base->GetVersionString)
    {
      if(const char* v = base->GetVersionString())
        version = v;
    }

    // sherpa calls GetApi(ORT_API_VERSION) through onnxruntime's header-only
    // wrapper and dereferences the result unchecked, so catch a too-old runtime
    // here rather than crashing inside sherpa.
    if(!base->GetApi(SHERPA_PLUGINS_ORT_API_VERSION))
    {
      std::fprintf(
          stderr,
          "sherpa-plugins: %s is onnxruntime %s, which does not provide API "
          "version %d (sherpa-onnx was built against %s)\n",
          resolved_path.c_str(), version.empty() ? "?" : version.c_str(),
          (int)SHERPA_PLUGINS_ORT_API_VERSION, SHERPA_PLUGINS_ORT_VERSION);
      return;
    }

    available = true;
    if(trace)
      std::fprintf(
          stderr, "sherpa-plugins: onnxruntime %s from %s\n",
          version.empty() ? "?" : version.c_str(), resolved_path.c_str());
  }
};

}
