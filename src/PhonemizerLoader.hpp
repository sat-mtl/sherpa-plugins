#pragma once
// Runtime loader for libsherpa-plugins-phonemizer (phonemizer/ in this repo):
// piper-phonemize + espeak-ng behind a C ABI, shipped next to sherpa-onnx.
//
// sherpa-onnx stopped phonemizing text itself; a TTS model without a lexicon
// now takes espeak-ng phonemes from the caller (see Tts.hpp). Same dlopen
// approach as SherpaLoader: a missing library only leaves `available` false, and
// those models then report that they cannot speak.

#include "compat/dylib_loader.hpp"

#include <cstdio>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <sherpa-plugins/phonemizer.h>

namespace sherpa
{

struct PhonemizerLoader
{
  decltype(&::SherpaPluginsPhonemize) SherpaPluginsPhonemize = nullptr;
  decltype(&::SherpaPluginsPhonemizerFree) SherpaPluginsPhonemizerFree = nullptr;
  decltype(&::SherpaPluginsPhonemizerVersion) SherpaPluginsPhonemizerVersion = nullptr;

  bool available = false;

  static const PhonemizerLoader& instance() noexcept
  {
    static const PhonemizerLoader self;
    return self;
  }

private:
#if defined(_WIN32)
  static constexpr const char* lib_name = "sherpa-plugins-phonemizer.dll";
#elif defined(__APPLE__)
  static constexpr const char* lib_name = "libsherpa-plugins-phonemizer.dylib";
#else
  static constexpr const char* lib_name = "libsherpa-plugins-phonemizer.so";
#endif

  std::vector<std::string> m_paths;
  std::unique_ptr<ossia::dylib_loader> m_lib;

  PhonemizerLoader() noexcept
  {
    // The same places SherpaLoader looks: it ships in the same directory.
    const std::string mod = ossia::get_module_folder();
    m_paths.push_back(mod + "/" + lib_name);
    m_paths.push_back(mod + "/../" + lib_name);
    m_paths.push_back(mod + "/../lib/" + lib_name);
    m_paths.push_back(mod + "/../../Frameworks/" + lib_name);
    m_paths.push_back(ossia::get_exe_folder() + "/" + lib_name);
    m_paths.emplace_back(lib_name);

    std::vector<std::string_view> candidates;
    candidates.reserve(m_paths.size());
    for(auto& p : m_paths)
      candidates.emplace_back(p);

    try
    {
      m_lib = std::make_unique<ossia::dylib_loader>(candidates);
    }
    catch(const std::exception& e)
    {
      std::fprintf(
          stderr, "sherpa-plugins: %s could not be loaded (%s)\n", lib_name, e.what());
      return;
    }

    SherpaPluginsPhonemize
        = m_lib->symbol<decltype(&::SherpaPluginsPhonemize)>("SherpaPluginsPhonemize");
    SherpaPluginsPhonemizerFree = m_lib->symbol<decltype(&::SherpaPluginsPhonemizerFree)>(
        "SherpaPluginsPhonemizerFree");
    SherpaPluginsPhonemizerVersion
        = m_lib->symbol<decltype(&::SherpaPluginsPhonemizerVersion)>(
            "SherpaPluginsPhonemizerVersion");
    available = SherpaPluginsPhonemize && SherpaPluginsPhonemizerFree;
  }
};

}
