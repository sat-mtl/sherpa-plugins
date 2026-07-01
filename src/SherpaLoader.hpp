#pragma once
// Runtime loader for the prebuilt sherpa-onnx C API.
//
// sherpa-onnx is hard to build cross-platform (notably with clang), so we never
// link it. We include only the pure-C header and dlopen the prebuilt
// libsherpa-onnx-c-api.{so,dll,dylib} at runtime, resolving the handful of
// SherpaOnnx* symbols the objects use. A missing library just flips `available`
// to false -- the objects then no-op instead of failing to load the host.
//
// The function-pointer member TYPES are derived from the real prototypes with
// decltype(&::Name), so they can never drift from the header. We never CALL the
// functions through `::Name` (that would create a link/import dependency), only
// through these resolved pointers.
//
// c-api.h is included without defining SHERPA_ONNX_BUILD_SHARED_LIBS, so on
// Windows SHERPA_ONNX_API expands to nothing (no dllimport) -- exactly what we
// want for a dlopen'd library.

#include "compat/dylib_loader.hpp"

#include <sherpa-onnx/c-api/c-api.h>

#include <cstdio>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace sherpa
{

#define SHERPA_FN(name) decltype(&::name) name = nullptr

#define SHERPA_REQ(name)                                                  \
  do                                                                      \
  {                                                                       \
    this->name = m_lib->symbol<decltype(&::name)>(#name);                 \
    if(!this->name)                                                       \
    {                                                                     \
      std::fprintf(stderr, "sherpa-plugins: missing symbol %s\n", #name); \
      available = false;                                                  \
      return;                                                             \
    }                                                                     \
  } while(0)

struct SherpaLoader
{
  // ---- version / wave / resampler utilities ----
  SHERPA_FN(SherpaOnnxGetVersionStr);
  SHERPA_FN(SherpaOnnxReadWave);
  SHERPA_FN(SherpaOnnxFreeWave);
  SHERPA_FN(SherpaOnnxCreateLinearResampler);
  SHERPA_FN(SherpaOnnxDestroyLinearResampler);
  SHERPA_FN(SherpaOnnxLinearResamplerReset);
  SHERPA_FN(SherpaOnnxLinearResamplerResample);
  SHERPA_FN(SherpaOnnxLinearResamplerResampleFree);

  // ---- offline (batch) ASR ----
  SHERPA_FN(SherpaOnnxCreateOfflineRecognizer);
  SHERPA_FN(SherpaOnnxDestroyOfflineRecognizer);
  SHERPA_FN(SherpaOnnxCreateOfflineStream);
  SHERPA_FN(SherpaOnnxDestroyOfflineStream);
  SHERPA_FN(SherpaOnnxAcceptWaveformOffline);
  SHERPA_FN(SherpaOnnxDecodeOfflineStream);
  SHERPA_FN(SherpaOnnxGetOfflineStreamResult);
  SHERPA_FN(SherpaOnnxDestroyOfflineRecognizerResult);

  // ---- online (streaming) ASR ----
  SHERPA_FN(SherpaOnnxCreateOnlineRecognizer);
  SHERPA_FN(SherpaOnnxDestroyOnlineRecognizer);
  SHERPA_FN(SherpaOnnxCreateOnlineStream);
  SHERPA_FN(SherpaOnnxDestroyOnlineStream);
  SHERPA_FN(SherpaOnnxOnlineStreamAcceptWaveform);
  SHERPA_FN(SherpaOnnxIsOnlineStreamReady);
  SHERPA_FN(SherpaOnnxDecodeOnlineStream);
  SHERPA_FN(SherpaOnnxGetOnlineStreamResult);
  SHERPA_FN(SherpaOnnxDestroyOnlineRecognizerResult);
  SHERPA_FN(SherpaOnnxOnlineStreamReset);
  SHERPA_FN(SherpaOnnxOnlineStreamInputFinished);
  SHERPA_FN(SherpaOnnxOnlineStreamIsEndpoint);

  // ---- VAD ----
  SHERPA_FN(SherpaOnnxCreateVoiceActivityDetector);
  SHERPA_FN(SherpaOnnxDestroyVoiceActivityDetector);
  SHERPA_FN(SherpaOnnxVoiceActivityDetectorAcceptWaveform);
  SHERPA_FN(SherpaOnnxVoiceActivityDetectorEmpty);
  SHERPA_FN(SherpaOnnxVoiceActivityDetectorDetected);
  SHERPA_FN(SherpaOnnxVoiceActivityDetectorFront);
  SHERPA_FN(SherpaOnnxVoiceActivityDetectorPop);
  SHERPA_FN(SherpaOnnxDestroySpeechSegment);
  SHERPA_FN(SherpaOnnxVoiceActivityDetectorFlush);
  SHERPA_FN(SherpaOnnxVoiceActivityDetectorReset);

  // ---- TTS ----
  SHERPA_FN(SherpaOnnxCreateOfflineTts);
  SHERPA_FN(SherpaOnnxDestroyOfflineTts);
  SHERPA_FN(SherpaOnnxOfflineTtsSampleRate);
  SHERPA_FN(SherpaOnnxOfflineTtsNumSpeakers);
  SHERPA_FN(SherpaOnnxOfflineTtsGenerateWithConfig);
  SHERPA_FN(SherpaOnnxDestroyOfflineTtsGeneratedAudio);

  bool available = true;

  // noexcept: the constructor catches the dlopen failure internally and just sets
  // available=false, so instance() never throws. This is what makes the objects
  // nothrow-default-constructible, which the Max/Pd (C object model) bindings
  // static_assert on.
  static const SherpaLoader& instance() noexcept
  {
    static const SherpaLoader self;
    return self;
  }

private:
#if defined(_WIN32)
  static constexpr const char* lib_name = "sherpa-onnx-c-api.dll";
#elif defined(__APPLE__)
  static constexpr const char* lib_name = "libsherpa-onnx-c-api.dylib";
#else
  static constexpr const char* lib_name = "libsherpa-onnx-c-api.so";
#endif

  // Owns the candidate path strings; the dylib_loader takes string_views into
  // them, used only during construction.
  std::vector<std::string> m_paths;
  std::unique_ptr<ossia::dylib_loader> m_lib;

  SherpaLoader()
  {
    m_paths.push_back(ossia::get_module_folder() + "/" + lib_name);
    m_paths.push_back(ossia::get_exe_folder() + "/" + lib_name);
    m_paths.emplace_back(lib_name); // system search path as a last resort

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
      std::fprintf(stderr, "sherpa-plugins: %s could not be loaded (%s)\n",
                   lib_name, e.what());
      available = false;
      return;
    }

    load();
  }

  void load()
  {
    SHERPA_REQ(SherpaOnnxGetVersionStr);
    SHERPA_REQ(SherpaOnnxReadWave);
    SHERPA_REQ(SherpaOnnxFreeWave);
    SHERPA_REQ(SherpaOnnxCreateLinearResampler);
    SHERPA_REQ(SherpaOnnxDestroyLinearResampler);
    SHERPA_REQ(SherpaOnnxLinearResamplerReset);
    SHERPA_REQ(SherpaOnnxLinearResamplerResample);
    SHERPA_REQ(SherpaOnnxLinearResamplerResampleFree);

    SHERPA_REQ(SherpaOnnxCreateOfflineRecognizer);
    SHERPA_REQ(SherpaOnnxDestroyOfflineRecognizer);
    SHERPA_REQ(SherpaOnnxCreateOfflineStream);
    SHERPA_REQ(SherpaOnnxDestroyOfflineStream);
    SHERPA_REQ(SherpaOnnxAcceptWaveformOffline);
    SHERPA_REQ(SherpaOnnxDecodeOfflineStream);
    SHERPA_REQ(SherpaOnnxGetOfflineStreamResult);
    SHERPA_REQ(SherpaOnnxDestroyOfflineRecognizerResult);

    SHERPA_REQ(SherpaOnnxCreateOnlineRecognizer);
    SHERPA_REQ(SherpaOnnxDestroyOnlineRecognizer);
    SHERPA_REQ(SherpaOnnxCreateOnlineStream);
    SHERPA_REQ(SherpaOnnxDestroyOnlineStream);
    SHERPA_REQ(SherpaOnnxOnlineStreamAcceptWaveform);
    SHERPA_REQ(SherpaOnnxIsOnlineStreamReady);
    SHERPA_REQ(SherpaOnnxDecodeOnlineStream);
    SHERPA_REQ(SherpaOnnxGetOnlineStreamResult);
    SHERPA_REQ(SherpaOnnxDestroyOnlineRecognizerResult);
    SHERPA_REQ(SherpaOnnxOnlineStreamReset);
    SHERPA_REQ(SherpaOnnxOnlineStreamInputFinished);
    SHERPA_REQ(SherpaOnnxOnlineStreamIsEndpoint);

    SHERPA_REQ(SherpaOnnxCreateVoiceActivityDetector);
    SHERPA_REQ(SherpaOnnxDestroyVoiceActivityDetector);
    SHERPA_REQ(SherpaOnnxVoiceActivityDetectorAcceptWaveform);
    SHERPA_REQ(SherpaOnnxVoiceActivityDetectorEmpty);
    SHERPA_REQ(SherpaOnnxVoiceActivityDetectorDetected);
    SHERPA_REQ(SherpaOnnxVoiceActivityDetectorFront);
    SHERPA_REQ(SherpaOnnxVoiceActivityDetectorPop);
    SHERPA_REQ(SherpaOnnxDestroySpeechSegment);
    SHERPA_REQ(SherpaOnnxVoiceActivityDetectorFlush);
    SHERPA_REQ(SherpaOnnxVoiceActivityDetectorReset);

    SHERPA_REQ(SherpaOnnxCreateOfflineTts);
    SHERPA_REQ(SherpaOnnxDestroyOfflineTts);
    SHERPA_REQ(SherpaOnnxOfflineTtsSampleRate);
    SHERPA_REQ(SherpaOnnxOfflineTtsNumSpeakers);
    SHERPA_REQ(SherpaOnnxOfflineTtsGenerateWithConfig);
    SHERPA_REQ(SherpaOnnxDestroyOfflineTtsGeneratedAudio);
  }
};

#undef SHERPA_REQ
#undef SHERPA_FN

}
