#pragma once
// Move-only RAII wrappers around sherpa-onnx opaque handles. Each destructor
// calls the matching SherpaOnnxDestroy* through the runtime loader. Wrap a handle
// in std::shared_ptr<...> when it must outlive the object that created it (e.g.
// while a worker job is still using it on the inference thread).

#include "SherpaLoader.hpp"

namespace sherpa
{

#define SHERPA_DEFINE_HANDLE(Name, CType, DestroyFn)                       \
  struct Name                                                             \
  {                                                                       \
    const CType* h = nullptr;                                            \
    Name() = default;                                                    \
    explicit Name(const CType* p) noexcept                              \
        : h{p}                                                          \
    {                                                                    \
    }                                                                    \
    Name(const Name&) = delete;                                          \
    Name& operator=(const Name&) = delete;                              \
    Name(Name&& o) noexcept                                             \
        : h{o.h}                                                        \
    {                                                                    \
      o.h = nullptr;                                                    \
    }                                                                    \
    Name& operator=(Name&& o) noexcept                                  \
    {                                                                    \
      if(this != &o)                                                    \
      {                                                                  \
        reset();                                                        \
        h = o.h;                                                        \
        o.h = nullptr;                                                  \
      }                                                                  \
      return *this;                                                     \
    }                                                                    \
    ~Name() { reset(); }                                                \
    void reset() noexcept                                               \
    {                                                                    \
      if(h)                                                             \
      {                                                                  \
        SherpaLoader::instance().DestroyFn(h);                         \
        h = nullptr;                                                   \
      }                                                                  \
    }                                                                    \
    explicit operator bool() const noexcept { return h != nullptr; }    \
    const CType* get() const noexcept { return h; }                     \
  }

SHERPA_DEFINE_HANDLE(
    OfflineRecognizerHandle, SherpaOnnxOfflineRecognizer,
    SherpaOnnxDestroyOfflineRecognizer);
SHERPA_DEFINE_HANDLE(
    OfflineStreamHandle, SherpaOnnxOfflineStream, SherpaOnnxDestroyOfflineStream);
SHERPA_DEFINE_HANDLE(
    OnlineRecognizerHandle, SherpaOnnxOnlineRecognizer,
    SherpaOnnxDestroyOnlineRecognizer);
SHERPA_DEFINE_HANDLE(
    OnlineStreamHandle, SherpaOnnxOnlineStream, SherpaOnnxDestroyOnlineStream);
SHERPA_DEFINE_HANDLE(
    VadHandle, SherpaOnnxVoiceActivityDetector,
    SherpaOnnxDestroyVoiceActivityDetector);
SHERPA_DEFINE_HANDLE(
    TtsHandle, SherpaOnnxOfflineTts, SherpaOnnxDestroyOfflineTts);
SHERPA_DEFINE_HANDLE(
    LinearResamplerHandle, SherpaOnnxLinearResampler,
    SherpaOnnxDestroyLinearResampler);

// Extended features. (For a given feature, create* and destroy* load together via
// SHERPA_OPT, so a non-null handle always has a valid destroy.)
SHERPA_DEFINE_HANDLE(
    KeywordSpotterHandle, SherpaOnnxKeywordSpotter, SherpaOnnxDestroyKeywordSpotter);
SHERPA_DEFINE_HANDLE(
    SpeakerEmbeddingExtractorHandle, SherpaOnnxSpeakerEmbeddingExtractor,
    SherpaOnnxDestroySpeakerEmbeddingExtractor);
SHERPA_DEFINE_HANDLE(
    SpeakerEmbeddingManagerHandle, SherpaOnnxSpeakerEmbeddingManager,
    SherpaOnnxDestroySpeakerEmbeddingManager);
SHERPA_DEFINE_HANDLE(
    OfflineSpeakerDiarizationHandle, SherpaOnnxOfflineSpeakerDiarization,
    SherpaOnnxDestroyOfflineSpeakerDiarization);
SHERPA_DEFINE_HANDLE(
    OfflinePunctuationHandle, SherpaOnnxOfflinePunctuation,
    SherpaOnnxDestroyOfflinePunctuation);
SHERPA_DEFINE_HANDLE(
    OnlinePunctuationHandle, SherpaOnnxOnlinePunctuation,
    SherpaOnnxDestroyOnlinePunctuation);
SHERPA_DEFINE_HANDLE(
    OfflineDiacritizationHandle, SherpaOnnxOfflineDiacritization,
    SherpaOnnxDestroyOfflineDiacritization);
SHERPA_DEFINE_HANDLE(
    AudioTaggingHandle, SherpaOnnxAudioTagging, SherpaOnnxDestroyAudioTagging);
SHERPA_DEFINE_HANDLE(
    SpokenLanguageIdHandle, SherpaOnnxSpokenLanguageIdentification,
    SherpaOnnxDestroySpokenLanguageIdentification);
SHERPA_DEFINE_HANDLE(
    OfflineDenoiserHandle, SherpaOnnxOfflineSpeechDenoiser,
    SherpaOnnxDestroyOfflineSpeechDenoiser);
SHERPA_DEFINE_HANDLE(
    OnlineDenoiserHandle, SherpaOnnxOnlineSpeechDenoiser,
    SherpaOnnxDestroyOnlineSpeechDenoiser);
SHERPA_DEFINE_HANDLE(
    SourceSeparationHandle, SherpaOnnxOfflineSourceSeparation,
    SherpaOnnxDestroyOfflineSourceSeparation);

#undef SHERPA_DEFINE_HANDLE

}
