// Coverage guard: when the runtime library is present, every sherpa-onnx feature
// entry point we wrap must have resolved in the loader. Catches a typo'd symbol
// name (SHERPA_OPT loads silently) and proves we cover 100% of the workflows.
#include "TestHelpers.hpp"

#include "SherpaLoader.hpp"

#include <catch2/catch_all.hpp>

using namespace sherpa;

TEST_CASE("Every sherpa-onnx feature entry point is wired", "[coverage]")
{
  if(!test::loader_ready())
    SHERPA_SKIP("sherpa-onnx runtime library not available");
  const auto& L = SherpaLoader::instance();

  // Core (required by the loader).
  CHECK(L.SherpaOnnxCreateOfflineRecognizer);
  CHECK(L.SherpaOnnxCreateOnlineRecognizer);
  CHECK(L.SherpaOnnxCreateVoiceActivityDetector);
  CHECK(L.SherpaOnnxCreateOfflineTts);

  // Extended (optional in the loader; must be present in the pinned lib).
  CHECK(L.SherpaOnnxCreateKeywordSpotter);
  CHECK(L.SherpaOnnxCreateSpeakerEmbeddingExtractor);
  CHECK(L.SherpaOnnxCreateSpeakerEmbeddingManager);
  CHECK(L.SherpaOnnxCreateOfflineSpeakerDiarization);
  CHECK(L.SherpaOnnxCreateOfflinePunctuation);
  CHECK(L.SherpaOnnxCreateOnlinePunctuation);
  CHECK(L.SherpaOnnxCreateOfflineDiacritization);
  CHECK(L.SherpaOnnxCreateAudioTagging);
  CHECK(L.SherpaOnnxCreateSpokenLanguageIdentification);
  CHECK(L.SherpaOnnxCreateOfflineSpeechDenoiser);
  CHECK(L.SherpaOnnxCreateOnlineSpeechDenoiser);
  CHECK(L.SherpaOnnxCreateOfflineSourceSeparation);
}
