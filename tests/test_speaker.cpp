// Workflow test for sherpa_speaker: capture a spoken utterance, compute its
// speaker embedding, then enroll it and identify the same clip back.
//
// Requires:
//   SHERPA_TEST_SPEAKER_MODEL = a speaker-embedding model .onnx (file) or a dir
//                               containing it (e.g. 3dspeaker campplus)
//   SHERPA_TEST_WAV           = a mono speech wav
#include "TestHelpers.hpp"

#include "Speaker.hpp"

#include <catch2/catch_all.hpp>

#include <string>

using namespace sherpa;

// Capture one utterance: enable Capture, feed the audio, then release Capture
// (the falling edge dispatches the embedding computation).
template <typename Obj>
static void capture_clip(
    Obj& obj, const std::vector<float>& audio, int block,
    const std::string& enroll_as)
{
  obj.inputs.enroll_name.value = enroll_as;
  obj.inputs.capture.value = true;
  test::drive_audio(obj, audio, block);
  obj.inputs.capture.value = false;
  test::drive_silence(obj, block); // falling edge -> dispatch
}

TEST_CASE("Speaker extracts an embedding and identifies a clip", "[speaker][workflow]")
{
  if(!test::loader_ready())
    SHERPA_SKIP("sherpa-onnx runtime library not available");
  auto model = test::env("SHERPA_TEST_SPEAKER_MODEL");
  auto wav = test::env("SHERPA_TEST_WAV");
  if(!model || !wav)
    SHERPA_SKIP("set SHERPA_TEST_SPEAKER_MODEL and SHERPA_TEST_WAV");

  int sr = 0;
  auto audio = test::load_wav_mono(*wav, sr);
  REQUIRE(sr > 0);
  REQUIRE_FALSE(audio.empty());

  // Independently query the model's embedding dimension for the size check.
  int dim = 0;
  {
    auto ex = sherpa::make_speaker_extractor(*model, 1);
    if(ex && SherpaLoader::instance().SherpaOnnxSpeakerEmbeddingExtractorDim)
      dim = SherpaLoader::instance().SherpaOnnxSpeakerEmbeddingExtractorDim(ex.get());
  }
  if(dim <= 0)
    SHERPA_SKIP("speaker model produced no embedding dimension");

  Speaker obj;
  test::wire_sync_worker(obj);

  const int block = 1600;
  obj.prepare(halp::setup{
      .input_channels = 1, .output_channels = 0, .frames = block, .rate = (double)sr});
  obj.inputs.model.value = *model;
  obj.inputs.threads.value = 1;
  obj.inputs.threshold.value = 0.3f;

  std::string matched;
  obj.outputs.speaker.call.context = &matched;
  obj.outputs.speaker.call.function
      = +[](void* c, std::string_view s) { *static_cast<std::string*>(c) = std::string(s); };

  // 1) Compute an embedding (no enrollment yet -> identify against nobody).
  capture_clip(obj, audio, block, "");
  INFO("embedding size: " << obj.outputs.embedding.value.size() << ", model dim: " << dim);
  REQUIRE_FALSE(obj.outputs.embedding.value.empty());
  REQUIRE(obj.outputs.embedding.value.size() == static_cast<std::size_t>(dim));

  // 2) Enroll the same clip under a name, then 3) identify it back.
  capture_clip(obj, audio, block, "alice");
  CHECK(obj.outputs.count.value >= 1);

  matched.clear();
  capture_clip(obj, audio, block, ""); // identify
  INFO("identified speaker: '" << matched << "'");
  CHECK(matched == "alice");
}
