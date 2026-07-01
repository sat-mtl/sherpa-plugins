// Workflow test for sherpa_speaker: compute a speaker embedding, enroll it, then
//   * identify the same voice back  -> must match the enrolled name (positive), and
//   * identify a white-noise clip   -> must NOT match the enrolled name (negative,
//     i.e. no false accept -- what makes speaker ID actually discriminative).
//
// Requires:
//   SHERPA_TEST_SPEAKER_MODEL = a speaker-embedding model .onnx (file) or a dir
//                               containing it (e.g. 3dspeaker campplus)
//   SHERPA_TEST_WAV           = a mono speech wav
#include "TestHelpers.hpp"

#include "Speaker.hpp"

#include <catch2/catch_all.hpp>

#include <string>
#include <vector>

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

TEST_CASE("Speaker embeds, identifies its own voice, and rejects noise",
          "[speaker][workflow]")
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
  obj.inputs.threshold.value = 0.4f; // typical verification operating point

  std::string matched;
  obj.outputs.speaker.call.context = &matched;
  obj.outputs.speaker.call.function = +[](void* c, std::string_view s) {
    *static_cast<std::string*>(c) = std::string(s);
  };

  // 1) Embedding has the model's dimension.
  capture_clip(obj, audio, block, "");
  INFO("embedding size: " << obj.outputs.embedding.value.size() << ", dim: " << dim);
  REQUIRE_FALSE(obj.outputs.embedding.value.empty());
  REQUIRE(obj.outputs.embedding.value.size() == static_cast<std::size_t>(dim));

  // 2) Enroll the voice, then identify the SAME voice -> must match.
  capture_clip(obj, audio, block, "alice");
  REQUIRE(obj.outputs.count.value >= 1);

  matched.clear();
  capture_clip(obj, audio, block, ""); // identify same voice
  INFO("self-identify matched: '" << matched << "'");
  REQUIRE(matched == "alice");

  // 3) Identify white noise -> must NOT be accepted as the enrolled speaker.
  auto noise = test::white_noise(audio.size(), 0.2f);
  matched = "sentinel";
  capture_clip(obj, noise, block, ""); // identify noise
  INFO("noise-identify matched: '" << matched << "'");
  CHECK(matched != "alice"); // no false accept (empty, or a non-match)
}
