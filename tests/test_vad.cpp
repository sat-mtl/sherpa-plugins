// Workflow test for sherpa_vad: stream a spoken utterance and expect the detector
// to report speech (a SpeechStart, and a completed Segment once silence follows).
//
// Requires:
//   SHERPA_TEST_VAD_MODEL = silero_vad.onnx (file) or a dir containing it
//   SHERPA_TEST_WAV       = a 16 kHz mono speech wav
#include "TestHelpers.hpp"

#include "Vad.hpp"

#include <catch2/catch_all.hpp>

using namespace sherpa;

TEST_CASE("VAD detects speech in a spoken utterance", "[vad][workflow]")
{
  if(!test::loader_ready())
    SHERPA_SKIP("sherpa-onnx runtime library not available");
  auto model = test::env("SHERPA_TEST_VAD_MODEL");
  auto wav = test::env("SHERPA_TEST_WAV");
  if(!model || !wav)
    SHERPA_SKIP("set SHERPA_TEST_VAD_MODEL and SHERPA_TEST_WAV");

  int sr = 0;
  auto audio = test::load_wav_mono(*wav, sr);
  REQUIRE(sr > 0);
  REQUIRE_FALSE(audio.empty());

  Vad obj;
  test::wire_sync_worker(obj);

  const int block = 1600;
  obj.prepare(halp::setup{
      .input_channels = 1, .output_channels = 0, .frames = block, .rate = (double)sr});
  obj.inputs.model.value = *model;
  obj.inputs.threshold.value = 0.5f;
  obj.inputs.threads.value = 1;

  int starts = 0, segments = 0;
  obj.outputs.on_start.call.context = &starts;
  obj.outputs.on_start.call.function = +[](void* c) { (*static_cast<int*>(c))++; };
  obj.outputs.on_segment.call.context = &segments;
  obj.outputs.on_segment.call.function
      = +[](void* c, float, float) { (*static_cast<int*>(c))++; };

  test::drive_audio(obj, audio, block);
  // Trailing silence so the final speech segment is closed and emitted.
  std::vector<float> tail(static_cast<std::size_t>(sr), 0.f); // 1 s
  test::drive_audio(obj, tail, block);

  INFO("speech starts: " << starts << ", completed segments: " << segments);
  REQUIRE(starts >= 1);
  CHECK(segments >= 1);
}
