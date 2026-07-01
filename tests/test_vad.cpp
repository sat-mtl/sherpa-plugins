// Workflow tests for sherpa_vad.
//   Positive: a spoken utterance produces a SpeechStart + a completed Segment,
//             and the reported segment lands inside the clip (not in the silence).
//   Negative: pure silence produces NO speech at all (no false positives).
//
// Requires:
//   SHERPA_TEST_VAD_MODEL = silero_vad.onnx (file) or a dir containing it
//   SHERPA_TEST_WAV       = a 16 kHz mono speech wav
#include "TestHelpers.hpp"

#include "Vad.hpp"

#include <catch2/catch_all.hpp>

#include <vector>

using namespace sherpa;

namespace
{
struct VadTap
{
  int starts = 0;
  int segments = 0;
  float first_start = -1.f;
  float total_speech = 0.f;
};

void wire(Vad& obj, VadTap& tap)
{
  obj.outputs.on_start.call.context = &tap;
  obj.outputs.on_start.call.function
      = +[](void* c) { static_cast<VadTap*>(c)->starts++; };
  obj.outputs.on_segment.call.context = &tap;
  obj.outputs.on_segment.call.function = +[](void* c, float start, float dur) {
    auto* t = static_cast<VadTap*>(c);
    if(t->segments == 0)
      t->first_start = start;
    t->segments++;
    t->total_speech += dur;
  };
}
}

TEST_CASE("VAD detects speech and locates it inside the clip", "[vad][workflow]")
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

  // Prepend 0.5 s of silence so a correct detector must start speech *after* it.
  const std::size_t lead = static_cast<std::size_t>(sr / 2);
  std::vector<float> clip(lead, 0.f);
  clip.insert(clip.end(), audio.begin(), audio.end());
  const float clip_dur = static_cast<float>(clip.size()) / static_cast<float>(sr);

  Vad obj;
  test::wire_sync_worker(obj);
  const int block = 1600;
  obj.prepare(halp::setup{
      .input_channels = 1, .output_channels = 0, .frames = block, .rate = (double)sr});
  obj.inputs.model.value = *model;
  obj.inputs.threshold.value = 0.5f;
  obj.inputs.threads.value = 1;

  VadTap tap;
  wire(obj, tap);

  test::drive_audio(obj, clip, block);
  std::vector<float> tail(static_cast<std::size_t>(sr), 0.f); // 1 s -> close segment
  test::drive_audio(obj, tail, block);

  INFO(
      "starts: " << tap.starts << ", segments: " << tap.segments
                 << ", first_start: " << tap.first_start
                 << "s, total_speech: " << tap.total_speech << "s");
  REQUIRE(tap.starts >= 1);
  REQUIRE(tap.segments >= 1);
  // Speech must begin after the injected lead silence, and within the clip.
  CHECK(tap.first_start >= 0.2f);
  CHECK(tap.first_start < clip_dur);
  // A real speech span was captured (not a 0-length blip).
  CHECK(tap.total_speech > 0.2f);
}

TEST_CASE("VAD reports no speech for pure silence (no false positives)",
          "[vad][workflow]")
{
  if(!test::loader_ready())
    SHERPA_SKIP("sherpa-onnx runtime library not available");
  auto model = test::env("SHERPA_TEST_VAD_MODEL");
  if(!model)
    SHERPA_SKIP("set SHERPA_TEST_VAD_MODEL");

  const int sr = 16000;
  Vad obj;
  test::wire_sync_worker(obj);
  const int block = 1600;
  obj.prepare(halp::setup{
      .input_channels = 1, .output_channels = 0, .frames = block, .rate = (double)sr});
  obj.inputs.model.value = *model;
  obj.inputs.threshold.value = 0.5f;
  obj.inputs.threads.value = 1;

  VadTap tap;
  wire(obj, tap);

  std::vector<float> silence(static_cast<std::size_t>(sr * 3), 0.f); // 3 s
  test::drive_audio(obj, silence, block);

  INFO("starts: " << tap.starts << ", segments: " << tap.segments);
  CHECK(tap.starts == 0);
  CHECK(tap.segments == 0);
}
