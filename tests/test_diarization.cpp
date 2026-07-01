// Workflow tests for sherpa_diarization.
//   Positive: a spoken passage yields segments that satisfy the diarization
//             invariants -- times within the clip, sorted, non-negative speaker
//             ids, and one callback per segment.
//   Negative: pure silence yields NO segments (no hallucinated speakers).
//
// Requires:
//   SHERPA_TEST_DIARIZATION_MODEL = a dir holding BOTH a pyannote segmentation
//                                   .onnx and a speaker embedding .onnx
//   SHERPA_TEST_WAV               = a speech wav (any rate; resampled internally)
#include "TestHelpers.hpp"

#include "Diarization.hpp"

#include <catch2/catch_all.hpp>

#include <vector>

using namespace sherpa;

TEST_CASE("Diarization segments a passage with valid, ordered segments",
          "[diarization][offline][workflow]")
{
  if(!test::loader_ready())
    SHERPA_SKIP("sherpa-onnx runtime library not available");
  auto model = test::env("SHERPA_TEST_DIARIZATION_MODEL");
  auto wav = test::env("SHERPA_TEST_WAV");
  if(!model || !wav)
    SHERPA_SKIP("set SHERPA_TEST_DIARIZATION_MODEL and SHERPA_TEST_WAV");

  int sr = 0;
  auto audio = test::load_wav_mono(*wav, sr);
  REQUIRE(sr > 0);
  REQUIRE_FALSE(audio.empty());
  const float duration = static_cast<float>(audio.size()) / static_cast<float>(sr);

  const int block = 1600;
  Diarization obj;
  test::wire_sync_worker(obj);
  obj.prepare(halp::setup{
      .input_channels = 1, .output_channels = 0, .frames = block, .rate = (double)sr});
  obj.inputs.model.value = *model;
  obj.inputs.num_speakers.value = -1; // auto
  obj.inputs.cluster_threshold.value = 0.5f;
  obj.inputs.threads.value = 1;

  int fired = 0;
  obj.outputs.on_segment.call.context = &fired;
  obj.outputs.on_segment.call.function
      = +[](void* c, float, float, int) { (*static_cast<int*>(c))++; };

  obj.inputs.record.value = true;
  test::drive_audio(obj, audio, block);
  obj.inputs.record.value = false;
  test::drive_silence(obj, block); // falling edge -> dispatch

  const auto& segs = obj.outputs.segments.value;
  INFO("num speakers: " << obj.outputs.num_speakers_out.value
                        << ", segments: " << segs.size()
                        << ", duration: " << duration << "s");
  REQUIRE_FALSE(segs.empty());
  REQUIRE(obj.outputs.num_speakers_out.value >= 1);
  CHECK(fired == static_cast<int>(segs.size()));
  CHECK(obj.outputs.progress.value == 1.f);

  // Diarization invariants: each segment is a valid, in-bounds interval with a
  // non-negative speaker id; segments are sorted by start time (the object sorts).
  const float tol = 0.75f; // model resampling / padding slack
  int max_speaker = -1;
  float prev_start = -1.f;
  for(const auto& s : segs)
  {
    CHECK(s.start >= 0.f);
    CHECK(s.end >= s.start);
    CHECK(s.end <= duration + tol);
    CHECK(s.speaker >= 0);
    CHECK(s.start >= prev_start - 1e-4f); // non-decreasing start
    prev_start = s.start;
    if(s.speaker > max_speaker)
      max_speaker = s.speaker;
  }
  // Speaker ids are dense 0..num_speakers-1.
  CHECK(max_speaker < obj.outputs.num_speakers_out.value);
}

TEST_CASE("Diarization finds no speakers in pure silence",
          "[diarization][offline][workflow]")
{
  if(!test::loader_ready())
    SHERPA_SKIP("sherpa-onnx runtime library not available");
  auto model = test::env("SHERPA_TEST_DIARIZATION_MODEL");
  if(!model)
    SHERPA_SKIP("set SHERPA_TEST_DIARIZATION_MODEL");

  const int sr = 16000;
  const int block = 1600;
  Diarization obj;
  test::wire_sync_worker(obj);
  obj.prepare(halp::setup{
      .input_channels = 1, .output_channels = 0, .frames = block, .rate = (double)sr});
  obj.inputs.model.value = *model;
  obj.inputs.num_speakers.value = -1;
  obj.inputs.cluster_threshold.value = 0.5f;
  obj.inputs.threads.value = 1;

  obj.inputs.record.value = true;
  std::vector<float> silence(static_cast<std::size_t>(sr * 3), 0.f); // 3 s
  test::drive_audio(obj, silence, block);
  obj.inputs.record.value = false;
  test::drive_silence(obj, block);

  INFO("silence -> segments: " << obj.outputs.segments.value.size()
                               << ", speakers: " << obj.outputs.num_speakers_out.value);
  CHECK(obj.outputs.segments.value.empty());
  CHECK(obj.outputs.num_speakers_out.value == 0);
}
