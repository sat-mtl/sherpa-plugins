// Workflow test for sherpa_diarization: record a passage, diarize it in one shot,
// expect a non-empty list of {start, end, speaker} segments.
//
// Requires the runtime library and:
//   SHERPA_TEST_DIARIZATION_MODEL = a directory holding BOTH a pyannote
//                                   segmentation .onnx and a speaker embedding
//                                   .onnx
//   SHERPA_TEST_WAV               = a speech wav (any rate; resampled internally)
#include "TestHelpers.hpp"

#include "Diarization.hpp"

#include <catch2/catch_all.hpp>

using namespace sherpa;

TEST_CASE(
    "Diarization segments a spoken passage", "[diarization][offline][workflow]")
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
  test::drive_silence(obj, block); // falling edge -> dispatch (synchronous)

  INFO("num speakers: " << obj.outputs.num_speakers_out.value
                        << ", segments: " << obj.outputs.segments.value.size());
  REQUIRE_FALSE(obj.outputs.segments.value.empty());
  CHECK(fired == (int)obj.outputs.segments.value.size());
  CHECK(obj.outputs.progress.value == 1.f);
}
