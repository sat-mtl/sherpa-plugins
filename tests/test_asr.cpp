// Workflow test for the unified sherpa_asr object. Exercises both engines:
//   * Streaming: a streaming model finalizes via endpointing.
//   * Batch:     a non-streaming model + an internal VAD segments the live stream
//                and finalizes each detected utterance.
//
// Requires the runtime library and (per section):
//   SHERPA_TEST_ASR_ONLINE_MODEL  = streaming model dir      (Streaming section)
//   SHERPA_TEST_ASR_OFFLINE_MODEL = non-streaming model dir  (Batch section)
//   SHERPA_TEST_VAD_MODEL         = silero/ten VAD model dir  (Batch section)
//   SHERPA_TEST_WAV               = a 16 kHz mono speech wav
#include "TestHelpers.hpp"

#include "Asr.hpp"

#include <catch2/catch_all.hpp>

#include <string>
#include <string_view>

using namespace sherpa;

namespace
{
// Capture the latest "Final" utterance emitted by the object.
void wire_final(Asr& obj, std::string& sink)
{
  obj.outputs.final_text.call.context = &sink;
  obj.outputs.final_text.call.function = +[](void* c, std::string_view sv) {
    *static_cast<std::string*>(c) = std::string(sv);
  };
}
}

TEST_CASE("Unified ASR: streaming model finalizes via endpointing",
          "[asr][unified][streaming][workflow]")
{
  if(!test::loader_ready())
    SHERPA_SKIP("sherpa-onnx runtime library not available");
  auto model = test::env("SHERPA_TEST_ASR_ONLINE_MODEL");
  auto wav = test::env("SHERPA_TEST_WAV");
  if(!model || !wav)
    SHERPA_SKIP("set SHERPA_TEST_ASR_ONLINE_MODEL and SHERPA_TEST_WAV");

  int sr = 0;
  auto audio = test::load_wav_mono(*wav, sr);
  REQUIRE(sr > 0);
  REQUIRE_FALSE(audio.empty());

  const int block = 3200; // 0.2 s at 16 kHz
  Asr obj;
  test::wire_sync_worker(obj);
  obj.prepare(halp::setup{
      .input_channels = 1, .output_channels = 1, .frames = block, .rate = (double)sr});
  obj.inputs.model.value = *model;
  obj.inputs.mode.value = Asr::Mode::Streaming;
  obj.inputs.endpointing.value = true;

  std::string partial, final_text;
  wire_final(obj, final_text);
  obj.outputs.partial.call.context = &partial;
  obj.outputs.partial.call.function = +[](void* c, std::string_view sv) {
    *static_cast<std::string*>(c) = std::string(sv);
  };

  test::drive_audio(obj, audio, block);
  std::vector<float> tail(static_cast<std::size_t>(sr), 0.f); // 1 s -> endpoint
  test::drive_audio(obj, tail, block);

  INFO("partial: " << partial);
  INFO("final:   " << final_text);
  REQUIRE((!partial.empty() || !final_text.empty()));
}

TEST_CASE("Unified ASR: non-streaming model finalizes via internal VAD",
          "[asr][unified][batch][workflow]")
{
  if(!test::loader_ready())
    SHERPA_SKIP("sherpa-onnx runtime library not available");
  auto model = test::env("SHERPA_TEST_ASR_OFFLINE_MODEL");
  auto vad = test::env("SHERPA_TEST_VAD_MODEL");
  auto wav = test::env("SHERPA_TEST_WAV");
  if(!model || !vad || !wav)
    SHERPA_SKIP("set SHERPA_TEST_ASR_OFFLINE_MODEL, SHERPA_TEST_VAD_MODEL, SHERPA_TEST_WAV");

  int sr = 0;
  auto audio = test::load_wav_mono(*wav, sr);
  REQUIRE(sr > 0);
  REQUIRE_FALSE(audio.empty());

  const int block = 1600;
  Asr obj;
  test::wire_sync_worker(obj);
  obj.prepare(halp::setup{
      .input_channels = 1, .output_channels = 1, .frames = block, .rate = (double)sr});
  obj.inputs.model.value = *model;
  obj.inputs.vad_model.value = *vad;
  obj.inputs.mode.value = Asr::Mode::Batch;

  std::string final_text;
  wire_final(obj, final_text);

  test::drive_audio(obj, audio, block);
  // Trailing silence so the VAD closes the last speech segment and it is decoded.
  std::vector<float> tail(static_cast<std::size_t>(sr), 0.f); // 1 s
  test::drive_audio(obj, tail, block);

  INFO("final: " << final_text);
  REQUIRE_FALSE(final_text.empty());
}
