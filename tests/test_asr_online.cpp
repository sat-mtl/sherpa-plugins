// Workflow test for sherpa_asr_online: stream a spoken utterance in blocks and
// expect a non-empty hypothesis (partial and/or final on endpoint).
//
// Requires:
//   SHERPA_TEST_ASR_ONLINE_MODEL = streaming model dir (transducer / paraformer /
//                                  zipformer2-ctc ...)
//   SHERPA_TEST_WAV              = a 16 kHz mono speech wav
#include "TestHelpers.hpp"

#include "OnlineRecognizer.hpp"

#include <catch2/catch_all.hpp>

#include <string>
#include <string_view>

using namespace sherpa;

TEST_CASE("Streaming ASR emits a hypothesis for a spoken utterance",
          "[asr][online][workflow]")
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

  OnlineRecognizer obj;
  test::wire_sync_worker(obj);

  const int block = 3200; // 0.2 s at 16 kHz
  obj.prepare(halp::setup{
      .input_channels = 1, .output_channels = 1, .frames = block, .rate = (double)sr});
  obj.inputs.model.value = *model;
  obj.inputs.threads.value = 1;
  obj.inputs.endpointing.value = true;

  std::string last_partial, final_text;
  obj.outputs.partial.call.context = &last_partial;
  obj.outputs.partial.call.function
      = +[](void* c, std::string_view sv) { *static_cast<std::string*>(c) = std::string(sv); };
  obj.outputs.final_text.call.context = &final_text;
  obj.outputs.final_text.call.function
      = +[](void* c, std::string_view sv) { *static_cast<std::string*>(c) = std::string(sv); };

  test::drive_audio(obj, audio, block);
  // Trailing silence to flush the tail / trigger an endpoint.
  std::vector<float> tail(static_cast<std::size_t>(sr), 0.f); // 1 s
  test::drive_audio(obj, tail, block);

  INFO("partial: " << last_partial);
  INFO("final:   " << final_text);
  const bool got_text = !last_partial.empty() || !final_text.empty();
  REQUIRE(got_text);
}
