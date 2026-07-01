// Workflow test for sherpa_asr_offline: record an utterance, decode in one shot,
// expect non-empty transcribed text.
//
// Requires the runtime library and:
//   SHERPA_TEST_ASR_OFFLINE_MODEL = model directory (transducer / paraformer /
//                                   whisper / sense-voice / ctc ...)
//   SHERPA_TEST_WAV               = a 16 kHz mono speech wav
#include "TestHelpers.hpp"

#include "OfflineRecognizer.hpp"

#include <catch2/catch_all.hpp>

using namespace sherpa;

TEST_CASE("Offline ASR transcribes a spoken utterance", "[asr][offline][workflow]")
{
  if(!test::loader_ready())
    SHERPA_SKIP("sherpa-onnx runtime library not available");
  auto model = test::env("SHERPA_TEST_ASR_OFFLINE_MODEL");
  auto wav = test::env("SHERPA_TEST_WAV");
  if(!model || !wav)
    SHERPA_SKIP("set SHERPA_TEST_ASR_OFFLINE_MODEL and SHERPA_TEST_WAV");

  int sr = 0;
  auto audio = test::load_wav_mono(*wav, sr);
  REQUIRE(sr > 0);
  REQUIRE_FALSE(audio.empty());

  const int block = 1600;

  SECTION("through the object (record -> operator() -> text)")
  {
    OfflineRecognizer obj;
    test::wire_sync_worker(obj);
    obj.prepare(halp::setup{
        .input_channels = 1, .output_channels = 1, .frames = block, .rate = (double)sr});
    obj.inputs.model.value = *model;
    obj.inputs.threads.value = 1;

    obj.inputs.record.value = true;
    test::drive_audio(obj, audio, block);
    obj.inputs.record.value = false;
    test::drive_silence(obj, block); // falling edge -> dispatch (synchronous)

    INFO("recognized: " << obj.outputs.text.value);
    REQUIRE_FALSE(obj.outputs.text.value.empty());
  }

  SECTION("through the worker job directly")
  {
    auto job = std::make_shared<OfflineRecognizer::Job>();
    job->samples = audio;
    job->rate = sr;
    job->num_threads = 1;
    job->reload = true;
    job->want_model = *model;

    auto cb = OfflineRecognizer::worker::work(job);
    OfflineRecognizer obj;
    REQUIRE(cb);
    cb(obj);
    INFO("recognized: " << obj.outputs.text.value);
    REQUIRE_FALSE(obj.outputs.text.value.empty());
  }
}
