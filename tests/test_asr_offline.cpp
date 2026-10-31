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
    cb(obj); // the closure only stashes the result now...
    obj(0);  // ...outputs are emitted from operator() (the run() cycle); flush it
    INFO("recognized: " << obj.outputs.text.value);
    REQUIRE_FALSE(obj.outputs.text.value.empty());
  }
}

// HLG decoding for an offline CTC model, through the Advanced port: with a
// graph attached the decoder may only emit words the graph contains.
//
// Requires (not run in CI: the model is ~380 MB):
//   SHERPA_TEST_ASR_HLG_MODEL = sherpa-onnx-zipformer-ctc-en-2023-10-02 (with HLG.fst)
TEST_CASE("Offline CTC decodes through an HLG graph", "[asr][offline][workflow]")
{
  if(!test::loader_ready())
    SHERPA_SKIP("sherpa-onnx runtime library not available");
  auto model = test::env("SHERPA_TEST_ASR_HLG_MODEL");
  if(!model)
    SHERPA_SKIP("set SHERPA_TEST_ASR_HLG_MODEL");

  const std::string dir = *model;
  int sr = 0;
  auto audio = test::load_wav_mono(dir + "/test_wavs/0.wav", sr);
  REQUIRE(sr > 0);

  auto job = std::make_shared<OfflineRecognizer::Job>();
  job->samples = audio;
  job->rate = sr;
  job->num_threads = 1;
  job->reload = true;
  job->want_model = dir;
  job->advanced = "ctc_graph = " + dir + "/HLG.fst";

  auto cb = OfflineRecognizer::worker::work(job);
  OfflineRecognizer obj;
  REQUIRE(cb);
  cb(obj);
  obj(0);
  INFO("recognized: " << obj.outputs.text.value);
  REQUIRE_FALSE(obj.outputs.text.value.empty());

  // The key does reach sherpa: a graph that does not exist fails the model.
  job->reload = true;
  job->advanced = "ctc_graph = " + dir + "/does-not-exist.fst";
  auto cb2 = OfflineRecognizer::worker::work(job);
  OfflineRecognizer obj2;
  cb2(obj2);
  obj2(0);
  CHECK(obj2.outputs.text.value.empty());
}
