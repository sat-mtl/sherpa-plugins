// Workflow test for sherpa_keyword_spotter: stream audio in blocks and, if the
// configured keyword occurs, expect the "Keyword" callback to fire with a
// non-empty string. A random wav may not contain the keyword, so absent a
// trigger the test only asserts the pipeline ran without error.
//
// Requires:
//   SHERPA_TEST_KWS_MODEL = streaming KWS transducer model dir (encoder/decoder/
//                           joiner + tokens.txt)
//   SHERPA_TEST_WAV       = a 16 kHz mono wav
// Optional:
//   SHERPA_TEST_KEYWORDS  = inline keywords (tokenized, one per line). When set,
//                           they are fed via the "Keywords" line-edit.
#include "TestHelpers.hpp"

#include "KeywordSpotter.hpp"

#include <catch2/catch_all.hpp>

#include <string>
#include <string_view>

using namespace sherpa;

TEST_CASE("Keyword spotter runs over streamed audio", "[kws][online][workflow]")
{
  if(!test::loader_ready())
    SHERPA_SKIP("sherpa-onnx runtime library not available");
  if(!SherpaLoader::instance().SherpaOnnxCreateKeywordSpotter)
    SHERPA_SKIP("keyword spotting symbols not present in this runtime");

  auto model = test::env("SHERPA_TEST_KWS_MODEL");
  auto wav = test::env("SHERPA_TEST_WAV");
  if(!model || !wav)
    SHERPA_SKIP("set SHERPA_TEST_KWS_MODEL and SHERPA_TEST_WAV");

  int sr = 0;
  auto audio = test::load_wav_mono(*wav, sr);
  REQUIRE(sr > 0);
  REQUIRE_FALSE(audio.empty());

  KeywordSpotter obj;
  test::wire_sync_worker(obj);

  const int block = 3200; // 0.2 s at 16 kHz
  obj.prepare(halp::setup{
      .input_channels = 1, .output_channels = 1, .frames = block, .rate = (double)sr});
  obj.inputs.model.value = *model;
  obj.inputs.threads.value = 1;
  obj.inputs.threshold.value = 0.25f;
  if(auto kw = test::env("SHERPA_TEST_KEYWORDS"))
    obj.inputs.keywords.value = *kw;

  std::string detected_keyword;
  int detections = 0;
  obj.outputs.keyword.call.context = &detected_keyword;
  obj.outputs.keyword.call.function
      = +[](void* c, std::string_view sv) { *static_cast<std::string*>(c) = std::string(sv); };
  obj.outputs.detected.call.context = &detections;
  obj.outputs.detected.call.function = +[](void* c) { (*static_cast<int*>(c))++; };

  test::drive_audio(obj, audio, block);
  // Trailing silence to flush the tail.
  std::vector<float> tail(static_cast<std::size_t>(sr), 0.f); // 1 s
  test::drive_audio(obj, tail, block);

  INFO("detections: " << detections << ", last keyword: " << detected_keyword);
  if(detections > 0)
  {
    // A keyword actually triggered: the reported string must be non-empty.
    CHECK_FALSE(detected_keyword.empty());
  }
  else
  {
    // No keyword in this wav: the pipeline just needs to have run cleanly.
    SUCCEED("ran without detecting a keyword");
  }
}
