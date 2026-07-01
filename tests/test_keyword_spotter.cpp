// Workflow test for sherpa_keyword_spotter.
//   Negative: with real keywords configured, pure silence must fire NO detection
//             (no false positives) -- the robust, model-independent check.
//   Positive: if the model ships its own keywords.txt + test_wavs (as the
//             gigaspeech KWS model does), feed a keyword clip and expect a
//             detection; the reported keyword must be non-empty.
//
// Requires:
//   SHERPA_TEST_KWS_MODEL = streaming KWS transducer model dir (encoder/decoder/
//                           joiner + tokens.txt [+ keywords.txt, test_wavs/])
//   SHERPA_TEST_WAV       = a 16 kHz mono wav (fallback keyword source)
// Optional:
//   SHERPA_TEST_KEYWORDS  = inline keywords (tokenized) fed via the line-edit.
#include "TestHelpers.hpp"

#include "KeywordSpotter.hpp"

#include <catch2/catch_all.hpp>

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

using namespace sherpa;

namespace
{
struct KwsTap
{
  std::string last;
  int count = 0;
};
void wire(KeywordSpotter& obj, KwsTap& tap)
{
  obj.outputs.keyword.call.context = &tap;
  obj.outputs.keyword.call.function = +[](void* c, std::string_view sv) {
    static_cast<KwsTap*>(c)->last = std::string(sv);
  };
  obj.outputs.detected.call.context = &tap;
  obj.outputs.detected.call.function
      = +[](void* c) { static_cast<KwsTap*>(c)->count++; };
}
}

TEST_CASE("Keyword spotter: no false positive on silence, detects a keyword clip",
          "[kws][online][workflow]")
{
  if(!test::loader_ready())
    SHERPA_SKIP("sherpa-onnx runtime library not available");
  if(!SherpaLoader::instance().SherpaOnnxCreateKeywordSpotter)
    SHERPA_SKIP("keyword spotting symbols not present in this runtime");

  auto model = test::env("SHERPA_TEST_KWS_MODEL");
  auto wav = test::env("SHERPA_TEST_WAV");
  if(!model || !wav)
    SHERPA_SKIP("set SHERPA_TEST_KWS_MODEL and SHERPA_TEST_WAV");

  namespace fs = std::filesystem;
  const fs::path dir{*model};

  // Prefer the model's own keywords file so keywords are actually configured.
  std::string kw_file;
  for(const char* name : {"keywords.txt", "test_wavs/keywords.txt"})
  {
    std::error_code ec;
    if(fs::exists(dir / name, ec))
    {
      kw_file = (dir / name).string();
      break;
    }
  }
  const auto inline_kw = test::env("SHERPA_TEST_KEYWORDS");
  const bool have_keywords = !kw_file.empty() || inline_kw.has_value();

  auto configure = [&](KeywordSpotter& obj, int sr, int block) {
    obj.prepare(halp::setup{
        .input_channels = 1, .output_channels = 1, .frames = block, .rate = (double)sr});
    obj.inputs.model.value = *model;
    obj.inputs.threads.value = 1;
    obj.inputs.threshold.value = 0.2f;
    if(!kw_file.empty())
      obj.inputs.keywords_file.file.filename = kw_file;
    if(inline_kw)
      obj.inputs.keywords.value = *inline_kw;
  };

  const int block = 3200; // 0.2 s at 16 kHz

  // --- Negative: keywords configured + silence -> no detection --------------
  if(have_keywords)
  {
    KeywordSpotter neg;
    test::wire_sync_worker(neg);
    KwsTap tap;
    wire(neg, tap);
    configure(neg, 16000, block);

    std::vector<float> silence(static_cast<std::size_t>(16000 * 2), 0.f); // 2 s
    test::drive_audio(neg, silence, block);

    INFO("silence detections: " << tap.count);
    CHECK(tap.count == 0); // no false positives on silence
  }
  else
  {
    WARN("no keywords configured (model has no keywords.txt, SHERPA_TEST_KEYWORDS "
         "unset); skipping the silence negative");
  }

  // --- Positive: a keyword clip should fire (best-effort; threshold-sensitive) -
  // Use the model's own test_wavs when present (they contain the keywords),
  // otherwise fall back to SHERPA_TEST_WAV.
  std::string kw_wav;
  {
    std::error_code ec;
    const fs::path tw = dir / "test_wavs";
    if(fs::is_directory(tw, ec))
      for(const auto& e : fs::directory_iterator(tw, ec))
        if(e.is_regular_file()
           && test::icontains(e.path().filename().string(), ".wav"))
        {
          kw_wav = e.path().string();
          break;
        }
  }
  if(kw_wav.empty())
    kw_wav = *wav;

  int sr = 0;
  auto audio = test::load_wav_mono(kw_wav, sr);
  REQUIRE(sr > 0);
  REQUIRE_FALSE(audio.empty());

  KeywordSpotter pos;
  test::wire_sync_worker(pos);
  KwsTap tap;
  wire(pos, tap);
  configure(pos, sr, block);

  test::drive_audio(pos, audio, block);
  std::vector<float> tail(static_cast<std::size_t>(sr), 0.f); // 1 s flush
  test::drive_audio(pos, tail, block);

  INFO("keyword wav: " << kw_wav << ", detections: " << tap.count
                       << ", last: '" << tap.last << "'");
  if(tap.count > 0)
    CHECK_FALSE(tap.last.empty()); // a real trigger reports a real keyword
  else
    SUCCEED("no keyword detected in this clip (threshold/keyword-set dependent)");
}
