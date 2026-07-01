// End-to-end semantic round-trip: text -> TTS -> audio -> ASR -> text.
//
// Unlike the per-object workflow tests (which assert only that *some* output is
// produced), this validates CONTENT: a known phrase is synthesized to real audio
// by sherpa_tts, that audio is fed through sherpa_asr_offline, and the recognised
// transcript is required to contain most of the original words. Random/garbage
// output cannot pass a word-overlap threshold, so this checks the two pipelines
// and their audio I/O actually agree.
//
// Requires the runtime library and:
//   SHERPA_TEST_TTS_MODEL         = an English TTS model dir (vits / piper ...)
//   SHERPA_TEST_ASR_OFFLINE_MODEL = an English non-streaming ASR dir (whisper ...)
#include "TestHelpers.hpp"

#include "OfflineRecognizer.hpp"
#include "Tts.hpp"

#include <catch2/catch_all.hpp>

#include <cmath>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

using namespace sherpa;

namespace
{
// Lowercase, keep [a-z0-9], split on everything else.
std::vector<std::string> words(std::string_view s)
{
  std::vector<std::string> out;
  std::string cur;
  for(char c : s)
  {
    if(c >= 'A' && c <= 'Z')
      c = static_cast<char>(c - 'A' + 'a');
    if((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9'))
      cur += c;
    else if(!cur.empty())
    {
      out.push_back(cur);
      cur.clear();
    }
  }
  if(!cur.empty())
    out.push_back(cur);
  return out;
}
}

TEST_CASE("Round-trip: text -> TTS -> ASR recovers the spoken words",
          "[roundtrip][workflow]")
{
  if(!test::loader_ready())
    SHERPA_SKIP("sherpa-onnx runtime library not available");
  auto tts_model = test::env("SHERPA_TEST_TTS_MODEL");
  auto asr_model = test::env("SHERPA_TEST_ASR_OFFLINE_MODEL");
  if(!tts_model || !asr_model)
    SHERPA_SKIP("set SHERPA_TEST_TTS_MODEL and SHERPA_TEST_ASR_OFFLINE_MODEL");

  const std::string phrase = "The quick brown fox jumps over the lazy dog.";

  // ---- 1. Synthesize the phrase to real host-rate mono audio ---------------
  const int block = 512;
  const double rate = 16000.; // keep it at the ASR working rate to avoid resamples
  Tts tts;
  test::wire_sync_worker(tts);
  tts.prepare(halp::setup{
      .input_channels = 0, .output_channels = 1, .frames = block, .rate = rate});
  tts.inputs.model.value = *tts_model;
  tts.inputs.speaker.value = 0;
  tts.inputs.speed.value = 1.f;
  tts.inputs.text.value = phrase;

  std::vector<float> speech;
  speech.reserve(static_cast<std::size_t>(rate * 12));
  std::vector<float> outbuf(static_cast<std::size_t>(block), 0.f);
  double energy = 0.;
  const int max_blocks = static_cast<int>(rate * 12 / block); // up to 12 s
  for(int k = 0; k < max_blocks; ++k)
  {
    std::fill(outbuf.begin(), outbuf.end(), 0.f);
    float* och[1]{outbuf.data()};
    tts.outputs.audio.samples = och;
    tts(block);
    for(float s : outbuf)
    {
      speech.push_back(s);
      energy += std::fabs(s);
    }
  }
  INFO("TTS output energy: " << energy << ", samples: " << speech.size());
  REQUIRE(energy > 0.0); // TTS actually produced sound

  // ---- 2. Recognize the synthesized audio ----------------------------------
  OfflineRecognizer asr;
  test::wire_sync_worker(asr);
  asr.prepare(halp::setup{
      .input_channels = 1, .output_channels = 1, .frames = block, .rate = rate});
  asr.inputs.model.value = *asr_model;
  asr.inputs.threads.value = 1;

  asr.inputs.record.value = true;
  test::drive_audio(asr, speech, block);
  asr.inputs.record.value = false;
  test::drive_silence(asr, block); // falling edge -> synchronous decode

  const std::string recognized = asr.outputs.text.value;
  INFO("phrase:     " << phrase);
  INFO("recognized: " << recognized);
  REQUIRE_FALSE(recognized.empty());

  // ---- 3. Assert the transcript recovered most of the content words --------
  const std::unordered_set<std::string> stop{"the", "over", "a", "an"};
  std::vector<std::string> content;
  for(auto& w : words(phrase))
    if(!stop.contains(w))
      content.push_back(w); // quick brown fox jumps lazy dog

  std::unordered_set<std::string> got;
  for(auto& w : words(recognized))
    got.insert(w);

  int hits = 0;
  for(auto& w : content)
    if(got.contains(w))
      ++hits;

  INFO("content words matched: " << hits << " / " << content.size());
  // A real pipeline recovers the clear majority; garbage cannot reach half.
  REQUIRE(hits * 2 >= static_cast<int>(content.size()));
}
