// Workflow test for sherpa_audio_labels: tag a clip of speech and expect a
// speech-family label near the top (content check, not just "some label"). The
// AudioSet ontology used by the tagging models has "Speech", "Narration,
// monologue", "Conversation", etc.; a speech clip must surface one of them.
//
// Requires:
//   SHERPA_TEST_AUDIO_LABELS_MODEL = an audio-tagging model dir (zipformer/ced +
//                                    labels csv) OR a whisper dir for language id
//   SHERPA_TEST_WAV                = a speech wav (resampled internally)
#include "TestHelpers.hpp"

#include "AudioLabels.hpp"

#include <catch2/catch_all.hpp>

#include <string>
#include <string_view>

using namespace sherpa;

TEST_CASE("AudioLabels tags speech with a speech-family label",
          "[audio_labels][workflow]")
{
  if(!test::loader_ready())
    SHERPA_SKIP("sherpa-onnx runtime library not available");
  auto model = test::env("SHERPA_TEST_AUDIO_LABELS_MODEL");
  auto wav = test::env("SHERPA_TEST_WAV");
  if(!model || !wav)
    SHERPA_SKIP("set SHERPA_TEST_AUDIO_LABELS_MODEL and SHERPA_TEST_WAV");

  int sr = 0;
  auto audio = test::load_wav_mono(*wav, sr);
  REQUIRE(sr > 0);
  REQUIRE_FALSE(audio.empty());

  AudioLabels obj;
  test::wire_sync_worker(obj);

  const int block = 1600;
  obj.prepare(halp::setup{
      .input_channels = 1, .output_channels = 0, .frames = block, .rate = (double)sr});
  obj.inputs.model.value = *model;
  obj.inputs.type.value = AudioLabels::Type::Auto;
  obj.inputs.top_k.value = 5;
  obj.inputs.threads.value = 1;

  std::string top_label;
  float top_score = -1.f;
  struct Top { std::string* n; float* s; } cap{&top_label, &top_score};
  obj.outputs.top.call.context = &cap;
  obj.outputs.top.call.function = +[](void* c, std::string_view n, float s) {
    auto* t = static_cast<Top*>(c);
    *t->n = std::string(n);
    *t->s = s;
  };
  std::string language;
  obj.outputs.language.call.context = &language;
  obj.outputs.language.call.function = +[](void* c, std::string_view sv) {
    *static_cast<std::string*>(c) = std::string(sv);
  };

  obj.inputs.listen.value = true;
  test::drive_audio(obj, audio, block);
  obj.inputs.listen.value = false;
  test::drive_silence(obj, block); // falling edge -> dispatch

  const auto& labels = obj.outputs.labels.value;
  INFO("labels: " << labels.size() << ", top: '" << top_label << "' ("
                  << top_score << "), language: '" << language << "'");

  // Language-ID branch (whisper model): a language code was produced.
  if(!language.empty())
  {
    SUCCEED("spoken-language-id produced a language: " + language);
    return;
  }

  // Tagging branch: labels must be present, sorted by descending score, and a
  // speech-family label must appear for a speech clip.
  REQUIRE_FALSE(labels.empty());
  for(std::size_t i = 1; i < labels.size(); ++i)
    CHECK(labels[i - 1].score >= labels[i].score - 1e-6f); // non-increasing
  CHECK(labels.front().score > 0.f);

  bool speechy = false;
  for(const auto& l : labels)
    if(test::icontains(l.name, "speech") || test::icontains(l.name, "narration")
       || test::icontains(l.name, "conversation")
       || test::icontains(l.name, "speak"))
      speechy = true;
  INFO("top-" << labels.size() << " labels checked for a speech family match");
  CHECK(speechy);
}
