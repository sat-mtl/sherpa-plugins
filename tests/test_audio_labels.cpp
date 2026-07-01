// Workflow test for sherpa_audio_labels: record a clip and expect either audio
// tagging or spoken language identification to produce a label.
//
// Requires:
//   SHERPA_TEST_AUDIO_LABELS_MODEL = an audio-tagging model dir (zipformer/ced +
//                                    labels csv) OR a whisper dir for language id
//   SHERPA_TEST_WAV                = any mono wav (resampled internally)
#include "TestHelpers.hpp"

#include "AudioLabels.hpp"

#include <catch2/catch_all.hpp>

#include <string>

using namespace sherpa;

TEST_CASE("AudioLabels classifies a recorded clip", "[audio_labels][workflow]")
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

  // Capture the "Top" and "Language" callbacks so a fired result also counts.
  int top_hits = 0, lang_hits = 0;
  obj.outputs.top.call.context = &top_hits;
  obj.outputs.top.call.function
      = +[](void* c, std::string_view, float) { (*static_cast<int*>(c))++; };
  obj.outputs.language.call.context = &lang_hits;
  obj.outputs.language.call.function
      = +[](void* c, std::string_view) { (*static_cast<int*>(c))++; };

  // Listen on -> drive the clip -> Listen off -> one silent block triggers the
  // falling edge and dispatches the accumulated audio to the worker.
  obj.inputs.listen.value = true;
  test::drive_audio(obj, audio, block);
  obj.inputs.listen.value = false;
  test::drive_silence(obj, block);

  INFO(
      "labels: " << obj.outputs.labels.value.size() << ", top hits: " << top_hits
                 << ", lang hits: " << lang_hits);
  REQUIRE(
      (!obj.outputs.labels.value.empty() || top_hits >= 1 || lang_hits >= 1));
}
