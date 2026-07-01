// Workflow test for sherpa_tts: synthesize text and expect non-silent audio out.
//
// Requires:
//   SHERPA_TEST_TTS_MODEL = TTS model dir (vits / matcha / kokoro / kitten ...)
#include "TestHelpers.hpp"

#include "Tts.hpp"

#include <catch2/catch_all.hpp>

#include <cmath>
#include <vector>

using namespace sherpa;

TEST_CASE("TTS synthesizes non-silent audio from text", "[tts][workflow]")
{
  if(!test::loader_ready())
    SHERPA_SKIP("sherpa-onnx runtime library not available");
  auto model = test::env("SHERPA_TEST_TTS_MODEL");
  if(!model)
    SHERPA_SKIP("set SHERPA_TEST_TTS_MODEL");

  Tts obj;
  test::wire_sync_worker(obj);

  const int block = 512;
  const double rate = 48000.;
  obj.prepare(halp::setup{
      .input_channels = 0, .output_channels = 1, .frames = block, .rate = rate});
  obj.inputs.model.value = *model;
  obj.inputs.speaker.value = 0;
  obj.inputs.speed.value = 1.f;
  obj.inputs.text.value = "Hello world, this is a test.";

  // First tick synthesizes (synchronous worker) and starts playback; the rest
  // drain the rendered buffer into the output bus. Sum the output energy.
  double energy = 0.;
  std::vector<float> outbuf(static_cast<std::size_t>(block), 0.f);
  const int max_blocks = 2000; // up to ~21 s at 48 kHz
  for(int k = 0; k < max_blocks; ++k)
  {
    std::fill(outbuf.begin(), outbuf.end(), 0.f);
    float* och[1]{outbuf.data()};
    obj.outputs.audio.samples = och;
    obj(block);
    for(float s : outbuf)
      energy += std::fabs(s);
  }

  INFO("output energy: " << energy);
  REQUIRE(energy > 0.0);
}
