// Workflow test for sherpa_denoiser: stream a speech wav through the denoiser and
// expect non-silent enhanced audio out.
//
// Requires:
//   SHERPA_TEST_DENOISER_MODEL = denoiser model dir/file (gtcrn / dpdfnet)
//   SHERPA_TEST_WAV            = a speech wav
#include "TestHelpers.hpp"

#include "Denoiser.hpp"

#include <catch2/catch_all.hpp>

#include <cmath>
#include <vector>

using namespace sherpa;

TEST_CASE("Denoiser enhances a speech wav to non-silent audio", "[denoiser][workflow]")
{
  if(!test::loader_ready())
    SHERPA_SKIP("sherpa-onnx runtime library not available");
  auto model = test::env("SHERPA_TEST_DENOISER_MODEL");
  auto wav = test::env("SHERPA_TEST_WAV");
  if(!model || !wav)
    SHERPA_SKIP("set SHERPA_TEST_DENOISER_MODEL and SHERPA_TEST_WAV");

  int sr = 0;
  auto audio = test::load_wav_mono(*wav, sr);
  REQUIRE(sr > 0);
  REQUIRE_FALSE(audio.empty());

  Denoiser obj;
  test::wire_sync_worker(obj);

  const int block = 512;
  obj.prepare(halp::setup{
      .input_channels = 1, .output_channels = 1, .frames = block, .rate = (double)sr});
  obj.inputs.model.value = *model;
  obj.inputs.mode.value = Denoiser::Mode::Auto; // prefer the streaming path
  obj.inputs.threads.value = 1;

  // Drive the wav block by block, feeding the input bus and collecting the
  // enhanced output-bus energy on each block (the synchronous worker fills the
  // playback queue during the same tick that operator() then drains).
  double energy = 0.;
  std::vector<float> outbuf(static_cast<std::size_t>(block), 0.f);
  for(std::size_t i = 0; i < audio.size(); i += static_cast<std::size_t>(block))
  {
    const int n
        = static_cast<int>(std::min<std::size_t>(block, audio.size() - i));

    float* ich[1]{const_cast<float*>(audio.data() + i)};
    obj.inputs.audio.samples = ich;
    obj.inputs.audio.channels = 1;

    std::fill(outbuf.begin(), outbuf.end(), 0.f);
    float* och[1]{outbuf.data()};
    obj.outputs.audio.samples = och;

    obj(n);

    for(int j = 0; j < n; ++j)
      energy += std::fabs(outbuf[static_cast<std::size_t>(j)]);
  }

  // Drain any output still buffered by the streaming denoiser with a short
  // silence tail so the queue is emptied into the output bus.
  std::vector<float> silence(static_cast<std::size_t>(block), 0.f);
  for(int k = 0; k < 200; ++k)
  {
    float* ich[1]{silence.data()};
    obj.inputs.audio.samples = ich;
    obj.inputs.audio.channels = 1;

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
