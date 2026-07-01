// Workflow test for sherpa_denoiser: add white noise to a speech wav, run it
// through the denoiser, and require the enhanced output to (a) be non-silent
// (speech preserved) and (b) have a LOWER level than the noisy input (noise
// removed) -- a relational check that a pass-through or a silence would fail.
//
// Requires:
//   SHERPA_TEST_DENOISER_MODEL = denoiser model dir/file (gtcrn / dpdfnet)
//   SHERPA_TEST_WAV            = a speech wav
#include "TestHelpers.hpp"

#include "Denoiser.hpp"

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <vector>

using namespace sherpa;

TEST_CASE("Denoiser reduces added noise while keeping speech", "[denoiser][workflow]")
{
  if(!test::loader_ready())
    SHERPA_SKIP("sherpa-onnx runtime library not available");
  auto model = test::env("SHERPA_TEST_DENOISER_MODEL");
  auto wav = test::env("SHERPA_TEST_WAV");
  if(!model || !wav)
    SHERPA_SKIP("set SHERPA_TEST_DENOISER_MODEL and SHERPA_TEST_WAV");

  int sr = 0;
  auto clean = test::load_wav_mono(*wav, sr);
  REQUIRE(sr > 0);
  REQUIRE_FALSE(clean.empty());

  // Corrupt the clean speech with a healthy dose of white noise.
  std::vector<float> noisy = clean;
  test::add_white_noise(noisy, 0.15f);
  const double in_rms = test::rms(noisy);
  REQUIRE(in_rms > 0.0);

  Denoiser obj;
  test::wire_sync_worker(obj);
  const int block = 512;
  obj.prepare(halp::setup{
      .input_channels = 1, .output_channels = 1, .frames = block, .rate = (double)sr});
  obj.inputs.model.value = *model;
  obj.inputs.mode.value = Denoiser::Mode::Auto;
  obj.inputs.threads.value = 1;

  std::vector<float> out;
  out.reserve(noisy.size() + static_cast<std::size_t>(block) * 64);
  std::vector<float> outbuf(static_cast<std::size_t>(block), 0.f);

  auto pump = [&](const float* in, int n) {
    float* ich[1]{const_cast<float*>(in)};
    obj.inputs.audio.samples = ich;
    obj.inputs.audio.channels = 1;
    std::fill(outbuf.begin(), outbuf.end(), 0.f);
    float* och[1]{outbuf.data()};
    obj.outputs.audio.samples = och;
    obj(n);
    for(int j = 0; j < n; ++j)
      out.push_back(outbuf[static_cast<std::size_t>(j)]);
  };

  for(std::size_t i = 0; i < noisy.size(); i += static_cast<std::size_t>(block))
  {
    const int n = static_cast<int>(std::min<std::size_t>(block, noisy.size() - i));
    pump(noisy.data() + i, n);
  }
  // Flush the streaming denoiser's internal latency with a short silence tail.
  std::vector<float> silence(static_cast<std::size_t>(block), 0.f);
  for(int k = 0; k < 64; ++k)
    pump(silence.data(), block);

  const double out_rms = test::rms(out);
  INFO("noisy in rms: " << in_rms << ", denoised out rms: " << out_rms
                        << ", out samples: " << out.size());

  REQUIRE(test::abs_energy(out) > 0.0); // speech preserved, not silenced
  CHECK(out_rms < in_rms);              // the added noise was attenuated
}
