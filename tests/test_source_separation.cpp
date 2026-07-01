// Workflow test for sherpa_source_separation (Spleeter 2-stems: vocals +
// accompaniment). Feeding a SPEECH clip, we expect exactly two stems, a
// non-silent vocals stem, and -- since the input is voice, not music -- the
// vocals stem to carry more energy than the accompaniment stem. That relational
// check verifies separation actually routed the voice, not just "some audio out".
//
// Requires:
//   SHERPA_TEST_SOURCESEP_MODEL = source-separation model dir (Spleeter / UVR)
//   SHERPA_TEST_WAV             = a speech wav (read mono, duplicated to stereo)
#include "TestHelpers.hpp"

#include "SourceSeparation.hpp"

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

using namespace sherpa;

TEST_CASE("Source separation yields two stems and routes speech to vocals",
          "[sourcesep][workflow]")
{
  if(!test::loader_ready())
    SHERPA_SKIP("sherpa-onnx runtime library not available");
  if(!SherpaLoader::instance().SherpaOnnxCreateOfflineSourceSeparation)
    SHERPA_SKIP("source-separation symbols not present in this build");

  auto model = test::env("SHERPA_TEST_SOURCESEP_MODEL");
  auto wavpath = test::env("SHERPA_TEST_WAV");
  if(!model || !wavpath)
    SHERPA_SKIP("set SHERPA_TEST_SOURCESEP_MODEL and SHERPA_TEST_WAV");

  int wav_rate = 0;
  auto mono = test::load_wav_mono(*wavpath, wav_rate);
  if(mono.empty() || wav_rate <= 0)
    SHERPA_SKIP("could not read SHERPA_TEST_WAV");

  SourceSeparation obj;
  test::wire_sync_worker(obj);
  const int block = 512;
  obj.prepare(halp::setup{
      .input_channels = 2, .output_channels = 8, .frames = block, .rate = (double)wav_rate});
  obj.inputs.model.value = *model;
  obj.inputs.threads.value = 1;

  // Record the (mono duplicated to stereo) clip.
  obj.inputs.record.value = true;
  for(std::size_t i = 0; i < mono.size(); i += static_cast<std::size_t>(block))
  {
    const int n = static_cast<int>(std::min<std::size_t>(block, mono.size() - i));
    float* slice = const_cast<float*>(mono.data() + i);
    float* in[2]{slice, slice};
    obj.inputs.audio.samples = in;
    obj(n);
  }
  obj.inputs.record.value = false;

  // Drain the rendered stems; accumulate per-stem energy + measure stem length.
  std::vector<float> zeros(static_cast<std::size_t>(block), 0.f);
  std::array<std::vector<float>, 8> outs;
  for(auto& b : outs)
    b.assign(static_cast<std::size_t>(block), 0.f);

  double e_vocals = 0., e_accomp = 0.;
  std::size_t vocal_len = 0; // samples until the vocals stem goes quiet
  const int max_blocks = static_cast<int>(mono.size() / block) + 16;
  for(int k = 0; k < max_blocks; ++k)
  {
    for(auto& b : outs)
      std::fill(b.begin(), b.end(), 0.f);
    float* s1[2]{outs[0].data(), outs[1].data()};
    float* s2[2]{outs[2].data(), outs[3].data()};
    float* s3[2]{outs[4].data(), outs[5].data()};
    float* s4[2]{outs[6].data(), outs[7].data()};
    obj.outputs.stem1.samples = s1;
    obj.outputs.stem2.samples = s2;
    obj.outputs.stem3.samples = s3;
    obj.outputs.stem4.samples = s4;

    float* in[2]{zeros.data(), zeros.data()};
    obj.inputs.audio.samples = in;
    obj(block);

    for(int j = 0; j < block; ++j)
    {
      const double v = std::fabs(outs[0][j]) + std::fabs(outs[1][j]);
      e_vocals += v;
      e_accomp += std::fabs(outs[2][j]) + std::fabs(outs[3][j]);
      if(v > 1e-6)
        vocal_len = static_cast<std::size_t>(k) * block + j + 1;
    }
  }

  INFO(
      "num_stems: " << obj.outputs.num_stems.value << ", vocals E: " << e_vocals
                    << ", accomp E: " << e_accomp << ", vocal_len: " << vocal_len
                    << " / in: " << mono.size());

  REQUIRE(obj.outputs.num_stems.value == 2); // Spleeter 2-stems
  REQUIRE(e_vocals > 0.0);                    // vocals stem carries the speech
  // Speech input -> vocals stem should dominate the (near-silent) accompaniment.
  CHECK(e_vocals > e_accomp);
  // Separated stem spans roughly the input length (not truncated/empty).
  CHECK(vocal_len >= mono.size() / 2);
}
