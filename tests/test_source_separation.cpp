// Workflow test for sherpa_source_separation: separate a stereo passage and
// expect non-silent audio out of the first stem.
//
// Requires:
//   SHERPA_TEST_SOURCESEP_MODEL = source-separation model dir (Spleeter / UVR)
//   SHERPA_TEST_WAV             = a music wav (read mono, duplicated to stereo)
#include "TestHelpers.hpp"

#include "SourceSeparation.hpp"

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

using namespace sherpa;

TEST_CASE(
    "Source separation produces non-silent stem audio",
    "[sourcesep][workflow]")
{
  if(!test::loader_ready())
    SHERPA_SKIP("sherpa-onnx runtime library not available");
  if(!SherpaLoader::instance().SherpaOnnxCreateOfflineSourceSeparation)
    SHERPA_SKIP("source-separation symbols not present in this build");

  auto model = test::env("SHERPA_TEST_SOURCESEP_MODEL");
  if(!model)
    SHERPA_SKIP("set SHERPA_TEST_SOURCESEP_MODEL");
  auto wavpath = test::env("SHERPA_TEST_WAV");
  if(!wavpath)
    SHERPA_SKIP("set SHERPA_TEST_WAV");

  int wav_rate = 0;
  auto mono = test::load_wav_mono(*wavpath, wav_rate);
  if(mono.empty() || wav_rate <= 0)
    SHERPA_SKIP("could not read SHERPA_TEST_WAV");

  SourceSeparation obj;
  test::wire_sync_worker(obj);

  const int block = 512;
  const double rate = static_cast<double>(wav_rate);
  obj.prepare(halp::setup{
      .input_channels = 2, .output_channels = 8, .frames = block, .rate = rate});
  obj.inputs.model.value = *model;
  obj.inputs.threads.value = 1;

  // Record: feed the (mono, duplicated to both channels) wav while Record is on.
  obj.inputs.record.value = true;
  for(std::size_t i = 0; i < mono.size(); i += static_cast<std::size_t>(block))
  {
    const int n
        = static_cast<int>(std::min<std::size_t>(block, mono.size() - i));
    float* slice = const_cast<float*>(mono.data() + i);
    float* in[2]{slice, slice};
    obj.inputs.audio.samples = in;
    obj(n);
  }

  // Record off + drive further blocks: the falling edge (first iteration below)
  // dispatches to the synchronous worker, which separates the buffer; subsequent
  // blocks drain the rendered stem buffers into the output buses.
  obj.inputs.record.value = false;

  std::vector<float> zeros(static_cast<std::size_t>(block), 0.f);
  std::array<std::vector<float>, 8> outs;
  for(auto& b : outs)
    b.assign(static_cast<std::size_t>(block), 0.f);

  double energy = 0.;
  const int max_blocks = static_cast<int>(mono.size() / block) + 8;
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

    for(float v : outs[0]) // stem 1, left
      energy += std::fabs(v);
    for(float v : outs[1]) // stem 1, right
      energy += std::fabs(v);
  }

  INFO("stem-1 output energy: " << energy);
  REQUIRE(energy > 0.0);
}
