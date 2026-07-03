// Reproduce score's runtime conditions for sherpa_asr: host rate 48 kHz (not the
// model's 16 kHz), audio fed in small blocks, exactly like the DSP graph. The
// 16 kHz reference wav is upsampled 3x to 48 kHz so the object must resample
// internally (streaming: sherpa-onnx online stream; batch: our LinearResampler).
#include "TestHelpers.hpp"

#include "Asr.hpp"

#include <catch2/catch_all.hpp>

#include <string>
#include <string_view>

using namespace sherpa;

namespace
{
// Naive 3x linear upsample 16 kHz -> 48 kHz.
std::vector<float> upsample_3x(const std::vector<float>& in)
{
  std::vector<float> out;
  if(in.empty())
    return out;
  out.reserve(in.size() * 3);
  for(std::size_t i = 0; i + 1 < in.size(); ++i)
  {
    const float a = in[i], b = in[i + 1];
    out.push_back(a);
    out.push_back(a + (b - a) * (1.f / 3.f));
    out.push_back(a + (b - a) * (2.f / 3.f));
  }
  out.push_back(in.back());
  out.push_back(in.back());
  out.push_back(in.back());
  return out;
}

void wire_final(Asr& obj, std::string& sink)
{
  obj.outputs.final_text.call.context = &sink;
  obj.outputs.final_text.call.function = +[](void* c, std::string_view sv) {
    static_cast<std::string*>(c)->assign(sv);
  };
}
void wire_partial(Asr& obj, std::string& sink)
{
  obj.outputs.partial.call.context = &sink;
  obj.outputs.partial.call.function = +[](void* c, std::string_view sv) {
    static_cast<std::string*>(c)->assign(sv);
  };
}
}

TEST_CASE("ASR @48k: streaming model at host rate 48000", "[asr][48k][streaming]")
{
  if(!test::loader_ready())
    SHERPA_SKIP("runtime not available");
  auto model = test::env("SHERPA_TEST_ASR_ONLINE_MODEL");
  auto wav = test::env("SHERPA_TEST_WAV");
  if(!model || !wav)
    SHERPA_SKIP("set SHERPA_TEST_ASR_ONLINE_MODEL + SHERPA_TEST_WAV");

  int sr = 0;
  auto a16 = test::load_wav_mono(*wav, sr);
  REQUIRE(sr == 16000);
  auto a48 = upsample_3x(a16);
  REQUIRE_FALSE(a48.empty());

  const int block = 512; // small DSP block like score
  Asr obj;
  test::wire_sync_worker(obj);
  obj.prepare(halp::setup{
      .input_channels = 1, .output_channels = 1, .frames = block, .rate = 48000.});
  obj.inputs.model.value = *model;
  obj.inputs.mode.value = Asr::Mode::Streaming;
  obj.inputs.endpointing.value = true;

  std::string partial, final_text;
  wire_partial(obj, partial);
  wire_final(obj, final_text);

  test::drive_audio(obj, a48, block);
  std::vector<float> tail(48000u, 0.f); // 1 s silence -> endpoint
  test::drive_audio(obj, tail, block);

  INFO("partial: " << partial);
  INFO("final:   " << final_text);
  REQUIRE((!partial.empty() || !final_text.empty()));
}

TEST_CASE("ASR @48k: batch model + VAD at host rate 48000", "[asr][48k][batch]")
{
  if(!test::loader_ready())
    SHERPA_SKIP("runtime not available");
  auto model = test::env("SHERPA_TEST_ASR_OFFLINE_MODEL");
  auto vad = test::env("SHERPA_TEST_VAD_MODEL");
  auto wav = test::env("SHERPA_TEST_WAV");
  if(!model || !vad || !wav)
    SHERPA_SKIP("set SHERPA_TEST_ASR_OFFLINE_MODEL + SHERPA_TEST_VAD_MODEL + SHERPA_TEST_WAV");

  int sr = 0;
  auto a16 = test::load_wav_mono(*wav, sr);
  REQUIRE(sr == 16000);
  auto a48 = upsample_3x(a16);
  REQUIRE_FALSE(a48.empty());

  const int block = 512;
  Asr obj;
  test::wire_sync_worker(obj);
  obj.prepare(halp::setup{
      .input_channels = 1, .output_channels = 1, .frames = block, .rate = 48000.});
  obj.inputs.model.value = *model;
  obj.inputs.vad_model.value = *vad;
  obj.inputs.mode.value = Asr::Mode::Batch;

  std::string final_text;
  wire_final(obj, final_text);

  test::drive_audio(obj, a48, block);
  std::vector<float> tail(48000u, 0.f); // 1 s silence -> close last segment
  test::drive_audio(obj, tail, block);

  INFO("final: " << final_text);
  REQUIRE_FALSE(final_text.empty());
}

TEST_CASE("ASR @48k: Auto mode picks an engine at host rate 48000", "[asr][48k][auto]")
{
  if(!test::loader_ready())
    SHERPA_SKIP("runtime not available");
  auto model = test::env("SHERPA_TEST_ASR_ONLINE_MODEL");
  auto wav = test::env("SHERPA_TEST_WAV");
  if(!model || !wav)
    SHERPA_SKIP("set SHERPA_TEST_ASR_ONLINE_MODEL + SHERPA_TEST_WAV");

  int sr = 0;
  auto a16 = test::load_wav_mono(*wav, sr);
  auto a48 = upsample_3x(a16);
  const int block = 512;
  Asr obj;
  test::wire_sync_worker(obj);
  obj.prepare(halp::setup{
      .input_channels = 1, .output_channels = 1, .frames = block, .rate = 48000.});
  obj.inputs.model.value = *model;
  obj.inputs.mode.value = Asr::Mode::Auto;

  std::string partial, final_text;
  wire_partial(obj, partial);
  wire_final(obj, final_text);

  test::drive_audio(obj, a48, block);
  std::vector<float> tail(48000u, 0.f);
  test::drive_audio(obj, tail, block);

  INFO("partial: " << partial);
  INFO("final:   " << final_text);
  REQUIRE((!partial.empty() || !final_text.empty()));
}
