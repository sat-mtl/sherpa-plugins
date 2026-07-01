// Structural tests: run everywhere, no library or models needed. They lock in the
// invariants the host bindings rely on (notably nothrow construction, required by
// the Max/Pd C object model) and the RT-safe audio accumulator.
#include "TestHelpers.hpp"

#include "OfflineRecognizer.hpp"
#include "OnlineRecognizer.hpp"
#include "Tts.hpp"
#include "Vad.hpp"
#include "helpers/AudioChunker.hpp"

#include <catch2/catch_all.hpp>

#include <string_view>
#include <type_traits>

using namespace sherpa;

TEMPLATE_TEST_CASE(
    "Objects are nothrow default-constructible (Max/Pd requirement)", "[structural]",
    OfflineRecognizer, OnlineRecognizer, Vad, Tts)
{
  STATIC_REQUIRE(std::is_nothrow_default_constructible_v<TestType>);
}

TEMPLATE_TEST_CASE(
    "Objects expose non-empty metadata", "[structural]", OfflineRecognizer,
    OnlineRecognizer, Vad, Tts)
{
  CHECK(std::string_view{TestType::name()}.size() > 0);
  CHECK(std::string_view{TestType::c_name()}.size() > 0);
  CHECK(std::string_view{TestType::uuid()}.size() == 36); // canonical uuid
}

TEST_CASE("MonoAccum downmixes and respects its capacity", "[structural]")
{
  MonoAccum acc;
  acc.prepare(8);

  // Two channels -> averaged to mono.
  float l[4] = {1.f, 1.f, 1.f, 1.f};
  float r[4] = {-1.f, 3.f, -1.f, 3.f};
  float* ch[2]{l, r};
  struct
  {
    float** samples;
    int channels;
    float* operator[](int i) const { return samples[i]; }
  } bus{ch, 2};

  acc.push(bus, 2, 4);
  REQUIRE(acc.size() == 4);
  CHECK(acc.buf[0] == Catch::Approx(0.f));
  CHECK(acc.buf[1] == Catch::Approx(2.f));

  // Overflow past the reserved capacity (8) is dropped, never reallocates.
  acc.push(bus, 2, 4); // now 8, at capacity
  acc.push(bus, 2, 4); // would exceed -> dropped
  REQUIRE(acc.size() == 8);

  std::vector<float> out;
  out.reserve(8);
  acc.drain_into(out);
  CHECK(out.size() == 8);
  CHECK(acc.empty());
}

TEST_CASE("Driving audio with no worker wired is a safe no-op", "[structural]")
{
  // Without a host wiring worker.request, operator() must accumulate and never
  // dispatch/crash (dispatch is guarded on worker.request).
  OnlineRecognizer obj;
  obj.prepare(halp::setup{
      .input_channels = 1, .output_channels = 0, .frames = 256, .rate = 48000.});
  std::vector<float> tone(48000, 0.25f);
  REQUIRE_NOTHROW(test::drive_audio(obj, tone, 256));
}

TEST_CASE("SherpaLoader singleton never throws and reports availability",
          "[structural]")
{
  REQUIRE_NOTHROW(SherpaLoader::instance());
  const auto& L = SherpaLoader::instance();
  if(L.available)
  {
    REQUIRE(L.SherpaOnnxGetVersionStr != nullptr);
    INFO("sherpa-onnx version: " << L.SherpaOnnxGetVersionStr());
    CHECK(std::string_view{L.SherpaOnnxGetVersionStr()}.size() > 0);
  }
  else
  {
    WARN("libsherpa-onnx-c-api not loadable here; workflow tests will skip");
    SUCCEED("loader unavailable (expected without the runtime library)");
  }
}
