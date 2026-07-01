#pragma once
// Shared helpers for the Catch2 tests.
//
// Workflow tests need (a) the runtime sherpa-onnx shared library (dlopen'd by
// SherpaLoader) and (b) a model + a speech wav, both supplied via environment
// variables. When either is missing the test reports itself skipped and passes,
// so the suite is green without models and does real validation when they are
// present (see .github/workflows/tests.yml).

#include "SherpaLoader.hpp"

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace sherpa::test
{

inline std::optional<std::string> env(const char* key)
{
  const char* v = std::getenv(key);
  if(v && *v)
    return std::string(v);
  return std::nullopt;
}

inline bool loader_ready()
{
  return SherpaLoader::instance().available;
}

// Report a test as skipped (Catch2 3.4 has no SKIP()): warn + a passing
// assertion so it does not count as "no assertions".
#define SHERPA_SKIP(msg)  \
  do                      \
  {                       \
    WARN("skipped: " msg);\
    SUCCEED("skipped");   \
    return;               \
  } while(0)

// Wire an object's worker to run synchronously (what the host does async): run
// work() on the job and immediately apply the returned closure to the object.
template <typename Obj>
void wire_sync_worker(Obj& obj)
{
  obj.worker.request = [&obj](std::shared_ptr<typename Obj::Job> job) {
    auto cb = Obj::worker::work(std::move(job));
    if(cb)
      cb(obj);
  };
}

// Load a mono wav via sherpa's own reader (requires the library to be loaded).
inline std::vector<float> load_wav_mono(const std::string& path, int& sample_rate)
{
  const auto& L = SherpaLoader::instance();
  std::vector<float> out;
  sample_rate = 0;
  if(!L.SherpaOnnxReadWave)
    return out;
  const auto* w = L.SherpaOnnxReadWave(path.c_str());
  if(w)
  {
    sample_rate = w->sample_rate;
    out.assign(w->samples, w->samples + w->num_samples);
    L.SherpaOnnxFreeWave(w);
  }
  return out;
}

// Feed a mono buffer through an object's dynamic_audio_bus<"In", float> input,
// one block of `block` frames at a time.
template <typename Obj>
void drive_audio(Obj& obj, const std::vector<float>& mono, int block)
{
  float* ch[1];
  for(std::size_t i = 0; i < mono.size(); i += static_cast<std::size_t>(block))
  {
    const int n = static_cast<int>(std::min<std::size_t>(block, mono.size() - i));
    ch[0] = const_cast<float*>(mono.data() + i);
    obj.inputs.audio.samples = ch;
    obj.inputs.audio.channels = 1;
    obj(n);
  }
}

// One block of silence through the audio input (e.g. to trigger a falling edge).
template <typename Obj>
void drive_silence(Obj& obj, int block)
{
  std::vector<float> z(static_cast<std::size_t>(block), 0.f);
  float* ch[1]{z.data()};
  obj.inputs.audio.samples = ch;
  obj.inputs.audio.channels = 1;
  obj(block);
}

// --- Signal helpers for relational / reference-based assertions -------------

// Sum of |samples| -- a cheap "is there any sound" measure.
inline double abs_energy(const std::vector<float>& v)
{
  double s = 0.;
  for(float x : v)
    s += std::fabs(static_cast<double>(x));
  return s;
}

// Root-mean-square level (length-independent, so buffers of different length are
// comparable -- useful across resampling / latency).
inline double rms(const std::vector<float>& v)
{
  if(v.empty())
    return 0.;
  double s = 0.;
  for(float x : v)
    s += static_cast<double>(x) * static_cast<double>(x);
  return std::sqrt(s / static_cast<double>(v.size()));
}

// Deterministic pseudo-random white noise in [-amp, amp] (fixed seed ->
// reproducible across runs; no <random>/global state).
inline std::vector<float>
white_noise(std::size_t n, float amp, std::uint32_t seed = 12345u)
{
  std::vector<float> o;
  o.reserve(n);
  std::uint32_t s = seed ? seed : 1u;
  for(std::size_t i = 0; i < n; ++i)
  {
    s = s * 1664525u + 1013904223u; // Numerical Recipes LCG
    const float u = static_cast<float>(s >> 8) * (1.0f / 16777216.0f); // [0,1)
    o.push_back((u * 2.f - 1.f) * amp);
  }
  return o;
}

// Add deterministic white noise to a signal in place.
inline void
add_white_noise(std::vector<float>& sig, float amp, std::uint32_t seed = 6789u)
{
  std::uint32_t s = seed ? seed : 1u;
  for(float& x : sig)
  {
    s = s * 1664525u + 1013904223u;
    const float u = static_cast<float>(s >> 8) * (1.0f / 16777216.0f);
    x += (u * 2.f - 1.f) * amp;
  }
}

// Lowercase word split keeping [a-z0-9]; for content assertions on transcripts /
// labels.
inline std::vector<std::string> words(std::string_view s)
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

// Case-insensitive substring test.
inline bool icontains(std::string_view hay, std::string_view needle)
{
  auto low = [](char c) {
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
  };
  if(needle.size() > hay.size())
    return false;
  for(std::size_t i = 0; i + needle.size() <= hay.size(); ++i)
  {
    std::size_t j = 0;
    for(; j < needle.size(); ++j)
      if(low(hay[i + j]) != low(needle[j]))
        break;
    if(j == needle.size())
      return true;
  }
  return false;
}

}
