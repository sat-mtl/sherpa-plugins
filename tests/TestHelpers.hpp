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
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <string>
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

}
