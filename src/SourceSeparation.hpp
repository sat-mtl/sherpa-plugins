#pragma once
// sherpa_source_separation -- music source separation (Spleeter / UVR).
//
// Record a stereo passage into the audio input (toggle "Record"); on the falling
// edge the accumulated stereo audio is sent to a worker thread and separated in
// one shot by a sherpa-onnx offline source-separation model into N stems (e.g.
// vocals + accompaniment). Each stem is copied (and resampled to the host rate if
// needed) into a per-stem playback buffer, which is then streamed out of the
// matching stereo output bus over the following blocks -- exactly like Tts drains
// its single playback buffer, but for up to four stereo stems at once.
//
// operator() (audio thread) only accumulates the stereo input, drains the
// pre-rendered stem buffers into the output buses, and dispatches on the falling
// edge of Record. All model loading, separation (slow!) and resampling happen on
// the worker; the worker result is applied on the processing thread, so the
// playback buffers need no locking.

#include "helpers/Common.hpp"
#include "helpers/ModelConfig.hpp"

#include <halp/audio.hpp>
#include <halp/controls.hpp>
#include <halp/controls.enums.hpp>
#include <halp/file_port.hpp>
#include <halp/meta.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace sherpa
{

class SourceSeparation
{
public:
  halp_meta(name, "Sherpa Source Separation")
  halp_meta(c_name, "sherpa_source_separation")
  halp_meta(category, "AI/Audio")
  halp_meta(author, "Jean-Michaël Celerier")
  halp_meta(description, "Music source separation with sherpa-onnx (Spleeter / UVR)")
  halp_meta(uuid, "6d041efd-5832-46bc-9dd3-3ee66b0eede9")

  static constexpr int max_stems = 4;

  struct
  {
    halp::fixed_audio_bus<"In", float, 2> audio;
    halp::folder_port<"Model"> model;
    halp::toggle<"Record"> record;
    halp::enum_t<Provider, "Provider"> provider;
    halp::hslider_i32<"Threads", halp::range{1., 8., 1.}> threads;
  } inputs;

  struct
  {
    halp::fixed_audio_bus<"Stem 1", float, 2> stem1;
    halp::fixed_audio_bus<"Stem 2", float, 2> stem2;
    halp::fixed_audio_bus<"Stem 3", float, 2> stem3;
    halp::fixed_audio_bus<"Stem 4", float, 2> stem4;
    halp::val_port<"Num stems", int> num_stems;
  } outputs;

  struct StemOut
  {
    std::vector<float> l, r; // host-rate stereo
  };

  struct Job
  {
    std::vector<float> L, R; // stereo input at host rate
    double rate = 48000.;
    int num_threads = 1;
    bool reload = false;
    Provider provider = Provider::CPU;
    std::string want_model;
    std::shared_ptr<SourceSeparationHandle> ss;
    std::array<StemOut, max_stems> out;
    int num_stems = 0;
  };

  struct worker
  {
    std::function<void(std::shared_ptr<Job>)> request;
    static std::function<void(SourceSeparation&)> work(std::shared_ptr<Job> job);
  } worker;

  void prepare(halp::setup info);
  void operator()(int frames);

private:
  void dispatch();
  void push_stereo(int frames) noexcept;

  static SourceSeparationHandle create_engine(
      std::string_view dir, int num_threads, const char* provider = "cpu");
  static void copy_or_resample(
      const SherpaLoader& L, const float* src, int n, int sr, int host,
      std::vector<float>& dst);

  bool m_available = SherpaLoader::instance().available;
  double m_host_rate = 48000.;
  std::shared_ptr<SourceSeparationHandle> m_ss;
  std::shared_ptr<Job> m_job;
  std::atomic<bool> m_inflight{false};
  std::vector<float> m_accumL, m_accumR; // RT stereo accumulator
  std::array<std::vector<float>, max_stems> m_playL, m_playR; // audio-thread reads
  std::size_t m_play_pos = 0;
  std::string m_requested_model, m_loaded_model;
  bool m_reload = false;
  bool m_was_recording = false;
};

inline void SourceSeparation::prepare(halp::setup info)
{
  m_host_rate = info.rate > 0 ? info.rate : 48000.;
  // Cap the recording buffer (per channel) -- e.g. 300 s at the host rate.
  const auto cap = static_cast<std::size_t>(m_host_rate * 300.0);
  m_accumL.clear();
  m_accumL.reserve(cap);
  m_accumR.clear();
  m_accumR.reserve(cap);
  if(!m_job)
  {
    m_job = std::make_shared<Job>();
    m_job->L.reserve(cap);
    m_job->R.reserve(cap);
    m_job->want_model.reserve(512);
  }
}

inline void SourceSeparation::push_stereo(int frames) noexcept
{
  if(frames <= 0)
    return;
  const std::size_t add = static_cast<std::size_t>(frames);
  if(m_accumL.size() + add > m_accumL.capacity())
    return; // RT cap reached: drop rather than reallocate
  const float* inL = inputs.audio[0];
  const float* inR = inputs.audio[1];
  if(!inL)
    return;
  if(!inR)
    inR = inL;
  for(int i = 0; i < frames; ++i)
  {
    m_accumL.push_back(inL[i]);
    m_accumR.push_back(inR[i]);
  }
}

inline void SourceSeparation::operator()(int frames)
{
  // Audio-thread bookkeeping: model change, accumulate, falling-edge dispatch.
  if(m_available)
  {
    if(inputs.model.value != m_requested_model)
    {
      m_requested_model = inputs.model.value;
      m_reload = true;
    }

    const bool rec_on = inputs.record.value;
    if(rec_on)
      push_stereo(frames);

    if(m_was_recording && !rec_on)
      dispatch();
    m_was_recording = rec_on;
  }

  // Stream the four per-stem stereo buffers out; silence when exhausted/absent.
  float** buses[max_stems]{
      outputs.stem1.samples, outputs.stem2.samples, outputs.stem3.samples,
      outputs.stem4.samples};

  const std::size_t pos = m_play_pos;
  for(int k = 0; k < max_stems; ++k)
  {
    float** bus = buses[k];
    if(!bus)
      continue;
    const std::vector<float>& bl = m_playL[k];
    const std::vector<float>& br = m_playR[k];
    float* o0 = bus[0];
    float* o1 = bus[1];
    for(int j = 0; j < frames; ++j)
    {
      const std::size_t p = pos + static_cast<std::size_t>(j);
      if(o0)
        o0[j] = (p < bl.size()) ? bl[p] : 0.f;
      if(o1)
        o1[j] = (p < br.size()) ? br[p] : 0.f;
    }
  }
  if(frames > 0)
    m_play_pos = pos + static_cast<std::size_t>(frames);
}

inline void SourceSeparation::dispatch()
{
  if(m_inflight.load(std::memory_order_acquire) || !m_job)
    return;
  if(m_accumL.empty() && !m_reload)
    return;

  auto& job = *m_job;
  job.L.assign(m_accumL.begin(), m_accumL.end());
  job.R.assign(m_accumR.begin(), m_accumR.end());
  m_accumL.clear();
  m_accumR.clear();
  job.rate = m_host_rate;
  job.num_threads = inputs.threads.value;
  job.provider = inputs.provider.value;
  job.want_model = m_requested_model;
  job.reload = m_reload;
  job.ss = m_ss;
  m_reload = false;

  m_inflight.store(true, std::memory_order_release);
  if(worker.request)
    worker.request(m_job);
}

inline SourceSeparationHandle SourceSeparation::create_engine(
    std::string_view dir, int num_threads, const char* provider)
{
  const auto& L = SherpaLoader::instance();
  if(!L.SherpaOnnxCreateOfflineSourceSeparation)
    return {};

  std::filesystem::path d{dir};

  // UVR ships a single MDX-Net model; Spleeter ships a vocals + accompaniment pair.
  std::string uvr = model::find_in_dir(d, {"uvr"});
  std::string vocals = model::find_in_dir(d, {"vocals"});
  std::string accompaniment = model::find_in_dir(d, {"accompaniment"});

  SherpaOnnxOfflineSourceSeparationConfig cfg;
  std::memset(&cfg, 0, sizeof(cfg));
  cfg.model.num_threads = num_threads;
  cfg.model.debug = 0;
  cfg.model.provider = provider;

  if(!uvr.empty())
  {
    cfg.model.uvr.model = uvr.c_str();
  }
  else if(!vocals.empty() && !accompaniment.empty())
  {
    cfg.model.spleeter.vocals = vocals.c_str();
    cfg.model.spleeter.accompaniment = accompaniment.c_str();
  }
  else
  {
    return {};
  }

  return SourceSeparationHandle{L.SherpaOnnxCreateOfflineSourceSeparation(&cfg)};
}

inline void SourceSeparation::copy_or_resample(
    const SherpaLoader& L, const float* src, int n, int sr, int host,
    std::vector<float>& dst)
{
  dst.clear();
  if(!src || n <= 0)
    return;
  if(sr <= 0 || sr == host || !L.SherpaOnnxCreateLinearResampler)
  {
    dst.assign(src, src + n);
    return;
  }
  // One-shot resample stem rate -> host rate (fresh, stateless).
  LinearResamplerHandle res{L.SherpaOnnxCreateLinearResampler(sr, host, 0.f, 0)};
  if(res)
  {
    const auto* ro = L.SherpaOnnxLinearResamplerResample(res.get(), src, n, 1);
    if(ro)
    {
      dst.assign(ro->samples, ro->samples + ro->n);
      L.SherpaOnnxLinearResamplerResampleFree(ro);
      return;
    }
  }
  dst.assign(src, src + n);
}

inline std::function<void(SourceSeparation&)>
SourceSeparation::worker::work(std::shared_ptr<Job> job)
{
  const auto& L = SherpaLoader::instance();

  if(job->reload || !job->ss || !*job->ss)
  {
    job->ss = std::make_shared<SourceSeparationHandle>(
        create_engine(job->want_model, job->num_threads, provider_str(job->provider)));
  }

  for(auto& s : job->out)
  {
    s.l.clear();
    s.r.clear();
  }
  job->num_stems = 0;

  if(job->ss && *job->ss && !job->L.empty()
     && L.SherpaOnnxOfflineSourceSeparationProcess
     && L.SherpaOnnxDestroySourceSeparationOutput)
  {
    const int n = static_cast<int>(job->L.size());
    const float* samples[2]{job->L.data(), job->R.data()};
    const auto* output = L.SherpaOnnxOfflineSourceSeparationProcess(
        job->ss->get(), samples, 2, n, static_cast<int>(job->rate));
    if(output)
    {
      const int host = static_cast<int>(job->rate);
      const int sr = output->sample_rate > 0 ? output->sample_rate : host;
      job->num_stems = output->num_stems;
      const int count = std::min(output->num_stems, max_stems);
      for(int k = 0; k < count && output->stems; ++k)
      {
        const auto& stem = output->stems[k];
        if(!stem.samples || stem.n <= 0 || stem.num_channels <= 0)
          continue;
        const float* c0 = stem.samples[0];
        const float* c1 = stem.num_channels >= 2 ? stem.samples[1] : c0;
        copy_or_resample(L, c0, stem.n, sr, host, job->out[k].l);
        copy_or_resample(L, c1, stem.n, sr, host, job->out[k].r);
      }
      L.SherpaOnnxDestroySourceSeparationOutput(output);
    }
  }

  return [job](SourceSeparation& self) {
    self.m_ss = job->ss;
    self.m_loaded_model = job->want_model;
    for(int k = 0; k < max_stems; ++k)
    {
      self.m_playL[k].swap(job->out[k].l); // adopt host-rate buffers; no alloc
      self.m_playR[k].swap(job->out[k].r);
    }
    self.m_play_pos = 0;
    self.outputs.num_stems.value = job->num_stems;
    self.m_inflight.store(false, std::memory_order_release);
  };
}

}
