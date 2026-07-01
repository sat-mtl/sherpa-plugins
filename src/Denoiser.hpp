#pragma once
// sherpa_denoiser -- speech enhancement (denoising), audio -> audio.
//
// Incoming audio (downmixed to mono at the host rate) is streamed through a
// sherpa-onnx speech denoiser and the enhanced signal is played back out of the
// mono audio output over the following blocks. Both denoiser families are
// covered behind a single "Mode" enum:
//   * Online (streaming): the natural realtime default. The stateful online
//     denoiser is fed small chunks continuously and emits enhanced audio with a
//     short internal latency (Run may return NULL until it has buffered enough).
//   * Offline (batch): each accumulated chunk is denoised independently by the
//     stateless offline Run.
//   * Auto: prefer the streaming path (create the online denoiser); fall back to
//     the offline denoiser if the model cannot be loaded online.
//
// The denoiser runs at its own sample rate (GetSampleRate); we hold two stateful
// sherpa linear resamplers (host->model on input, model->host on output) so
// resampling stays continuous across blocks. All sherpa calls happen on the
// worker thread; operator() only accumulates input, dispatches, and drains the
// pre-rendered playback buffer into the output bus (exactly Tts.hpp's
// output-draining pattern combined with Vad.hpp's input+worker pattern).
//
// The denoiser + both resamplers are carried through the Job as shared_ptr so
// they persist across chunks and are recreated only on a model/mode change.

#include "helpers/AudioChunker.hpp"
#include "helpers/ModelConfig.hpp"

#include <halp/audio.hpp>
#include <halp/controls.hpp>
#include <halp/controls.enums.hpp>
#include <halp/file_port.hpp>
#include <halp/meta.hpp>

#include <atomic>
#include <cstddef>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace sherpa
{

class Denoiser
{
public:
  halp_meta(name, "Sherpa Denoiser")
  halp_meta(c_name, "sherpa_denoiser")
  halp_meta(category, "AI/Audio")
  halp_meta(author, "Jean-Michaël Celerier")
  halp_meta(description, "Speech enhancement with sherpa-onnx (offline / streaming gtcrn / dpdfnet)")
  halp_meta(uuid, "1c2e4510-8740-48b6-b424-5dad68548fcb")

  enum class Mode
  {
    Auto,
    Offline,
    Online
  };

  struct
  {
    halp::dynamic_audio_bus<"In", float> audio;
    halp::folder_port<"Model"> model;
    halp::enum_t<Mode, "Mode"> mode;
    halp::hslider_i32<"Threads", halp::range{1., 8., 1.}> threads;
  } inputs;

  struct
  {
    halp::fixed_audio_bus<"Out", float, 1> audio;
  } outputs;

  struct Job
  {
    std::vector<float> samples; // accumulated host-rate mono input
    std::vector<float> model_in; // input resampled to the model rate (reused)
    std::vector<float> out; // enhanced audio produced this dispatch, host rate
    double rate = 48000.;
    int num_threads = 1;
    int model_rate = 16000;
    bool use_online = false;
    bool reload = false;
    Mode mode = Mode::Auto;
    std::string want_model;
    std::shared_ptr<OnlineDenoiserHandle> online;
    std::shared_ptr<OfflineDenoiserHandle> offline;
    std::shared_ptr<LinearResamplerHandle> res_in;  // host  -> model rate
    std::shared_ptr<LinearResamplerHandle> res_out; // model -> host rate
  };

  struct worker
  {
    std::function<void(std::shared_ptr<Job>)> request;
    static std::function<void(Denoiser&)> work(std::shared_ptr<Job> job);
  } worker;

  void prepare(halp::setup info);
  void operator()(int frames);

private:
  void dispatch();

  bool m_available = SherpaLoader::instance().available;
  double m_host_rate = 48000.;
  int m_model_rate = 16000;
  bool m_use_online = false;
  std::shared_ptr<OnlineDenoiserHandle> m_online;
  std::shared_ptr<OfflineDenoiserHandle> m_offline;
  std::shared_ptr<LinearResamplerHandle> m_res_in;
  std::shared_ptr<LinearResamplerHandle> m_res_out;
  std::shared_ptr<Job> m_job;
  std::atomic<bool> m_inflight{false};
  MonoAccum m_accum;
  std::vector<float> m_playback; // host-rate queue drained on the audio thread
  std::size_t m_play_pos = 0;
  std::string m_requested_model, m_loaded_model;
  Mode m_requested_mode = Mode::Auto, m_loaded_mode = Mode::Auto;
  bool m_reload = false;
};

inline void Denoiser::prepare(halp::setup info)
{
  m_host_rate = info.rate > 0 ? info.rate : 48000.;
  m_accum.prepare(static_cast<std::size_t>(m_host_rate * 4.0));
  if(!m_job)
  {
    m_job = std::make_shared<Job>();
    m_job->samples.reserve(static_cast<std::size_t>(m_host_rate * 4.0));
    m_job->model_in.reserve(static_cast<std::size_t>(m_host_rate * 4.0));
    m_job->out.reserve(static_cast<std::size_t>(m_host_rate * 4.0));
    m_job->want_model.reserve(512);
  }
  m_playback.reserve(static_cast<std::size_t>(m_host_rate * 30.0));
}

inline void Denoiser::operator()(int frames)
{
  float* out = outputs.audio[0];

  if(m_available)
  {
    if(inputs.model.value != m_requested_model)
    {
      m_requested_model = inputs.model.value;
      m_reload = true;
    }
    if(inputs.mode.value != m_requested_mode)
    {
      m_requested_mode = inputs.mode.value;
      m_reload = true;
    }

    m_accum.push(inputs.audio, inputs.audio.channels, frames);

    if(!m_inflight.load(std::memory_order_acquire) && (!m_accum.empty() || m_reload))
      dispatch();
  }

  // Drain the pre-rendered playback queue; silence when exhausted.
  std::size_t pos = m_play_pos;
  const std::size_t n = m_playback.size();
  for(int j = 0; j < frames; ++j)
    out[j] = (pos < n) ? m_playback[pos++] : 0.f;
  m_play_pos = pos;
}

inline void Denoiser::dispatch()
{
  if(!m_job)
    return;
  auto& job = *m_job;
  m_accum.drain_into(job.samples);
  job.rate = m_host_rate;
  job.num_threads = inputs.threads.value;
  job.want_model = m_requested_model;
  job.mode = m_requested_mode;
  job.reload = m_reload;
  job.model_rate = m_model_rate;
  job.use_online = m_use_online;
  job.online = m_online;
  job.offline = m_offline;
  job.res_in = m_res_in;
  job.res_out = m_res_out;
  m_reload = false;

  m_inflight.store(true, std::memory_order_release);
  if(worker.request)
    worker.request(m_job);
}

inline std::function<void(Denoiser&)> Denoiser::worker::work(std::shared_ptr<Job> job)
{
  const auto& L = SherpaLoader::instance();
  namespace fs = std::filesystem;

  job->out.clear();

  // Append a model-rate denoised chunk to job->out, resampled to the host rate.
  auto append_resampled = [&](const SherpaOnnxDenoisedAudio* da) {
    if(!da || da->n <= 0)
      return;
    if(job->res_out && *job->res_out)
    {
      const auto* ro = L.SherpaOnnxLinearResamplerResample(
          job->res_out->get(), da->samples, da->n, 0);
      if(ro)
      {
        job->out.insert(job->out.end(), ro->samples, ro->samples + ro->n);
        L.SherpaOnnxLinearResamplerResampleFree(ro);
      }
    }
    else
    {
      job->out.insert(job->out.end(), da->samples, da->samples + da->n);
    }
  };

  // (Re)create the denoiser + resamplers on a model/mode change.
  if(job->reload)
  {
    job->online.reset();
    job->offline.reset();
    job->res_in.reset();
    job->res_out.reset();
    job->use_online = false;
    job->model_rate = 16000;

    // Locate the model file and pick the family: dpdfnet if the file/dir name
    // mentions dpdf/dpcrn, otherwise gtcrn (the default).
    fs::path d{job->want_model};
    std::error_code ec;
    std::string model_file;
    if(fs::is_regular_file(d, ec))
      model_file = d.string();
    else
    {
      model_file = model::find_in_dir(d, {"gtcrn", "dpdf", "dpcrn"}, ".onnx");
      if(model_file.empty())
        model_file = model::find_in_dir(d, {}, ".onnx");
    }

    if(!model_file.empty())
    {
      const std::string dirname = model::to_lower(d.filename().string());
      const bool is_dpdf = model::icontains(model_file, "dpdf")
                           || model::icontains(model_file, "dpcrn")
                           || model::icontains(dirname, "dpdf")
                           || model::icontains(dirname, "dpcrn");

      SherpaOnnxOfflineSpeechDenoiserModelConfig mc;
      std::memset(&mc, 0, sizeof(mc));
      mc.num_threads = job->num_threads;
      mc.debug = 0;
      mc.provider = "cpu";
      if(is_dpdf)
        mc.dpdfnet.model = model_file.c_str();
      else
        mc.gtcrn.model = model_file.c_str();

      const bool try_online
          = (job->mode == Mode::Online) || (job->mode == Mode::Auto);
      const bool try_offline
          = (job->mode == Mode::Offline) || (job->mode == Mode::Auto);

      if(try_online && L.SherpaOnnxCreateOnlineSpeechDenoiser)
      {
        SherpaOnnxOnlineSpeechDenoiserConfig cfg;
        std::memset(&cfg, 0, sizeof(cfg));
        cfg.model = mc;
        const auto* h = L.SherpaOnnxCreateOnlineSpeechDenoiser(&cfg);
        if(h)
        {
          job->online = std::make_shared<OnlineDenoiserHandle>(h);
          job->use_online = true;
          if(L.SherpaOnnxOnlineSpeechDenoiserGetSampleRate)
            job->model_rate = L.SherpaOnnxOnlineSpeechDenoiserGetSampleRate(h);
        }
      }
      // Offline (explicit, or Auto fallback when the online create failed).
      if(!job->use_online && try_offline && L.SherpaOnnxCreateOfflineSpeechDenoiser)
      {
        SherpaOnnxOfflineSpeechDenoiserConfig cfg;
        std::memset(&cfg, 0, sizeof(cfg));
        cfg.model = mc;
        const auto* h = L.SherpaOnnxCreateOfflineSpeechDenoiser(&cfg);
        if(h)
        {
          job->offline = std::make_shared<OfflineDenoiserHandle>(h);
          if(L.SherpaOnnxOfflineSpeechDenoiserGetSampleRate)
            job->model_rate = L.SherpaOnnxOfflineSpeechDenoiserGetSampleRate(h);
        }
      }

      if(job->model_rate <= 0)
        job->model_rate = 16000;

      const int host = static_cast<int>(job->rate);
      if((job->online || job->offline) && host > 0 && host != job->model_rate)
      {
        job->res_in = std::make_shared<LinearResamplerHandle>(
            L.SherpaOnnxCreateLinearResampler(host, job->model_rate, 0.f, 0));
        job->res_out = std::make_shared<LinearResamplerHandle>(
            L.SherpaOnnxCreateLinearResampler(job->model_rate, host, 0.f, 0));
      }
    }
  }

  // Denoise this dispatch's accumulated samples.
  if(!job->samples.empty() && (job->online || job->offline))
  {
    // Resample host -> model rate (stateful, continuous across dispatches).
    if(job->res_in && *job->res_in)
    {
      const auto* ro = L.SherpaOnnxLinearResamplerResample(
          job->res_in->get(), job->samples.data(),
          static_cast<int>(job->samples.size()), 0);
      job->model_in.clear();
      if(ro)
      {
        job->model_in.assign(ro->samples, ro->samples + ro->n);
        L.SherpaOnnxLinearResamplerResampleFree(ro);
      }
    }
    else
    {
      job->model_in.assign(job->samples.begin(), job->samples.end());
    }

    if(!job->model_in.empty())
    {
      const int n = static_cast<int>(job->model_in.size());
      if(job->use_online && job->online && *job->online
         && L.SherpaOnnxOnlineSpeechDenoiserRun)
      {
        const auto* da = L.SherpaOnnxOnlineSpeechDenoiserRun(
            job->online->get(), job->model_in.data(), n, job->model_rate);
        if(da)
        {
          append_resampled(da);
          if(L.SherpaOnnxDestroyDenoisedAudio)
            L.SherpaOnnxDestroyDenoisedAudio(da);
        }
      }
      else if(job->offline && *job->offline && L.SherpaOnnxOfflineSpeechDenoiserRun)
      {
        const auto* da = L.SherpaOnnxOfflineSpeechDenoiserRun(
            job->offline->get(), job->model_in.data(), n, job->model_rate);
        if(da)
        {
          append_resampled(da);
          if(L.SherpaOnnxDestroyDenoisedAudio)
            L.SherpaOnnxDestroyDenoisedAudio(da);
        }
      }
    }
  }

  return [job](Denoiser& self) {
    if(job->reload)
    {
      self.m_online = job->online;
      self.m_offline = job->offline;
      self.m_res_in = job->res_in;
      self.m_res_out = job->res_out;
      self.m_model_rate = job->model_rate;
      self.m_use_online = job->use_online;
      self.m_loaded_model = job->want_model;
      self.m_loaded_mode = job->mode;
    }

    // Enqueue the newly produced host-rate samples for the audio thread. When the
    // consumer has fully drained the queue we adopt the new buffer wholesale (a
    // swap, like Tts.hpp -- no copy); otherwise drop the consumed prefix and
    // append, keeping a continuous stream.
    if(!job->out.empty())
    {
      if(self.m_play_pos >= self.m_playback.size())
      {
        self.m_playback.swap(job->out);
        self.m_play_pos = 0;
      }
      else
      {
        if(self.m_play_pos > 0)
        {
          self.m_playback.erase(
              self.m_playback.begin(),
              self.m_playback.begin() + static_cast<std::ptrdiff_t>(self.m_play_pos));
          self.m_play_pos = 0;
        }
        self.m_playback.insert(
            self.m_playback.end(), job->out.begin(), job->out.end());
      }
    }
    else if(self.m_play_pos >= self.m_playback.size() && !self.m_playback.empty())
    {
      self.m_playback.clear();
      self.m_play_pos = 0;
    }

    self.m_inflight.store(false, std::memory_order_release);
  };
}

}
