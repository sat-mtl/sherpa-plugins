#pragma once
// sherpa_tts -- text to speech.
//
// When the "Text" changes, the text is synthesized on a worker thread by a
// sherpa-onnx offline TTS model, resampled to the host rate, and streamed out of
// the mono audio output over the following blocks. Speaker id and speed are
// exposed as controls.
//
// operator() (audio thread) only copies pre-rendered samples into the output bus;
// synthesis, model loading and resampling happen on the worker. The worker result
// is applied on the processing thread, so the playback buffer needs no locking.

#include "helpers/Common.hpp"
#include "helpers/ModelConfig.hpp"

#include <halp/audio.hpp>
#include <halp/callback.hpp>
#include <halp/controls.enums.hpp>
#include <halp/controls.hpp>
#include <halp/file_port.hpp>
#include <halp/meta.hpp>
#include <halp/soundfile_port.hpp>

#include <atomic>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace sherpa
{

class Tts
{
public:
  halp_meta(name, "Sherpa TTS")
  halp_meta(c_name, "sherpa_tts")
  halp_meta(category, "AI/Audio")
  halp_meta(author, "Jean-Michaël Celerier")
  halp_meta(description, "Text-to-speech with sherpa-onnx (vits / matcha / kokoro)")
  halp_meta(uuid, "c4c33e8c-9eb5-4065-bec6-86c46c4d8854")

  struct
  {
    halp::lineedit<"Text", ""> text;
    halp::folder_port<"Model"> model;
    halp::hslider_i32<"Speaker", halp::range{0., 512., 0.}> speaker;
    halp::knob_f32<"Speed", halp::range{0.1, 3.0, 1.0}> speed;
    halp::knob_f32<"Silence", halp::range{0., 2., 0.2}> silence;
    halp::hslider_i32<"Steps", halp::range{0., 50., 0.}> num_steps;
    // Zero-shot voice cloning (e.g. zipvoice): a reference clip + its transcript.
    halp::soundfile_port<"Reference"> reference;
    halp::lineedit<"Reference text", ""> reference_text;
    halp::enum_t<Provider, "Provider"> provider;
    halp::hslider_i32<"Threads", halp::range{1., 8., 1.}> threads;
    halp::lineedit<"Advanced", ""> advanced;
  } inputs;

  struct
  {
    halp::fixed_audio_bus<"Out", float, 1> audio;
    halp::callback<"Done"> done;
  } outputs;

  struct Job
  {
    std::string text;
    std::string want_model;
    bool reload = false;
    int sid = 0;
    float speed = 1.f;
    int num_threads = 1;
    Provider provider = Provider::CPU;
    float silence = 0.2f;
    int num_steps = 0;
    std::vector<float> ref;
    int ref_rate = 0;
    std::string ref_text;
    std::string advanced;
    double host_rate = 48000.;
    std::shared_ptr<TtsHandle> tts;
    std::vector<float> out; // synthesized, resampled to host_rate, mono
  };

  struct worker
  {
    std::function<void(std::shared_ptr<Job>)> request;
    static std::function<void(Tts&)> work(std::shared_ptr<Job> job);
  } worker;

  void prepare(halp::setup info);
  void operator()(int frames);

private:
  void dispatch();

  bool m_available = SherpaLoader::instance().available;
  double m_host_rate = 48000.;
  std::shared_ptr<TtsHandle> m_tts;
  std::shared_ptr<Job> m_job;
  std::atomic<bool> m_inflight{false};
  std::vector<float> m_playback; // host-rate mono, read on the audio thread
  std::size_t m_play_pos = 0;
  std::string m_requested_model, m_loaded_model;
  std::string m_last_text;
  bool m_reload = false;
  bool m_speak = false;
  // The "Done" bang, like every value/callback outlet, must fire from operator()
  // (the run() cycle) -- a callback invoked from the async worker closure does not
  // propagate. The closure sets this; operator() fires outputs.done() once.
  bool m_emit_done = false;
};

inline void Tts::prepare(halp::setup info)
{
  m_host_rate = info.rate > 0 ? info.rate : 48000.;
  if(!m_job)
  {
    m_job = std::make_shared<Job>();
    m_job->text.reserve(4096);
    m_job->want_model.reserve(512);
    m_job->out.reserve(static_cast<std::size_t>(m_host_rate * 30.0));
  }
  m_playback.reserve(static_cast<std::size_t>(m_host_rate * 30.0));
}

inline void Tts::operator()(int frames)
{
  float* out = outputs.audio[0];

  if(m_available)
  {
    if(inputs.model.value != m_requested_model)
    {
      m_requested_model = inputs.model.value;
      m_reload = true;
    }
    if(inputs.text.value != m_last_text)
    {
      m_last_text = inputs.text.value;
      if(!m_last_text.empty())
        m_speak = true;
    }
    if((m_speak || m_reload) && !m_inflight.load(std::memory_order_acquire))
      dispatch();
  }

  // Stream the pre-rendered buffer out; silence when exhausted.
  std::size_t pos = m_play_pos;
  const std::size_t n = m_playback.size();
  for(int j = 0; j < frames; ++j)
    out[j] = (pos < n) ? m_playback[pos++] : 0.f;
  m_play_pos = pos;

  if(m_emit_done) // fire from the run() cycle so the callback propagates
  {
    outputs.done();
    m_emit_done = false;
  }
}

inline void Tts::dispatch()
{
  if(!m_job)
    return;
  auto& job = *m_job;
  job.text = inputs.text.value;
  job.want_model = m_requested_model;
  job.reload = m_reload;
  job.sid = inputs.speaker.value;
  job.speed = inputs.speed.value;
  job.num_threads = inputs.threads.value;
  job.provider = inputs.provider.value;
  job.silence = inputs.silence.value;
  job.num_steps = inputs.num_steps.value;
  job.ref_text = inputs.reference_text.value;
  job.advanced = inputs.advanced.value;
  job.ref.clear();
  {
    const int rch = inputs.reference.channels();
    const long long rfr = inputs.reference.frames();
    if(rch > 0 && rfr > 0)
    {
      job.ref_rate = inputs.reference.soundfile.rate;
      job.ref.reserve(static_cast<std::size_t>(rfr));
      for(long long i = 0; i < rfr; ++i)
      {
        float s = 0.f;
        for(int c = 0; c < rch; ++c)
          s += inputs.reference[c][i];
        job.ref.push_back(rch > 1 ? s / static_cast<float>(rch) : s);
      }
    }
  }
  job.host_rate = m_host_rate;
  job.tts = m_tts;
  m_reload = false;
  m_speak = false;

  m_inflight.store(true, std::memory_order_release);
  if(worker.request)
    worker.request(m_job);
}

inline std::function<void(Tts&)> Tts::worker::work(std::shared_ptr<Job> job)
{
  const auto& L = SherpaLoader::instance();

  if(job->reload || !job->tts || !*job->tts)
  {
    job->tts = std::make_shared<TtsHandle>(
        model::create_tts(job->want_model, job->num_threads,
                          provider_str(job->provider), job->advanced));
  }

  job->out.clear();
  if(job->tts && *job->tts && !job->text.empty())
  {
    SherpaOnnxGenerationConfig cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.sid = job->sid;
    cfg.speed = job->speed;
    cfg.silence_scale = job->silence;
    if(job->num_steps > 0)
      cfg.num_steps = job->num_steps;
    if(!job->ref.empty())
    {
      cfg.reference_audio = job->ref.data();
      cfg.reference_audio_len = static_cast<int32_t>(job->ref.size());
      cfg.reference_sample_rate = job->ref_rate;
    }
    if(!job->ref_text.empty())
      cfg.reference_text = job->ref_text.c_str();

    const auto* audio = L.SherpaOnnxOfflineTtsGenerateWithConfig(
        job->tts->get(), job->text.c_str(), &cfg, nullptr, nullptr);
    if(audio)
    {
      const int sr = audio->sample_rate;
      const int host = static_cast<int>(job->host_rate);
      if(sr <= 0 || sr == host)
      {
        job->out.assign(audio->samples, audio->samples + audio->n);
      }
      else
      {
        // One-shot resample model rate -> host rate (fresh, stateless).
        LinearResamplerHandle res{
            L.SherpaOnnxCreateLinearResampler(sr, host, 0.f, 0)};
        if(res)
        {
          const auto* ro = L.SherpaOnnxLinearResamplerResample(
              res.get(), audio->samples, audio->n, 1);
          if(ro)
          {
            job->out.assign(ro->samples, ro->samples + ro->n);
            L.SherpaOnnxLinearResamplerResampleFree(ro);
          }
        }
        else
        {
          job->out.assign(audio->samples, audio->samples + audio->n);
        }
      }
      L.SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
    }
  }

  return [job](Tts& self) {
    self.m_tts = job->tts;
    self.m_loaded_model = job->want_model;
    self.m_playback.swap(job->out); // adopt host-rate buffer; no allocation
    self.m_play_pos = 0;
    self.m_emit_done = true; // fired from operator() (see m_emit_done)
    self.m_inflight.store(false, std::memory_order_release);
  };
}

}
