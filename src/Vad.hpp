#pragma once
// sherpa_vad -- voice activity detection.
//
// Streams the incoming audio (downmixed to mono, resampled to 16 kHz on the
// worker) through a sherpa-onnx VAD and emits: a bang on "SpeechStart" when
// speech begins, a "Segment" message (start_sec, duration_sec) for each completed
// speech segment, optionally the segment's audio on "SegmentAudio", and a live
// "Active" boolean. Pairs naturally with the offline recognizer for a VAD -> ASR
// chain.
//
// The VAD's AcceptWaveform expects the configured (16 kHz) rate; we hold a
// stateful sherpa linear resampler so resampling stays continuous across blocks.
// All sherpa calls run on the worker thread.

#include "helpers/AudioChunker.hpp"
#include "helpers/ModelConfig.hpp"

#include <halp/audio.hpp>
#include <halp/callback.hpp>
#include <halp/controls.hpp>
#include <halp/file_port.hpp>
#include <halp/meta.hpp>

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace sherpa
{

class Vad
{
public:
  halp_meta(name, "Sherpa VAD")
  halp_meta(c_name, "sherpa_vad")
  halp_meta(category, "AI/Audio")
  halp_meta(author, "Jean-Michaël Celerier")
  halp_meta(description, "Voice activity detection with sherpa-onnx (silero / ten-vad)")
  halp_meta(uuid, "b44d484a-86cf-4a4d-9398-ddfda6eafc18")

  struct
  {
    halp::dynamic_audio_bus<"In", float> audio;
    halp::folder_port<"Model"> model;
    halp::knob_f32<"Threshold", halp::range{0., 1., 0.5}> threshold;
    halp::toggle<"Emit audio"> emit_audio;
    halp::hslider_i32<"Threads", halp::range{1., 8., 1.}> threads;
  } inputs;

  struct
  {
    halp::callback<"SpeechStart"> on_start;
    halp::callback<"Segment", float, float> on_segment; // start_sec, duration_sec
    halp::callback<"SegmentAudio", std::vector<float>> on_audio;
    halp::val_port<"Active", bool> active;
  } outputs;

  struct Seg
  {
    float start = 0.f;
    float dur = 0.f;
  };

  struct Job
  {
    std::vector<float> samples;
    double rate = 16000.;
    int num_threads = 1;
    float threshold = 0.5f;
    bool want_audio = false;
    bool reload = false;
    std::string want_model;
    std::shared_ptr<VadHandle> vad;
    std::shared_ptr<LinearResamplerHandle> resampler;
    std::vector<Seg> segments;
    std::vector<std::vector<float>> seg_audio;
    bool active = false;
  };

  struct worker
  {
    std::function<void(std::shared_ptr<Job>)> request;
    static std::function<void(Vad&)> work(std::shared_ptr<Job> job);
  } worker;

  void prepare(halp::setup info);
  void operator()(int frames);

private:
  void dispatch();

  bool m_available = SherpaLoader::instance().available;
  double m_host_rate = 48000.;
  std::shared_ptr<VadHandle> m_vad;
  std::shared_ptr<LinearResamplerHandle> m_resampler;
  std::shared_ptr<Job> m_job;
  std::atomic<bool> m_inflight{false};
  MonoAccum m_accum;
  std::string m_requested_model, m_loaded_model;
  bool m_reload = false;
  bool m_active_state = false;
};

inline void Vad::prepare(halp::setup info)
{
  m_host_rate = info.rate > 0 ? info.rate : 48000.;
  m_accum.prepare(static_cast<std::size_t>(m_host_rate * 4.0));
  if(!m_job)
  {
    m_job = std::make_shared<Job>();
    m_job->samples.reserve(static_cast<std::size_t>(m_host_rate * 4.0));
    m_job->want_model.reserve(512);
  }
}

inline void Vad::operator()(int frames)
{
  if(!m_available)
    return;

  if(inputs.model.value != m_requested_model)
  {
    m_requested_model = inputs.model.value;
    m_reload = true;
  }

  m_accum.push(inputs.audio, inputs.audio.channels, frames);

  if(!m_inflight.load(std::memory_order_acquire) && (!m_accum.empty() || m_reload))
    dispatch();
}

inline void Vad::dispatch()
{
  if(!m_job)
    return;
  auto& job = *m_job;
  m_accum.drain_into(job.samples);
  job.rate = m_host_rate;
  job.num_threads = inputs.threads.value;
  job.threshold = inputs.threshold.value;
  job.want_audio = inputs.emit_audio.value;
  job.want_model = m_requested_model;
  job.reload = m_reload;
  job.vad = m_vad;
  job.resampler = m_resampler;
  m_reload = false;

  m_inflight.store(true, std::memory_order_release);
  if(worker.request)
    worker.request(m_job);
}

inline std::function<void(Vad&)> Vad::worker::work(std::shared_ptr<Job> job)
{
  const auto& L = SherpaLoader::instance();

  if(job->reload || !job->vad || !*job->vad)
  {
    job->vad = std::make_shared<VadHandle>(model::create_vad(
        job->want_model, job->threshold, 16000, job->num_threads));
    job->resampler.reset();
  }
  if(job->reload || !job->resampler || !*job->resampler)
  {
    job->resampler = std::make_shared<LinearResamplerHandle>(
        L.SherpaOnnxCreateLinearResampler(
            static_cast<int>(job->rate), 16000, 0.f, 0));
  }

  job->segments.clear();
  job->seg_audio.clear();
  job->active = false;

  if(job->vad && *job->vad && job->resampler && *job->resampler
     && !job->samples.empty())
  {
    auto* vad = job->vad->get();
    const auto* ro = L.SherpaOnnxLinearResamplerResample(
        job->resampler->get(), job->samples.data(),
        static_cast<int>(job->samples.size()), 0);
    if(ro)
    {
      if(ro->n > 0)
        L.SherpaOnnxVoiceActivityDetectorAcceptWaveform(vad, ro->samples, ro->n);
      L.SherpaOnnxLinearResamplerResampleFree(ro);
    }

    job->active = L.SherpaOnnxVoiceActivityDetectorDetected(vad) != 0;
    while(L.SherpaOnnxVoiceActivityDetectorEmpty(vad) == 0)
    {
      const auto* seg = L.SherpaOnnxVoiceActivityDetectorFront(vad);
      if(seg)
      {
        job->segments.push_back(
            Seg{seg->start / 16000.f, seg->n / 16000.f});
        if(job->want_audio)
          job->seg_audio.emplace_back(seg->samples, seg->samples + seg->n);
        L.SherpaOnnxDestroySpeechSegment(seg);
      }
      L.SherpaOnnxVoiceActivityDetectorPop(vad);
    }
  }

  return [job](Vad& self) {
    if(job->reload)
    {
      self.m_vad = job->vad;
      self.m_resampler = job->resampler;
      self.m_loaded_model = job->want_model;
    }
    if(job->active && !self.m_active_state)
      self.outputs.on_start();
    self.m_active_state = job->active;
    self.outputs.active.value = job->active;

    for(std::size_t i = 0; i < job->segments.size(); ++i)
    {
      self.outputs.on_segment(job->segments[i].start, job->segments[i].dur);
      if(i < job->seg_audio.size())
        self.outputs.on_audio(std::move(job->seg_audio[i]));
    }
    self.m_inflight.store(false, std::memory_order_release);
  };
}

}
