#pragma once
// sherpa_asr_offline -- batch speech recognition.
//
// Record an utterance into the audio input (toggle "Record"); on the falling
// edge the accumulated mono audio is sent to a worker thread, decoded in one
// shot by a non-streaming sherpa-onnx recognizer, and the text is emitted.
//
// All inference and model loading happen on the worker; operator() (audio thread)
// only downmixes + accumulates and dispatches. See helpers/AudioChunker.hpp.

#include "helpers/AudioChunker.hpp"
#include "helpers/Common.hpp"
#include "helpers/ModelConfig.hpp"

#include <halp/audio.hpp>
#include <halp/callback.hpp>
#include <halp/controls.enums.hpp>
#include <halp/controls.hpp>
#include <halp/file_port.hpp>
#include <halp/meta.hpp>

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace sherpa
{

class OfflineRecognizer
{
public:
  halp_meta(name, "Sherpa Offline ASR")
  halp_meta(c_name, "sherpa_asr_offline")
  halp_meta(category, "AI/Audio")
  halp_meta(author, "Jean-Michaël Celerier")
  halp_meta(description, "Batch speech-to-text with sherpa-onnx (record -> text)")
  halp_meta(uuid, "d9aba387-3ccc-4726-b006-21ab2eefc63d")

  struct
  {
    halp::dynamic_audio_bus<"In", float> audio;
    halp::folder_port<"Model"> model;
    halp::toggle<"Record"> record;
    halp::enum_t<Decoding, "Decoding"> decoding;
    halp::enum_t<Provider, "Provider"> provider;
    halp::hslider_i32<"Threads", halp::range{1., 16., 1.}> threads;
    halp::lineedit<"Advanced", ""> advanced;
  } inputs;

  struct
  {
    halp::val_port<"Text", std::string> text;
    halp::val_port<"Tokens", std::vector<std::string>> tokens;
    halp::val_port<"Timestamps", std::vector<float>> timestamps;
    halp::callback<"Lang", std::string_view> lang;
    halp::callback<"Emotion", std::string_view> emotion;
    halp::callback<"Event", std::string_view> event;
    halp::callback<"Done"> done;
  } outputs;

  struct Job
  {
    std::vector<float> samples;
    double rate = 16000.;
    int num_threads = 1;
    Provider provider = Provider::CPU;
    Decoding decoding = Decoding::GreedySearch;
    bool reload = false;
    std::string want_model;
    std::string advanced;
    std::shared_ptr<OfflineRecognizerHandle> rec;
    std::string text;
    std::vector<std::string> tokens;
    std::vector<float> timestamps;
    std::string lang, emotion, event;
  };

  struct worker
  {
    std::function<void(std::shared_ptr<Job>)> request;
    static std::function<void(OfflineRecognizer&)> work(std::shared_ptr<Job> job);
  } worker;

  void prepare(halp::setup info);
  void operator()(int frames);

private:
  void dispatch();

  bool m_available = SherpaLoader::instance().available;
  double m_host_rate = 48000.;
  std::shared_ptr<OfflineRecognizerHandle> m_rec;
  std::shared_ptr<Job> m_job;
  std::atomic<bool> m_inflight{false};
  MonoAccum m_accum;
  std::string m_requested_model, m_loaded_model;
  bool m_reload = false;
  bool m_was_recording = false;
};

inline void OfflineRecognizer::prepare(halp::setup info)
{
  m_host_rate = info.rate > 0 ? info.rate : 48000.;
  // Cap the recording buffer (mono) -- e.g. 120 s at the host rate.
  m_accum.prepare(static_cast<std::size_t>(m_host_rate * 120.0));
  if(!m_job)
  {
    m_job = std::make_shared<Job>();
    m_job->samples.reserve(static_cast<std::size_t>(m_host_rate * 120.0));
    m_job->want_model.reserve(512);
  }
}

inline void OfflineRecognizer::operator()(int frames)
{
  if(!m_available)
    return;

  if(inputs.model.value != m_requested_model)
  {
    m_requested_model = inputs.model.value;
    m_reload = true;
  }

  const bool rec_on = inputs.record.value;
  if(rec_on)
    m_accum.push(inputs.audio, inputs.audio.channels, frames);

  // Falling edge of Record -> transcribe what we captured.
  if(m_was_recording && !rec_on)
    dispatch();
  m_was_recording = rec_on;
}

inline void OfflineRecognizer::dispatch()
{
  if(m_inflight.load(std::memory_order_acquire) || !m_job)
    return;
  if(m_accum.empty() && !m_reload)
    return;

  auto& job = *m_job;
  m_accum.drain_into(job.samples);
  job.rate = m_host_rate;
  job.num_threads = inputs.threads.value;
  job.provider = inputs.provider.value;
  job.decoding = inputs.decoding.value;
  job.want_model = m_requested_model;
  job.advanced = inputs.advanced.value;
  job.reload = m_reload;
  job.rec = m_rec;
  m_reload = false;

  m_inflight.store(true, std::memory_order_release);
  if(worker.request)
    worker.request(m_job);
}

inline std::function<void(OfflineRecognizer&)>
OfflineRecognizer::worker::work(std::shared_ptr<Job> job)
{
  const auto& L = SherpaLoader::instance();

  if(job->reload || !job->rec || !*job->rec)
  {
    job->rec = std::make_shared<OfflineRecognizerHandle>(
        model::create_offline_recognizer(
            job->want_model, job->num_threads, decoding_str(job->decoding),
            provider_str(job->provider), job->advanced));
  }

  job->text.clear();
  job->tokens.clear();
  job->timestamps.clear();
  job->lang.clear();
  job->emotion.clear();
  job->event.clear();
  if(job->rec && *job->rec && !job->samples.empty())
  {
    OfflineStreamHandle stream{
        L.SherpaOnnxCreateOfflineStream(job->rec->get())};
    if(stream)
    {
      L.SherpaOnnxAcceptWaveformOffline(
          stream.get(), static_cast<int>(job->rate), job->samples.data(),
          static_cast<int>(job->samples.size()));
      L.SherpaOnnxDecodeOfflineStream(job->rec->get(), stream.get());
      const auto* r = L.SherpaOnnxGetOfflineStreamResult(stream.get());
      if(r)
      {
        if(r->text)
          job->text = r->text;
        if(r->tokens_arr && r->count > 0)
          for(int i = 0; i < r->count; ++i)
            if(r->tokens_arr[i])
              job->tokens.emplace_back(r->tokens_arr[i]);
        if(r->timestamps && r->count > 0)
          job->timestamps.assign(r->timestamps, r->timestamps + r->count);
        if(r->lang)
          job->lang = r->lang;
        if(r->emotion)
          job->emotion = r->emotion;
        if(r->event)
          job->event = r->event;
        L.SherpaOnnxDestroyOfflineRecognizerResult(r);
      }
    }
  }

  return [job](OfflineRecognizer& self) {
    self.m_rec = job->rec;
    self.m_loaded_model = job->want_model;
    self.outputs.text.value = std::move(job->text);
    self.outputs.tokens.value = std::move(job->tokens);
    self.outputs.timestamps.value = std::move(job->timestamps);
    if(!job->lang.empty())
      self.outputs.lang(std::string_view{job->lang});
    if(!job->emotion.empty())
      self.outputs.emotion(std::string_view{job->emotion});
    if(!job->event.empty())
      self.outputs.event(std::string_view{job->event});
    self.outputs.done();
    self.m_inflight.store(false, std::memory_order_release);
  };
}

}
