#pragma once
// sherpa_asr_online -- streaming speech recognition.
//
// Continuously feeds the incoming audio (downmixed to mono) to a persistent
// sherpa-onnx online recognizer + stream and emits the growing hypothesis on
// "Partial", a finalized utterance on "Final" at each detected endpoint, and a
// bang on "Endpoint". sherpa resamples the input internally, so the host rate is
// passed through as-is.
//
// The recognizer/stream are stateful and live across calls; only one decode is in
// flight at a time (a new block's audio accumulates until the previous decode
// returns, so nothing is dropped). All sherpa calls run on the worker thread.

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
#include <string_view>
#include <vector>

namespace sherpa
{

class OnlineRecognizer
{
public:
  halp_meta(name, "Sherpa Streaming ASR")
  halp_meta(c_name, "sherpa_asr_online")
  halp_meta(category, "AI/Audio")
  halp_meta(author, "Jean-Michaël Celerier")
  halp_meta(description, "Streaming speech-to-text with sherpa-onnx")
  halp_meta(uuid, "75d7ca6d-e68c-4001-b844-9116d4e4c5dc")

  struct
  {
    halp::dynamic_audio_bus<"In", float> audio;
    halp::folder_port<"Model"> model;
    halp::toggle<"Endpointing"> endpointing;
    halp::hslider_i32<"Threads", halp::range{1., 16., 1.}> threads;
  } inputs;

  struct
  {
    halp::callback<"Partial", std::string_view> partial;
    halp::callback<"Final", std::string_view> final_text;
    halp::callback<"Endpoint"> endpoint;
  } outputs;

  struct Job
  {
    std::vector<float> samples;
    double rate = 16000.;
    int num_threads = 1;
    bool reload = false;
    bool endpointing = true;
    std::string want_model;
    std::shared_ptr<OnlineRecognizerHandle> rec;
    std::shared_ptr<OnlineStreamHandle> stream;
    std::string text;
    bool is_endpoint = false;
    bool has_text = false;
  };

  struct worker
  {
    std::function<void(std::shared_ptr<Job>)> request;
    static std::function<void(OnlineRecognizer&)> work(std::shared_ptr<Job> job);
  } worker;

  void prepare(halp::setup info);
  void operator()(int frames);

private:
  void dispatch();

  bool m_available = SherpaLoader::instance().available;
  double m_host_rate = 48000.;
  std::shared_ptr<OnlineRecognizerHandle> m_rec;
  std::shared_ptr<OnlineStreamHandle> m_stream;
  std::shared_ptr<Job> m_job;
  std::atomic<bool> m_inflight{false};
  MonoAccum m_accum;
  std::string m_requested_model, m_loaded_model;
  bool m_reload = false;
};

inline void OnlineRecognizer::prepare(halp::setup info)
{
  m_host_rate = info.rate > 0 ? info.rate : 48000.;
  // A few seconds of mono headroom in case a decode runs long.
  m_accum.prepare(static_cast<std::size_t>(m_host_rate * 4.0));
  if(!m_job)
  {
    m_job = std::make_shared<Job>();
    m_job->samples.reserve(static_cast<std::size_t>(m_host_rate * 4.0));
    m_job->want_model.reserve(512);
    m_job->text.reserve(1024);
  }
}

inline void OnlineRecognizer::operator()(int frames)
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

inline void OnlineRecognizer::dispatch()
{
  if(!m_job)
    return;
  auto& job = *m_job;
  m_accum.drain_into(job.samples);
  job.rate = m_host_rate;
  job.num_threads = inputs.threads.value;
  job.endpointing = inputs.endpointing.value;
  job.want_model = m_requested_model;
  job.reload = m_reload;
  job.rec = m_rec;
  job.stream = m_stream;
  m_reload = false;

  m_inflight.store(true, std::memory_order_release);
  if(worker.request)
    worker.request(m_job);
}

inline std::function<void(OnlineRecognizer&)>
OnlineRecognizer::worker::work(std::shared_ptr<Job> job)
{
  const auto& L = SherpaLoader::instance();

  if(job->reload || !job->rec || !*job->rec)
  {
    job->rec = std::make_shared<OnlineRecognizerHandle>(
        model::create_online_recognizer(
            job->want_model, job->num_threads, "greedy_search", job->endpointing,
            ""));
    job->stream.reset();
  }
  if(job->rec && *job->rec && (!job->stream || !*job->stream))
  {
    job->stream = std::make_shared<OnlineStreamHandle>(
        L.SherpaOnnxCreateOnlineStream(job->rec->get()));
  }

  job->text.clear();
  job->has_text = false;
  job->is_endpoint = false;

  if(job->rec && *job->rec && job->stream && *job->stream && !job->samples.empty())
  {
    auto* rec = job->rec->get();
    auto* st = job->stream->get();
    L.SherpaOnnxOnlineStreamAcceptWaveform(
        st, static_cast<int>(job->rate), job->samples.data(),
        static_cast<int>(job->samples.size()));
    while(L.SherpaOnnxIsOnlineStreamReady(rec, st))
      L.SherpaOnnxDecodeOnlineStream(rec, st);

    const auto* r = L.SherpaOnnxGetOnlineStreamResult(rec, st);
    if(r)
    {
      if(r->text && r->text[0] != '\0')
      {
        job->text = r->text;
        job->has_text = true;
      }
      L.SherpaOnnxDestroyOnlineRecognizerResult(r);
    }

    if(job->endpointing && L.SherpaOnnxOnlineStreamIsEndpoint(rec, st))
    {
      job->is_endpoint = true;
      L.SherpaOnnxOnlineStreamReset(rec, st);
    }
  }

  return [job](OnlineRecognizer& self) {
    if(job->reload)
    {
      self.m_rec = job->rec;
      self.m_stream = job->stream;
      self.m_loaded_model = job->want_model;
    }
    if(job->has_text)
    {
      if(job->is_endpoint)
        self.outputs.final_text(std::string_view{job->text});
      else
        self.outputs.partial(std::string_view{job->text});
    }
    if(job->is_endpoint)
      self.outputs.endpoint();
    self.m_inflight.store(false, std::memory_order_release);
  };
}

}
