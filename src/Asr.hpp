#pragma once
// sherpa_asr -- unified speech recognition (streaming + batch behind one object).
//
// A single object that presents identical I/O regardless of the underlying model
// family: continuous audio in, finalized utterances out on "Final" (plus a live
// "Partial" hypothesis and per-utterance metadata). Two engines back it, chosen by
// the "Mode" enum:
//   * Streaming : a stateful sherpa-onnx ONLINE recognizer decodes the running
//     stream; endpoint detection finalizes each utterance (like sherpa_asr_online).
//   * Batch     : a NON-streaming (offline) recognizer -- whisper / sense-voice /
//     paraformer / moonshine / ... -- is paired with an INTERNAL VAD that segments
//     the live stream; each completed speech segment is decoded in one shot and
//     emitted on "Final". This makes non-streaming models usable on a continuous
//     input without a push-to-talk toggle. Needs a VAD model (the "VAD model" port,
//     or a silero/ten VAD found inside the ASR model dir).
//   * Auto      : try to build the streaming engine first; if the model is not
//     streaming-capable, fall back to the batch (offline + VAD) engine.
//
// operator() (audio thread) only downmixes + accumulates and dispatches; every
// sherpa call (model load, resampling, VAD, decoding) runs on the worker thread.
// Only one decode is in flight at a time -- new blocks accumulate until it returns.
//
// This object COMPLEMENTS the dedicated sherpa_asr_online / sherpa_asr_offline
// objects (which stay for explicit streaming and push-to-talk workflows); it does
// not replace them.

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

class Asr
{
public:
  halp_meta(name, "Sherpa ASR")
  halp_meta(c_name, "sherpa_asr")
  halp_meta(category, "AI/Audio")
  halp_meta(author, "Jean-Michaël Celerier")
  halp_meta(
      description,
      "Unified speech-to-text with sherpa-onnx: streaming models finalize via "
      "endpointing, non-streaming models via an internal VAD (continuous audio "
      "in -> utterances out)")
  halp_meta(uuid, "f4ef68b4-cf87-4558-9605-217ab3db2e53")

  enum class Mode
  {
    Auto,
    Streaming,
    Batch
  };
  enum class Engine
  {
    None,
    Streaming,
    Batch
  };

  struct
  {
    halp::dynamic_audio_bus<"In", float> audio;
    halp::folder_port<"Model"> model;
    halp::folder_port<"VAD model"> vad_model; // for Batch/non-streaming models
    halp::enum_t<Mode, "Mode"> mode;
    halp::toggle<"Endpointing", halp::toggle_setup{.init = true}> endpointing;
    halp::enum_t<Decoding, "Decoding"> decoding;
    halp::enum_t<Provider, "Provider"> provider;
    halp::hslider_i32<"Threads", halp::range{1., 16., 1.}> threads;
    halp::lineedit<"Model type", ""> model_type; // force a family; empty = auto
    halp::lineedit<"Advanced", ""> advanced;
  } inputs;

  struct
  {
    halp::callback<"Partial", std::string_view> partial;
    halp::callback<"Final", std::string_view> final_text;
    halp::callback<"Endpoint"> endpoint;
    halp::val_port<"Tokens", std::vector<std::string>> tokens;
    halp::val_port<"Timestamps", std::vector<float>> timestamps;
    halp::callback<"Lang", std::string_view> lang;
    halp::callback<"Emotion", std::string_view> emotion;
    halp::callback<"Event", std::string_view> event;
  } outputs;

  // One finalized utterance + its metadata.
  struct Result
  {
    std::string text;
    std::vector<std::string> tokens;
    std::vector<float> timestamps;
    std::string lang, emotion, event;
  };

  struct Job
  {
    std::vector<float> samples;
    double rate = 16000.;
    int num_threads = 1;
    Provider provider = Provider::CPU;
    Decoding decoding = Decoding::GreedySearch;
    Mode mode = Mode::Auto;
    bool endpointing = true;
    bool reload = false;
    std::string want_model;
    std::string want_vad;
    std::string advanced;
    std::string model_type;

    Engine engine = Engine::None;
    std::shared_ptr<OnlineRecognizerHandle> online;
    std::shared_ptr<OnlineStreamHandle> stream;
    std::shared_ptr<OfflineRecognizerHandle> offline;
    std::shared_ptr<VadHandle> vad;
    std::shared_ptr<LinearResamplerHandle> resampler; // host -> 16k, batch only

    // Results produced this dispatch.
    std::string partial;
    bool has_partial = false;
    std::vector<Result> finals;
  };

  struct worker
  {
    std::function<void(std::shared_ptr<Job>)> request;
    static std::function<void(Asr&)> work(std::shared_ptr<Job> job);
  } worker;

  void prepare(halp::setup info);
  void operator()(int frames);

private:
  void dispatch();

  bool m_available = SherpaLoader::instance().available;
  double m_host_rate = 48000.;
  Engine m_engine = Engine::None;
  std::shared_ptr<OnlineRecognizerHandle> m_online;
  std::shared_ptr<OnlineStreamHandle> m_stream;
  std::shared_ptr<OfflineRecognizerHandle> m_offline;
  std::shared_ptr<VadHandle> m_vad;
  std::shared_ptr<LinearResamplerHandle> m_resampler;
  std::shared_ptr<Job> m_job;
  std::atomic<bool> m_inflight{false};
  MonoAccum m_accum;
  std::string m_requested_model, m_loaded_model;
  std::string m_requested_vad, m_loaded_vad;
  Mode m_requested_mode = Mode::Auto, m_loaded_mode = Mode::Auto;
  bool m_reload = false;

  // Results handed from the worker-result closure to operator(). Outputs MUST be
  // emitted from the node's run() (operator()), not from the async closure: value
  // outlets written outside the run() cycle never propagate to score's ports (the
  // closure runs via the execution queue, a different point than the node tick).
  // The closure (also on the DSP thread) stashes here; operator() drains + emits.
  std::string m_emit_partial;
  bool m_emit_has_partial = false;
  std::vector<Result> m_emit_finals;

  void emit_pending();
};

inline void Asr::prepare(halp::setup info)
{
  m_host_rate = info.rate > 0 ? info.rate : 48000.;
  m_accum.prepare(static_cast<std::size_t>(m_host_rate * 4.0));
  if(!m_job)
  {
    m_job = std::make_shared<Job>();
    m_job->samples.reserve(static_cast<std::size_t>(m_host_rate * 4.0));
    m_job->want_model.reserve(512);
    m_job->want_vad.reserve(512);
    m_job->partial.reserve(1024);
  }
  m_emit_partial.reserve(1024);
  m_emit_finals.reserve(16);
}

inline void Asr::emit_pending()
{
  // Runs inside operator() (the node run() cycle), so these value-outlet writes
  // actually propagate to score's ports.
  if(m_emit_has_partial)
  {
    outputs.partial(std::string_view{m_emit_partial});
    m_emit_has_partial = false;
  }
  if(!m_emit_finals.empty())
  {
    SHERPA_DBG("[sherpa] ASR EMIT finals=%zu partial-pending=%d\n",
               m_emit_finals.size(), m_emit_has_partial ? 1 : 0);
    for(auto& res : m_emit_finals)
    {
      outputs.final_text(std::string_view{res.text});
      outputs.endpoint();
      if(!res.lang.empty())
        outputs.lang(std::string_view{res.lang});
      if(!res.emotion.empty())
        outputs.emotion(std::string_view{res.emotion});
      if(!res.event.empty())
        outputs.event(std::string_view{res.event});
      outputs.tokens.value = std::move(res.tokens);
      outputs.timestamps.value = std::move(res.timestamps);
    }
    m_emit_finals.clear();
  }
}

inline void Asr::operator()(int frames)
{
  if(!m_available)
    return;

  if(inputs.model.value != m_requested_model)
  {
    m_requested_model = inputs.model.value;
    m_reload = true;
  }
  if(inputs.vad_model.value != m_requested_vad)
  {
    m_requested_vad = inputs.vad_model.value;
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

  // Emit results from the run() cycle so the value outlets propagate. Placed after
  // dispatch() so that with a synchronous worker (tests) the result stashed during
  // dispatch is flushed the same tick; in score the async closure stashes and the
  // next operator() flushes it.
  emit_pending();
}

inline void Asr::dispatch()
{
  if(!m_job)
    return;
  auto& job = *m_job;
  m_accum.drain_into(job.samples);
  job.rate = m_host_rate;
  job.num_threads = inputs.threads.value;
  job.provider = inputs.provider.value;
  job.decoding = inputs.decoding.value;
  job.mode = m_requested_mode;
  job.endpointing = inputs.endpointing.value;
  job.want_model = m_requested_model;
  job.want_vad = m_requested_vad;
  job.advanced = inputs.advanced.value;
  job.model_type = inputs.model_type.value;
  job.reload = m_reload;
  job.engine = m_engine;
  job.online = m_online;
  job.stream = m_stream;
  job.offline = m_offline;
  job.vad = m_vad;
  job.resampler = m_resampler;
  m_reload = false;

  m_inflight.store(true, std::memory_order_release);
  if(worker.request)
    worker.request(m_job);
}

inline std::function<void(Asr&)> Asr::worker::work(std::shared_ptr<Job> job)
{
  const auto& L = SherpaLoader::instance();

  // ---- (Re)build the engine on a model / VAD / mode change -----------------
  if(job->reload)
  {
    job->online.reset();
    job->stream.reset();
    job->offline.reset();
    job->vad.reset();
    job->resampler.reset();
    job->engine = Engine::None;

    const bool try_stream
        = (job->mode == Mode::Streaming) || (job->mode == Mode::Auto);
    const bool try_batch
        = (job->mode == Mode::Batch) || (job->mode == Mode::Auto);

    if(try_stream)
    {
      auto rec = model::create_online_recognizer(
          job->want_model, job->num_threads, decoding_str(job->decoding),
          job->endpointing, "", provider_str(job->provider), job->advanced);
      if(rec)
      {
        job->online = std::make_shared<OnlineRecognizerHandle>(std::move(rec));
        job->stream = std::make_shared<OnlineStreamHandle>(
            L.SherpaOnnxCreateOnlineStream(job->online->get()));
        job->engine = Engine::Streaming;
      }
    }

    if(job->engine == Engine::None && try_batch)
    {
      auto rec = model::create_offline_recognizer(
          job->want_model, job->num_threads, decoding_str(job->decoding),
          provider_str(job->provider), job->advanced, job->model_type);
      if(rec)
      {
        job->offline = std::make_shared<OfflineRecognizerHandle>(std::move(rec));
        // The VAD segments the live stream. Prefer the dedicated port; else look
        // for a VAD model inside the ASR model dir.
        const std::string& vad_dir
            = !job->want_vad.empty() ? job->want_vad : job->want_model;
        job->vad = std::make_shared<VadHandle>(model::create_vad(
            vad_dir, 0.5f, 16000, job->num_threads, 30.f,
            provider_str(job->provider), job->advanced));
        job->resampler = std::make_shared<LinearResamplerHandle>(
            L.SherpaOnnxCreateLinearResampler(
                static_cast<int>(job->rate), 16000, 0.f, 0));
        job->engine = Engine::Batch;
      }
    }
    SHERPA_DBG(
        "[sherpa] ASR reload mode=%d -> engine=%d (online=%d stream=%d "
        "offline=%d vad=%d) model='%s' vad='%s'\n",
        (int)job->mode, (int)job->engine, job->online && *job->online,
        job->stream && *job->stream, job->offline && *job->offline,
        job->vad && *job->vad, job->want_model.c_str(), job->want_vad.c_str());
  }

  job->partial.clear();
  job->has_partial = false;
  job->finals.clear();

  if(job->samples.empty())
    return [job](Asr& self) {
      if(job->reload)
      {
        self.m_engine = job->engine;
        self.m_online = job->online;
        self.m_stream = job->stream;
        self.m_offline = job->offline;
        self.m_vad = job->vad;
        self.m_resampler = job->resampler;
        self.m_loaded_model = job->want_model;
        self.m_loaded_vad = job->want_vad;
        self.m_loaded_mode = job->mode;
      }
      self.m_inflight.store(false, std::memory_order_release);
    };

  // ---- Streaming engine ----------------------------------------------------
  if(job->engine == Engine::Streaming && job->online && *job->online
     && job->stream && *job->stream)
  {
    auto* rec = job->online->get();
    auto* st = job->stream->get();
    L.SherpaOnnxOnlineStreamAcceptWaveform(
        st, static_cast<int>(job->rate), job->samples.data(),
        static_cast<int>(job->samples.size()));
    while(L.SherpaOnnxIsOnlineStreamReady(rec, st))
      L.SherpaOnnxDecodeOnlineStream(rec, st);

    const bool is_endpoint
        = job->endpointing && L.SherpaOnnxOnlineStreamIsEndpoint(rec, st);

    const auto* r = L.SherpaOnnxGetOnlineStreamResult(rec, st);
    if(r)
    {
      if(r->text && r->text[0] != '\0')
      {
        if(is_endpoint)
        {
          Result res;
          res.text = r->text;
          if(r->tokens_arr && r->count > 0)
            for(int i = 0; i < r->count; ++i)
              if(r->tokens_arr[i])
                res.tokens.emplace_back(r->tokens_arr[i]);
          if(r->timestamps && r->count > 0)
            res.timestamps.assign(r->timestamps, r->timestamps + r->count);
          job->finals.push_back(std::move(res));
        }
        else
        {
          job->partial = r->text;
          job->has_partial = true;
        }
      }
      L.SherpaOnnxDestroyOnlineRecognizerResult(r);
    }

    if(is_endpoint)
      L.SherpaOnnxOnlineStreamReset(rec, st);
  }
  // ---- Batch engine (offline recognizer + internal VAD) --------------------
  else if(
      job->engine == Engine::Batch && job->offline && *job->offline && job->vad
      && *job->vad && job->resampler && *job->resampler)
  {
    auto* off = job->offline->get();
    auto* vad = job->vad->get();

    // Feed the VAD at 16 kHz (stateful resampler, continuous across dispatches).
    const auto* ro = L.SherpaOnnxLinearResamplerResample(
        job->resampler->get(), job->samples.data(),
        static_cast<int>(job->samples.size()), 0);
    if(ro)
    {
      if(ro->n > 0)
        L.SherpaOnnxVoiceActivityDetectorAcceptWaveform(vad, ro->samples, ro->n);
      L.SherpaOnnxLinearResamplerResampleFree(ro);
    }

    // Decode every completed speech segment in one shot.
    while(L.SherpaOnnxVoiceActivityDetectorEmpty(vad) == 0)
    {
      const auto* seg = L.SherpaOnnxVoiceActivityDetectorFront(vad);
      if(seg && seg->samples && seg->n > 0)
      {
        OfflineStreamHandle st{L.SherpaOnnxCreateOfflineStream(off)};
        if(st)
        {
          L.SherpaOnnxAcceptWaveformOffline(st.get(), 16000, seg->samples, seg->n);
          L.SherpaOnnxDecodeOfflineStream(off, st.get());
          const auto* r = L.SherpaOnnxGetOfflineStreamResult(st.get());
          if(r)
          {
            Result res;
            if(r->text)
              res.text = r->text;
            if(r->tokens_arr && r->count > 0)
              for(int i = 0; i < r->count; ++i)
                if(r->tokens_arr[i])
                  res.tokens.emplace_back(r->tokens_arr[i]);
            if(r->timestamps && r->count > 0)
              res.timestamps.assign(r->timestamps, r->timestamps + r->count);
            if(r->lang)
              res.lang = r->lang;
            if(r->emotion)
              res.emotion = r->emotion;
            if(r->event)
              res.event = r->event;
            if(!res.text.empty())
              job->finals.push_back(std::move(res));
            L.SherpaOnnxDestroyOfflineRecognizerResult(r);
          }
        }
      }
      if(seg)
        L.SherpaOnnxDestroySpeechSegment(seg);
      L.SherpaOnnxVoiceActivityDetectorPop(vad);
    }
  }

  // Per-dispatch trace (only when audio flows or results appear, to avoid
  // spamming the silent reload-only ticks): tells "audio never reached the
  // object" (no lines with samples>0) apart from "audio flows but nothing is
  // emitted" (samples>0 but finals=0/partial=0).
  if(!job->samples.empty() || job->has_partial || !job->finals.empty())
    SHERPA_DBG(
        "[sherpa] ASR dispatch engine=%d samples=%zu partial=%d finals=%zu\n",
        (int)job->engine, job->samples.size(), job->has_partial ? 1 : 0,
        job->finals.size());

  return [job](Asr& self) {
    if(job->reload)
    {
      self.m_engine = job->engine;
      self.m_online = job->online;
      self.m_stream = job->stream;
      self.m_offline = job->offline;
      self.m_vad = job->vad;
      self.m_resampler = job->resampler;
      self.m_loaded_model = job->want_model;
      self.m_loaded_vad = job->want_vad;
      self.m_loaded_mode = job->mode;
    }

    // Hand results to operator() rather than emitting here: this closure runs on
    // the DSP thread but via the execution queue, outside the node run() cycle, so
    // value outlets written here would not propagate. emit_pending() emits them.
    if(job->has_partial)
    {
      self.m_emit_partial = std::move(job->partial);
      self.m_emit_has_partial = true;
    }
    for(auto& res : job->finals)
      self.m_emit_finals.push_back(std::move(res));

    self.m_inflight.store(false, std::memory_order_release);
  };
}

}
