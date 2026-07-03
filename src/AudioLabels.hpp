#pragma once
// sherpa_audio_labels -- audio clip -> labels.
//
// One object covering BOTH sherpa-onnx audio tagging (a sound-event classifier:
// "Dog", "Speech", "Music", ...) and spoken language identification (whisper:
// "en", "de", "zh", ...). Both are "audio clip -> labels" and both feed their
// result through a reused SherpaOnnxOfflineStream, so a single Type enum + model
// auto-detection selects the head.
//
// Record a clip by toggling "Listen": while it is on the incoming audio is
// downmixed to mono and accumulated (RT-safe, no allocation); on the falling
// edge the buffer is handed to the worker, which (re)loads the model, feeds the
// offline stream (it resamples internally, so the host rate is passed straight
// through), computes, and publishes the labels back on the processing thread.
//
// All model loading and inference run on the worker; operator() only accumulates
// and dispatches. See helpers/AudioChunker.hpp for the threading contract.

#include "helpers/AudioChunker.hpp"
#include "helpers/Common.hpp"
#include "helpers/ModelConfig.hpp"

#include <halp/audio.hpp>
#include <halp/callback.hpp>
#include <halp/controls.hpp>
#include <halp/controls.enums.hpp>
#include <halp/file_port.hpp>
#include <halp/meta.hpp>

#include <atomic>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sherpa
{

class AudioLabels
{
public:
  halp_meta(name, "Sherpa Audio Labels")
  halp_meta(c_name, "sherpa_audio_labels")
  halp_meta(category, "AI/Audio")
  halp_meta(author, "Jean-Michaël Celerier")
  halp_meta(
      description,
      "Audio tagging and spoken language identification (clip -> labels)")
  halp_meta(uuid, "046ecb17-c823-42d8-9730-6fe6a9282c5e")

  enum class Type
  {
    Auto,
    AudioTagging,
    LanguageID
  };

  struct Label
  {
    std::string name;
    float score{};
    halp_field_names(name, score);
  };

  struct
  {
    halp::dynamic_audio_bus<"In", float> audio;
    halp::folder_port<"Model"> model;
    halp::enum_t<Type, "Type"> type;
    halp::hslider_i32<"Top K", halp::range{1., 20., 5.}> top_k;
    halp::enum_t<Provider, "Provider"> provider;
    halp::hslider_i32<"Threads", halp::range{1., 8., 1.}> threads;
    halp::toggle<"Listen"> listen;
  } inputs;

  struct
  {
    halp::val_port<"Labels", std::vector<Label>> labels;
    halp::callback<"Top", std::string_view, float> top;  // best label + score
    halp::callback<"Language", std::string_view> language; // for LID
  } outputs;

  // kind: 0 audio-tagging, 1 language-id, -1 none
  struct Job
  {
    std::vector<float> samples;
    double rate = 16000.;
    int num_threads = 1;
    int top_k = 5;
    bool reload = false;
    Provider provider = Provider::CPU;
    Type type = Type::Auto;
    std::string want_model;
    int kind = -1;
    std::shared_ptr<AudioTaggingHandle> tagger;
    std::shared_ptr<SpokenLanguageIdHandle> slid;
    std::vector<Label> labels;
    std::string top_name;
    float top_score = 0.f;
    std::string language;
    bool have_top = false;
    bool have_lang = false;
  };

  struct worker
  {
    std::function<void(std::shared_ptr<Job>)> request;
    static std::function<void(AudioLabels&)> work(std::shared_ptr<Job> job);
  } worker;

  void prepare(halp::setup info);
  void operator()(int frames);

private:
  void dispatch();

  bool m_available = SherpaLoader::instance().available;
  double m_host_rate = 48000.;
  int m_kind = -1;
  std::shared_ptr<AudioTaggingHandle> m_tagger;
  std::shared_ptr<SpokenLanguageIdHandle> m_slid;
  std::shared_ptr<Job> m_job;
  std::atomic<bool> m_inflight{false};
  MonoAccum m_accum;
  std::string m_requested_model, m_loaded_model;
  Type m_requested_type = Type::Auto;
  Type m_loaded_type = Type::Auto;
  bool m_reload = false;
  bool m_was_listening = false;

  // Outputs must be emitted from operator() (the run() cycle), not from the async
  // worker closure -- value outlets written outside run() do not propagate.
  std::vector<Label> m_emit_labels;
  std::string m_emit_top_name;
  float m_emit_top_score = 0.f;
  std::string m_emit_language;
  bool m_emit_have_top = false;
  bool m_emit_have_lang = false;
  bool m_emit_ready = false;

  void emit_pending();
};

inline void AudioLabels::prepare(halp::setup info)
{
  m_host_rate = info.rate > 0 ? info.rate : 48000.;
  // Cap the clip buffer (mono) -- e.g. 60 s at the host rate.
  m_accum.prepare(static_cast<std::size_t>(m_host_rate * 60.0));
  if(!m_job)
  {
    m_job = std::make_shared<Job>();
    m_job->samples.reserve(static_cast<std::size_t>(m_host_rate * 60.0));
    m_job->want_model.reserve(512);
  }
}

inline void AudioLabels::emit_pending()
{
  if(!m_emit_ready)
    return;
  outputs.labels.value = std::move(m_emit_labels);
  if(m_emit_have_top)
    outputs.top(std::string_view{m_emit_top_name}, m_emit_top_score);
  if(m_emit_have_lang)
    outputs.language(std::string_view{m_emit_language});
  m_emit_ready = false;
}

inline void AudioLabels::operator()(int frames)
{
  if(!m_available)
    return;

  if(inputs.model.value != m_requested_model || inputs.type.value != m_requested_type)
  {
    m_requested_model = inputs.model.value;
    m_requested_type = inputs.type.value;
    m_reload = true;
  }

  const bool on = inputs.listen.value;
  if(on)
    m_accum.push(inputs.audio, inputs.audio.channels, frames);

  // Falling edge of Listen -> classify the captured clip.
  if(m_was_listening && !on)
    dispatch();
  m_was_listening = on;

  emit_pending(); // emit from the run() cycle so value outlets propagate
}

inline void AudioLabels::dispatch()
{
  if(m_inflight.load(std::memory_order_acquire) || !m_job)
    return;
  if(m_accum.empty() && !m_reload)
    return;

  auto& job = *m_job;
  m_accum.drain_into(job.samples);
  job.rate = m_host_rate;
  job.num_threads = inputs.threads.value;
  job.top_k = inputs.top_k.value;
  job.provider = inputs.provider.value;
  job.want_model = m_requested_model;
  job.type = m_requested_type;
  job.reload = m_reload;
  job.kind = m_kind;
  job.tagger = m_tagger;
  job.slid = m_slid;
  m_reload = false;

  m_inflight.store(true, std::memory_order_release);
  if(worker.request)
    worker.request(m_job);
}

inline std::function<void(AudioLabels&)>
AudioLabels::worker::work(std::shared_ptr<Job> job)
{
  namespace fs = std::filesystem;
  const auto& L = SherpaLoader::instance();

  if(job->reload)
  {
    job->tagger.reset();
    job->slid.reset();
    job->kind = -1;

    fs::path d{job->want_model};
    const std::string dirname = model::to_lower(d.filename().string());
    const std::string enc = model::find_in_dir(d, {"encoder"}, ".onnx");
    const std::string dec = model::find_in_dir(d, {"decoder"}, ".onnx");

    Type t = job->type;
    if(t == Type::Auto)
    {
      if((!enc.empty() && !dec.empty())
         || dirname.find("whisper") != std::string::npos
         || dirname.find("lang") != std::string::npos
         || dirname.find("lid") != std::string::npos)
        t = Type::LanguageID;
      else
        t = Type::AudioTagging;
    }

    if(t == Type::LanguageID && L.SherpaOnnxCreateSpokenLanguageIdentification)
    {
      SherpaOnnxSpokenLanguageIdentificationConfig cfg;
      std::memset(&cfg, 0, sizeof(cfg));
      cfg.whisper.encoder = enc.c_str();
      cfg.whisper.decoder = dec.c_str();
      cfg.num_threads = job->num_threads;
      cfg.provider = provider_str(job->provider);
      job->slid = std::make_shared<SpokenLanguageIdHandle>(
          L.SherpaOnnxCreateSpokenLanguageIdentification(&cfg));
      if(*job->slid)
        job->kind = 1;
    }
    else if(L.SherpaOnnxCreateAudioTagging)
    {
      std::string ced = model::find_in_dir(d, {"ced"}, ".onnx");
      std::string mdl = model::find_in_dir(d, {"model"}, ".onnx");
      if(mdl.empty())
        mdl = model::find_in_dir(d, {}, ".onnx");
      std::string labels
          = model::find_in_dir(d, {"class_labels", "labels", "class"}, ".csv");
      if(labels.empty())
        labels = model::find_in_dir(d, {}, ".csv");

      SherpaOnnxAudioTaggingConfig cfg;
      std::memset(&cfg, 0, sizeof(cfg));
      if(!ced.empty())
        cfg.model.ced = ced.c_str();
      else
        cfg.model.zipformer.model = mdl.c_str();
      cfg.model.num_threads = job->num_threads;
      cfg.model.provider = provider_str(job->provider);
      cfg.labels = labels.c_str();
      cfg.top_k = job->top_k;
      job->tagger = std::make_shared<AudioTaggingHandle>(
          L.SherpaOnnxCreateAudioTagging(&cfg));
      if(*job->tagger)
        job->kind = 0;
    }
  }

  job->labels.clear();
  job->top_name.clear();
  job->top_score = 0.f;
  job->language.clear();
  job->have_top = false;
  job->have_lang = false;

  if(job->kind == 0 && job->tagger && *job->tagger && !job->samples.empty()
     && L.SherpaOnnxAudioTaggingCreateOfflineStream
     && L.SherpaOnnxAudioTaggingCompute)
  {
    OfflineStreamHandle stream{
        L.SherpaOnnxAudioTaggingCreateOfflineStream(job->tagger->get())};
    if(stream)
    {
      L.SherpaOnnxAcceptWaveformOffline(
          stream.get(), static_cast<int>(job->rate), job->samples.data(),
          static_cast<int>(job->samples.size()));
      const SherpaOnnxAudioEvent* const* res = L.SherpaOnnxAudioTaggingCompute(
          job->tagger->get(), stream.get(), job->top_k);
      if(res)
      {
        for(int i = 0; res[i] != nullptr; ++i)
        {
          Label lab;
          lab.name = res[i]->name ? res[i]->name : "";
          lab.score = res[i]->prob;
          job->labels.push_back(std::move(lab));
        }
        if(!job->labels.empty())
        {
          job->top_name = job->labels.front().name;
          job->top_score = job->labels.front().score;
          job->have_top = true;
        }
        L.SherpaOnnxAudioTaggingFreeResults(res);
      }
    }
  }
  else if(
      job->kind == 1 && job->slid && *job->slid && !job->samples.empty()
      && L.SherpaOnnxSpokenLanguageIdentificationCreateOfflineStream
      && L.SherpaOnnxSpokenLanguageIdentificationCompute)
  {
    OfflineStreamHandle stream{
        L.SherpaOnnxSpokenLanguageIdentificationCreateOfflineStream(
            job->slid->get())};
    if(stream)
    {
      L.SherpaOnnxAcceptWaveformOffline(
          stream.get(), static_cast<int>(job->rate), job->samples.data(),
          static_cast<int>(job->samples.size()));
      const auto* r = L.SherpaOnnxSpokenLanguageIdentificationCompute(
          job->slid->get(), stream.get());
      if(r)
      {
        if(r->lang && *r->lang)
        {
          job->language = r->lang;
          job->have_lang = true;
          Label lab;
          lab.name = r->lang;
          lab.score = 1.f;
          job->labels.push_back(std::move(lab));
          job->top_name = job->language;
          job->top_score = 1.f;
          job->have_top = true;
        }
        L.SherpaOnnxDestroySpokenLanguageIdentificationResult(r);
      }
    }
  }

  return [job](AudioLabels& self) {
    if(job->reload)
    {
      self.m_tagger = job->tagger;
      self.m_slid = job->slid;
      self.m_kind = job->kind;
      self.m_loaded_model = job->want_model;
      self.m_loaded_type = job->type;
    }
    // Stash for operator() to emit from the run() cycle (see emit_pending).
    self.m_emit_labels = std::move(job->labels);
    self.m_emit_have_top = job->have_top;
    self.m_emit_top_name = std::move(job->top_name);
    self.m_emit_top_score = job->top_score;
    self.m_emit_have_lang = job->have_lang;
    self.m_emit_language = std::move(job->language);
    self.m_emit_ready = true;
    self.m_inflight.store(false, std::memory_order_release);
  };
}

}
