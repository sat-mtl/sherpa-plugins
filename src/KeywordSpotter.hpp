#pragma once
// sherpa_keyword_spotter -- streaming keyword / wake-word spotting.
//
// Continuously feeds the incoming audio (downmixed to mono) to a persistent
// sherpa-onnx keyword spotter + online stream and fires a callback each time a
// configured keyword is detected. sherpa resamples the input internally, so the
// host rate is passed through as-is.
//
// Keywords come from either the "Keywords" line-edit (inline, tokenized, one per
// line -> SherpaOnnxCreateKeywordStreamWithKeywords) or, when that is empty, the
// "Keywords file" port (a path baked into the spotter config -> plain
// SherpaOnnxCreateKeywordStream).
//
// The spotter/stream are stateful and live across calls; only one decode is in
// flight at a time (a new block's audio accumulates until the previous decode
// returns, so nothing is dropped). All sherpa calls run on the worker thread.
// After a keyword fires the stream is reset so the next keyword is detected
// independently.

#include "helpers/AudioChunker.hpp"
#include "helpers/Common.hpp"
#include "helpers/ModelConfig.hpp"

#include <halp/audio.hpp>
#include <halp/callback.hpp>
#include <halp/controls.hpp>
#include <halp/controls.basic.hpp>
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
#include <vector>

namespace sherpa
{

// Build a keyword spotter from a model directory. KWS models are streaming
// transducers (encoder/decoder/joiner + tokens); anything else yields an empty
// (invalid) handle. Runs on the worker thread. Null-checks the extended symbol.
inline KeywordSpotterHandle create_keyword_spotter(
    std::string_view dir, int num_threads, float threshold,
    std::string_view keywords_file, const char* provider = "cpu")
{
  const auto& L = SherpaLoader::instance();
  if(!L.SherpaOnnxCreateKeywordSpotter)
    return {};

  std::filesystem::path d{dir};

  std::string tokens = model::file_if_exists(d, "tokens.txt");
  if(tokens.empty())
    tokens = model::find_in_dir(d, {"tokens"}, ".txt");
  std::string enc = model::find_in_dir(d, {"encoder"});
  std::string dec = model::find_in_dir(d, {"decoder"});
  std::string joiner = model::find_in_dir(d, {"joiner"});

  if(enc.empty() || dec.empty() || joiner.empty() || tokens.empty())
    return {};

  std::string kwfile{keywords_file};

  SherpaOnnxKeywordSpotterConfig cfg;
  std::memset(&cfg, 0, sizeof(cfg));
  cfg.feat_config.sample_rate = 16000; // input is resampled to this internally
  cfg.feat_config.feature_dim = 80;
  cfg.model_config.tokens = tokens.c_str();
  cfg.model_config.num_threads = num_threads;
  cfg.model_config.provider = provider;
  cfg.model_config.transducer.encoder = enc.c_str();
  cfg.model_config.transducer.decoder = dec.c_str();
  cfg.model_config.transducer.joiner = joiner.c_str();
  cfg.max_active_paths = 4;
  cfg.num_trailing_blanks = 1;
  cfg.keywords_score = 1.0f;
  cfg.keywords_threshold = threshold;
  if(!kwfile.empty())
    cfg.keywords_file = kwfile.c_str();

  return KeywordSpotterHandle{L.SherpaOnnxCreateKeywordSpotter(&cfg)};
}

class KeywordSpotter
{
public:
  halp_meta(name, "Sherpa Keyword Spotter")
  halp_meta(c_name, "sherpa_keyword_spotter")
  halp_meta(category, "AI/Audio")
  halp_meta(author, "Jean-Michaël Celerier")
  halp_meta(description, "Streaming keyword / wake-word spotting with sherpa-onnx")
  halp_meta(uuid, "0edcb900-a505-4426-a54c-8eade2216e41")

  struct
  {
    halp::dynamic_audio_bus<"In", float> audio;
    halp::folder_port<"Model"> model;
    halp::file_port<"Keywords file"> keywords_file;
    halp::lineedit<"Keywords", ""> keywords;
    halp::knob_f32<"Threshold", halp::range{0., 1., 0.25}> threshold;
    halp::enum_t<Provider, "Provider"> provider;
    halp::hslider_i32<"Threads", halp::range{1., 8., 1.}> threads;
  } inputs;

  struct
  {
    halp::callback<"Keyword", std::string_view> keyword;
    halp::callback<"Detected"> detected;
    halp::val_port<"Last", std::string> last;
  } outputs;

  struct Job
  {
    std::vector<float> samples;
    double rate = 16000.;
    int num_threads = 1;
    float threshold = 0.25f;
    bool reload = false;
    Provider provider = Provider::CPU;
    std::string want_model;
    std::string want_keywords;      // inline keywords (line-edit)
    std::string want_keywords_file; // keywords file path
    std::shared_ptr<KeywordSpotterHandle> spotter;
    std::shared_ptr<OnlineStreamHandle> stream;
    std::string keyword;
    bool has_keyword = false;
  };

  struct worker
  {
    std::function<void(std::shared_ptr<Job>)> request;
    static std::function<void(KeywordSpotter&)> work(std::shared_ptr<Job> job);
  } worker;

  void prepare(halp::setup info);
  void operator()(int frames);

private:
  void dispatch();

  bool m_available = SherpaLoader::instance().available
                     && SherpaLoader::instance().SherpaOnnxCreateKeywordSpotter != nullptr;
  double m_host_rate = 48000.;
  std::shared_ptr<KeywordSpotterHandle> m_spotter;
  std::shared_ptr<OnlineStreamHandle> m_stream;
  std::shared_ptr<Job> m_job;
  std::atomic<bool> m_inflight{false};
  MonoAccum m_accum;
  std::string m_requested_model, m_loaded_model;
  std::string m_requested_keywords, m_loaded_keywords;
  std::string m_requested_keywords_file, m_loaded_keywords_file;
  float m_requested_threshold = 0.25f, m_loaded_threshold = 0.25f;
  bool m_reload = false;
};

inline void KeywordSpotter::prepare(halp::setup info)
{
  m_host_rate = info.rate > 0 ? info.rate : 48000.;
  // A few seconds of mono headroom in case a decode runs long.
  m_accum.prepare(static_cast<std::size_t>(m_host_rate * 4.0));
  if(!m_job)
  {
    m_job = std::make_shared<Job>();
    m_job->samples.reserve(static_cast<std::size_t>(m_host_rate * 4.0));
    m_job->want_model.reserve(512);
    m_job->want_keywords.reserve(512);
    m_job->want_keywords_file.reserve(512);
    m_job->keyword.reserve(256);
  }
}

inline void KeywordSpotter::operator()(int frames)
{
  if(!m_available)
    return;

  if(inputs.model.value != m_requested_model)
  {
    m_requested_model = inputs.model.value;
    m_reload = true;
  }
  if(inputs.keywords.value != m_requested_keywords)
  {
    m_requested_keywords = inputs.keywords.value;
    m_reload = true;
  }
  if(inputs.keywords_file.file.filename != m_requested_keywords_file)
  {
    m_requested_keywords_file = inputs.keywords_file.file.filename;
    m_reload = true;
  }
  if(inputs.threshold.value != m_requested_threshold)
  {
    m_requested_threshold = inputs.threshold.value;
    m_reload = true;
  }

  m_accum.push(inputs.audio, inputs.audio.channels, frames);

  if(!m_inflight.load(std::memory_order_acquire) && (!m_accum.empty() || m_reload))
    dispatch();
}

inline void KeywordSpotter::dispatch()
{
  if(!m_job)
    return;
  auto& job = *m_job;
  m_accum.drain_into(job.samples);
  job.rate = m_host_rate;
  job.num_threads = inputs.threads.value;
  job.threshold = m_requested_threshold;
  job.provider = inputs.provider.value;
  job.want_model = m_requested_model;
  job.want_keywords = m_requested_keywords;
  job.want_keywords_file = m_requested_keywords_file;
  job.reload = m_reload;
  job.spotter = m_spotter;
  job.stream = m_stream;
  m_reload = false;

  m_inflight.store(true, std::memory_order_release);
  if(worker.request)
    worker.request(m_job);
}

inline std::function<void(KeywordSpotter&)>
KeywordSpotter::worker::work(std::shared_ptr<Job> job)
{
  const auto& L = SherpaLoader::instance();

  // Guard the extended keyword-spotting symbols: if any is missing, the feature
  // is unavailable in this runtime -> no-op and clear the in-flight flag.
  const bool have_kws
      = L.SherpaOnnxCreateKeywordSpotter && L.SherpaOnnxCreateKeywordStream
        && L.SherpaOnnxCreateKeywordStreamWithKeywords
        && L.SherpaOnnxIsKeywordStreamReady && L.SherpaOnnxDecodeKeywordStream
        && L.SherpaOnnxResetKeywordStream && L.SherpaOnnxGetKeywordResult
        && L.SherpaOnnxDestroyKeywordResult
        && L.SherpaOnnxOnlineStreamAcceptWaveform;

  if(have_kws && (job->reload || !job->spotter || !*job->spotter))
  {
    job->spotter = std::make_shared<KeywordSpotterHandle>(create_keyword_spotter(
        job->want_model, job->num_threads, job->threshold,
        job->want_keywords_file, provider_str(job->provider)));
    job->stream.reset();
  }
  if(have_kws && job->spotter && *job->spotter && (!job->stream || !*job->stream))
  {
    auto* kws = job->spotter->get();
    const SherpaOnnxOnlineStream* st = nullptr;
    if(!job->want_keywords.empty())
      st = L.SherpaOnnxCreateKeywordStreamWithKeywords(kws, job->want_keywords.c_str());
    else
      st = L.SherpaOnnxCreateKeywordStream(kws);
    job->stream = std::make_shared<OnlineStreamHandle>(st);
  }

  job->keyword.clear();
  job->has_keyword = false;

  if(have_kws && job->spotter && *job->spotter && job->stream && *job->stream
     && !job->samples.empty())
  {
    auto* kws = job->spotter->get();
    auto* st = job->stream->get();
    L.SherpaOnnxOnlineStreamAcceptWaveform(
        st, static_cast<int>(job->rate), job->samples.data(),
        static_cast<int>(job->samples.size()));
    while(L.SherpaOnnxIsKeywordStreamReady(kws, st))
      L.SherpaOnnxDecodeKeywordStream(kws, st);

    const auto* r = L.SherpaOnnxGetKeywordResult(kws, st);
    if(r)
    {
      if(r->keyword && r->keyword[0] != '\0')
      {
        job->keyword = r->keyword;
        job->has_keyword = true;
      }
      L.SherpaOnnxDestroyKeywordResult(r);
    }

    if(job->has_keyword)
      L.SherpaOnnxResetKeywordStream(kws, st); // re-arm for the next trigger
  }

  return [job](KeywordSpotter& self) {
    if(job->reload)
    {
      self.m_spotter = job->spotter;
      self.m_stream = job->stream;
      self.m_loaded_model = job->want_model;
      self.m_loaded_keywords = job->want_keywords;
      self.m_loaded_keywords_file = job->want_keywords_file;
      self.m_loaded_threshold = job->threshold;
    }
    if(job->has_keyword)
    {
      self.outputs.last.value = job->keyword;
      self.outputs.keyword(std::string_view{job->keyword});
      self.outputs.detected();
    }
    self.m_inflight.store(false, std::memory_order_release);
  };
}

}
