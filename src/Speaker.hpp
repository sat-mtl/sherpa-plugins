#pragma once
// sherpa_speaker -- speaker identification / verification.
//
// Record an utterance into the audio input (toggle "Capture"); on the falling
// edge the accumulated mono audio is sent to a worker thread which feeds it
// through a sherpa-onnx SpeakerEmbeddingExtractor and produces a fixed-length
// speaker embedding vector. The embedding is published, and then -- depending on
// the "Enroll as" field / the enroll/remove/clear messages -- either enrolled
// into an in-memory SpeakerEmbeddingManager under a name, or searched against the
// already-enrolled speakers to identify who spoke (emitting the matched name +
// similarity score).
//
// Threading (see helpers/AudioChunker.hpp): operator() (audio thread) only
// downmixes + accumulates and dispatches on the Capture falling edge. The
// extractor (model load + inference) runs on the worker thread. The manager is
// cheap, so all of its operations (create / add / search / remove) run on the
// processing thread -- in the worker's returned closure and in the message
// handlers -- keeping the manager owned by a single thread.

#include "helpers/AudioChunker.hpp"
#include "helpers/ModelConfig.hpp"

#include <halp/audio.hpp>
#include <halp/callback.hpp>
#include <halp/controls.hpp>
#include <halp/file_port.hpp>
#include <halp/messages.hpp>
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

// Build a SpeakerEmbeddingExtractor from a "Model" folder (or a direct .onnx
// path). Runs on the worker thread (std::filesystem + allocation). Null-checks
// the extended create symbol and reports an empty handle when unavailable.
inline SpeakerEmbeddingExtractorHandle
make_speaker_extractor(std::string_view dir, int num_threads)
{
  const auto& L = SherpaLoader::instance();
  if(!L.SherpaOnnxCreateSpeakerEmbeddingExtractor)
    return {};

  namespace fs = std::filesystem;
  fs::path p{dir};
  std::error_code ec;

  std::string model;
  if(fs::is_regular_file(p, ec))
  {
    model = p.string();
  }
  else
  {
    // Prefer a file whose name hints at a speaker-embedding model, else the
    // first .onnx in the directory.
    model = model::find_in_dir(
        p, {"embedding", "campplus", "speaker", "eres2net", "wespeaker",
            "3dspeaker", "voxceleb", "nemo"});
    if(model.empty())
      model = model::find_in_dir(p, {}); // first .onnx
  }
  if(model.empty())
    return {};

  SherpaOnnxSpeakerEmbeddingExtractorConfig cfg;
  std::memset(&cfg, 0, sizeof(cfg));
  cfg.model = model.c_str();
  cfg.num_threads = num_threads;
  cfg.debug = 0;
  cfg.provider = "cpu";

  return SpeakerEmbeddingExtractorHandle{
      L.SherpaOnnxCreateSpeakerEmbeddingExtractor(&cfg)};
}

class Speaker
{
public:
  halp_meta(name, "Sherpa Speaker ID")
  halp_meta(c_name, "sherpa_speaker")
  halp_meta(category, "AI/Audio")
  halp_meta(author, "Jean-Michaël Celerier")
  halp_meta(
      description,
      "Speaker identification / verification with sherpa-onnx "
      "(capture -> embedding, enroll + identify)")
  halp_meta(uuid, "c719a44c-81e9-4a8b-a974-4ee6bd4334d0")

  struct
  {
    halp::dynamic_audio_bus<"In", float> audio;
    halp::folder_port<"Model"> model;
    halp::hslider_i32<"Threads", halp::range{1., 8., 1.}> threads;
    halp::knob_f32<"Threshold", halp::range{0., 1., 0.5}> threshold;
    halp::toggle<"Capture"> capture;
    halp::lineedit<"Enroll as", ""> enroll_name;
  } inputs;

  struct
  {
    halp::val_port<"Embedding", std::vector<float>> embedding;
    halp::callback<"Speaker", std::string_view> speaker;
    halp::val_port<"Score", float> score;
    halp::val_port<"Count", int> count;
  } outputs;

  // Messages (Max/Pd/ossia): operate on the LAST computed embedding.
  //  enroll <name> : store the last embedding under <name>
  //  remove <name> : drop <name> from the manager
  //  clear         : forget all enrolled speakers
  struct messages
  {
    struct
    {
      halp_meta(name, "enroll")
      void operator()(Speaker& self, std::string name) { self.enroll(name); }
    } enroll;

    struct
    {
      halp_meta(name, "remove")
      void operator()(Speaker& self, std::string name) { self.remove(name); }
    } remove;

    struct
    {
      halp_meta(name, "clear")
      void operator()(Speaker& self) { self.clear(); }
    } clear;
  };

  struct Job
  {
    std::vector<float> samples;
    double rate = 16000.;
    int num_threads = 1;
    float threshold = 0.5f;
    bool reload = false;
    std::string want_model;
    std::string enroll_name;
    std::shared_ptr<SpeakerEmbeddingExtractorHandle> extractor;
    std::vector<float> embedding;
    int dim = 0;
  };

  struct worker
  {
    std::function<void(std::shared_ptr<Job>)> request;
    static std::function<void(Speaker&)> work(std::shared_ptr<Job> job);
  } worker;

  void prepare(halp::setup info);
  void operator()(int frames);

  // Manager operations -- cheap, run on the processing thread.
  void enroll(const std::string& name);
  void remove(const std::string& name);
  void clear();

private:
  void dispatch();
  void apply(Job& job);
  void ensure_manager();
  void identify(const std::vector<float>& v, float threshold);
  void update_count();

  bool m_available = SherpaLoader::instance().available;
  double m_host_rate = 48000.;
  std::shared_ptr<SpeakerEmbeddingExtractorHandle> m_extractor;
  std::shared_ptr<SpeakerEmbeddingManagerHandle> m_manager;
  std::vector<float> m_last_embedding;
  int m_last_dim = 0;
  std::shared_ptr<Job> m_job;
  std::atomic<bool> m_inflight{false};
  MonoAccum m_accum;
  std::string m_requested_model, m_loaded_model;
  bool m_reload = false;
  bool m_was_capturing = false;
};

inline void Speaker::prepare(halp::setup info)
{
  m_host_rate = info.rate > 0 ? info.rate : 48000.;
  // Cap the capture buffer (mono) -- e.g. 60 s at the host rate.
  m_accum.prepare(static_cast<std::size_t>(m_host_rate * 60.0));
  if(!m_job)
  {
    m_job = std::make_shared<Job>();
    m_job->samples.reserve(static_cast<std::size_t>(m_host_rate * 60.0));
    m_job->want_model.reserve(512);
    m_job->enroll_name.reserve(256);
  }
}

inline void Speaker::operator()(int frames)
{
  if(!m_available)
    return;

  if(inputs.model.value != m_requested_model)
  {
    m_requested_model = inputs.model.value;
    m_reload = true;
  }

  const bool cap = inputs.capture.value;
  if(cap)
    m_accum.push(inputs.audio, inputs.audio.channels, frames);

  // Falling edge of Capture -> compute the embedding for the utterance.
  if(m_was_capturing && !cap)
    dispatch();
  m_was_capturing = cap;
}

inline void Speaker::dispatch()
{
  if(m_inflight.load(std::memory_order_acquire) || !m_job)
    return;
  if(m_accum.empty() && !m_reload)
    return;

  auto& job = *m_job;
  m_accum.drain_into(job.samples);
  job.rate = m_host_rate;
  job.num_threads = inputs.threads.value;
  job.threshold = inputs.threshold.value;
  job.want_model = m_requested_model;
  job.enroll_name = inputs.enroll_name.value; // snapshot at release
  job.reload = m_reload;
  job.extractor = m_extractor;
  m_reload = false;

  m_inflight.store(true, std::memory_order_release);
  if(worker.request)
    worker.request(m_job);
}

inline std::function<void(Speaker&)> Speaker::worker::work(std::shared_ptr<Job> job)
{
  const auto& L = SherpaLoader::instance();

  if(job->reload || !job->extractor || !*job->extractor)
  {
    job->extractor = std::make_shared<SpeakerEmbeddingExtractorHandle>(
        make_speaker_extractor(job->want_model, job->num_threads));
  }

  job->embedding.clear();
  job->dim = 0;

  if(job->extractor && *job->extractor && !job->samples.empty()
     && L.SherpaOnnxSpeakerEmbeddingExtractorDim
     && L.SherpaOnnxSpeakerEmbeddingExtractorCreateStream
     && L.SherpaOnnxSpeakerEmbeddingExtractorComputeEmbedding
     && L.SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding)
  {
    auto* ex = job->extractor->get();
    const int dim = L.SherpaOnnxSpeakerEmbeddingExtractorDim(ex);

    // The extractor feeds audio through a reused SherpaOnnxOnlineStream.
    OnlineStreamHandle stream{
        L.SherpaOnnxSpeakerEmbeddingExtractorCreateStream(ex)};
    if(stream && dim > 0)
    {
      L.SherpaOnnxOnlineStreamAcceptWaveform(
          stream.get(), static_cast<int>(job->rate), job->samples.data(),
          static_cast<int>(job->samples.size()));
      L.SherpaOnnxOnlineStreamInputFinished(stream.get());

      const float* v
          = L.SherpaOnnxSpeakerEmbeddingExtractorComputeEmbedding(ex, stream.get());
      if(v)
      {
        job->embedding.assign(v, v + dim);
        job->dim = dim;
        L.SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding(v);
      }
    }
  }

  return [job](Speaker& self) { self.apply(*job); };
}

inline void Speaker::apply(Job& job)
{
  if(job.reload)
  {
    m_extractor = job.extractor;
    m_loaded_model = job.want_model;
  }

  if(!job.embedding.empty())
  {
    m_last_dim = job.dim;
    m_last_embedding = std::move(job.embedding);
    ensure_manager();

    outputs.embedding.value = m_last_embedding;

    if(!job.enroll_name.empty())
    {
      // Enroll the freshly computed embedding under the requested name.
      const auto& L = SherpaLoader::instance();
      if(m_manager && *m_manager && L.SherpaOnnxSpeakerEmbeddingManagerAdd)
        L.SherpaOnnxSpeakerEmbeddingManagerAdd(
            m_manager->get(), job.enroll_name.c_str(), m_last_embedding.data());
    }
    else
    {
      // Identify against the enrolled speakers.
      identify(m_last_embedding, job.threshold);
    }

    update_count();
  }

  m_inflight.store(false, std::memory_order_release);
}

inline void Speaker::ensure_manager()
{
  if((!m_manager || !*m_manager) && m_last_dim > 0)
  {
    const auto& L = SherpaLoader::instance();
    if(L.SherpaOnnxCreateSpeakerEmbeddingManager)
      m_manager = std::make_shared<SpeakerEmbeddingManagerHandle>(
          L.SherpaOnnxCreateSpeakerEmbeddingManager(m_last_dim));
  }
}

inline void Speaker::identify(const std::vector<float>& v, float threshold)
{
  if(!m_manager || !*m_manager || v.empty())
    return;
  const auto& L = SherpaLoader::instance();

  // Best-matches gives both the name and the similarity score in one call.
  if(L.SherpaOnnxSpeakerEmbeddingManagerGetBestMatches
     && L.SherpaOnnxSpeakerEmbeddingManagerFreeBestMatches)
  {
    const auto* r = L.SherpaOnnxSpeakerEmbeddingManagerGetBestMatches(
        m_manager->get(), v.data(), threshold, 1);
    if(r)
    {
      if(r->count > 0 && r->matches)
      {
        outputs.score.value = r->matches[0].score;
        outputs.speaker(
            std::string_view{r->matches[0].name ? r->matches[0].name : ""});
      }
      L.SherpaOnnxSpeakerEmbeddingManagerFreeBestMatches(r);
    }
  }
  // Fallback: plain search (name only, no score).
  else if(
      L.SherpaOnnxSpeakerEmbeddingManagerSearch
      && L.SherpaOnnxSpeakerEmbeddingManagerFreeSearch)
  {
    const char* name = L.SherpaOnnxSpeakerEmbeddingManagerSearch(
        m_manager->get(), v.data(), threshold);
    if(name)
    {
      outputs.speaker(std::string_view{name});
      L.SherpaOnnxSpeakerEmbeddingManagerFreeSearch(name);
    }
  }
}

inline void Speaker::update_count()
{
  const auto& L = SherpaLoader::instance();
  int n = 0;
  if(m_manager && *m_manager && L.SherpaOnnxSpeakerEmbeddingManagerNumSpeakers)
    n = L.SherpaOnnxSpeakerEmbeddingManagerNumSpeakers(m_manager->get());
  outputs.count.value = n;
}

inline void Speaker::enroll(const std::string& name)
{
  if(m_last_embedding.empty() || name.empty())
    return;
  ensure_manager();
  const auto& L = SherpaLoader::instance();
  if(m_manager && *m_manager && L.SherpaOnnxSpeakerEmbeddingManagerAdd)
    L.SherpaOnnxSpeakerEmbeddingManagerAdd(
        m_manager->get(), name.c_str(), m_last_embedding.data());
  update_count();
}

inline void Speaker::remove(const std::string& name)
{
  const auto& L = SherpaLoader::instance();
  if(m_manager && *m_manager && !name.empty()
     && L.SherpaOnnxSpeakerEmbeddingManagerRemove)
    L.SherpaOnnxSpeakerEmbeddingManagerRemove(m_manager->get(), name.c_str());
  update_count();
}

inline void Speaker::clear()
{
  // No bulk-clear in the C API: drop the manager so it is recreated empty.
  m_manager.reset();
  outputs.count.value = 0;
}

}
