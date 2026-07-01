#pragma once
// sherpa_diarization -- offline speaker diarization ("who spoke when").
//
// Record a whole passage into the audio input (toggle "Record"); on the falling
// edge the accumulated mono audio is handed to a worker thread, resampled to the
// diarization model's rate, and processed in one shot by a sherpa-onnx offline
// speaker-diarization pipeline (a pyannote segmentation model + a speaker
// embedding extractor + fast clustering). The result is a list of
// {start, end, speaker} segments, emitted as a value port and one callback per
// segment.
//
// All inference and model loading happen on the worker; operator() (audio thread)
// only downmixes + accumulates and dispatches -- Process is slow and must stay off
// the audio thread. See helpers/AudioChunker.hpp for the threading contract.

#include "helpers/AudioChunker.hpp"
#include "helpers/ModelConfig.hpp"

#include <halp/audio.hpp>
#include <halp/callback.hpp>
#include <halp/controls.hpp>
#include <halp/file_port.hpp>
#include <halp/meta.hpp>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace sherpa
{

// A diarization model DIR holds BOTH a pyannote segmentation .onnx and a speaker
// embedding .onnx. Detect each by filename, zero-init the config, and create the
// pipeline. Runs on the worker thread (filesystem + allocation) -- never on audio.
inline OfflineSpeakerDiarizationHandle create_diarization(
    std::string_view dir, int num_clusters, float threshold, int num_threads)
{
  const auto& L = SherpaLoader::instance();
  if(!L.SherpaOnnxCreateOfflineSpeakerDiarization)
    return {}; // extended symbol not present in this runtime

  namespace fs = std::filesystem;
  fs::path d{dir};

  // Gather the .onnx files so we can fall back to a positional guess when the
  // filenames do not match the heuristics below.
  std::vector<std::string> onnx;
  {
    std::error_code ec;
    if(fs::is_directory(d, ec))
    {
      for(const auto& e : fs::directory_iterator(d, ec))
      {
        if(!e.is_regular_file())
          continue;
        if(model::to_lower(e.path().filename().string()).ends_with(".onnx"))
          onnx.push_back(e.path().string());
      }
    }
  }

  std::string seg = model::find_in_dir(d, {"segment", "pyannote"});
  std::string emb = model::find_in_dir(
      d, {"embedding", "speaker", "3dspeaker", "nemo", "wespeaker"});

  auto other_than = [&](const std::string& taken) -> std::string {
    for(const auto& f : onnx)
      if(f != taken)
        return f;
    return {};
  };
  if(seg.empty() && !emb.empty())
    seg = other_than(emb);
  else if(emb.empty() && !seg.empty())
    emb = other_than(seg);
  else if(seg.empty() && emb.empty() && onnx.size() >= 2)
  {
    seg = onnx[0];
    emb = onnx[1];
  }

  if(seg.empty() || emb.empty() || seg == emb)
    return {};

  SherpaOnnxOfflineSpeakerDiarizationConfig cfg;
  std::memset(&cfg, 0, sizeof(cfg));
  cfg.segmentation.pyannote.model = seg.c_str();
  cfg.segmentation.num_threads = num_threads;
  cfg.segmentation.provider = "cpu";
  cfg.embedding.model = emb.c_str();
  cfg.embedding.num_threads = num_threads;
  cfg.embedding.provider = "cpu";
  cfg.clustering.num_clusters = num_clusters; // <=0 => auto via threshold
  cfg.clustering.threshold = threshold;
  cfg.min_duration_on = 0.3f;
  cfg.min_duration_off = 0.5f;

  return OfflineSpeakerDiarizationHandle{
      L.SherpaOnnxCreateOfflineSpeakerDiarization(&cfg)};
}

class Diarization
{
public:
  halp_meta(name, "Sherpa Diarization")
  halp_meta(c_name, "sherpa_diarization")
  halp_meta(category, "AI/Audio")
  halp_meta(author, "Jean-Michaël Celerier")
  halp_meta(description, "Offline speaker diarization with sherpa-onnx (who spoke when)")
  halp_meta(uuid, "f7e40b28-70ac-4431-abf1-69dde53665dd")

  struct Segment
  {
    float start;
    float end;
    int speaker;
    halp_field_names(start, end, speaker);
  };

  struct
  {
    halp::dynamic_audio_bus<"In", float> audio;
    halp::folder_port<"Model"> model;
    halp::toggle<"Record"> record;
    halp::spinbox_i32<"Num speakers", halp::range{-1., 32., -1.}> num_speakers;
    halp::knob_f32<"Cluster threshold", halp::range{0., 1., 0.5}> cluster_threshold;
    halp::hslider_i32<"Threads", halp::range{1., 8., 1.}> threads;
  } inputs;

  struct
  {
    halp::val_port<"Segments", std::vector<Segment>> segments;
    halp::val_port<"Num speakers", int> num_speakers_out;
    halp::callback<"Segment", float, float, int> on_segment; // start, end, speaker
    halp::val_port<"Progress", float> progress;
  } outputs;

  struct Job
  {
    std::vector<float> samples;
    double rate = 48000.;
    int num_threads = 1;
    int num_clusters = -1;
    float threshold = 0.5f;
    bool reload = false;
    std::string want_model;
    std::shared_ptr<OfflineSpeakerDiarizationHandle> sd;
    std::vector<Segment> segments;
    int num_speakers = 0;
    float progress = 0.f;
  };

  struct worker
  {
    std::function<void(std::shared_ptr<Job>)> request;
    static std::function<void(Diarization&)> work(std::shared_ptr<Job> job);
  } worker;

  void prepare(halp::setup info);
  void operator()(int frames);

private:
  void dispatch();

  bool m_available = SherpaLoader::instance().available;
  double m_host_rate = 48000.;
  std::shared_ptr<OfflineSpeakerDiarizationHandle> m_sd;
  std::shared_ptr<Job> m_job;
  std::atomic<bool> m_inflight{false};
  MonoAccum m_accum;
  std::string m_requested_model, m_loaded_model;
  bool m_reload = false;
  bool m_was_recording = false;
};

inline void Diarization::prepare(halp::setup info)
{
  m_host_rate = info.rate > 0 ? info.rate : 48000.;
  // Cap the recording buffer (mono) -- e.g. 300 s at the host rate.
  m_accum.prepare(static_cast<std::size_t>(m_host_rate * 300.0));
  if(!m_job)
  {
    m_job = std::make_shared<Job>();
    m_job->samples.reserve(static_cast<std::size_t>(m_host_rate * 300.0));
    m_job->want_model.reserve(512);
  }
}

inline void Diarization::operator()(int frames)
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

  // Falling edge of Record -> diarize what we captured.
  if(m_was_recording && !rec_on)
    dispatch();
  m_was_recording = rec_on;
}

inline void Diarization::dispatch()
{
  if(m_inflight.load(std::memory_order_acquire) || !m_job)
    return;
  if(m_accum.empty() && !m_reload)
    return;

  auto& job = *m_job;
  m_accum.drain_into(job.samples);
  job.rate = m_host_rate;
  job.num_threads = inputs.threads.value;
  job.num_clusters = inputs.num_speakers.value;
  job.threshold = inputs.cluster_threshold.value;
  job.want_model = m_requested_model;
  job.reload = m_reload;
  job.sd = m_sd;
  m_reload = false;

  m_inflight.store(true, std::memory_order_release);
  if(worker.request)
    worker.request(m_job);
}

inline std::function<void(Diarization&)>
Diarization::worker::work(std::shared_ptr<Job> job)
{
  const auto& L = SherpaLoader::instance();

  if(job->reload || !job->sd || !*job->sd)
  {
    job->sd = std::make_shared<OfflineSpeakerDiarizationHandle>(create_diarization(
        job->want_model, job->num_clusters, job->threshold, job->num_threads));
  }

  job->segments.clear();
  job->num_speakers = 0;
  job->progress = 0.f;

  if(job->sd && *job->sd && !job->samples.empty()
     && L.SherpaOnnxOfflineSpeakerDiarizationProcessWithCallback
     && L.SherpaOnnxOfflineSpeakerDiarizationGetSampleRate)
  {
    auto* sd = job->sd->get();

    int model_rate = L.SherpaOnnxOfflineSpeakerDiarizationGetSampleRate(sd);
    if(model_rate <= 0)
      model_rate = 16000;

    // Resample the whole accumulated buffer host-rate -> model-rate. flush=1
    // because this is the entire utterance. Keep the resampler alive until after
    // Process (the ResampleOut owns its own samples, freed separately).
    const float* in = job->samples.data();
    int n = static_cast<int>(job->samples.size());
    LinearResamplerHandle rs;
    const SherpaOnnxResampleOut* ro = nullptr;
    if(static_cast<int>(job->rate) != model_rate)
    {
      rs = LinearResamplerHandle{L.SherpaOnnxCreateLinearResampler(
          static_cast<int>(job->rate), model_rate, 0.f, 0)};
      if(rs)
      {
        ro = L.SherpaOnnxLinearResamplerResample(rs.get(), in, n, 1);
        if(ro)
        {
          in = ro->samples;
          n = ro->n;
        }
      }
    }

    if(n > 0)
    {
      auto cb = +[](int32_t done, int32_t total, void* arg) -> int32_t {
        auto* j = static_cast<Job*>(arg);
        if(total > 0)
          j->progress = static_cast<float>(done) / static_cast<float>(total);
        return 0;
      };
      const auto* r = L.SherpaOnnxOfflineSpeakerDiarizationProcessWithCallback(
          sd, in, n, cb, job.get());
      if(r)
      {
        if(L.SherpaOnnxOfflineSpeakerDiarizationResultGetNumSpeakers)
          job->num_speakers
              = L.SherpaOnnxOfflineSpeakerDiarizationResultGetNumSpeakers(r);

        int num_seg = 0;
        if(L.SherpaOnnxOfflineSpeakerDiarizationResultGetNumSegments)
          num_seg = L.SherpaOnnxOfflineSpeakerDiarizationResultGetNumSegments(r);

        if(num_seg > 0
           && L.SherpaOnnxOfflineSpeakerDiarizationResultSortByStartTime)
        {
          const auto* segs
              = L.SherpaOnnxOfflineSpeakerDiarizationResultSortByStartTime(r);
          if(segs)
          {
            job->segments.reserve(static_cast<std::size_t>(num_seg));
            for(int i = 0; i < num_seg; ++i)
              job->segments.push_back(
                  Segment{segs[i].start, segs[i].end, segs[i].speaker});
            if(L.SherpaOnnxOfflineSpeakerDiarizationDestroySegment)
              L.SherpaOnnxOfflineSpeakerDiarizationDestroySegment(segs);
          }
        }

        if(L.SherpaOnnxOfflineSpeakerDiarizationDestroyResult)
          L.SherpaOnnxOfflineSpeakerDiarizationDestroyResult(r);
      }
    }

    if(ro)
      L.SherpaOnnxLinearResamplerResampleFree(ro);
  }

  return [job](Diarization& self) {
    if(job->reload)
    {
      self.m_sd = job->sd;
      self.m_loaded_model = job->want_model;
    }
    self.outputs.num_speakers_out.value = job->num_speakers;
    for(const auto& s : job->segments)
      self.outputs.on_segment(s.start, s.end, s.speaker);
    self.outputs.segments.value = std::move(job->segments);
    self.outputs.progress.value = 1.f;
    self.m_inflight.store(false, std::memory_order_release);
  };
}

}
