#pragma once
// sherpa_text -- text -> text: punctuation restoration (offline ct-transformer or
// online cnn-bilstm) and Arabic diacritization (catt). One object; the model (or
// the Type override) selects the transform. Model loading + inference run on the
// worker (they may execute on the audio thread in some hosts).

#include "helpers/Handles.hpp"

#include <halp/callback.hpp>
#include <halp/controls.hpp>
#include <halp/controls.enums.hpp>
#include <halp/file_port.hpp>
#include <halp/meta.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>

namespace sherpa
{

class TextProcessor
{
public:
  halp_meta(name, "Sherpa Text Processor")
  halp_meta(c_name, "sherpa_text")
  halp_meta(category, "AI/Text")
  halp_meta(author, "Jean-Michaël Celerier")
  halp_meta(description, "Punctuation restoration and Arabic diacritization")
  halp_meta(uuid, "3f9d5b12-7a4e-4c88-9b0d-6e2a1f8c3d47")

  enum class Type
  {
    Auto,
    PunctuationOffline,
    PunctuationOnline,
    Diacritization
  };

  struct
  {
    struct : halp::val_port<"Text", std::string>
    {
      void update(TextProcessor& self) { self.on_input(); }
    } text;
    halp::folder_port<"Model"> model;
    halp::enum_t<Type, "Type"> type;
    halp::hslider_i32<"Threads", halp::range{1., 8., 1.}> threads;
  } inputs;

  struct
  {
    halp::val_port<"Text", std::string> result;
    halp::callback<"Done"> done;
  } outputs;

  // kind: 0 offline-punct, 1 online-punct, 2 diacritization, -1 none
  struct Job
  {
    std::string text;
    std::string want_model;
    Type type = Type::Auto;
    int num_threads = 1;
    bool reload = false;
    int kind = -1;
    std::shared_ptr<OfflinePunctuationHandle> punct_off;
    std::shared_ptr<OnlinePunctuationHandle> punct_on;
    std::shared_ptr<OfflineDiacritizationHandle> diac;
    std::string result;
  };

  struct worker
  {
    std::function<void(std::shared_ptr<Job>)> request;
    static std::function<void(TextProcessor&)> work(std::shared_ptr<Job> job);
  } worker;

  void on_input() { (*this)(); }
  void operator()();

private:
  void dispatch();

  bool m_available = SherpaLoader::instance().available;
  std::shared_ptr<Job> m_job;
  std::atomic<bool> m_inflight{false};
  int m_kind = -1;
  std::shared_ptr<OfflinePunctuationHandle> m_punct_off;
  std::shared_ptr<OnlinePunctuationHandle> m_punct_on;
  std::shared_ptr<OfflineDiacritizationHandle> m_diac;
  std::string m_loaded_model;
  Type m_loaded_type = Type::Auto;
  bool m_reload = false;
};

inline void TextProcessor::operator()()
{
  if(!m_available)
    return;
  if(!m_job)
  {
    m_job = std::make_shared<Job>();
    m_job->text.reserve(4096);
    m_job->want_model.reserve(512);
    m_job->result.reserve(4096);
  }
  if(inputs.model.value != m_loaded_model || inputs.type.value != m_loaded_type)
    m_reload = true;
  dispatch();
}

inline void TextProcessor::dispatch()
{
  if(m_inflight.load(std::memory_order_acquire) || !m_job)
    return;
  auto& job = *m_job;
  job.text = inputs.text.value;
  job.want_model = inputs.model.value;
  job.type = inputs.type.value;
  job.num_threads = inputs.threads.value;
  job.reload = m_reload;
  job.kind = m_kind;
  job.punct_off = m_punct_off;
  job.punct_on = m_punct_on;
  job.diac = m_diac;
  m_reload = false;
  m_inflight.store(true, std::memory_order_release);
  if(worker.request)
    worker.request(m_job);
}

inline std::function<void(TextProcessor&)>
TextProcessor::worker::work(std::shared_ptr<Job> job)
{
  namespace fs = std::filesystem;
  const auto& L = SherpaLoader::instance();

  auto lower = [](std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    return s;
  };
  auto find = [&](const fs::path& d, std::initializer_list<std::string_view> needles,
                  std::string_view ext) -> std::string {
    std::error_code ec;
    if(!fs::is_directory(d, ec))
      return {};
    for(const auto& e : fs::directory_iterator(d, ec))
    {
      if(!e.is_regular_file())
        continue;
      auto n = lower(e.path().filename().string());
      if(!ext.empty() && !n.ends_with(ext))
        continue;
      for(auto s : needles)
        if(n.find(s) != std::string::npos)
          return e.path().string();
      if(needles.size() == 0)
        return e.path().string();
    }
    return {};
  };

  if(job->reload)
  {
    job->punct_off.reset();
    job->punct_on.reset();
    job->diac.reset();
    job->kind = -1;

    fs::path d{job->want_model};
    const std::string dirname = lower(d.filename().string());
    Type t = job->type;
    if(t == Type::Auto)
    {
      if(dirname.find("diacrit") != std::string::npos
         || dirname.find("catt") != std::string::npos
         || dirname.find("arabic") != std::string::npos
         || !find(d, {"catt"}, ".onnx").empty())
        t = Type::Diacritization;
      else if(dirname.find("online") != std::string::npos
              || !find(d, {"cnn_bilstm", "bilstm"}, ".onnx").empty())
        t = Type::PunctuationOnline;
      else
        t = Type::PunctuationOffline;
    }

    if(t == Type::Diacritization && L.SherpaOnnxCreateOfflineDiacritization)
    {
      std::string enc = find(d, {"encoder"}, ".onnx");
      std::string dec = find(d, {"decoder"}, ".onnx");
      SherpaOnnxOfflineDiacritizationConfig cfg;
      std::memset(&cfg, 0, sizeof(cfg));
      cfg.model.catt_encoder = enc.c_str();
      cfg.model.catt_decoder = dec.c_str();
      cfg.model.num_threads = job->num_threads;
      cfg.model.provider = "cpu";
      job->diac = std::make_shared<OfflineDiacritizationHandle>(
          L.SherpaOnnxCreateOfflineDiacritization(&cfg));
      if(*job->diac)
        job->kind = 2;
    }
    else if(t == Type::PunctuationOnline && L.SherpaOnnxCreateOnlinePunctuation)
    {
      std::string m = find(d, {"cnn_bilstm", "bilstm"}, ".onnx");
      if(m.empty())
        m = find(d, {}, ".onnx");
      std::string bpe = find(d, {"bpe"}, "");
      SherpaOnnxOnlinePunctuationConfig cfg;
      std::memset(&cfg, 0, sizeof(cfg));
      cfg.model.cnn_bilstm = m.c_str();
      cfg.model.bpe_vocab = bpe.c_str();
      cfg.model.num_threads = job->num_threads;
      cfg.model.provider = "cpu";
      job->punct_on = std::make_shared<OnlinePunctuationHandle>(
          L.SherpaOnnxCreateOnlinePunctuation(&cfg));
      if(*job->punct_on)
        job->kind = 1;
    }
    else if(L.SherpaOnnxCreateOfflinePunctuation)
    {
      std::string m = find(d, {"ct_transformer", "punct"}, ".onnx");
      if(m.empty())
        m = find(d, {}, ".onnx");
      SherpaOnnxOfflinePunctuationConfig cfg;
      std::memset(&cfg, 0, sizeof(cfg));
      cfg.model.ct_transformer = m.c_str();
      cfg.model.num_threads = job->num_threads;
      cfg.model.provider = "cpu";
      job->punct_off = std::make_shared<OfflinePunctuationHandle>(
          L.SherpaOnnxCreateOfflinePunctuation(&cfg));
      if(*job->punct_off)
        job->kind = 0;
    }
  }

  job->result.clear();
  if(!job->text.empty())
  {
    const char* out = nullptr;
    if(job->kind == 0 && job->punct_off && *job->punct_off
       && L.SherpaOfflinePunctuationAddPunct)
      out = L.SherpaOfflinePunctuationAddPunct(job->punct_off->get(), job->text.c_str());
    else if(job->kind == 1 && job->punct_on && *job->punct_on
            && L.SherpaOnnxOnlinePunctuationAddPunct)
      out = L.SherpaOnnxOnlinePunctuationAddPunct(job->punct_on->get(), job->text.c_str());
    else if(job->kind == 2 && job->diac && *job->diac
            && L.SherpaOfflineDiacritizationAddDiacritics)
      out = L.SherpaOfflineDiacritizationAddDiacritics(job->diac->get(), job->text.c_str());
    if(out)
    {
      job->result = out;
      if(job->kind == 0)
        L.SherpaOfflinePunctuationFreeText(out);
      else if(job->kind == 1)
        L.SherpaOnnxOnlinePunctuationFreeText(out);
      else if(job->kind == 2)
        L.SherpaOfflineDiacritizationFreeText(out);
    }
  }

  return [job](TextProcessor& self) {
    if(job->reload)
    {
      self.m_punct_off = job->punct_off;
      self.m_punct_on = job->punct_on;
      self.m_diac = job->diac;
      self.m_kind = job->kind;
      self.m_loaded_model = job->want_model;
      self.m_loaded_type = job->type;
    }
    self.outputs.result.value = std::move(job->result);
    self.outputs.done();
    self.m_inflight.store(false, std::memory_order_release);
  };
}

}
