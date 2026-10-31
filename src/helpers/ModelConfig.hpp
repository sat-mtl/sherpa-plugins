#pragma once
// Best-effort model-directory detection + config builders for the sherpa-onnx
// C API. Given a directory (the object's "Model" folder port), probe for the
// conventional filenames k2-fsa ships and build the matching config, then create
// the handle. Runs on the worker/inference thread (uses std::filesystem and
// allocates freely) -- NEVER call these from operator() on the audio thread.
//
// Detection is heuristic and covers the common families; for unusual layouts,
// extend the rules below or expose an explicit override on the object.
//
// Config strings are local std::strings: sherpa-onnx copies them during the
// Create* call, so they only need to outlive that call (they do).

#include "helpers/Handles.hpp"
#include "helpers/Options.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <string>
#include <vector>
#include <string_view>

namespace sherpa::model
{
namespace fs = std::filesystem;

// Optional detection tracing: set the environment variable SHERPA_PLUGINS_DEBUG=1
// to log model-directory scanning and the chosen family to stderr. Helps diagnose
// "no output" / "Errors in config" from a specific model at runtime.
inline bool debug_enabled() noexcept
{
  static const bool on = [] {
    const char* v = std::getenv("SHERPA_PLUGINS_DEBUG");
    return v && v[0] && v[0] != '0';
  }();
  return on;
}

#define SHERPA_DBG(...)                     \
  do                                       \
  {                                        \
    if(::sherpa::model::debug_enabled())   \
      std::fprintf(stderr, __VA_ARGS__);   \
  } while(0)

inline std::string to_lower(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s;
}

inline bool icontains(std::string_view hay, std::string_view needle)
{
  if(needle.empty())
    return true;
  auto h = to_lower(std::string{hay});
  auto n = to_lower(std::string{needle});
  return h.find(n) != std::string::npos;
}

// First regular file in dir whose lowercased name contains any needle (or any,
// when needles is empty) and ends with ext. Returns "" if none.
inline std::string find_in_dir(
    const fs::path& dir, std::initializer_list<std::string_view> needles,
    std::string_view ext = ".onnx")
{
  std::error_code ec;
  if(!fs::is_directory(dir, ec))
    return {};
  for(const auto& e : fs::directory_iterator(dir, ec))
  {
    if(!e.is_regular_file())
      continue;
    auto name = to_lower(e.path().filename().string());
    if(!ext.empty() && !(name.size() >= ext.size() && name.ends_with(ext)))
      continue;
    bool ok = needles.size() == 0;
    for(auto n : needles)
      if(name.find(to_lower(std::string{n})) != std::string::npos)
      {
        ok = true;
        break;
      }
    if(ok)
      return e.path().string();
  }
  return {};
}

inline std::string file_if_exists(const fs::path& dir, std::string_view name)
{
  std::error_code ec;
  auto p = dir / name;
  return fs::exists(p, ec) ? p.string() : std::string{};
}

// The LLM-based recognizers (qwen3-asr, funasr-nano) take a HuggingFace tokenizer
// DIRECTORY rather than a tokens.txt: a "tokenizer" subfolder for qwen3-asr, a
// "Qwen3-0.6B" one for funasr-nano. Any subfolder holding tokenizer.json will do.
inline std::string find_tokenizer_dir(const fs::path& dir)
{
  std::error_code ec;
  if(!fs::is_directory(dir, ec))
    return {};
  if(fs::is_directory(dir / "tokenizer", ec))
    return (dir / "tokenizer").string();
  for(const auto& e : fs::directory_iterator(dir, ec))
  {
    if(e.is_directory(ec) && fs::exists(e.path() / "tokenizer.json", ec))
      return e.path().string();
  }
  return {};
}

// Lowercased last path component, robust to a trailing separator: a folder path
// like ".../my-model/" has an EMPTY filename(), which would break the family
// auto-detection that keys on the directory name. Fall back to the parent then.
inline std::string dir_name(const fs::path& d)
{
  std::string n = d.filename().string();
  if(n.empty())
    n = d.parent_path().filename().string();
  return to_lower(n);
}

// ---------------------------------------------------------------------------
// Offline (batch) ASR
// ---------------------------------------------------------------------------
inline OfflineRecognizerHandle create_offline_recognizer(
    std::string_view dir, int num_threads, std::string_view decoding,
    const char* provider = "cpu", std::string_view advanced = {},
    std::string_view model_type = {})
{
  const auto& L = SherpaLoader::instance();
  fs::path d{dir};

  std::string tokens = file_if_exists(d, "tokens.txt");
  if(tokens.empty())
    tokens = find_in_dir(d, {"tokens"}, ".txt");
  std::string enc = find_in_dir(d, {"encoder"});
  std::string dec = find_in_dir(d, {"decoder"});
  std::string joiner = find_in_dir(d, {"joiner"});
  std::string any = find_in_dir(d, {});
  std::string dirname = dir_name(d);

  std::string dec_method{decoding};

  SherpaOnnxOfflineRecognizerConfig cfg;
  std::memset(&cfg, 0, sizeof(cfg));
  cfg.feat_config.sample_rate = 16000; // input is resampled to this internally
  cfg.feat_config.feature_dim = 80;
  cfg.model_config.tokens = tokens.c_str();
  cfg.model_config.num_threads = num_threads;
  cfg.model_config.provider = provider;
  cfg.decoding_method = dec_method.c_str();

  // Extra files for multi-part architectures (moonshine).
  std::string preproc = find_in_dir(d, {"preprocess"});
  std::string uncached = find_in_dir(d, {"uncached"});
  std::string cached = find_in_dir(d, {"cached"});
  std::string mencoder = enc.empty() ? find_in_dir(d, {"encode"}) : enc;
  // LLM-based recognizers: qwen3-asr (conv_frontend + encoder + decoder) and
  // funasr-nano (encoder_adaptor + llm + embedding), both with a tokenizer dir.
  std::string conv_frontend = find_in_dir(d, {"conv_frontend"});
  std::string enc_adaptor = find_in_dir(d, {"encoder_adaptor"});
  std::string llm = find_in_dir(d, {"llm"});
  std::string embedding = find_in_dir(d, {"embedding"});
  std::string tokenizer = find_tokenizer_dir(d);
  // "encoder" also matches funasr-nano's encoder_adaptor; it is not an encoder.
  if(!enc_adaptor.empty() && enc == enc_adaptor)
    enc.clear();
  auto has = [&](std::string_view w) {
    return icontains(any, w) || icontains(dirname, w);
  };

  auto set_qwen3 = [&] {
    cfg.model_config.qwen3_asr.conv_frontend = conv_frontend.c_str();
    cfg.model_config.qwen3_asr.encoder = enc.c_str();
    cfg.model_config.qwen3_asr.decoder = dec.c_str();
    cfg.model_config.qwen3_asr.tokenizer = tokenizer.c_str();
  };
  auto set_funasr_nano = [&] {
    cfg.model_config.funasr_nano.encoder_adaptor = enc_adaptor.c_str();
    cfg.model_config.funasr_nano.llm = llm.c_str();
    cfg.model_config.funasr_nano.embedding = embedding.c_str();
    cfg.model_config.funasr_nano.tokenizer = tokenizer.c_str();
  };

  // Populate the config for an explicit sherpa model-type name; returns false if
  // the files that family needs are absent. Covers every offline family exposed
  // by the C API.
  auto set_family = [&](std::string_view t) -> bool {
    if(t == "qwen3_asr" || t == "qwen3")
    {
      if(conv_frontend.empty() || enc.empty() || dec.empty() || tokenizer.empty())
        return false;
      set_qwen3();
    }
    else if(t == "funasr_nano")
    {
      if(enc_adaptor.empty() || llm.empty() || embedding.empty() || tokenizer.empty())
        return false;
      set_funasr_nano();
    }
    else if(t == "transducer")
    {
      if(enc.empty() || dec.empty() || joiner.empty())
        return false;
      cfg.model_config.transducer.encoder = enc.c_str();
      cfg.model_config.transducer.decoder = dec.c_str();
      cfg.model_config.transducer.joiner = joiner.c_str();
    }
    else if(t == "whisper")
    {
      if(enc.empty() || dec.empty())
        return false;
      cfg.model_config.whisper.encoder = enc.c_str();
      cfg.model_config.whisper.decoder = dec.c_str();
    }
    else if(t == "fire_red_asr" || t == "fire_red" || t == "firered")
    {
      if(enc.empty() || dec.empty())
        return false;
      cfg.model_config.fire_red_asr.encoder = enc.c_str();
      cfg.model_config.fire_red_asr.decoder = dec.c_str();
    }
    else if(t == "canary")
    {
      if(enc.empty() || dec.empty())
        return false;
      cfg.model_config.canary.encoder = enc.c_str();
      cfg.model_config.canary.decoder = dec.c_str();
      cfg.model_config.canary.use_pnc = 1;
    }
    else if(t == "cohere" || t == "cohere_transcribe")
    {
      if(enc.empty() || dec.empty())
        return false;
      cfg.model_config.cohere_transcribe.encoder = enc.c_str();
      cfg.model_config.cohere_transcribe.decoder = dec.c_str();
      cfg.model_config.cohere_transcribe.use_punct = 1;
      cfg.model_config.cohere_transcribe.use_itn = 1;
    }
    else if(t == "moonshine")
    {
      if(preproc.empty())
        return false;
      cfg.model_config.moonshine.preprocessor = preproc.c_str();
      cfg.model_config.moonshine.encoder = mencoder.c_str();
      cfg.model_config.moonshine.uncached_decoder = uncached.c_str();
      cfg.model_config.moonshine.cached_decoder = cached.c_str();
    }
    else if(any.empty())
      return false;
    else if(t == "paraformer")
      cfg.model_config.paraformer.model = any.c_str();
    else if(t == "sense_voice" || t == "sensevoice" || t == "sense")
    {
      cfg.model_config.sense_voice.model = any.c_str();
      cfg.model_config.sense_voice.use_itn = 1;
    }
    else if(t == "telespeech" || t == "telespeech_ctc")
      cfg.model_config.telespeech_ctc = any.c_str();
    else if(t == "wenet_ctc" || t == "wenet")
      cfg.model_config.wenet_ctc.model = any.c_str();
    else if(t == "dolphin")
      cfg.model_config.dolphin.model = any.c_str();
    else if(t == "zipformer_ctc")
      cfg.model_config.zipformer_ctc.model = any.c_str();
    else if(t == "nemo_ctc" || t == "nemo")
      cfg.model_config.nemo_ctc.model = any.c_str();
    else if(t == "tdnn" || t == "yesno")
    {
      cfg.model_config.tdnn.model = any.c_str();
      // The yes/no TDNN is trained on 23-dim fbank at 8 kHz: with the default
      // 80-dim features it never returns a result.
      cfg.feat_config.sample_rate = 8000;
      cfg.feat_config.feature_dim = 23;
    }
    else if(t == "omnilingual")
      cfg.model_config.omnilingual.model = any.c_str();
    else if(t == "medasr")
      cfg.model_config.medasr.model = any.c_str();
    else if(t == "fire_red_asr_ctc" || t == "firered_ctc")
      cfg.model_config.fire_red_asr_ctc.model = any.c_str();
    else
      return false;
    return true;
  };

  // An explicit "Model type" forces the family; empty -> filename auto-detect.
  const std::string mt = to_lower(std::string{model_type});
  const bool forced = !mt.empty() && set_family(mt);

  if(!forced)
  {
    if(!enc.empty() && !dec.empty() && !joiner.empty())
    {
      cfg.model_config.transducer.encoder = enc.c_str();
      cfg.model_config.transducer.decoder = dec.c_str();
      cfg.model_config.transducer.joiner = joiner.c_str();
    }
    else if(!conv_frontend.empty() && !enc.empty() && !dec.empty() && !tokenizer.empty())
    {
      set_qwen3();
    }
    else if(!enc_adaptor.empty() && !llm.empty() && !embedding.empty() && !tokenizer.empty())
    {
      set_funasr_nano();
    }
    else if(has("moonshine") || (!preproc.empty() && !uncached.empty() && !cached.empty()))
    {
      cfg.model_config.moonshine.preprocessor = preproc.c_str();
      cfg.model_config.moonshine.encoder = mencoder.c_str();
      cfg.model_config.moonshine.uncached_decoder = uncached.c_str();
      cfg.model_config.moonshine.cached_decoder = cached.c_str();
    }
    else if(!enc.empty() && !dec.empty()
            && (icontains(enc, "whisper") || has("whisper")))
    {
      cfg.model_config.whisper.encoder = enc.c_str();
      cfg.model_config.whisper.decoder = dec.c_str();
    }
    else if(!enc.empty() && !dec.empty() && (has("firered") || has("fire_red") || has("fire-red")))
    {
      cfg.model_config.fire_red_asr.encoder = enc.c_str();
      cfg.model_config.fire_red_asr.decoder = dec.c_str();
    }
    else if(!enc.empty() && !dec.empty() && has("canary"))
    {
      cfg.model_config.canary.encoder = enc.c_str();
      cfg.model_config.canary.decoder = dec.c_str();
      cfg.model_config.canary.use_pnc = 1;
    }
    else if(!enc.empty() && !dec.empty() && has("cohere"))
    {
      cfg.model_config.cohere_transcribe.encoder = enc.c_str();
      cfg.model_config.cohere_transcribe.decoder = dec.c_str();
      cfg.model_config.cohere_transcribe.use_punct = 1;
      cfg.model_config.cohere_transcribe.use_itn = 1;
    }
    else if(has("paraformer"))
    {
      cfg.model_config.paraformer.model = any.c_str();
    }
    else if(has("sense"))
    {
      cfg.model_config.sense_voice.model = any.c_str();
      cfg.model_config.sense_voice.use_itn = 1;
    }
    else if(has("telespeech"))
    {
      cfg.model_config.telespeech_ctc = any.c_str();
    }
    else if(has("wenet"))
    {
      cfg.model_config.wenet_ctc.model = any.c_str();
    }
    else if(has("dolphin"))
    {
      cfg.model_config.dolphin.model = any.c_str();
    }
    else if(has("tdnn") || has("yesno"))
    {
      cfg.model_config.tdnn.model = any.c_str();
      cfg.feat_config.sample_rate = 8000; // see the forced "tdnn" case above
      cfg.feat_config.feature_dim = 23;
    }
    else if(has("omnilingual"))
    {
      cfg.model_config.omnilingual.model = any.c_str();
    }
    else if(has("medasr"))
    {
      cfg.model_config.medasr.model = any.c_str();
    }
    else if(has("firered") || has("fire_red") || has("fire-red"))
    {
      // enc/dec FireRed matched above; a single-model FireRed is the CTC variant.
      cfg.model_config.fire_red_asr_ctc.model = any.c_str();
    }
    else if(has("zipformer") && has("ctc"))
    {
      cfg.model_config.zipformer_ctc.model = any.c_str();
    }
    else if(has("nemo") || has("ctc"))
    {
      // Generic CTC without a more specific match: NeMo enc-dec CTC is the norm.
      cfg.model_config.nemo_ctc.model = any.c_str();
    }
    else if(!any.empty())
    {
      // Last resort: SenseVoice (multi-lingual, common). Use the "Model type"
      // override to force any family the filename heuristics miss.
      cfg.model_config.sense_voice.model = any.c_str();
      cfg.model_config.sense_voice.use_itn = 1;
    }
    else
    {
      return {};
    }
  }

  // Advanced (key=value) overrides. `adv` owns the strings until the Create call.
  opts::KeyValues adv{advanced};
  if(!adv.empty())
  {
    adv.apply_cstr("hotwords_file", cfg.hotwords_file);
    adv.apply_cstr("rule_fsts", cfg.rule_fsts);
    adv.apply_cstr("rule_fars", cfg.rule_fars);
    adv.apply_float("hotwords_score", cfg.hotwords_score);
    adv.apply_float("blank_penalty", cfg.blank_penalty);
    adv.apply_int("max_active_paths", cfg.max_active_paths);
    // Whisper-only knobs (ignored by the other offline model types).
    adv.apply_cstr("language", cfg.model_config.whisper.language);
    adv.apply_cstr("task", cfg.model_config.whisper.task);
    adv.apply_int("tail_paddings", cfg.model_config.whisper.tail_paddings);
    // Cohere Transcribe.
    adv.apply_cstr("language", cfg.model_config.cohere_transcribe.language);
    adv.apply_bool01("use_punct", cfg.model_config.cohere_transcribe.use_punct);
    adv.apply_bool01("use_itn", cfg.model_config.cohere_transcribe.use_itn);
    // LLM-based recognizers (qwen3-asr / funasr-nano); unset ones keep
    // sherpa's defaults.
    auto& q = cfg.model_config.qwen3_asr;
    auto& f = cfg.model_config.funasr_nano;
    adv.apply_cstr("hotwords", q.hotwords);
    adv.apply_cstr("hotwords", f.hotwords);
    adv.apply_int("max_new_tokens", q.max_new_tokens);
    adv.apply_int("max_new_tokens", f.max_new_tokens);
    adv.apply_float("temperature", q.temperature);
    adv.apply_float("temperature", f.temperature);
    adv.apply_float("top_p", q.top_p);
    adv.apply_float("top_p", f.top_p);
    adv.apply_int("seed", q.seed);
    adv.apply_int("seed", f.seed);
    adv.apply_int("max_total_len", q.max_total_len);
    adv.apply_cstr("language", f.language);
    adv.apply_bool01("itn", f.itn);
    adv.apply_cstr("system_prompt", f.system_prompt);
    adv.apply_cstr("user_prompt", f.user_prompt);
    // HLG decoding for CTC models: decode against a graph (closed vocabulary).
    adv.apply_cstr("ctc_graph", cfg.ctc_fst_decoder_config.graph);
    adv.apply_int("ctc_max_active", cfg.ctc_fst_decoder_config.max_active);
  }

  // A relative file in Advanced (ctc_graph=HLG.fst) names a file of the model
  // folder: a preset or a document can then carry it without an absolute path.
  std::string hotwords_path, rule_fsts_path, graph_path;
  const auto in_model_dir = [&](const char*& value, std::string& storage) {
    if(!value || !*value)
      return;
    const fs::path p{value};
    std::error_code ec;
    if(p.is_absolute() || fs::exists(p, ec))
      return;
    if(const auto in_dir = d / p; fs::exists(in_dir, ec))
    {
      storage = in_dir.string();
      value = storage.c_str();
    }
  };
  in_model_dir(cfg.hotwords_file, hotwords_path);
  in_model_dir(cfg.rule_fsts, rule_fsts_path);
  in_model_dir(cfg.ctc_fst_decoder_config.graph, graph_path);

  OfflineRecognizerHandle h{L.SherpaOnnxCreateOfflineRecognizer(&cfg)};
  SHERPA_DBG(
      "[sherpa] offline dir='%.*s' dirname='%s' tokens='%s' enc='%s' dec='%s' "
      "joiner='%s' any='%s' forced=%d -> create %s\n",
      (int)dir.size(), dir.data(), dirname.c_str(), tokens.c_str(), enc.c_str(),
      dec.c_str(), joiner.c_str(), any.c_str(), forced ? 1 : 0,
      h ? "OK" : "FAILED");
  return h;
}

// ---------------------------------------------------------------------------
// Online (streaming) ASR
// ---------------------------------------------------------------------------
inline OnlineRecognizerHandle create_online_recognizer(
    std::string_view dir, int num_threads, std::string_view decoding,
    bool enable_endpoint, std::string_view hotwords_file,
    const char* provider = "cpu", std::string_view advanced = {})
{
  const auto& L = SherpaLoader::instance();
  fs::path d{dir};

  std::string tokens = file_if_exists(d, "tokens.txt");
  if(tokens.empty())
    tokens = find_in_dir(d, {"tokens"}, ".txt");
  std::string enc = find_in_dir(d, {"encoder"});
  std::string dec = find_in_dir(d, {"decoder"});
  std::string joiner = find_in_dir(d, {"joiner"});
  std::string any = find_in_dir(d, {});
  std::string dirname = dir_name(d);

  std::string dec_method{decoding};
  std::string hotwords{hotwords_file};

  SherpaOnnxOnlineRecognizerConfig cfg;
  std::memset(&cfg, 0, sizeof(cfg));
  cfg.feat_config.sample_rate = 16000;
  cfg.feat_config.feature_dim = 80;
  cfg.model_config.tokens = tokens.c_str();
  cfg.model_config.num_threads = num_threads;
  cfg.model_config.provider = provider;
  cfg.decoding_method = dec_method.c_str();
  cfg.enable_endpoint = enable_endpoint ? 1 : 0;
  cfg.rule1_min_trailing_silence = 2.4f;
  cfg.rule2_min_trailing_silence = 1.2f;
  cfg.rule3_min_utterance_length = 20.0f;
  if(!hotwords.empty())
    cfg.hotwords_file = hotwords.c_str();

  if(!enc.empty() && !dec.empty() && !joiner.empty())
  {
    cfg.model_config.transducer.encoder = enc.c_str();
    cfg.model_config.transducer.decoder = dec.c_str();
    cfg.model_config.transducer.joiner = joiner.c_str();
  }
  else if(!enc.empty() && !dec.empty()
          && (icontains(any, "paraformer") || icontains(dirname, "paraformer")))
  {
    cfg.model_config.paraformer.encoder = enc.c_str();
    cfg.model_config.paraformer.decoder = dec.c_str();
  }
  // Before the generic "ctc" test: a streaming NeMo CTC dir has "ctc" in its
  // name too, and sherpa exits the process on a zipformer2 model without its
  // metadata.
  else if(icontains(any, "nemo") || icontains(dirname, "nemo"))
  {
    cfg.model_config.nemo_ctc.model = any.c_str();
  }
  else if(icontains(any, "ctc") || icontains(dirname, "ctc")
          || icontains(any, "zipformer"))
  {
    cfg.model_config.zipformer2_ctc.model = any.c_str();
  }
  else if(icontains(any, "t-one") || icontains(any, "tone")
          || icontains(dirname, "t-one"))
  {
    cfg.model_config.t_one_ctc.model = any.c_str();
  }
  else
  {
    return {};
  }

  // Advanced (key=value) overrides. `adv` owns the strings until the Create call.
  opts::KeyValues adv{advanced};
  if(!adv.empty())
  {
    adv.apply_cstr("hotwords_file", cfg.hotwords_file);
    adv.apply_cstr("rule_fsts", cfg.rule_fsts);
    adv.apply_cstr("rule_fars", cfg.rule_fars);
    adv.apply_float("hotwords_score", cfg.hotwords_score);
    adv.apply_float("blank_penalty", cfg.blank_penalty);
    adv.apply_int("max_active_paths", cfg.max_active_paths);
    adv.apply_float("rule1_min_trailing_silence", cfg.rule1_min_trailing_silence);
    adv.apply_float("rule2_min_trailing_silence", cfg.rule2_min_trailing_silence);
    adv.apply_float("rule3_min_utterance_length", cfg.rule3_min_utterance_length);
  }

  OnlineRecognizerHandle h{L.SherpaOnnxCreateOnlineRecognizer(&cfg)};
  SHERPA_DBG(
      "[sherpa] online dir='%.*s' dirname='%s' tokens='%s' enc='%s' dec='%s' "
      "joiner='%s' any='%s' -> create %s\n",
      (int)dir.size(), dir.data(), dirname.c_str(), tokens.c_str(), enc.c_str(),
      dec.c_str(), joiner.c_str(), any.c_str(), h ? "OK" : "FAILED");
  return h;
}

// ---------------------------------------------------------------------------
// VAD. `path` may be a directory (we find silero_vad.onnx / ten-vad.onnx inside)
// or a direct .onnx path.
// ---------------------------------------------------------------------------
inline VadHandle create_vad(
    std::string_view path, float threshold, int sample_rate, int num_threads,
    float buffer_seconds = 30.f, const char* provider = "cpu",
    std::string_view advanced = {})
{
  const auto& L = SherpaLoader::instance();
  fs::path p{path};
  std::error_code ec;

  std::string model;
  bool is_ten = false;
  if(fs::is_regular_file(p, ec))
  {
    model = p.string();
    is_ten = icontains(model, "ten");
  }
  else
  {
    model = find_in_dir(p, {"silero"});
    if(model.empty())
    {
      model = find_in_dir(p, {"ten"});
      is_ten = !model.empty();
    }
    if(model.empty())
      model = find_in_dir(p, {"vad"});
  }
  if(model.empty())
    return {};

  SherpaOnnxVadModelConfig cfg;
  std::memset(&cfg, 0, sizeof(cfg));
  cfg.sample_rate = sample_rate;
  cfg.num_threads = num_threads;
  cfg.provider = provider;
  // Advanced (key=value) overrides applied to whichever VAD family is active.
  opts::KeyValues adv{advanced};
  auto apply_vad = [&](auto& v) {
    adv.apply_float("threshold", v.threshold);
    adv.apply_float("min_silence_duration", v.min_silence_duration);
    adv.apply_float("min_speech_duration", v.min_speech_duration);
    adv.apply_float("max_speech_duration", v.max_speech_duration);
    adv.apply_int("window_size", v.window_size);
  };

  if(is_ten)
  {
    cfg.ten_vad.model = model.c_str();
    cfg.ten_vad.threshold = threshold;
    cfg.ten_vad.min_silence_duration = 0.5f;
    cfg.ten_vad.min_speech_duration = 0.25f;
    cfg.ten_vad.max_speech_duration = 20.f;
    cfg.ten_vad.window_size = 256;
    apply_vad(cfg.ten_vad);
  }
  else
  {
    cfg.silero_vad.model = model.c_str();
    cfg.silero_vad.threshold = threshold;
    cfg.silero_vad.min_silence_duration = 0.5f;
    cfg.silero_vad.min_speech_duration = 0.25f;
    cfg.silero_vad.max_speech_duration = 20.f;
    cfg.silero_vad.window_size = 512;
    apply_vad(cfg.silero_vad);
  }

  VadHandle h{L.SherpaOnnxCreateVoiceActivityDetector(&cfg, buffer_seconds)};
  SHERPA_DBG(
      "[sherpa] vad path='%.*s' model='%s' is_ten=%d sr=%d -> create %s\n",
      (int)path.size(), path.data(), model.c_str(), is_ten ? 1 : 0, sample_rate,
      h ? "OK" : "FAILED");
  return h;
}

// Window size (samples at the model rate) the VAD expects per AcceptWaveform.
inline int vad_window_size(std::string_view path)
{
  return icontains(std::string{path}, "ten") ? 256 : 512;
}

// ---------------------------------------------------------------------------
// TTS
// ---------------------------------------------------------------------------

// Every lexicon*.txt in dir, comma-separated (the form sherpa takes several in):
// kokoro multi-lang ships lexicon-us-en.txt + lexicon-zh.txt, most others a
// single lexicon.txt.
inline std::string find_lexicons(const fs::path& dir)
{
  std::vector<std::string> found;
  std::error_code ec;
  if(fs::is_directory(dir, ec))
  {
    for(const auto& e : fs::directory_iterator(dir, ec))
    {
      if(!e.is_regular_file())
        continue;
      auto name = to_lower(e.path().filename().string());
      if(name.starts_with("lexicon") && name.ends_with(".txt"))
        found.push_back(e.path().string());
    }
  }
  std::sort(found.begin(), found.end());
  std::string out;
  for(const auto& f : found)
  {
    if(!out.empty())
      out += ',';
    out += f;
  }
  return out;
}

// A loaded TTS model, and how its input text has to be handed to it.
//
// sherpa-onnx no longer phonemizes (k2-fsa/sherpa-onnx#4055): a model reads text
// through a lexicon, or takes espeak-ng phonemes computed by the caller. The
// latter is `phonemize`: run the text through PhonemizerLoader with `voice`,
// using the model's own espeak-ng-data (`espeak_data`).
struct TtsModel
{
  TtsHandle tts;
  bool phonemize = false;
  std::string espeak_data;
  std::string voice;
};

inline TtsModel create_tts(
    std::string_view dir, int num_threads, const char* provider = "cpu",
    std::string_view advanced = {})
{
  const auto& L = SherpaLoader::instance();
  fs::path d{dir};

  std::string tokens = file_if_exists(d, "tokens.txt");
  std::string voices = find_in_dir(d, {"voices"}, ""); // .bin
  std::string vocoder = find_in_dir(d, {"vocoder", "vocos", "hifigan"});
  std::string acoustic = find_in_dir(d, {"acoustic", "matcha", "steps"});
  std::string model = find_in_dir(d, {"model"});
  if(model.empty())
    model = find_in_dir(d, {}); // first .onnx
  std::string lexicon = find_lexicons(d);
  std::string data_dir;
  {
    std::error_code ec;
    auto espeak = d / "espeak-ng-data";
    if(fs::is_directory(espeak, ec))
      data_dir = espeak.string();
  }
  std::string dirname = dir_name(d);

  std::string enc = find_in_dir(d, {"encoder"});
  std::string dec = find_in_dir(d, {"decoder"});

  std::error_code _dbgec;
  const bool _isdir = fs::is_directory(d, _dbgec);
  (void)_isdir; // only read under SHERPA_PLUGINS_DEBUG
  SHERPA_DBG(
      "[sherpa] TTS scan dir='%s' isdir=%d\n"
      "  dirname='%s' tokens='%s' voices='%s' vocoder='%s'\n"
      "  acoustic='%s' model='%s' enc='%s' dec='%s' lexicon='%s' data_dir='%s'\n",
      std::string{dir}.c_str(), static_cast<int>(_isdir), dirname.c_str(),
      tokens.c_str(), voices.c_str(), vocoder.c_str(), acoustic.c_str(),
      model.c_str(), enc.c_str(), dec.c_str(), lexicon.c_str(), data_dir.c_str());

  SherpaOnnxOfflineTtsConfig cfg;
  std::memset(&cfg, 0, sizeof(cfg));
  cfg.model.num_threads = num_threads;
  cfg.model.provider = provider;
  cfg.max_num_sentences = 1;

  // Advanced (key=value) overrides. `adv` owns the strings until the Create call.
  opts::KeyValues adv{advanced};
  const char* family = "(none)";

  if(icontains(dirname, "zipvoice") || (!enc.empty() && !dec.empty() && !vocoder.empty()))
  {
    family = "zipvoice";
    cfg.model.zipvoice.tokens = tokens.c_str();
    cfg.model.zipvoice.encoder = enc.c_str();
    cfg.model.zipvoice.decoder = dec.c_str();
    cfg.model.zipvoice.vocoder = vocoder.c_str();
    if(!lexicon.empty())
      cfg.model.zipvoice.lexicon = lexicon.c_str();
  }
  else if(icontains(dirname, "supertonic"))
  {
    family = "supertonic";
    // Named locals: the c_str() must outlive the Create* call below.
    static thread_local std::string dp, te, ve, tj, ui, vs;
    dp = find_in_dir(d, {"duration"});
    te = find_in_dir(d, {"text_encoder"});
    ve = find_in_dir(d, {"vector"});
    tj = find_in_dir(d, {"tts"}, ".json");
    ui = find_in_dir(d, {"unicode", "indexer"}, ".bin");
    vs = find_in_dir(d, {"voice"}, ".bin"); // voice style / speaker embedding
    cfg.model.supertonic.duration_predictor = dp.c_str();
    cfg.model.supertonic.text_encoder = te.c_str();
    cfg.model.supertonic.vector_estimator = ve.c_str();
    cfg.model.supertonic.vocoder = vocoder.c_str();
    if(!tj.empty())
      cfg.model.supertonic.tts_json = tj.c_str();
    if(!ui.empty())
      cfg.model.supertonic.unicode_indexer = ui.c_str();
    if(!vs.empty())
      cfg.model.supertonic.voice_style = vs.c_str();
  }
  else if(icontains(dirname, "pocket"))
  {
    family = "pocket";
    static thread_local std::string lf, lm, tc, vj, ts;
    lf = find_in_dir(d, {"lm_flow"});
    lm = find_in_dir(d, {"lm_main"});
    tc = find_in_dir(d, {"text_conditioner", "conditioner"});
    vj = find_in_dir(d, {"vocab"}, ".json");
    ts = find_in_dir(d, {"token_scores", "token_score"}, ".json");
    cfg.model.pocket.lm_flow = lf.c_str();
    cfg.model.pocket.lm_main = lm.c_str();
    cfg.model.pocket.encoder = enc.c_str();
    cfg.model.pocket.decoder = dec.c_str();
    cfg.model.pocket.text_conditioner = tc.c_str();
    if(!vj.empty())
      cfg.model.pocket.vocab_json = vj.c_str();
    if(!ts.empty())
      cfg.model.pocket.token_scores_json = ts.c_str();
  }
  else if(!voices.empty() && (icontains(dirname, "kitten") || icontains(model, "kitten")))
  {
    family = "kitten";
    cfg.model.kitten.model = model.c_str();
    cfg.model.kitten.voices = voices.c_str();
    cfg.model.kitten.tokens = tokens.c_str();
    if(!lexicon.empty())
      cfg.model.kitten.lexicon = lexicon.c_str();
    cfg.model.kitten.length_scale = 1.0f;
    adv.apply_float("length_scale", cfg.model.kitten.length_scale);
  }
  else if(!voices.empty())
  {
    family = "kokoro";
    // Kokoro (model.onnx + voices.bin + tokens [+ lexicons or espeak-ng-data]).
    cfg.model.kokoro.model = model.c_str();
    cfg.model.kokoro.voices = voices.c_str();
    cfg.model.kokoro.tokens = tokens.c_str();
    if(!lexicon.empty())
      cfg.model.kokoro.lexicon = lexicon.c_str();
    adv.apply_cstr("lang", cfg.model.kokoro.lang);
    cfg.model.kokoro.length_scale = 1.0f;
    adv.apply_float("length_scale", cfg.model.kokoro.length_scale);
  }
  else if(!vocoder.empty())
  {
    family = "matcha";
    // Matcha: an acoustic model + a separate vocoder (vocos / hifigan). The
    // acoustic file may be named acoustic / matcha / model-steps-*; fall back to
    // the first non-vocoder .onnx (`model`) when no explicit name matched.
    const std::string& am = !acoustic.empty() ? acoustic : model;
    cfg.model.matcha.acoustic_model = am.c_str();
    cfg.model.matcha.vocoder = vocoder.c_str();
    cfg.model.matcha.tokens = tokens.c_str();
    if(!lexicon.empty())
      cfg.model.matcha.lexicon = lexicon.c_str();
    cfg.model.matcha.noise_scale = 0.667f;
    cfg.model.matcha.length_scale = 1.0f;
    adv.apply_float("noise_scale", cfg.model.matcha.noise_scale);
    adv.apply_float("length_scale", cfg.model.matcha.length_scale);
  }
  else if(!model.empty())
  {
    family = "vits";
    // VITS (single model.onnx + tokens [+ lexicon or espeak-ng-data]).
    cfg.model.vits.model = model.c_str();
    cfg.model.vits.tokens = tokens.c_str();
    if(!lexicon.empty())
      cfg.model.vits.lexicon = lexicon.c_str();
    cfg.model.vits.noise_scale = 0.667f;
    cfg.model.vits.noise_scale_w = 0.8f;
    cfg.model.vits.length_scale = 1.0f;
    adv.apply_float("noise_scale", cfg.model.vits.noise_scale);
    adv.apply_float("noise_scale_w", cfg.model.vits.noise_scale_w);
    adv.apply_float("length_scale", cfg.model.vits.length_scale);
  }
  else
  {
    SHERPA_DBG("[sherpa] TTS: no family matched (no valid model files found) -> "
               "returning empty handle, object stays unavailable\n");
    return {};
  }

  // Top-level text-normalization / chunking overrides (all model families).
  adv.apply_cstr("rule_fsts", cfg.rule_fsts);
  adv.apply_cstr("rule_fars", cfg.rule_fars);
  adv.apply_int("max_num_sentences", cfg.max_num_sentences);

  SHERPA_DBG("[sherpa] TTS family='%s' -> SherpaOnnxCreateOfflineTts...\n", family);
  TtsModel m;
  m.tts = TtsHandle{L.SherpaOnnxCreateOfflineTts(&cfg)};
  SHERPA_DBG("[sherpa] TTS create %s (family='%s')\n", m.tts ? "OK" : "FAILED", family);
  if(!m.tts)
    return m;

  // The espeak-based families read text through a lexicon when the model ships
  // one, and need the caller's phonemes otherwise. zipvoice, supertonic and
  // pocket have their own text frontends.
  const std::string_view fam{family};
  const bool espeak_family
      = fam == "vits" || fam == "matcha" || fam == "kokoro" || fam == "kitten";
  if(espeak_family && lexicon.empty() && !data_dir.empty())
  {
    m.phonemize = true;
    m.espeak_data = data_dir;
    // The espeak-ng voice: an explicit `voice = ...`, else what the model's
    // metadata says, else en-us (what upstream falls back to as well).
    if(const char* v = adv.get_cstr("voice"))
      m.voice = v;
    else if(L.SherpaOnnxOfflineTtsLang && L.SherpaOnnxDestroyOfflineTtsLang)
    {
      if(const char* lang = L.SherpaOnnxOfflineTtsLang(m.tts.get()))
      {
        m.voice = lang;
        L.SherpaOnnxDestroyOfflineTtsLang(lang);
      }
    }
    if(m.voice.empty())
      m.voice = "en-us";
    SHERPA_DBG("[sherpa] TTS phonemizes with espeak-ng voice '%s'\n", m.voice.c_str());
  }
  return m;
}

}
