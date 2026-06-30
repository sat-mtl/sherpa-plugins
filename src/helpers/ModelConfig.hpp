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

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <string>
#include <string_view>

namespace sherpa::model
{
namespace fs = std::filesystem;

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

// ---------------------------------------------------------------------------
// Offline (batch) ASR
// ---------------------------------------------------------------------------
inline OfflineRecognizerHandle create_offline_recognizer(
    std::string_view dir, int num_threads, std::string_view decoding)
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
  std::string dirname = to_lower(d.filename().string());

  std::string dec_method{decoding};

  SherpaOnnxOfflineRecognizerConfig cfg;
  std::memset(&cfg, 0, sizeof(cfg));
  cfg.feat_config.sample_rate = 16000; // input is resampled to this internally
  cfg.feat_config.feature_dim = 80;
  cfg.model_config.tokens = tokens.c_str();
  cfg.model_config.num_threads = num_threads;
  cfg.model_config.provider = "cpu";
  cfg.decoding_method = dec_method.c_str();

  if(!enc.empty() && !dec.empty() && !joiner.empty())
  {
    cfg.model_config.transducer.encoder = enc.c_str();
    cfg.model_config.transducer.decoder = dec.c_str();
    cfg.model_config.transducer.joiner = joiner.c_str();
  }
  else if(!enc.empty() && !dec.empty()
          && (icontains(enc, "whisper") || icontains(dirname, "whisper")))
  {
    cfg.model_config.whisper.encoder = enc.c_str();
    cfg.model_config.whisper.decoder = dec.c_str();
  }
  else if(icontains(any, "paraformer") || icontains(dirname, "paraformer"))
  {
    cfg.model_config.paraformer.model = any.c_str();
  }
  else if(icontains(any, "sense") || icontains(dirname, "sense"))
  {
    cfg.model_config.sense_voice.model = any.c_str();
    cfg.model_config.sense_voice.use_itn = 1;
  }
  else if(icontains(any, "ctc") || icontains(any, "nemo")
          || icontains(dirname, "ctc") || icontains(dirname, "nemo"))
  {
    cfg.model_config.nemo_ctc.model = any.c_str();
  }
  else if(!any.empty())
  {
    // Last resort: try SenseVoice (multi-lingual, common).
    cfg.model_config.sense_voice.model = any.c_str();
    cfg.model_config.sense_voice.use_itn = 1;
  }
  else
  {
    return {};
  }

  return OfflineRecognizerHandle{L.SherpaOnnxCreateOfflineRecognizer(&cfg)};
}

// ---------------------------------------------------------------------------
// Online (streaming) ASR
// ---------------------------------------------------------------------------
inline OnlineRecognizerHandle create_online_recognizer(
    std::string_view dir, int num_threads, std::string_view decoding,
    bool enable_endpoint, std::string_view hotwords_file)
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
  std::string dirname = to_lower(d.filename().string());

  std::string dec_method{decoding};
  std::string hotwords{hotwords_file};

  SherpaOnnxOnlineRecognizerConfig cfg;
  std::memset(&cfg, 0, sizeof(cfg));
  cfg.feat_config.sample_rate = 16000;
  cfg.feat_config.feature_dim = 80;
  cfg.model_config.tokens = tokens.c_str();
  cfg.model_config.num_threads = num_threads;
  cfg.model_config.provider = "cpu";
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
  else if(icontains(any, "ctc") || icontains(dirname, "ctc")
          || icontains(any, "zipformer"))
  {
    cfg.model_config.zipformer2_ctc.model = any.c_str();
  }
  else if(icontains(any, "nemo"))
  {
    cfg.model_config.nemo_ctc.model = any.c_str();
  }
  else
  {
    return {};
  }

  return OnlineRecognizerHandle{L.SherpaOnnxCreateOnlineRecognizer(&cfg)};
}

// ---------------------------------------------------------------------------
// VAD. `path` may be a directory (we find silero_vad.onnx / ten-vad.onnx inside)
// or a direct .onnx path.
// ---------------------------------------------------------------------------
inline VadHandle create_vad(
    std::string_view path, float threshold, int sample_rate, int num_threads,
    float buffer_seconds = 30.f)
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
  cfg.provider = "cpu";
  if(is_ten)
  {
    cfg.ten_vad.model = model.c_str();
    cfg.ten_vad.threshold = threshold;
    cfg.ten_vad.min_silence_duration = 0.5f;
    cfg.ten_vad.min_speech_duration = 0.25f;
    cfg.ten_vad.max_speech_duration = 20.f;
    cfg.ten_vad.window_size = 256;
  }
  else
  {
    cfg.silero_vad.model = model.c_str();
    cfg.silero_vad.threshold = threshold;
    cfg.silero_vad.min_silence_duration = 0.5f;
    cfg.silero_vad.min_speech_duration = 0.25f;
    cfg.silero_vad.max_speech_duration = 20.f;
    cfg.silero_vad.window_size = 512;
  }

  return VadHandle{L.SherpaOnnxCreateVoiceActivityDetector(&cfg, buffer_seconds)};
}

// Window size (samples at the model rate) the VAD expects per AcceptWaveform.
inline int vad_window_size(std::string_view path)
{
  return icontains(std::string{path}, "ten") ? 256 : 512;
}

// ---------------------------------------------------------------------------
// TTS
// ---------------------------------------------------------------------------
inline TtsHandle create_tts(std::string_view dir, int num_threads)
{
  const auto& L = SherpaLoader::instance();
  fs::path d{dir};

  std::string tokens = file_if_exists(d, "tokens.txt");
  std::string voices = find_in_dir(d, {"voices"}, ""); // .bin
  std::string vocoder = find_in_dir(d, {"vocoder", "vocos", "hifigan"});
  std::string acoustic = find_in_dir(d, {"acoustic", "matcha"});
  std::string model = find_in_dir(d, {"model"});
  if(model.empty())
    model = find_in_dir(d, {}); // first .onnx
  std::string lexicon = file_if_exists(d, "lexicon.txt");
  std::string data_dir;
  {
    std::error_code ec;
    auto espeak = d / "espeak-ng-data";
    if(fs::is_directory(espeak, ec))
      data_dir = espeak.string();
  }
  std::string dirname = to_lower(d.filename().string());

  SherpaOnnxOfflineTtsConfig cfg;
  std::memset(&cfg, 0, sizeof(cfg));
  cfg.model.num_threads = num_threads;
  cfg.model.provider = "cpu";
  cfg.max_num_sentences = 1;

  if(!voices.empty() && (icontains(dirname, "kitten") || icontains(model, "kitten")))
  {
    cfg.model.kitten.model = model.c_str();
    cfg.model.kitten.voices = voices.c_str();
    cfg.model.kitten.tokens = tokens.c_str();
    if(!data_dir.empty())
      cfg.model.kitten.data_dir = data_dir.c_str();
    cfg.model.kitten.length_scale = 1.0f;
  }
  else if(!voices.empty())
  {
    // Kokoro (model.onnx + voices.bin + tokens [+ espeak-ng-data]).
    cfg.model.kokoro.model = model.c_str();
    cfg.model.kokoro.voices = voices.c_str();
    cfg.model.kokoro.tokens = tokens.c_str();
    if(!data_dir.empty())
      cfg.model.kokoro.data_dir = data_dir.c_str();
    cfg.model.kokoro.length_scale = 1.0f;
  }
  else if(!vocoder.empty() && !acoustic.empty())
  {
    cfg.model.matcha.acoustic_model = acoustic.c_str();
    cfg.model.matcha.vocoder = vocoder.c_str();
    cfg.model.matcha.tokens = tokens.c_str();
    if(!lexicon.empty())
      cfg.model.matcha.lexicon = lexicon.c_str();
    if(!data_dir.empty())
      cfg.model.matcha.data_dir = data_dir.c_str();
    cfg.model.matcha.noise_scale = 0.667f;
    cfg.model.matcha.length_scale = 1.0f;
  }
  else if(!model.empty())
  {
    // VITS (single model.onnx + tokens [+ lexicon or espeak-ng-data]).
    cfg.model.vits.model = model.c_str();
    cfg.model.vits.tokens = tokens.c_str();
    if(!lexicon.empty())
      cfg.model.vits.lexicon = lexicon.c_str();
    if(!data_dir.empty())
      cfg.model.vits.data_dir = data_dir.c_str();
    cfg.model.vits.noise_scale = 0.667f;
    cfg.model.vits.noise_scale_w = 0.8f;
    cfg.model.vits.length_scale = 1.0f;
  }
  else
  {
    return {};
  }

  return TtsHandle{L.SherpaOnnxCreateOfflineTts(&cfg)};
}

}
