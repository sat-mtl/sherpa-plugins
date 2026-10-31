// Implementation of sherpa-plugins/phonemizer.h: piper-phonemize over espeak-ng,
// exactly what sherpa-onnx's own frontend did before it removed it
// (sherpa-onnx/csrc/piper-phonemize-lexicon.cc, CallPhonemizeEspeak + InitEspeak,
// at k2-fsa/sherpa-onnx@0b36476^).

#include <sherpa-plugins/phonemizer.h>

#include <espeak-ng/speak_lib.h>
#include <phonemize.hpp>

#include <cstdio>
#include <exception>
#include <mutex>
#include <string>
#include <vector>

namespace
{
struct Phonemes : SherpaPluginsPhonemes
{
  std::vector<int32_t> cp;
  std::vector<int32_t> len;
};

// espeak-ng is a process-wide singleton with no thread safety at all.
std::mutex g_mutex;
std::string g_data_dir;
bool g_initialized = false;

bool ensure_initialized(const std::string& data_dir)
{
  if(g_initialized && data_dir == g_data_dir)
    return true;
  if(g_initialized)
  {
    espeak_Terminate();
    g_initialized = false;
  }
  // Returns the sample rate (22050) on success, a negative error otherwise.
  const int r = espeak_Initialize(
      AUDIO_OUTPUT_SYNCHRONOUS, 0, data_dir.empty() ? nullptr : data_dir.c_str(), 0);
  if(r <= 0)
  {
    std::fprintf(
        stderr, "sherpa-plugins phonemizer: espeak-ng cannot load '%s' (%d)\n",
        data_dir.c_str(), r);
    return false;
  }
  g_data_dir = data_dir;
  g_initialized = true;
  return true;
}
}

extern "C" {

const SherpaPluginsPhonemes*
SherpaPluginsPhonemize(const char* data_dir, const char* voice, const char* text)
{
  if(!text || !*text)
    return nullptr;

  try
  {
    std::vector<std::vector<piper::Phoneme>> sentences;
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      if(!ensure_initialized(data_dir ? data_dir : ""))
        return nullptr;

      piper::eSpeakPhonemeConfig config;
      if(voice && *voice)
        config.voice = voice;
      // Throws when espeak-ng does not know the voice.
      piper::phonemize_eSpeak(text, config, sentences);
    }

    auto* out = new Phonemes{};
    for(const auto& s : sentences)
    {
      if(s.empty())
        continue; // sherpa rejects zero-length sentences
      out->len.push_back(static_cast<int32_t>(s.size()));
      for(char32_t c : s)
        out->cp.push_back(static_cast<int32_t>(c));
    }
    if(out->len.empty())
    {
      delete out;
      return nullptr;
    }
    out->codepoints = out->cp.data();
    out->lens = out->len.data();
    out->num_sentences = static_cast<int32_t>(out->len.size());
    return out;
  }
  catch(const std::exception& e)
  {
    std::fprintf(
        stderr, "sherpa-plugins phonemizer: voice '%s': %s\n", voice ? voice : "",
        e.what());
    return nullptr;
  }
}

void SherpaPluginsPhonemizerFree(const SherpaPluginsPhonemes* p)
{
  delete static_cast<const Phonemes*>(p);
}

const char* SherpaPluginsPhonemizerVersion(void)
{
  const char* path = nullptr;
  return espeak_Info(&path);
}
}
