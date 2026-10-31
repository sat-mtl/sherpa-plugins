/* sherpa-plugins phonemizer: text -> espeak-ng phonemes, as Unicode codepoints.
 *
 * sherpa-onnx no longer phonemizes text itself (it dropped espeak-ng and
 * piper-phonemize to be GPL-free): models without a lexicon now expect
 * SherpaOnnxGenerationConfig.phoneme_codepoints, computed by the caller with
 * piper-phonemize. This library is that caller-side step, built as a shared
 * library with a plain C ABI so the plugin can dlopen it next to sherpa-onnx
 * whatever compiler built either side.
 *
 * Links espeak-ng, so the binary is GPL-3.0-or-later. */
#ifndef SHERPA_PLUGINS_PHONEMIZER_H_
#define SHERPA_PLUGINS_PHONEMIZER_H_

#include <stdint.h>

#if defined(SHERPA_PLUGINS_PHONEMIZER_BUILD)
#if defined(_WIN32)
#define SHERPA_PLUGINS_PHONEMIZER_API __declspec(dllexport)
#else
#define SHERPA_PLUGINS_PHONEMIZER_API __attribute__((visibility("default")))
#endif
#else
#define SHERPA_PLUGINS_PHONEMIZER_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Sentences of phonemes, flattened: sentence i is the lens[i] codepoints
 * following the ones of sentences 0..i-1. Laid out to be passed straight to
 * SherpaOnnxGenerationConfig.phoneme_codepoints{,_lens,_num_sentences}. */
typedef struct SherpaPluginsPhonemes {
  const int32_t *codepoints;
  const int32_t *lens;
  int32_t num_sentences;
} SherpaPluginsPhonemes;

/* Phonemize UTF-8 `text` with the espeak-ng `voice` (e.g. "en-us", as returned
 * by SherpaOnnxOfflineTtsLang), reading espeak-ng's data from `data_dir` (a
 * model's espeak-ng-data directory, or the directory containing one).
 *
 * Thread-safe (espeak-ng is not: calls are serialized). espeak-ng is
 * re-initialized when data_dir differs from the previous call.
 *
 * Returns NULL on failure: espeak-ng could not load data_dir, it does not know
 * the voice, or the text produced no phonemes. Free with
 * SherpaPluginsPhonemizerFree. */
SHERPA_PLUGINS_PHONEMIZER_API const SherpaPluginsPhonemes *
SherpaPluginsPhonemize(const char *data_dir, const char *voice,
                       const char *text);

SHERPA_PLUGINS_PHONEMIZER_API void SherpaPluginsPhonemizerFree(
    const SherpaPluginsPhonemes *p);

/* espeak-ng version string, for diagnostics. Static storage, do not free. */
SHERPA_PLUGINS_PHONEMIZER_API const char *SherpaPluginsPhonemizerVersion(void);

#ifdef __cplusplus
}
#endif

#endif
