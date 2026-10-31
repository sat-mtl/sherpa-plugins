# sherpa-plugins

[Avendish](https://github.com/celtera/avendish) objects exposing the speech
features of [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — offline &
streaming speech recognition, voice activity detection and text-to-speech —
compiled, without rewrite, to every host that has a real object model
(ossia score, Max/MSP, Pure Data, Python, TouchDesigner, Godot, …).

## Strategy

sherpa-onnx is painful to build cross-platform (it does not build with clang).
We never build it here. Instead:

- **sherpa-onnx already ships a complete C ABI** (`sherpa-onnx/c-api/c-api.h`,
  ~200 `extern "C"` functions) and a `BUILD_SHARED_LIBS=ON` switch that emits a
  single `libsherpa-onnx-c-api.{so,dll,dylib}` (+ a sibling `libonnxruntime`).
  That prebuilt shared library is built once per platform with each platform's
  supported toolchain (gcc / MSVC / Apple-clang).
- This (clang-built) addon only `#include`s the pure-C `c-api.h` and
  **`dlopen`s** the prebuilt library at runtime — it never links it. Missing
  library ⇒ the object simply reports unavailable instead of failing to load.

So three layers: the prebuilt sherpa C-API `.so` ← a dlopen loader
(`src/SherpaLoader.hpp`) ← RAII/config helpers (`src/helpers/`) ← the objects.

## Objects

| `c_name`                  | Class              | Flow                                        |
|---------------------------|--------------------|---------------------------------------------|
| `sherpa_asr`              | `Asr`              | audio stream → utterances (unified streaming + batch) |
| `sherpa_asr_offline`      | `OfflineRecognizer`| audio buffer → text (batch, on trigger)     |
| `sherpa_asr_online`       | `OnlineRecognizer` | audio stream → incremental text + endpoint  |
| `sherpa_vad`              | `Vad`              | audio stream → speech-segment events        |
| `sherpa_tts`              | `Tts`              | text → synthesized audio                    |
| `sherpa_text`             | `TextProcessor`    | text → punctuated / diacritized text        |
| `sherpa_audio_labels`     | `AudioLabels`      | audio → {label, score}[] (tagging · LID)    |
| `sherpa_denoiser`         | `Denoiser`         | audio → enhanced audio                       |
| `sherpa_speaker`          | `Speaker`          | audio → embedding + enroll / identify       |
| `sherpa_keyword_spotter`  | `KeywordSpotter`   | audio stream → keyword events               |
| `sherpa_diarization`      | `Diarization`      | audio → {start, end, speaker, confidence}[] |
| `sherpa_source_separation`| `SourceSeparation` | audio → N separated stems                    |

Every object exposes **Provider** (CPU/CUDA/CoreML/DirectML) and **Threads**; the
ASR / VAD / TTS objects add an **`Advanced`** `key=value` port for the long tail of
sherpa options (see `docs/COVERAGE.md`).

All inference, model loading and result allocation happen on an Avendish
**worker** thread — never on the audio/DSP thread.

## Real-time or offline

Seven objects work on live audio (or live text). Five only give a result once a
recording is closed: their **Record** / **Listen** / **Capture** toggle goes off.
Most offline speech models can still follow a live input hands-free through
`sherpa_asr` in Batch mode, one result per sentence.

### Real-time: continuous input, output while it runs

| Object | What happens live | Latency |
|---|---|---|
| Streaming ASR (`sherpa_asr_online`) | Partial words while speaking, Final at each pause (endpoint) | ~0.1–0.6 s, depending on the model |
| `sherpa_asr`, Streaming mode | Same as Streaming ASR | same |
| VAD | Active is live; SpeechStart fires right away; Segment comes when the speech ends | tens of ms; Segment after the speech ends |
| Keyword Spotter | Fires as soon as a keyword is said | ~0.3 s |
| Denoiser (Auto or Online mode) | Cleaned audio out continuously | short; Offline mode works in ~1 s chunks |
| TTS | Speaks when Text changes | the time to synthesize one sentence: fast for Piper, Kokoro, Kitten, Matcha, Supertonic; slower for the voice-cloning ZipVoice and Pocket |
| Text Processor (punctuation) | Processes each text as it arrives | very short |

Streaming speech models: Nemotron 0.6B (English, the most accurate), NeMo CTC
80 ms (English, the lowest latency), Zipformer 20M (English: light, but makes
many mistakes), Zipformer CTC (English), Paraformer (Chinese and English),
T-one (Russian).

### Offline: record a passage, get the result when it closes

| Object | Toggle | Result |
|---|---|---|
| Offline ASR (`sherpa_asr_offline`) | Record | Text once Record goes off; Status says what was recorded or why no text came out |
| Audio Labels (sound tagging, language ID) | Listen | Labels and language once Listen goes off |
| Speaker ID | Capture | Speaker, score and embedding once Capture goes off |
| Diarization | Record | Who spoke when, after processing the whole passage |
| Source Separation | Record | Stems play back after the whole passage is processed |

Offline-only speech models: Whisper (English and multilingual), Moonshine,
SenseVoice, Paraformer (Chinese), Zipformer transducer, Zipformer CTC (with or
without HLG), NeMo CTC conformer, Canary, Cohere, Dolphin, FireRed, TeleSpeech,
WeNet (Cantonese), Omnilingual, MedASR, TDNN, Qwen3-ASR and FunASR-Nano.

### Hands-free on live input: one result per sentence

These offline objects can follow a live input; they still wait for each sentence
to end, plus the decode time.

- **`sherpa_asr` in Batch mode**, with any offline model above: its built-in VAD
  cuts the input into sentences, and each is decoded when the speaker pauses.
- **Offline ASR, Audio Labels and Speaker ID driven by a VAD:** cable the VAD's
  Active output into Record, Listen or Capture.
- **Diarization and Source Separation** have no live mode: they need the whole
  passage.

## Building

```bash
# Point at a locally-built sherpa-onnx shared install (lib/ + include/), or let
# CMake fetch the prebuilt release for this version pair (SHERPA_ONNX_FETCH_LIBS,
# on by default in a score build).
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DSHERPA_ONNX_DIR=/path/to/sherpa-onnx-install
cmake --build build
```

The same `CMakeLists.txt` also builds as an ossia/score addon when dropped into
a score source tree (it reuses score's Avendish via `find_package(Avendish)`).

Neither the download nor `SHERPA_ONNX_DIR` is needed to *compile*: compilation
only uses the vendored C header. When the runtime library cannot be located CMake
warns and keeps going, and the objects report themselves unavailable at run time.
Pass `-DSHERPA_ONNX_REQUIRE_LIBS=ON` (implied by a score deployment build) to turn
that into a configuration error, so a release is never packaged without it.

Building the runtime library yourself, against the onnxruntime score deploys —
this is the only way to get one before the `prebuilt-v<sherpa>-ort<ort>` release
for the current version pair is published:

```bash
# SHERPA_ONNX_VERSION in cmake/sherpa-onnx.cmake: v<release>, or the commit of
# a "<release>-<commit>" snapshot
git clone https://github.com/k2-fsa/sherpa-onnx src && git -C src checkout a63f821
export SHERPA_ONNXRUNTIME_INCLUDE_DIR=<score-build>/_deps/onnxruntime-src/include
export SHERPA_ONNXRUNTIME_LIB_DIR=<score-build>/_deps/onnxruntime-src/lib
cmake -S src -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
      -DSHERPA_ONNX_ENABLE_C_API=ON -DSHERPA_ONNX_ENABLE_TTS=ON \
      -DSHERPA_ONNX_ENABLE_PYTHON=OFF -DSHERPA_ONNX_ENABLE_TESTS=OFF \
      -DSHERPA_ONNX_ENABLE_BINARY=OFF -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
      -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF -DSHERPA_ONNX_ENABLE_JNI=OFF \
      -DCMAKE_INSTALL_PREFIX=$PWD/install
cmake --build build --target install
# Linux: the result must reference score's onnxruntime, not one of sherpa's own
readelf -d install/lib/libsherpa-onnx-c-api.so | grep NEEDED   # libonnxruntime.so.1
nm -D --undefined-only install/lib/libsherpa-onnx-c-api.so | grep Ort # @VERS_1.27.1

# The phonemizer (see "TTS and the phonemizer" below), into the same tree
cmake -S phonemizer -B build-phonemizer -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=$PWD/install
cmake --build build-phonemizer --target install
```

then configure score (or this addon) with `-DSHERPA_ONNX_DIR=<...>/install`.

## Runtime

`libsherpa-onnx-c-api` is `dlopen`'d at startup (`src/SherpaLoader.hpp`), never
linked. It in turn *links* onnxruntime, so something has to provide one:

* **In ossia score** — the host already deploys onnxruntime (in `lib/`,
  `Contents/Frameworks` or `bin/` depending on the platform). The addon binds to
  that copy: `src/OnnxRuntimeLoader.hpp` maps it *before* opening
  `libsherpa-onnx-c-api`, so the latter's `DT_NEEDED` / `LC_LOAD_DYLIB` / import
  resolves to the image already in the process — no rpath, nothing to copy into
  the plugin folder, and one onnxruntime in the process rather than two. If
  something else in the host loaded onnxruntime first, that image is reused
  instead.
* **Standalone (Max / TouchDesigner / Godot / pd)** — those hosts provide no
  onnxruntime, so each package bundles `libsherpa-onnx-c-api.*` *and*
  `libonnxruntime.*` beside the module (`avnd_addon_package(... SUPPORT ...)`).

This is why the prebuilt archives are linked against the **same onnxruntime
release score deploys** (`SHERPA_ONNXRUNTIME_VERSION` in
`cmake/sherpa-onnx.cmake`); a different build has a different SONAME, symbol
version and install name, and cannot bind to score's copy at all. If the host's onnxruntime turns out to be
older than the one sherpa was built against, the objects report themselves
unavailable instead of crashing.

Set `SHERPA_ONNXRUNTIME_PATH` to override the search, and `SHERPA_PLUGINS_DEBUG=1`
to print which library was resolved and where from.

### TTS and the phonemizer

sherpa-onnx no longer turns text into phonemes itself (it dropped espeak-ng and
piper-phonemize to be GPL-free, k2-fsa/sherpa-onnx#4055). A TTS model now reads
text in one of two ways:

* through a **lexicon** — when the model folder has `lexicon*.txt` (all of them
  are passed, e.g. kokoro multi-lang's `lexicon-us-en.txt` + `lexicon-zh.txt`);
* otherwise, for the espeak-based families (vits/piper, matcha, kokoro, kitten)
  with an `espeak-ng-data/` folder, from **espeak-ng phonemes** that the plugin
  computes itself with `libsherpa-plugins-phonemizer` (`phonemizer/`: piper-phonemize
  + espeak-ng behind a C ABI, built and shipped next to sherpa-onnx, dlopen'd by
  `src/PhonemizerLoader.hpp`). The espeak voice comes from the model's metadata
  (`SherpaOnnxOfflineTtsLang`), falling back to `en-us`; override it with
  `voice = fr` in the TTS object's `Advanced` port.

espeak-ng is GPL-3.0-or-later, and so is the phonemizer library.

> Status: initial scaffold. Needs a first local compile pass; a few halp
> member spellings (soundfile port) and the prebuilt-release asset URLs are
> marked `TODO/VERIFY` in the sources.
