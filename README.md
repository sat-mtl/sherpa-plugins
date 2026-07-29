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
| `sherpa_diarization`      | `Diarization`      | audio → {start, end, speaker}[]             |
| `sherpa_source_separation`| `SourceSeparation` | audio → N separated stems                    |

Every object exposes **Provider** (CPU/CUDA/CoreML/DirectML) and **Threads**; the
ASR / VAD / TTS objects add an **`Advanced`** `key=value` port for the long tail of
sherpa options (see `docs/COVERAGE.md`).

All inference, model loading and result allocation happen on an Avendish
**worker** thread — never on the audio/DSP thread.

## Building

```bash
# Point at a locally-built sherpa-onnx shared install (lib/ + include/), or let
# CMake fetch a prebuilt release once the asset URLs are wired in sherpa-onnx.cmake.
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DSHERPA_ONNX_DIR=/path/to/sherpa-onnx-install
cmake --build build
```

The same `CMakeLists.txt` also builds as an ossia/score addon when dropped into
a score source tree (it reuses score's Avendish via `find_package(Avendish)`).

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

TTS additionally needs the model's `espeak-ng-data/` directory (supplied with the
model).

> Status: initial scaffold. Needs a first local compile pass; a few halp
> member spellings (soundfile port) and the prebuilt-release asset URLs are
> marked `TODO/VERIFY` in the sources.
