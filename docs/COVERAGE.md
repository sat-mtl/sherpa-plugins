# Coverage

sherpa-plugins wraps **100% of the sherpa-onnx workflow families** in 11 Avendish
objects, grouped by input→output signature (so related features share one object,
like `ImageProcessor` fronting many vision models).

## Objects ↔ sherpa-onnx features

| Object (`c_name`) | I/O | sherpa-onnx features covered |
|---|---|---|
| `sherpa_asr_offline` | audio → text (+meta) | non-streaming ASR |
| `sherpa_asr_online` | audio → text (+meta) | streaming ASR |
| `sherpa_tts` | text → audio | TTS |
| `sherpa_vad` | audio → speech segments | voice activity detection |
| `sherpa_text` | text → text | offline punctuation · online punctuation · diacritization |
| `sherpa_audio_labels` | audio → {label,score}[] | audio tagging · spoken-language-ID |
| `sherpa_denoiser` | audio → audio | offline denoise · online denoise |
| `sherpa_speaker` | audio → embedding + match | speaker embedding · verification/identification |
| `sherpa_keyword_spotter` | audio → keyword events | keyword spotting |
| `sherpa_diarization` | audio → {start,end,speaker}[] | speaker diarization |
| `sherpa_source_separation` | audio → N audio stems | source separation |

Every `SherpaOnnxCreate*` entry point is asserted resolved by `tests/test_coverage.cpp`.

## Model families

| Object | Auto-detected families | Notes |
|---|---|---|
| offline ASR | transducer, paraformer, whisper, sense-voice, moonshine, fire-red, canary, nemo-ctc, zipformer-ctc, wenet-ctc, telespeech, dolphin | exotic (qwen3/cohere/tdnn/med/omni/funasr) reachable by pointing the Model dir precisely |
| online ASR | transducer, paraformer, zipformer2-ctc, nemo-ctc, t-one-ctc | |
| TTS | vits, matcha, kokoro, kitten, zipvoice, pocket, supertonic | |
| VAD | silero, ten-vad | |
| text | ct-transformer (offline punct), cnn-bilstm (online punct), catt (diacritization) | via `Type` override or dir name |
| audio_labels | zipformer / CED (tagging), whisper (LID) | via `Type` override |
| denoiser | gtcrn, dpdfnet | `Mode` Auto/Offline/Online |
| speaker | 3d-speaker / nemo / wespeaker embeddings | |
| keyword_spotter | transducer KWS | keywords via file or inline lineedit |
| diarization | pyannote segmentation + speaker embedding | |
| source_separation | spleeter, uvr | |

Detection is heuristic (filenames + dir name); every multi-family object also has a
`Type`/`Mode` override enum so any family is reachable even when auto-detect misses.

## Options

- **Provider** (CPU / CUDA / CoreML / DirectML) — exposed on all 4 core objects
  (ASR offline/online, VAD, TTS); sherpa falls back to CPU if unavailable.
- **Threads** — everywhere.
- **ASR**: endpointing (online), result metadata out (tokens, timestamps,
  lang/emotion/event).
- **VAD**: threshold. **TTS**: speaker, speed. **KWS**: keywords, threshold.
- **Speaker**: enroll/remove/clear messages + threshold.

### Still hardcoded / not yet exposed (roadmap)

- Provider on the 6 extended objects (they build config inline — CPU only for now).
- decoding_method, hotwords, rule-FSTs (text normalization/ITN), blank_penalty,
  max_active_paths; VAD min/max-speech/window; TTS noise/length scales,
  silence_scale, and **zero-shot voice cloning** (reference_audio/reference_text);
  whisper language/task; an `Advanced (key=value)` escape-hatch port for the long
  tail of config fields.

## Verification

- **Structural** (`tests/test_structural.cpp`): every object is nothrow-default-
  constructible (Max/Pd requirement), has metadata, and no-ops safely without a
  worker/library.
- **Coverage** (`tests/test_coverage.cpp`): all feature entry points resolve.
- **Workflow** (`tests/test_<feature>.cpp`): each object is driven through its real
  pipeline against a model (gated on env vars; run in CI via `.github/workflows/
  tests.yml`, which downloads a small model per feature).
- Every object also builds Avendish's SDK-free `dump` backend and introspects the
  intended ports.
