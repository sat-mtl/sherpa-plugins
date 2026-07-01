# Coverage

sherpa-plugins wraps **100% of the sherpa-onnx workflow families** in 12 Avendish
objects, grouped by input→output signature (so related features share one object,
like `ImageProcessor` fronting many vision models).

## Objects ↔ sherpa-onnx features

| Object (`c_name`) | I/O | sherpa-onnx features covered |
|---|---|---|
| `sherpa_asr` | audio → utterances (+meta) | **unified** streaming + non-streaming ASR (streaming finalizes via endpointing; non-streaming via an internal VAD) |
| `sherpa_asr_offline` | audio → text (+meta) | non-streaming ASR (push-to-talk) |
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

## Model families & validation status

Two independent axes, enumerated against the pinned **sherpa-onnx 1.13.3** C-API:

- **Code** — can an object reach the family? Filename auto-detect, plus a
  `Model type` (ASR) / `Type` (text, labels) / `Mode` (denoiser) override so any
  family is reachable even when the heuristic misses.
- **CI** — is a real model of that family run through inference in
  `.github/workflows/tests.yml`?

**CI validates one model per feature** (the "CI" column). Every other supported
family shares the *same* code path but is not downloaded in CI — a deliberate
bandwidth choice, not a code gap. Expanding is just adding a model to `tests.yml`.

| Feature | Families reachable in code | CI-validated | Not wired |
|---|---|---|---|
| **Offline ASR** (18 in API) | transducer, paraformer, whisper, sense-voice, moonshine, fire-red-asr, canary, telespeech, wenet-ctc, dolphin, zipformer-ctc, nemo-ctc, **tdnn, omnilingual, medasr, fire-red-ctc, cohere** — all auto-detected; force any with the **`Model type`** port | whisper (whisper-tiny.en) | **funasr-nano** (LLM multi-file pipeline: encoder-adaptor + llm + embedding + tokenizer + prompts — use sherpa's own tooling) |
| **Online ASR** (5) | transducer, paraformer, zipformer2-ctc, nemo-ctc, t-one-ctc — all | transducer (streaming-zipformer) | — |
| **TTS** (7) | vits, matcha, kokoro, kitten, zipvoice, pocket, supertonic — all | vits (piper-amy) | — |
| **VAD** (2) | silero, ten-vad | silero | — |
| **Text** (3) | offline-punct (ct-transformer), online-punct (cnn-bilstm), diacritization (catt) — `Type` | offline-punct | — |
| **Audio labels** | tagging (zipformer, CED), language-ID (whisper) — `Type` | zipformer tagging | — |
| **Denoiser** (2) | gtcrn, dpdfnet — `Mode` | gtcrn | — |
| **Source sep** (2) | spleeter, uvr | spleeter | — |
| **Speaker** | any embedding model (3d-speaker / wespeaker / nemo — one API) | 3d-speaker | — |
| **KWS** | transducer KWS | gigaspeech KWS | — |
| **Diarization** | pyannote segmentation + speaker embedding | pyannote + 3d-speaker | — |

The **unified `sherpa_asr`** reaches every offline + online family above (`Mode`
Auto/Streaming/Batch; Batch pairs a non-streaming model with a VAD via the 2nd port).

**`Model type` override values** (offline ASR / unified ASR, empty = auto-detect):
`transducer, paraformer, whisper, sense_voice, moonshine, fire_red_asr, canary,
cohere, telespeech, wenet_ctc, dolphin, zipformer_ctc, nemo_ctc, tdnn, omnilingual,
medasr, fire_red_asr_ctc`.

So: **code coverage is complete** (every 1.13.3 family reachable except the
funasr-nano LLM pipeline); **CI coverage is one representative model per feature**.

## Options

- **Provider** (CPU / CUDA / CoreML / DirectML) — exposed on **all 11 objects**;
  sherpa falls back to CPU if the requested provider is unavailable.
- **Threads** — everywhere.
- **ASR**: decoding method (greedy / modified-beam-search), endpointing (online),
  result metadata out (tokens, timestamps, lang/emotion/event).
- **VAD**: threshold. **TTS**: speaker, speed, silence, steps, and zero-shot
  **voice cloning** (reference audio + text). **KWS**: keywords, threshold.
- **Speaker**: enroll/remove/clear messages + threshold.

### Advanced (key=value) escape hatch

The 4 core objects (ASR offline/online, VAD, TTS) expose an **`Advanced`** free-text
port for the long tail of sherpa config fields. Entries are `key = value`, separated
by newlines or `;` (not `,`, which appears inside values); keys are case-insensitive.
Parsed on the worker thread, applied over the auto-built config. See
`src/helpers/Options.hpp`.

| Object | Keys |
|---|---|
| ASR offline | `hotwords_file`, `hotwords_score`, `blank_penalty`, `max_active_paths`, `rule_fsts`, `rule_fars`, whisper `language` / `task` / `tail_paddings` |
| ASR online | `hotwords_file`, `hotwords_score`, `blank_penalty`, `max_active_paths`, `rule_fsts`, `rule_fars`, `rule1_min_trailing_silence`, `rule2_min_trailing_silence`, `rule3_min_utterance_length` |
| VAD | `threshold`, `min_silence_duration`, `min_speech_duration`, `max_speech_duration`, `window_size` |
| TTS | `noise_scale`, `noise_scale_w` (vits), `length_scale`, `rule_fsts`, `rule_fars`, `max_num_sentences` |

Example (TTS): `length_scale = 1.2 ; noise_scale = 0.5`

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
