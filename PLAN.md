# Meeting Transcribe — Project Plan

**Target:** Fully offline meeting transcription + speaker diarization
**Hardware:** Mac M4, 32GB RAM
**Status:** Stack finalized — July 6, 2026

---

## Requirements

- Transcription (realtime + accurate final pass)
- Speaker diarization (who spoke when)
- 100% offline (client requirement — no data leaves the machine)
- Confidence scores for transcript and speakers

---

## Final Stack

### 1. Realtime ASR (live captions)

| Model | Params | Engine | Speed | Languages |
|---|---|---|---|---|
| Nemotron 3.5 ASR Streaming | 0.6B | MLX or CoreML (ANE) | 112x RT, 80ms–1.1s chunks | 40 locales |

- Cache-aware FastConformer-RNNT: processes each frame once → ultra-low latency
- Native punctuation and capitalization
- Model: `nvidia/nemotron-3.5-asr-streaming-0.6b` (multilingual — NOT the `-en` variant)
- MLX port: `199-biotechnologies/nemotron-asr-mlx` · CoreML: `FluidInference/Nemotron-3.5-ASR-Streaming-Multilingual-0.6b-CoreML`

### 2. Chunked ASR (final accurate transcript — selectable, A/B test per language)

| Model | Params | Engine | WER | Speed | Languages |
|---|---|---|---|---|---|
| Qwen3-ASR | 1.7B | MLX (4/5/8-bit) | 1.32% (best) | 30–36x RT | 52 incl. Indonesian |
| Whisper large-v3 | 1.5B | MLX | ~7.4% FLEURS | ~5–8x RT | 99+ (widest) |
| Voxtral Mini Transcribe 2 | 3B | MLX | 5.9% FLEURS | ~10x RT | 13 |

- Qwen3-ASR = primary candidate. Known issue: Indonesian ↔ Malay confusion — validate on real client audio
- Whisper large-v3 = fallback for languages Qwen3/Voxtral don't cover
- All three provide word-level timestamps and logprob confidence

### 3. Diarization

| Model | Engine | DER (AMI SDM) |
|---|---|---|
| pyannote community-1 (pyannote.audio 4.0) | PyTorch MPS | 19.9% |

- Best open-source DER; tuned end-to-end pipeline (segmentation + WeSpeaker embeddings + clustering)
- Decision: do NOT build custom MLX diarization — MLX ports only cover the older 3.1 models (22.7% DER) and lose the tuned pipeline
- Confidence scores: not built-in → DIY (~50–100 lines):
  - Segmentation confidence: average frame-level speaker activity probabilities per segment
  - Speaker confidence: embedding distance to assigned cluster centroid

### 4. Supporting components

| Component | Choice | Notes |
|---|---|---|
| VAD | Silero VAD v5 | Pre-segment, cut silence |
| Word confidence | ASR logprobs (native) | All 4 models |
| Speaker confidence | DIY from community-1 internals | See above |
| Summarization (optional) | Qwen 8B 4-bit via MLX | ~5GB RAM |

---

## Pipeline

```
Mic → Silero VAD → Nemotron 0.6B ──────────────→ live captions
         ↓ (recording saved)
Recording → Qwen3-ASR / Whisper / Voxtral ─────→ accurate transcript + word confidence
Recording → pyannote community-1 ──────────────→ speaker segments + speaker confidence
         ↓
Merge by timestamps → final diarized transcript → (optional) LLM summary
```

---

## Engine strategy

| Engine | Used for | Why |
|---|---|---|
| MLX | Qwen3-ASR, Whisper, Voxtral, Nemotron, LLM | Best for models >1B params; GPU; direct HF downloads |
| CoreML | Nemotron (alternative) | ANE = lowest latency/power for <1B models |
| PyTorch MPS | pyannote community-1 | Only engine that runs it; ~10x faster than CPU |

Rule of thumb: <2B params → CoreML wins (ANE). >2B params → MLX. Diarization → tuned pipeline beats fast engine.

---

## Memory budget (32GB)

Worst case simultaneous: Voxtral 3B (~4GB) + pyannote (~2GB) + Nemotron (~1GB) + OS/app ≈ 12–14GB → comfortable headroom.

## Offline deployment notes

- One-time internet needed to download weights (~5–8GB total); pyannote community-1 is gated on HuggingFace (free token)
- After download: set `HF_HUB_OFFLINE=1` → fully air-gapped
- pyannoteAI precision-2 (cloud confidence scores) explicitly rejected — violates offline requirement

## Open items

- [ ] A/B test Qwen3-ASR vs Whisper large-v3 vs Voxtral on real client meeting audio (esp. Indonesian id/ms confusion)
- [ ] Implement DIY speaker confidence scores
- [ ] Decide on summarization LLM (if client wants it)
- [ ] Tune pyannote thresholds on client audio if DER unsatisfactory
