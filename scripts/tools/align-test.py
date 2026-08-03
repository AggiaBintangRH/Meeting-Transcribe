#!/usr/bin/env python3
"""Standalone quality/latency probe for Qwen3-ForcedAligner-0.6B (MLX).

Phase-1 de-risking tool for word-exact speaker attribution: it answers
"are the per-word timestamps good enough on the owner's real ATND-mic audio,
and how long does alignment take?" BEFORE any app wiring exists.

Forced alignment takes audio + ALREADY-KNOWN text and returns a start/end time
per word. It never sees the ASR, so any of the four chunked models can supply
the text — that is exactly what this script checks by letting you pick.

    # transcribe a 30s slice with Qwen3-ASR, then align it
    scripts/tools/align-test.py --audio recordings/foo.wav --start 0 --end 30

    # align text you already have (no ASR run at all)
    scripts/tools/align-test.py --audio recordings/foo.wav --text "hello there"

    # same slice through a different ASR, to compare alignment fidelity
    scripts/tools/align-test.py --audio recordings/foo.wav --end 30 --asr whisper

Not part of the app; nothing imports it.
"""

import argparse
import os
import pathlib
import sys
import time

# Pin the model cache to the project's own models/ dir before importing
# anything that reads HF_HOME, exactly like the sidecars do. This file lives at
# scripts/tools/, so the project root is THREE levels up (owner, 2026-07-29:
# one folder per service).
PROJECT = pathlib.Path(__file__).resolve().parent.parent.parent
os.environ.setdefault("HF_HOME", str(PROJECT / "models"))

SR = 16000

ASR_REPOS = {
    "qwen3": "mlx-community/Qwen3-ASR-1.7B-bf16",
    "whisper": "mlx-community/whisper-large-v3-mlx",
    "voxtral": "mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16",
    "granite": "mlx-community/granite-speech-4.1-2b-nar-mlx",
}

ALIGNER_REPO = "mlx-community/Qwen3-ForcedAligner-0.6B-bf16"


def transcribe(audio, asr: str, repo: str) -> str:
    """Run one chunked-ASR model over `audio`, mirroring what the per-model ASR
    sidecars do (whisper/ goes through mlx-whisper; qwen3/ granite/ voxtral/ through
    mlx-audio) so the text this script aligns is the text the app would align.

    The dispatch is kept as ONE branch here on purpose: this is a developer tool
    driving several models in one run, not the shipped path — which is one
    standalone sidecar per model since 2026-07-30."""
    import numpy as np

    if "whisper" in asr.lower():
        import mlx_whisper

        result = mlx_whisper.transcribe(
            np.asarray(audio, dtype=np.float32),
            path_or_hf_repo=repo,
            language="en",
        )
        return (result.get("text") or "").strip()

    from mlx_audio.stt import load

    model = load(repo)
    try:
        result = model.generate(audio, language="en")
    except TypeError:  # model takes no language kwarg
        result = model.generate(audio)
    return (result.text or "").strip()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio", required=True, help="path to a WAV file")
    p.add_argument("--start", type=float, default=0.0, help="slice start (seconds)")
    p.add_argument("--end", type=float, default=None, help="slice end (seconds)")
    p.add_argument("--text", default=None,
                   help="text to align; omit to transcribe the slice first")
    p.add_argument("--asr", default="qwen3", choices=sorted(ASR_REPOS),
                   help="which chunked ASR supplies the text (default: qwen3)")
    p.add_argument("--align-model", default=ALIGNER_REPO)
    args = p.parse_args()

    from mlx_audio.stt import load
    from mlx_audio.stt.utils import load_audio

    print(f"HF_HOME  {os.environ['HF_HOME']}")
    print(f"audio    {args.audio}")

    audio = load_audio(args.audio, sr=SR)
    total = len(audio) / SR
    end = total if args.end is None else min(args.end, total)
    start = max(0.0, args.start)
    if end <= start:
        print(f"!! empty slice: start={start} end={end} (file is {total:.1f}s)")
        return 1
    sliced = audio[int(start * SR):int(end * SR)]
    dur = len(sliced) / SR
    print(f"slice    {start:.1f}s .. {end:.1f}s  ({dur:.1f}s of {total:.1f}s)")

    # ---- text -------------------------------------------------------------
    text = args.text
    if text is None:
        repo = ASR_REPOS[args.asr]
        print(f"\n==> Transcribing with {args.asr} ({repo}) ...")
        t0 = time.time()
        text = transcribe(sliced, args.asr, repo)
        asr_sec = time.time() - t0
        print(f"    {asr_sec:.1f}s  ({dur / asr_sec:.1f}x realtime)")
    else:
        print("\n==> Using supplied text (no ASR run)")
    if not text:
        print("!! empty text — nothing to align")
        return 1
    print(f"\nTEXT ({len(text.split())} words):\n{text}\n")

    # ---- align ------------------------------------------------------------
    print(f"==> Loading aligner {args.align_model} ...")
    t0 = time.time()
    aligner = load(args.align_model)
    print(f"    loaded in {time.time() - t0:.1f}s "
          f"(sample_rate={getattr(aligner, 'sample_rate', '?')})")

    print("==> Aligning ...")
    t0 = time.time()
    result = aligner.generate(sliced, text, language="English")
    align_sec = time.time() - t0
    print(f"    {align_sec:.2f}s for {dur:.1f}s of audio "
          f"({dur / align_sec:.1f}x realtime)\n")

    # ---- report -----------------------------------------------------------
    items = list(result)
    print(f"{'#':>4}  {'start':>8}  {'end':>8}  {'dur':>6}  word")
    print("-" * 52)
    prev_end = None
    backwards = 0
    for i, it in enumerate(items):
        flag = ""
        if prev_end is not None and it.start_time < prev_end - 1e-6:
            flag = "  <-- OVERLAPS PREVIOUS"
            backwards += 1
        prev_end = it.end_time
        print(f"{i:>4}  {it.start_time:>8.2f}  {it.end_time:>8.2f}  "
              f"{it.end_time - it.start_time:>6.2f}  {it.text}{flag}")

    # ---- sanity gates (the same ones the app will apply) ------------------
    src_words = text.split()
    print("\n" + "=" * 52)
    print(f"items            {len(items)}")
    print(f"source words     {len(src_words)}")
    print(f"non-monotonic    {backwards}")
    if items:
        print(f"time span        {items[0].start_time:.2f}s .. "
              f"{items[-1].end_time:.2f}s  (audio is {dur:.1f}s)")
        if items[-1].end_time > dur + 0.5:
            print("!! WARNING: alignment runs past the end of the audio")
    joined = result.text
    fidelity = joined.replace(" ", "").lower() == text.replace(" ", "").lower()
    print(f"text fidelity    {'EXACT' if fidelity else 'DIFFERS (see below)'}")
    if not fidelity:
        print(f"  in : {text}")
        print(f"  out: {joined}")
    print(f"align speed      {dur / align_sec:.1f}x realtime "
          f"({align_sec:.2f}s per {dur:.0f}s chunk)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
