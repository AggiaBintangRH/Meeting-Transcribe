#!/usr/bin/env python3
"""
DiCoW v3.3 target-speaker ASR sidecar for MeetingTranscriber (persistent).

Overlap attempt #4 (owner-requested 2026-07-16). DiCoW (BUT-FIT,
diarization-conditioned Whisper) transcribes ONE speaker at a time: it is
conditioned on an STNO mask derived from the diarization turns, so overlaps
are handled inside the ASR instead of by separating the waveform. This is a
re-integration of the engine removed on 2026-07-14 (see CLAUDE.md) — the app
side now gates its output with leakage / duplicate / word-density checks.

Runs in its OWN venv (.venv-dicow), not the main .venv: DiCoW's remote code
needs transformers 4.55 (on 5.x, generate() dies with
`AttributeError: '_get_initial_cache_position'`), which conflicts with the
MLX stack. `pandas` is an undocumented import of that remote code.

Protocol:
  stdout: JSON lines
          {"type":"status","text":"LOADED"}
          {"type":"result","id":N,"targets":[{"sid":3,"text":"..."},...]}
          {"type":"error","id":N,"text":...}
  stdin (one JSON per line):
          {"cmd":"transcribe_targets","id":N,"audio":path,"start":s,"end":s,
           "language":"en"|null,
           "targets":[{"sid":3,"turns":[[s,e],...]},...]}   # absolute seconds
          {"cmd":"exit"}   (or EOF) exits.

Text only — DiCoW's internal timestamps are deliberately never returned (they
were unreliable in the 2026-07-14 attempt, see CLAUDE.md). Windows longer than
30s are rejected so generate() never enters its long-form seek loop, which was
the source of that attempt's runaway-span bug.

Fully offline: HF_HUB_OFFLINE=1, weights read from the project models/ folder.
"""
import json
import os
import sys
import time
import traceback

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Must be set BEFORE transformers is imported. The app normally passes these in
# (PythonRuntime.sidecarEnvironment); the defaults keep a hand-run sidecar
# pointed at the project's models/ folder. Getting HF_HOME wrong makes the
# dynamic-module cache (models/modules/transformers_modules) go stale and throw
# a confusing `FileNotFoundError: .../decoding.py`.
os.environ.setdefault("HF_HOME", os.path.join(BASE, "models"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(BASE, "models", "hub"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

REPO = "BUT-FIT/DiCoW_v3_3_large"
SR = 16_000
MAX_WINDOW_SEC = 30.0     # one Whisper frame — never trigger the seek loop
MIN_WINDOW_SEC = 1.0
FRAMES = 1500             # 30s @ 50 Hz — the STNO mask's fixed time resolution
FRAME_SEC = 0.02
MAX_NEW_TOKENS = 400      # Whisper's context is 448; a 30s frame never needs more

# STNO mask channel order: Silence / Target / Non-target / Overlap.
CH_SILENCE, CH_TARGET, CH_NON_TARGET, CH_OVERLAP = 0, 1, 2, 3


def emit(payload: dict) -> None:
    try:
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()
    except BrokenPipeError:
        sys.exit(0)


def fail(message: str) -> None:
    emit({"type": "error", "text": message})
    sys.exit(1)


def brief_traceback() -> str:
    lines = [l for l in traceback.format_exc().splitlines() if l.strip()]
    return " | ".join(lines[-3:])


def log(message: str) -> None:
    sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {message}\n")
    sys.stderr.flush()


def frame_activity(turns: list, ws: float, we: float, torch) -> "object":
    """Per-frame boolean activity over the 30s mask grid, window-relative.

    `turns` are absolute [start,end] pairs; frames past the window's real
    duration stay False (the processor zero-pads the audio out to 30s).
    """
    active = torch.zeros(FRAMES, dtype=torch.bool)
    span = we - ws
    for t in turns:
        try:
            ts, te = float(t[0]), float(t[1])
        except (TypeError, ValueError, IndexError):
            continue
        s = max(0.0, min(span, ts - ws))
        e = max(0.0, min(span, te - ws))
        if e <= s:
            continue
        i0 = int(s / FRAME_SEC)
        i1 = min(FRAMES, int(round(e / FRAME_SEC)))
        if i1 > i0:
            active[i0:i1] = True
    return active


def stno_mask(target_turns: list, other_turns: list, ws: float, we: float, torch):
    """(1, 4, 1500) one-hot STNO mask for one target speaker.

    T = target alone, O = target + someone else, N = others alone, S = nobody.
    """
    tgt = frame_activity(target_turns, ws, we, torch)
    oth = frame_activity(other_turns, ws, we, torch)
    mask = torch.zeros(1, 4, FRAMES, dtype=torch.float32)
    mask[0, CH_OVERLAP][tgt & oth] = 1.0
    mask[0, CH_TARGET][tgt & ~oth] = 1.0
    mask[0, CH_NON_TARGET][~tgt & oth] = 1.0
    mask[0, CH_SILENCE][~tgt & ~oth] = 1.0
    return mask


def main() -> None:
    try:
        import librosa
        import torch
        import transformers
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
        import pandas  # noqa: F401  — required by DiCoW's remote code, undocumented
    except Exception:  # noqa: BLE001
        fail(f"import failed: {brief_traceback()} — run download-best-models.sh "
             f"(DiCoW needs its own .venv-dicow)")

    if not transformers.__version__.startswith("4.55."):
        fail(f"transformers {transformers.__version__} is not supported — DiCoW "
             f"needs 4.55.x (generate() breaks on 5.x). Run download-best-models.sh.")

    # CPU float32 only: MPS is unverified for DiCoW's custom remote code, and a
    # wrong-device silent fallback would be worse than the slower CPU path.
    device = torch.device("cpu")
    log(f"loading {REPO} on {device} (transformers {transformers.__version__})")
    try:
        processor = AutoProcessor.from_pretrained(REPO, trust_remote_code=True)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            REPO, trust_remote_code=True, torch_dtype=torch.float32)
        model.to(device)
        model.eval()
        # MUST be set by hand: generation.py's _fix_timestamps_from_segmentation
        # reads self.tokenizer, which from_pretrained leaves as None -> crash.
        model.tokenizer = processor.tokenizer
    except Exception:  # noqa: BLE001
        fail(f"DiCoW load failed: {brief_traceback()} — run download-best-models.sh")

    log("model ready")
    emit({"type": "status", "text": "LOADED"})

    def transcribe_targets(job: dict) -> None:
        req_id = job.get("id")
        audio = job.get("audio", "")
        start = float(job.get("start", 0.0))
        end = float(job.get("end", 0.0))
        language = job.get("language") or None
        targets = job.get("targets") or []

        if not os.path.exists(audio):
            emit({"type": "error", "id": req_id, "text": f"Audio not found: {audio}"})
            return
        span = end - start
        if span < MIN_WINDOW_SEC:
            emit({"type": "error", "id": req_id,
                  "text": f"Window too short ({span:.2f}s < {MIN_WINDOW_SEC}s)"})
            return
        if span > MAX_WINDOW_SEC:
            # Hard limit: longer input makes generate() run its long-form seek
            # loop, whose timestamps we do not trust (see the module docstring).
            emit({"type": "error", "id": req_id,
                  "text": f"Window too long ({span:.2f}s > {MAX_WINDOW_SEC}s)"})
            return
        if not targets:
            emit({"type": "error", "id": req_id, "text": "No targets given"})
            return

        try:
            x, _ = librosa.load(audio, sr=SR, mono=True, offset=start, duration=span)
            if x.size == 0:
                emit({"type": "error", "id": req_id, "text": "Empty window slice"})
                return

            inputs = processor(x, sampling_rate=SR, return_tensors="pt",
                               return_attention_mask=True)
            feats = inputs.input_features.to(torch.float32).to(device)
            attn = inputs.get("attention_mask")
            if attn is not None:
                attn = attn.to(device)

            started = time.time()
            results = []
            for entry in targets:
                sid = entry.get("sid")
                own = entry.get("turns") or []
                # "Non-target" is every OTHER requested speaker in this window.
                other = [t for e in targets if e.get("sid") != sid
                         for t in (e.get("turns") or [])]
                mask = stno_mask(own, other, start, end, torch).to(device)

                with torch.no_grad():
                    out = model.generate(
                        input_features=feats,
                        attention_mask=attn,
                        stno_mask=mask,
                        tokenizer=processor.tokenizer,
                        language=language,
                        task="transcribe",
                        max_new_tokens=MAX_NEW_TOKENS,
                    )
                text = processor.batch_decode(out, skip_special_tokens=True)[0]
                results.append({"sid": sid, "text": text.strip()})

            log(f"transcribed id={req_id} [{start:.1f},{end:.1f}]s "
                f"x{len(results)} targets in {time.time() - started:.1f}s")
            emit({"type": "result", "id": req_id, "targets": results})
        except Exception:  # noqa: BLE001
            emit({"type": "error", "id": req_id,
                  "text": f"DiCoW transcription failed: {brief_traceback()}"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
        except Exception:  # noqa: BLE001
            emit({"type": "error", "text": f"Bad job line: {line[:120]}"})
            continue
        cmd = job.get("cmd", "transcribe_targets")
        if cmd == "exit":
            break
        if cmd == "transcribe_targets":
            transcribe_targets(job)
        else:
            emit({"type": "error", "id": job.get("id"), "text": f"Unknown cmd: {cmd}"})


if __name__ == "__main__":
    main()
