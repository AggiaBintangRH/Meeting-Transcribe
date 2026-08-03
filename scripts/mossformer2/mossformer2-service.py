#!/usr/bin/env python3
"""
MossFormer2 overlap-separation sidecar for MeetingTranscriber (persistent).

Overlap attempt #3 (owner-requested 2026-07-14). Uses the STANDALONE
`alibabasglab/MossFormer2` GitHub repo's vendored PyTorch code
(scripts/mossformer2/vendor/mossformer2/) with the `mossformer2-librimix-2spk` checkpoint
(8 kHz, 2-speaker LibriMix). This is NOT the earlier-removed `clearvoice` /
`MossFormer2_SS_16K` attempt — different repo, different weights, different code.

At stop the app asks this sidecar to separate a short window around a detected
overlap into two per-speaker tracks; each track is re-transcribed by the chunked
ASR sidecar and, only if quality gates pass, spliced back into the transcript.
Single-mic separation is unreliable on real audio (see CLAUDE.md); the app's
quality checks are expected to skip many/most windows.

Protocol:
  stdout: JSON lines
          {"type":"status","text":"LOADED"}
          {"type":"result","id":N,"tracks":[
              {"path":"...track1.wav","active":[[s,e],...]},
              {"path":"...track2.wav","active":[[s,e],...]}]}
          {"type":"error","id":N,"text":...}
  stdin (one JSON per line):
          {"cmd":"separate","id":N,"audio":path,"start":s,"end":s,"out_dir":dir}
          {"cmd":"exit"}   (or EOF) exits.

Fully offline: pure PyTorch, no network, no transformers.
"""
import contextlib
import json
import os
import sys
import time
import traceback

# This file lives at scripts/mossformer2/ (one folder per service, owner
# 2026-07-29), so the project root — which owns models/ — is THREE levels up,
# not two. The vendored PyTorch code sits NEXT TO this service, not in a shared
# scripts/vendor: each service owns its own vendored third-party code.
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, "scripts", "mossformer2", "vendor", "mossformer2"))

MODEL_DIR = os.path.join(BASE, "models", "mossformer2", "mossformer2-librimix-2spk")
OUT_SR = 16_000          # tracks written at 16 kHz so downstream ASR gets a normal rate
MIN_WINDOW_SEC = 2.0     # separation needs at least this much audio


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


def active_spans(signal, sr: int, np) -> list:
    """Frame-RMS energy thresholding → list of [start,end] active spans in
    window-relative seconds. Drives per-track speaker attribution downstream."""
    hop = max(1, int(0.05 * sr))          # 50 ms frames
    n = signal.shape[0]
    if n == 0:
        return []
    frames = []
    for i in range(0, n, hop):
        seg = signal[i:i + hop]
        frames.append(float(np.sqrt(np.mean(seg * seg)) if seg.size else 0.0))
    frames = np.asarray(frames, dtype="float32")
    peak = float(frames.max()) if frames.size else 0.0
    if peak <= 1e-6:
        return []
    threshold = 0.15 * peak
    active = frames >= threshold

    spans = []
    run_start = None
    for idx, on in enumerate(active):
        if on and run_start is None:
            run_start = idx
        elif not on and run_start is not None:
            spans.append([run_start * hop / sr, idx * hop / sr])
            run_start = None
    if run_start is not None:
        spans.append([run_start * hop / sr, n / sr])

    # Merge spans separated by < 0.3s, drop spans shorter than 0.2s.
    merged = []
    for s in spans:
        if merged and s[0] - merged[-1][1] < 0.3:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    merged = [[round(s, 3), round(e, 3)] for s, e in merged if e - s >= 0.2]
    return merged


def main() -> None:
    try:
        import numpy as np
        import torch
        import librosa
        import soundfile as sf
    except Exception:  # noqa: BLE001
        fail(f"import failed: {brief_traceback()} — run download-best-models.sh")

    try:
        with open(os.path.join(MODEL_DIR, "config.json")) as f:
            config = json.load(f)
    except Exception:  # noqa: BLE001
        fail(f"config.json load failed ({MODEL_DIR}): {brief_traceback()}")

    native_sr = int(config["sample_rate"])   # 8000 for librimix-2spk
    device = torch.device("cpu")

    log(f"loading {config['config_name']} (native {native_sr} Hz) on {device}")
    try:
        from model.mossformer2 import Mossformer2Wrapper
        # Mossformer2Wrapper.__init__ prints to stdout ("model initialised on ...")
        # which would corrupt the JSON-lines protocol — redirect it to stderr.
        with contextlib.redirect_stdout(sys.stderr):
            model = Mossformer2Wrapper(config)
            model.encoder.load_state_dict(torch.load(
                os.path.join(MODEL_DIR, "encoder.ckpt"), map_location=device))
            model.decoder.load_state_dict(torch.load(
                os.path.join(MODEL_DIR, "decoder.ckpt"), map_location=device))
            model.masknet.load_state_dict(torch.load(
                os.path.join(MODEL_DIR, "masknet.ckpt"), map_location=device))
            model.to(device)
            model.eval()
    except Exception:  # noqa: BLE001
        fail(f"MossFormer2 load failed: {brief_traceback()} — run download-best-models.sh")

    log(f"model ready ({model.num_spks} speakers)")
    emit({"type": "status", "text": "LOADED"})

    def separate(job: dict) -> None:
        req_id = job.get("id")
        audio = job.get("audio", "")
        start = float(job.get("start", 0.0))
        end = float(job.get("end", 0.0))
        out_dir = job.get("out_dir", "")

        if not os.path.exists(audio):
            emit({"type": "error", "id": req_id, "text": f"Audio not found: {audio}"})
            return
        if end - start < MIN_WINDOW_SEC:
            emit({"type": "error", "id": req_id,
                  "text": f"Window too short ({end - start:.2f}s < {MIN_WINDOW_SEC}s)"})
            return
        try:
            os.makedirs(out_dir, exist_ok=True)

            info = sf.info(audio)
            file_sr = info.samplerate
            a = max(0, int(start * file_sr))
            b = min(info.frames, int(end * file_sr))
            x, _ = sf.read(audio, start=a, stop=b, dtype="float32")
            if getattr(x, "ndim", 1) > 1:
                x = x.mean(axis=1)
            if x.size == 0:
                emit({"type": "error", "id": req_id, "text": "Empty window slice"})
                return

            # Resample to the model's fixed native rate (8 kHz).
            x8 = librosa.resample(x.astype("float32"), orig_sr=file_sr, target_sr=native_sr)
            mix = torch.from_numpy(x8).unsqueeze(0)

            started = time.time()
            with torch.no_grad():
                est = model.forward(mix.to(device))   # (1, T, num_spks) @ native_sr

            tracks = []
            for ns in range(model.num_spks):
                sig = est[0, :, ns]
                sig = sig / sig.abs().max().clamp(min=1e-8)     # per-track peak normalize
                out8 = sig.cpu().numpy().astype("float32")
                # Resample each output track back to 16 kHz for downstream ASR.
                out16 = librosa.resample(out8, orig_sr=native_sr, target_sr=OUT_SR)
                out_path = os.path.join(out_dir, f"track{ns + 1}.wav")
                sf.write(out_path, out16, OUT_SR)
                tracks.append({"path": out_path, "active": active_spans(out16, OUT_SR, np)})

            log(f"separated id={req_id} [{start:.1f},{end:.1f}]s in "
                f"{time.time() - started:.1f}s -> {len(tracks)} tracks")
            emit({"type": "result", "id": req_id, "tracks": tracks})
        except Exception:  # noqa: BLE001
            emit({"type": "error", "id": req_id,
                  "text": f"Separation failed: {brief_traceback()}"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
        except Exception:  # noqa: BLE001
            emit({"type": "error", "text": f"Bad job line: {line[:120]}"})
            continue
        cmd = job.get("cmd", "separate")
        if cmd == "exit":
            break
        if cmd == "separate":
            separate(job)
        else:
            emit({"type": "error", "id": job.get("id"), "text": f"Unknown cmd: {cmd}"})


if __name__ == "__main__":
    main()
