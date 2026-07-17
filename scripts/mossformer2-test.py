#!/usr/bin/env python3
"""
MossFormer2 (librimix-2spk) speech-separation quality-test CLI.
Runs in its own venv (.venv-mossformer2 — pure PyTorch, no transformers).

Splits ONE mixed-audio file into N per-speaker tracks (N = model's trained
speaker count; librimix-2spk = 2). No diarization input, no merge logic —
raw model output only, so the separation quality itself can be judged
(then each track gets transcribed by Whisper in a second pass, see
mossformer2-test.sh, to check whether ASR text comes out clean per speaker).

Usage: .venv-mossformer2/bin/python3 scripts/mossformer2-test.py <audio.wav> <out_dir>
"""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "scripts", "vendor", "mossformer2"))

MODEL_DIR = os.path.join(BASE, "models", "mossformer2", "mossformer2-librimix-2spk")


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: mossformer2-test.py <audio.wav> <out_dir>")
    audio_path, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    import json
    import torch
    import librosa
    import soundfile as sf
    from model.mossformer2 import Mossformer2Wrapper

    with open(os.path.join(MODEL_DIR, "config.json")) as f:
        config = json.load(f)

    print(f"loading {config['config_name']} "
          f"(native sample rate: {config['sample_rate']} Hz) ...")
    model = Mossformer2Wrapper(config)
    model.encoder.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "encoder.ckpt"), map_location=model.device))
    model.decoder.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "decoder.ckpt"), map_location=model.device))
    model.masknet.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "masknet.ckpt"), map_location=model.device))
    model.eval()

    # The model requires its exact training sample rate as input.
    x, sr = sf.read(audio_path)
    if getattr(x, "ndim", 1) > 1:
        x = x.mean(axis=1)
    x = librosa.resample(x.astype("float32"), orig_sr=sr,
                          target_sr=config["sample_rate"])
    mix = torch.from_numpy(x).unsqueeze(0)
    print(f"input: {audio_path} ({len(x) / config['sample_rate']:.1f}s "
          f"@ {sr} Hz -> resampled to {config['sample_rate']} Hz)")

    with torch.no_grad():
        est = model.forward(mix.to(model.device))

    for ns in range(model.num_spks):
        sig = est[0, :, ns]
        sig = sig / sig.abs().max().clamp(min=1e-8)
        out_path = os.path.join(out_dir, f"speaker{ns + 1}.wav")
        sf.write(out_path, sig.cpu().numpy(), config["sample_rate"])
        print(f"wrote {out_path}")

    print("done")


if __name__ == "__main__":
    main()
