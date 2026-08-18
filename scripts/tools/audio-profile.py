#!/usr/bin/env python3
"""Profile a recording on the properties that decide diarization quality.

WHY MEASURE RATHER THAN LISTEN. "The audio sounds fine" and "the diarizer works"
are different claims, and the gap between two machines is usually in numbers a
listener does not hear: a level 20 dB lower gives the VAD less to gate on, a
band-limited feed strips the high formants speaker embeddings lean on, and a
recording that clips has had its peaks replaced by a flat line that looks the
same for everyone who speaks loudly.

Every figure below is chosen because something in this project depends on it:

  RMS / peak        `PARTIAL_SILENCE_RMS` (0.004) gates the realtime lanes, and
                    the ATND position layer collects direction ONLY while our VAD
                    hears speech. A quiet feed silences both.
  clipping          clipped peaks are identical across speakers, so they carry no
                    identity — measured on a real file where sox and soundfile
                    disagreed by 0.103 purely on already-clipped samples.
  speech ratio      Silero's own verdict, the same one the app gates on.
  bandwidth         WeSpeaker and CAM++ embed 16 kHz mono; content above ~7 kHz
                    being absent means the source was band-limited or resampled
                    from something narrower, which costs the embedding.
  DC offset         a non-zero mean biases RMS and every energy gate built on it.

By default only the FIVE most recent files are profiled. A glob over a
recordings folder grows without bound, every file costs a full Silero pass, and
the question being asked is almost always about the last attempt or two — so the
default is the useful answer rather than the complete one. It says what it
skipped; `--all` profiles everything.

Usage:
    scripts/tools/audio-profile.py FILE [FILE ...]
    scripts/tools/audio-profile.py recordings/*.wav          # newest 5
    scripts/tools/audio-profile.py --last 2 recordings/*.wav
    scripts/tools/audio-profile.py --all recordings/*.wav
"""
from __future__ import annotations

import os
import sys

import numpy as np
import soundfile as sf


def profile(path: str) -> dict:
    info = sf.info(path)
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]

    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    rms = float(np.sqrt(np.mean(mono ** 2))) if mono.size else 0.0
    # A sample within one 16-bit step of full scale is at the rail.
    clipped = int(np.sum(np.abs(mono) >= 0.999)) if mono.size else 0

    # Speech share AND SNR, both from Silero's own verdict rather than a new
    # threshold — the same gate the app runs on.
    #
    # ⚠ SNR IS THE NUMBER THAT DECIDES, AND LEVEL IS NOT. Measured on one source
    # played into three captures:
    #
    #     signal -38.8 dB, noise -55.9 dB -> SNR 17.2 ->  9 turns, sane
    #     signal  -9.0 dB, noise -13.2 dB -> SNR  4.2 -> 31 turns, chaos
    #     signal -39.2 dB, noise -41.5 dB -> SNR  2.3 -> 20 turns, chaos
    #
    # The middle one is THIRTY dB louder than the first and far worse. Changing a
    # gain moves signal and noise together, so it cannot fix this — the owner set
    # the peak to -12 dB and nothing improved, exactly as these numbers predict.
    #
    # Why the diarizer cares: WeSpeaker and CAM++ embed whatever is in the speech
    # band. At 17 dB the embedding is dominated by the voice, so five people land
    # in five places. At 2-4 dB every embedding also carries the SAME noise, so
    # they all drag toward one point, the distances between voices collapse, and
    # clustering cuts arbitrarily. That is the 9 -> 31 turn jump: not more people,
    # just unstable boundaries.
    speech = snr = sig_db = noise_db = None
    try:
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad

        x = mono
        if sr != 16000:
            import scipy.signal as ss
            x = ss.resample_poly(x, 16000, sr).astype("float32")
        ts = get_speech_timestamps(torch.from_numpy(x), load_silero_vad(),
                                   sampling_rate=16000)
        speech = sum(t["end"] - t["start"] for t in ts) / 16000
        mask = np.zeros(len(x), dtype=bool)
        for t in ts:
            mask[t["start"]:t["end"]] = True
        rms_of = lambda a: float(np.sqrt(np.mean(a ** 2))) if a.size else 0.0
        db_of = lambda v: 20 * np.log10(v) if v > 0 else float("-inf")
        sig_db, noise_db = db_of(rms_of(x[mask])), db_of(rms_of(x[~mask]))
        # ⚠ NO SNR WHEN THERE IS NO SPEECH, rather than `-inf - -inf = nan`.
        # A digitally silent recording is a NORMAL outcome here — CLAUDE.md
        # records it as "nothing was recorded that session, not a bug" — and
        # printing `nan dB` invites someone to chase a number that means nothing.
        # Absent must read as absent, the same rule the transcript's confidence
        # figures follow.
        snr = (sig_db - noise_db
               if mask.any() and (~mask).any() and sig_db > float("-inf")
                  and noise_db > float("-inf")
               else None)
    except Exception as exc:  # noqa: BLE001 — reported, never fatal
        speech = f"unavailable ({type(exc).__name__})"

    # Where the energy stops. A real wideband voice feed still has content at
    # 6-8 kHz; a narrowband or upsampled one falls off a cliff much earlier.
    band = None
    if mono.size > sr:
        seg = mono[: sr * 60]
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        freqs = np.fft.rfftfreq(len(seg), 1 / sr)
        total = float(np.sum(spec ** 2))
        if total > 0:
            cum = np.cumsum(spec ** 2) / total
            band = float(freqs[int(np.searchsorted(cum, 0.99))])

    return {
        "file": os.path.basename(path),
        "sr": sr,
        "ch": info.channels,
        "subtype": info.subtype,
        "sec": len(mono) / sr if sr else 0,
        "rms": rms,
        "rms_db": 20 * np.log10(rms) if rms > 0 else float("-inf"),
        "peak": peak,
        "clipped": clipped,
        "clipped_pct": 100 * clipped / len(mono) if mono.size else 0,
        "dc": float(np.mean(mono)) if mono.size else 0.0,
        "speech_sec": speech,
        "snr_db": snr,
        "sig_db": sig_db,
        "noise_db": noise_db,
        "band99_hz": band,
    }


DEFAULT_LAST = 5


def main() -> int:
    args = sys.argv[1:]
    limit = DEFAULT_LAST
    files = []
    i = 0
    while i < len(args):
        if args[i] == "--all":
            limit = None
        elif args[i] == "--last" and i + 1 < len(args):
            i += 1
            try:
                limit = int(args[i])
            except ValueError:
                print(f"--last needs a number, got {args[i]!r}", file=sys.stderr)
                return 2
        else:
            files.append(args[i])
        i += 1

    if not files:
        print(__doc__)
        return 2

    # NEWEST FIRST, by modification time — a shell glob arrives alphabetically,
    # which for `meeting-<timestamp>.wav` happens to be chronological and for
    # anything else is arbitrary. Sorting by mtime is right for both.
    files = [f for f in files if os.path.isfile(f)]
    files.sort(key=os.path.getmtime, reverse=True)
    skipped = 0
    if limit is not None and len(files) > limit:
        skipped = len(files) - limit
        files = files[:limit]

    rows = [profile(p) for p in files]
    for r in rows:
        print(f"\n=== {r['file']}")
        print(f"  format        {r['sr']} Hz, {r['ch']} ch, {r['subtype']}, "
              f"{r['sec']:.1f} s")
        print(f"  level         RMS {r['rms']:.5f} ({r['rms_db']:.1f} dBFS)   "
              f"peak {r['peak']:.4f}")
        print(f"  clipping      {r['clipped']} samples ({r['clipped_pct']:.3f}%)")
        print(f"  DC offset     {r['dc']:+.6f}")
        if r["snr_db"] is not None:
            print(f"  SNR           {r['snr_db']:.1f} dB    "
                  f"(speech {r['sig_db']:.1f} dB, background {r['noise_db']:.1f} dB)")
        elif isinstance(r["speech_sec"], float):
            print("  SNR           n/a — no speech, or no silence to compare it against")
        if isinstance(r["speech_sec"], float):
            pct = 100 * r["speech_sec"] / r["sec"] if r["sec"] else 0
            print(f"  speech        {r['speech_sec']:.1f} s of {r['sec']:.1f} s "
                  f"({pct:.1f}%)  [Silero, the app's own gate]")
        else:
            print(f"  speech        {r['speech_sec']}")
        print(f"  99% energy    below {r['band99_hz']:.0f} Hz"
              if r["band99_hz"] else "  99% energy    (too short to measure)")
        # The one line that reads as a verdict, and it only ever flags what is
        # measured against a threshold this project already uses.
        flags = []
        # THE VERDICT LINE. Ranked first because it is the one that predicted the
        # outcome on all three measured captures, while level predicted nothing.
        if r["snr_db"] is not None:
            if r["snr_db"] < 10:
                flags.append(f"SNR {r['snr_db']:.1f} dB — speaker embeddings will "
                             "carry more background than voice, and diarization "
                             "will split people arbitrarily. Above 15 dB is the "
                             "target. Raising the gain will NOT help: it moves "
                             "signal and background together.")
            elif r["snr_db"] < 15:
                flags.append(f"SNR {r['snr_db']:.1f} dB — below the 15 dB that "
                             "measured well here; diarization may be unstable.")
        if r["rms"] < 0.004:
            flags.append("BELOW the realtime silence gate (0.004) — the VAD and "
                         "the ATND beam collector will see this as silence")
        elif r["rms"] < 0.01:
            flags.append("quiet; the silence gate is close")
        if r["clipped_pct"] > 0.01:
            flags.append("clipped peaks carry no speaker identity")
        # ⚠ 4000 Hz WAS THE FIRST THRESHOLD AND IT WAS WRONG. It fired on
        # `Meeting5People.wav` — the fixture every engine answers CORRECTLY on —
        # because 99 % of speech energy genuinely sits low: 2481 Hz there, 2580 Hz
        # on a real meeting, 5080 Hz on a loud close-mic one. A flag that fires on
        # known-good audio is worse than no flag, so this now marks only the
        # genuinely muffled. Measured range above; judge the NUMBER, not the flag.
        if r["band99_hz"] and r["band99_hz"] < 1500:
            flags.append("very narrowband — well below the 2.5-5 kHz seen on "
                         "recordings this project diarizes correctly")
        if abs(r["dc"]) > 0.001:
            flags.append("DC offset biases every energy gate")
        for f in flags:
            print(f"  ⚠ {f}")
    if skipped:
        print(f"\n({skipped} older file(s) not profiled — newest {len(files)} shown. "
              f"Use --all, or --last N.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
