#!/usr/bin/env python3
"""
Integration test suite for the three Python sidecars.

    .venv/bin/python3 scripts/sidecar-tests.py            # everything
    .venv/bin/python3 scripts/sidecar-tests.py --list
    .venv/bin/python3 scripts/sidecar-tests.py --only diarize
    .venv/bin/python3 scripts/sidecar-tests.py --only nemotron/lane-isolation

WHY THIS EXISTS
---------------
`swift test` covers pure Swift logic and touches no sidecar. The sidecars are
the part that actually changed the most (dual-stream lanes, the additive `-2`
frame, forced alignment, two profile stores, a partial-window cap, a silence
gate) and each of those was proven once by a throwaway harness that was then
deleted. This file is those proofs, kept.

Every check below corresponds to a REGRESSION THAT WAS REAL — not to a line of
code someone felt like covering. The docstring on each check says which one.

NOT part of the app; nothing in MeetingTranscriber imports it. Model loads take
3-30 s and need the downloaded weights, so this is a deliberate, developer-run
tool rather than something that can join `swift test`. Use --only to narrow a
run; a full run is minutes.

SAFETY
------
The diarization sidecar persists speaker profiles to models/speaker-profiles/.
Those are the OWNER'S real voices and a test that corrupts them would be worse
than no test at all, so:
  * every diarization subprocess gets MT_PROFILE_DIR pointed at a temp dir,
  * the four real profile files are SHA-256'd before and after the whole run
    and the suite fails loudly if a single byte moved,
  * nothing is ever written into recordings/ — fixtures are read-only inputs and
    all scratch WAVs live in a temp dir that is removed at exit.

AUDIO FIXTURES
--------------
Two checks (lane isolation, profile-store separation) need real, clearly
DIFFERENT speech; synthetic tones would make them vacuous. Audio is never
committed (recordings/ is gitignored and the repo is already big), so paths are
arguments that default to whatever is found in recordings/. With no usable
audio those two checks SKIP with a loud message — a skip is never a pass.
"""

import argparse
import hashlib
import json
import os
import pathlib
import queue
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import wave

PROJECT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS = PROJECT / "scripts"
VENV_PY = PROJECT / ".venv" / "bin" / "python3"
REAL_PROFILE_DIR = PROJECT / "models" / "speaker-profiles"
RECORDINGS = PROJECT / "recordings"

# Pin the model cache the same way every sidecar does, so the in-process
# white-box checks (which import mlx_audio directly) resolve offline too.
os.environ.setdefault("HF_HOME", str(PROJECT / "models"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SR = 16_000

DEFAULT_CHUNKED_MODEL = "mlx-community/Qwen3-ASR-1.7B-bf16"
DEFAULT_ALIGN_MODEL = "mlx-community/Qwen3-ForcedAligner-0.6B-bf16"

# Mirrors of the sidecars' own constants. Deliberately RE-DECLARED rather than
# imported: if someone changes PARTIAL_WINDOW in the sidecar the check should
# force a conscious decision here, not silently follow along.
PARTIAL_WINDOW_SEC = 10.0
REMOTE_ID_BASE = 10_000


# ----------------------------------------------------------------- reporting
PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


class Report:
    """Collects one line per check and decides the exit status."""

    def __init__(self):
        self.rows = []
        self.started = time.time()

    def add(self, status: str, check_id: str, detail: str = "") -> None:
        icon = {PASS: "ok  ", FAIL: "FAIL", SKIP: "skip"}[status]
        elapsed = time.time() - self.started
        line = f"[{elapsed:6.1f}s] {icon}  {check_id}"
        if detail:
            line += f"  —  {detail}"
        print(line, flush=True)
        self.rows.append((status, check_id, detail))

    def ok(self, check_id, detail=""):
        self.add(PASS, check_id, detail)

    def fail(self, check_id, detail=""):
        self.add(FAIL, check_id, detail)

    def skip(self, check_id, detail=""):
        self.add(SKIP, check_id, detail)

    def expect(self, check_id, condition: bool, detail_ok="", detail_bad="") -> bool:
        """Record one check from a boolean. Returns the boolean for chaining."""
        if condition:
            self.ok(check_id, detail_ok)
        else:
            self.fail(check_id, detail_bad or detail_ok)
        return condition

    def summarize(self) -> int:
        n_pass = sum(1 for s, _, _ in self.rows if s == PASS)
        n_fail = sum(1 for s, _, _ in self.rows if s == FAIL)
        n_skip = sum(1 for s, _, _ in self.rows if s == SKIP)
        print()
        print("=" * 72)
        print(f"{n_pass} passed, {n_fail} failed, {n_skip} skipped "
              f"in {time.time() - self.started:.1f}s")
        if n_skip:
            print("\nSKIPPED (these proved NOTHING — do not read them as passes):")
            for status, cid, detail in self.rows:
                if status == SKIP:
                    print(f"  - {cid}: {detail}")
        if n_fail:
            print("\nFAILED:")
            for status, cid, detail in self.rows:
                if status == FAIL:
                    print(f"  - {cid}: {detail}")
        print("=" * 72)
        return 1 if n_fail else 0


# ------------------------------------------------------------------- helpers
class Sidecar:
    """A running sidecar process: framed stdin, JSON-lines stdout, tee'd stderr.

    stdout and stderr are drained by threads. Without that, a sidecar that logs
    a lot (they all do) would block on a full stderr pipe while we sit waiting
    for a stdout line — a deadlock that looks exactly like a hung model.
    """

    def __init__(self, script: str, args=None, env_extra=None, text_stdin=False):
        env = dict(os.environ)
        env.update(env_extra or {})
        self.name = script
        self.proc = subprocess.Popen(
            [str(VENV_PY), str(SCRIPTS / script), *(args or [])],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, cwd=str(PROJECT),
        )
        self.text_stdin = text_stdin
        self.messages = queue.Queue()
        self.raw_lines = []      # every stdout line verbatim (for byte checks)
        self.stderr_lines = []
        self._t_out = threading.Thread(target=self._pump_stdout, daemon=True)
        self._t_err = threading.Thread(target=self._pump_stderr, daemon=True)
        self._t_out.start()
        self._t_err.start()

    def _pump_stdout(self):
        for raw in self.proc.stdout:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            self.raw_lines.append(line)
            try:
                self.messages.put(("json", json.loads(line), line))
            except json.JSONDecodeError:
                self.messages.put(("garbage", None, line))

    def _pump_stderr(self):
        for raw in self.proc.stderr:
            self.stderr_lines.append(raw.decode("utf-8", "replace").rstrip("\n"))

    # -- stdin -------------------------------------------------------------
    def write(self, data: bytes) -> None:
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    def send_json(self, obj: dict) -> None:
        self.write((json.dumps(obj) + "\n").encode("utf-8"))

    # -- stdout ------------------------------------------------------------
    def next_message(self, timeout=120.0):
        """Next parsed stdout message, or None on timeout/EOF."""
        try:
            kind, payload, line = self.messages.get(timeout=timeout)
        except queue.Empty:
            return None
        if kind == "garbage":
            raise AssertionError(f"{self.name} wrote non-JSON to stdout: {line!r}")
        return payload

    def wait_for(self, predicate, timeout=120.0):
        """Drain until a message satisfies `predicate` (or time runs out)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self.next_message(timeout=max(0.1, deadline - time.time()))
            if msg is None:
                return None
            if predicate(msg):
                return msg
        return None

    def drain(self, settle=0.4):
        """Collect whatever has arrived, then stop. Used between phases."""
        time.sleep(settle)
        out = []
        while True:
            try:
                _, payload, _ = self.messages.get_nowait()
            except queue.Empty:
                return out
            out.append(payload)

    def close(self):
        try:
            self.write(struct.pack("<i", -1) if not self.text_stdin else b"")
        except Exception:  # noqa: BLE001 — already dead is fine
            pass
        try:
            self.proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()


# -- framed stdin encoders (mirror of the sidecars' documented protocol) -----
def frame_office(samples) -> bytes:
    return struct.pack("<i", samples.size) + samples.astype("float32").tobytes()


def frame_remote(samples) -> bytes:
    return (struct.pack("<i", -2) + struct.pack("<i", samples.size)
            + samples.astype("float32").tobytes())


FLUSH_OFFICE = struct.pack("<i", 0)
FLUSH_REMOTE = struct.pack("<i", -3)


def sha256(path: pathlib.Path):
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile_hashes():
    """Fingerprint the OWNER'S real profile files (see SAFETY in the docstring)."""
    names = ["profiles.json", "embeddings.npz",
             "profiles-remote.json", "embeddings-remote.npz"]
    return {n: sha256(REAL_PROFILE_DIR / n) for n in names}


def write_wav(path, samples, sr=SR):
    """16-bit mono WAV — the format every sidecar's file path expects."""
    import numpy as np
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())
    return str(path)


def tone(seconds: float, base_hz=180.0, seed=0):
    """Synthetic voiced-ish audio: loud enough to clear the realtime silence
    gate, so checks about TIMING/SHAPE don't need a real recording."""
    import numpy as np
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    sig = (0.25 * np.sin(2 * np.pi * base_hz * t)
           + 0.12 * np.sin(2 * np.pi * base_hz * 2.5 * t)
           + 0.03 * rng.standard_normal(t.size))
    # Slow amplitude wobble so it is not a perfectly stationary drone.
    return (sig * (0.7 + 0.3 * np.sin(2 * np.pi * 2.0 * t))).astype("float32")


def near_silence(seconds: float, seed=1):
    """Below PARTIAL_SILENCE_RMS (0.004) — an idle conferencing channel."""
    import numpy as np
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(seconds * SR)) * 0.0004).astype("float32")


def load_clip(path, start=0.0, seconds=None):
    """Read a fixture as float32 16 kHz mono, same loader the aligner probe uses."""
    import numpy as np
    from mlx_audio.stt.utils import load_audio
    audio = np.asarray(load_audio(str(path), sr=SR), dtype=np.float32)
    a = int(start * SR)
    b = audio.size if seconds is None else min(audio.size, a + int(seconds * SR))
    return audio[a:b]


def find_fixtures():
    """Pick default real-speech fixtures out of recordings/, biggest first.

    Returns [] when nothing usable is there — the caller SKIPS rather than
    quietly substituting a tone, which would make the check meaningless.
    """
    if not RECORDINGS.is_dir():
        return []
    import numpy as np
    candidates = sorted((p for p in RECORDINGS.glob("*.wav")),
                        key=lambda p: p.stat().st_size, reverse=True)
    usable = []
    seen_meetings = set()
    for path in candidates[:12]:
        # One file per meeting: `foo.wav` and `foo-remote.wav` are two channels
        # of the SAME conversation and often carry the same words, which would
        # make the lane-isolation comparison vacuous.
        meeting = path.stem[:-len("-remote")] if path.stem.endswith("-remote") else path.stem
        if meeting in seen_meetings:
            continue
        try:
            clip = load_clip(path, 0.0, 30.0)
        except Exception:  # noqa: BLE001 — half-written recordings exist
            continue
        if clip.size < 20 * SR:
            continue
        if float(np.sqrt(np.mean(clip * clip))) < 0.01:  # no real speech in it
            continue
        usable.append(path)
        seen_meetings.add(meeting)
        if len(usable) >= 4:
            break
    return usable


def extract_nested(module, source: str, outer: str, inner: str, inject: dict):
    """Compile ONE nested function out of a sidecar, verbatim, for white-box use.

    Some sidecar logic is unreachable through the wire protocol — the chunked
    sidecar aligns whatever ITS OWN ASR produced, so there is no way to hand it
    a chosen sentence or a hallucinated timestamp from stdin. Rather than modify
    the sidecar to add a test hook (explicitly out of bounds), we lift the real
    function out of the real file by AST and run it against injected stubs.

    This still tests the shipped code: delete the `src` mapping in the sidecar
    and the check that uses this fails, which is the whole point.
    """
    import ast
    tree = ast.parse(source)
    outer_fn = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == outer)
    inner_fn = next(n for n in ast.walk(outer_fn)
                    if isinstance(n, ast.FunctionDef) and n.name == inner)
    namespace = dict(vars(module))
    namespace.update(inject)
    exec(compile(ast.Module(body=[inner_fn], type_ignores=[]),
                 f"<{outer}.{inner}>", "exec"), namespace)
    return namespace[inner]


def load_sidecar_module(filename: str, mod_name: str):
    """Import a sidecar for its module-level constants WITHOUT running main()."""
    import importlib.util
    path = SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # __name__ != "__main__", so main() is inert
    return module, path.read_text()


# ============================================================ nemotron group
NEMOTRON_CHECKS = [
    "nemotron/single-stream-bytes",
    "nemotron/lane-isolation",
    "nemotron/partial-window-cap",
    "nemotron/silence-gate",
]


def run_nemotron(rep: Report, ctx):
    # -- check 1: single-stream output is byte-identical to the pre-dual-stream
    #    contract. Its own process, fed ONLY office frames, so the assertion
    #    covers every byte the sidecar produced rather than a filtered subset.
    if ctx.wants("nemotron/single-stream-bytes"):
        sc = Sidecar("nemotron-asr-service.py", ["--language", "en-US"])
        try:
            ready = sc.wait_for(lambda m: m.get("type") in ("status", "error"),
                                timeout=ctx.load_timeout)
            if ready is None or ready.get("type") == "error":
                rep.fail("nemotron/single-stream-bytes",
                         f"sidecar never became READY: {ready}")
            else:
                audio = tone(4.0)
                for i in range(0, audio.size, SR // 2):
                    sc.write(frame_office(audio[i:i + SR // 2]))
                sc.write(FLUSH_OFFICE)
                sc.wait_for(lambda m: m.get("type") == "final", timeout=120)
                sc.drain()

                bad = []
                for line in sc.raw_lines:
                    msg = json.loads(line)
                    if set(msg) != {"type", "text"}:
                        bad.append(f"key set {sorted(msg)}")
                    elif json.dumps(msg) + "" != line:
                        # Re-serializing must reproduce the line exactly: that
                        # covers key ORDER too, which a set comparison misses.
                        bad.append(f"byte drift {line!r}")
                saw_final = any(json.loads(l).get("type") == "final"
                                for l in sc.raw_lines)
                rep.expect("nemotron/single-stream-bytes",
                           not bad and saw_final,
                           f'{len(sc.raw_lines)} lines, all exactly '
                           f'{{"type","text"}}, no "stream" key',
                           f"offending lines: {bad[:3]} (final seen: {saw_final})")
        finally:
            sc.close()

    remaining = [c for c in NEMOTRON_CHECKS[1:] if ctx.wants(c)]
    if not remaining:
        return

    sc = Sidecar("nemotron-asr-service.py", ["--language", "en-US"])
    try:
        ready = sc.wait_for(lambda m: m.get("type") in ("status", "error"),
                            timeout=ctx.load_timeout)
        if ready is None or ready.get("type") == "error":
            for cid in remaining:
                rep.fail(cid, f"sidecar never became READY: {ready}")
            return

        # -- check 2: the two lanes never mix. Two DIFFERENT real recordings go
        #    in INTERLEAVED, frame by frame; each lane's final must match that
        #    same audio's solo baseline. Feeding them sequentially would pass
        #    even with a single shared buffer, which is exactly the bug.
        if ctx.wants("nemotron/lane-isolation"):
            cid = "nemotron/lane-isolation"
            if not (ctx.audio_a and ctx.audio_b):
                rep.skip(cid, "needs two real speech recordings (--audio-a/--audio-b); "
                              "none found in recordings/ — LANE MIXING IS UNTESTED")
            else:
                clip_a = load_clip(ctx.audio_a, ctx.clip_start, ctx.clip_sec)
                clip_b = load_clip(ctx.audio_b, ctx.clip_start, ctx.clip_sec)
                step = SR // 2

                def solo(frame_fn, flush, clip, lane):
                    """Baseline: this clip alone in this lane. Returns None when
                    the lane never produced a final — that is a BROKEN LANE
                    (e.g. its audio was appended to the other one), which must
                    fail rather than skip as a fixture problem."""
                    for i in range(0, clip.size, step):
                        sc.write(frame_fn(clip[i:i + step]))
                    sc.write(flush)
                    msg = sc.wait_for(lambda m: m.get("type") == "final",
                                      timeout=ctx.final_timeout)
                    sc.drain()
                    if msg is None:
                        return None
                    if (msg.get("stream") or "office") != lane:
                        # A final on the wrong lane means the lanes are crossed.
                        return None
                    return msg.get("text", "")

                base_a = solo(frame_office, FLUSH_OFFICE, clip_a, "office")
                base_b = solo(frame_remote, FLUSH_REMOTE, clip_b, "remote")

                if base_a is None or base_b is None:
                    rep.fail(cid, "a lane fed on its own produced no final of its "
                                  f"own within {ctx.final_timeout:.0f}s "
                                  f"(office={base_a!r}, remote={base_b!r}) — the "
                                  "lanes are not independent")
                elif not base_a.strip() or not base_b.strip():
                    rep.skip(cid, "a baseline transcript came back empty — fixture "
                                  "has no speech in the sliced window")
                elif base_a.strip() == base_b.strip():
                    rep.skip(cid, "the two fixtures transcribe identically, so the "
                                  "check could not tell mixed lanes apart")
                else:
                    for i in range(0, max(clip_a.size, clip_b.size), step):
                        if i < clip_a.size:
                            sc.write(frame_office(clip_a[i:i + step]))
                        if i < clip_b.size:
                            sc.write(frame_remote(clip_b[i:i + step]))
                    sc.write(FLUSH_OFFICE)
                    sc.write(FLUSH_REMOTE)
                    finals = {}
                    deadline = time.time() + ctx.final_timeout
                    while len(finals) < 2 and time.time() < deadline:
                        msg = sc.next_message(timeout=max(1, deadline - time.time()))
                        if msg is None:
                            break
                        if msg.get("type") == "final":
                            finals[msg.get("stream") or "office"] = msg.get("text", "")
                    got_o, got_r = finals.get("office"), finals.get("remote")
                    if got_o is None or got_r is None:
                        rep.fail(cid, f"only got finals for {sorted(finals)} — a lane "
                                      "went silent while both were being fed")
                    else:
                        ok = (got_o.strip() == base_a.strip()
                              and got_r.strip() == base_b.strip())
                        rep.expect(cid, ok,
                                   "interleaved finals matched both solo baselines",
                                   f"office {got_o[:60]!r} vs baseline {base_a[:60]!r}; "
                                   f"remote {got_r[:60]!r} vs baseline {base_b[:60]!r}")
                sc.drain()

        # -- check 3: a partial transcribes at most PARTIAL_WINDOW no matter how
        #    long the buffer is. Without the cap, cost grew quadratically across
        #    an utterance (measured 1.5 s per partial at a 20 s buffer, against a
        #    1.5 s cadence) and the realtime loop fell behind realtime.
        #    Observed through the sidecar's own `buf=.. sent=..` stderr line.
        if ctx.wants("nemotron/partial-window-cap"):
            cid = "nemotron/partial-window-cap"
            mark = len(sc.stderr_lines)
            audio = tone(16.0, base_hz=200.0, seed=3)
            for i in range(0, audio.size, SR // 2):
                sc.write(frame_office(audio[i:i + SR // 2]))
            sc.write(FLUSH_OFFICE)
            sc.wait_for(lambda m: m.get("type") == "final", timeout=300)
            time.sleep(0.5)

            import re
            pattern = re.compile(
                r"(office|remote) (partial|final) buf=([\d.]+)s sent=([\d.]+)s")
            partials = []
            for line in sc.stderr_lines[mark:]:
                m = pattern.search(line)
                if m and m.group(2) == "partial":
                    partials.append((float(m.group(3)), float(m.group(4))))
            over = [p for p in partials if p[1] > PARTIAL_WINDOW_SEC + 0.05]
            long_buf = [p for p in partials if p[0] > PARTIAL_WINDOW_SEC + 0.5]
            if not partials:
                rep.fail(cid, "no partials were logged at all — cannot judge the cap")
            elif not long_buf:
                rep.fail(cid, f"buffer never grew past {PARTIAL_WINDOW_SEC}s "
                              f"(max {max(p[0] for p in partials):.1f}s), so the cap "
                              "was never exercised")
            else:
                rep.expect(cid, not over,
                           f"{len(partials)} partials, buffer reached "
                           f"{max(p[0] for p in partials):.1f}s, sent never exceeded "
                           f"{max(p[1] for p in partials):.1f}s",
                           f"{len(over)} partial(s) transcribed more than "
                           f"{PARTIAL_WINDOW_SEC}s: {over[:3]}")
            sc.drain()

        # -- check 4: a near-silent lane spends no compute. The owner's log shows
        #    an idle conferencing channel at rms 0.00000 for 47 minutes; before
        #    the gate that was a generate() every 1.5 s to transcribe nothing.
        #    FLUSH is deliberately NOT gated, so the final must still come back.
        if ctx.wants("nemotron/silence-gate"):
            cid = "nemotron/silence-gate"
            quiet = near_silence(9.0)
            for i in range(0, quiet.size, SR // 2):
                sc.write(frame_remote(quiet[i:i + SR // 2]))
            time.sleep(1.0)  # give any (wrongly) triggered partial time to appear
            early = sc.drain(settle=0.3)
            stray = [m for m in early if m.get("type") == "partial"]

            sc.write(FLUSH_REMOTE)
            final = sc.wait_for(lambda m: m.get("type") == "final", timeout=180)
            if final is None:
                rep.fail(cid, "FLUSH over a silent lane produced no final at all "
                              "(the gate must not swallow finals)")
            elif stray:
                rep.fail(cid, f"{len(stray)} partial(s) emitted for a silent lane: "
                              f"{[m.get('text') for m in stray][:3]}")
            elif final.get("stream") != "remote":
                rep.fail(cid, f"final came back on the wrong lane: {final}")
            else:
                rep.expect(cid, final.get("text", "").strip() == "",
                           "no partials for 9 s of silence; FLUSH returned "
                           "an empty final",
                           f"final text was not empty: {final.get('text')!r}")
            sc.drain()
    finally:
        sc.close()


# ============================================================= chunked group
CHUNKED_CHECKS = [
    "chunked/final-shape-without-aligner",
    "chunked/src-indices-skip-punctuation",
    "chunked/words-never-past-chunk",
]


def run_chunked(rep: Report, ctx):
    # -- check 5: with no --align-model, a final carries EXACTLY type and text.
    #    This is what keeps the single-stream wire format byte-identical; a
    #    stray "words"/"dur" here would be a silent protocol change.
    if ctx.wants("chunked/final-shape-without-aligner"):
        cid = "chunked/final-shape-without-aligner"
        sc = Sidecar("chunked-asr-service.py",
                     ["--model", ctx.chunked_model, "--language", "en"])
        try:
            ready = sc.wait_for(lambda m: m.get("type") in ("status", "error"),
                                timeout=ctx.load_timeout)
            if ready is None or ready.get("type") == "error":
                rep.fail(cid, f"sidecar never reported LOADED: {ready}")
            else:
                clip = (load_clip(ctx.audio_a, ctx.clip_start, 5.0)
                        if ctx.audio_a else tone(5.0))
                sc.write(frame_office(clip))
                sc.write(FLUSH_OFFICE)
                final = sc.wait_for(lambda m: m.get("type") == "final", timeout=300)
                if final is None:
                    rep.fail(cid, "no final came back")
                else:
                    rep.expect(cid, set(final) == {"type", "text"},
                               'final carried exactly {"type","text"}',
                               f"final key set was {sorted(final)} "
                               "(expected no words/dur without --align-model)")
        finally:
            sc.close()

    # The next two exercise code the wire protocol cannot reach: the chunked
    # sidecar only ever aligns text ITS OWN ASR produced, so neither a chosen
    # sentence nor a hallucinated timestamp can be injected from stdin. They run
    # the real functions lifted out of the real file — see extract_nested().
    needs_whitebox = [c for c in CHUNKED_CHECKS[1:] if ctx.wants(c)]
    if not needs_whitebox:
        return

    module, source = load_sidecar_module("chunked-asr-service.py", "mt_chunked_asr")

    # -- check 6: "src" indices point into the ORIGINAL text.split(), skipping
    #    punctuation-only source words. The aligner's tokenizer drops a
    #    standalone em-dash, so a naive item->word index shifts every later word
    #    one speaker to the left. That is the exact bug "src" exists to prevent.
    if ctx.wants("chunked/src-indices-skip-punctuation"):
        cid = "chunked/src-indices-skip-punctuation"
        try:
            from mlx_audio.stt.models.qwen3_asr.qwen3_forced_aligner import (
                ForceAlignProcessor,
            )
            proc = ForceAlignProcessor()
        except Exception as exc:  # noqa: BLE001
            rep.skip(cid, f"aligner processor unavailable ({exc})")
            proc = None

        if proc is not None:
            logs = []
            pair = extract_nested(module, source, "main", "pair_source_indices",
                                  {"align_proc": proc, "log": logs.append})
            text = "we shipped it — and then we tested it"
            words = text.split()
            assert words[3] == "—", "fixture must contain a standalone em-dash"

            class Item:  # minimal stand-in for the aligner's result items
                def __init__(self, t):
                    self.text = t

            tokens, _ = proc.encode_timestamp(text, "English")
            indices = pair(text, [Item(t) for t in tokens])

            if indices is None:
                rep.fail(cid, f"pairing returned None for {text!r}: {logs}")
            else:
                naive = list(range(len(indices)))
                mapped = [words[i] for i in indices]
                expected = [w for w in words if proc.clean_token(w)]
                ok = (mapped == expected and indices != naive
                      and 3 not in indices)
                rep.expect(cid, ok,
                           f"em-dash at word 3 consumes no item; indices "
                           f"{indices} skip it (naive would be {naive})",
                           f"indices {indices} map to {mapped}, expected {expected}")

    # -- check 7: alignment items are never allowed to run past the chunk.
    #    Forced alignment force-fits whatever text it is handed, so a real
    #    Whisper hallucination got stamped after the end of the audio: a few
    #    stragglers are dropped, but a bulk overrun rejects the whole alignment
    #    rather than emitting a truncated guess.
    if ctx.wants("chunked/words-never-past-chunk"):
        cid = "chunked/words-never-past-chunk"
        import numpy as np

        class Item:
            def __init__(self, text, start, end):
                self.text, self.start_time, self.end_time = text, start, end

        def make_align(items):
            """align_chunk with a fake aligner and an identity src mapping, so the
            only thing under test is the past-the-end gate itself."""
            fake = type("FakeAligner", (), {"generate": lambda self, a, t, language=None: items})()
            logs = []
            fn = extract_nested(
                module, source, "main", "align_chunk",
                {"aligner": fake,
                 "pair_source_indices": lambda text, its: list(range(len(its))),
                 "log": logs.append},
            )
            return fn, logs

        audio = np.zeros(int(10.0 * SR), dtype=np.float32)  # a 10 s chunk

        # (a) one straggler out of twenty: dropped, the rest survive.
        good = [Item(f"w{i}", i * 0.4, i * 0.4 + 0.3) for i in range(19)]
        good.append(Item("hallucination", 10.9, 11.4))
        fn, logs_a = make_align(good)
        kept = fn(audio, " ".join(it.text for it in good))

        # (b) a bulk overrun: no timestamps at all rather than a truncated guess.
        bad = [Item(f"w{i}", i * 0.4, i * 0.4 + 0.3) for i in range(10)]
        bad += [Item(f"h{i}", 11.0 + i, 11.5 + i) for i in range(6)]
        fn, logs_b = make_align(bad)
        rejected = fn(audio, " ".join(it.text for it in bad))

        problems = []
        if kept is None:
            problems.append(f"a single straggler wrongly rejected the chunk: {logs_a}")
        else:
            if len(kept) != 19:
                problems.append(f"kept {len(kept)} words, expected 19")
            past = [w for w in kept if w["end"] > 10.0 + module.ALIGN_END_TOLERANCE_SEC]
            if past:
                problems.append(f"kept {len(past)} word(s) past the chunk: {past[:2]}")
        if rejected is not None:
            problems.append(f"bulk overrun was NOT rejected ({len(rejected)} words)")

        rep.expect(cid, not problems,
                   "1/20 past-end item dropped, 6/16 overrun rejected wholesale",
                   "; ".join(problems))


# ============================================================= diarize group
DIARIZE_CHECKS = [
    "diarize/absent-stream-means-office",
    "diarize/profile-stores-are-disjoint",
    "diarize/native-rate-final-matches-16k-chunks",
    "diarize/reset-wipes-both-stores",
]


def run_diarize(rep: Report, ctx):
    wanted = [c for c in DIARIZE_CHECKS if ctx.wants(c)]
    if not wanted:
        return

    # EVERY diarization run is pointed at a throwaway profile dir. The real one
    # holds the owner's voices; see SAFETY in the module docstring.
    profile_dir = pathlib.Path(tempfile.mkdtemp(prefix="mt-profiles-", dir=ctx.tmp))
    audio_path = None
    if ctx.audio_diarize:
        clip = load_clip(ctx.audio_diarize, ctx.clip_start, ctx.diarize_sec)
        audio_path = write_wav(pathlib.Path(ctx.tmp) / "diarize-fixture.wav", clip)

    if audio_path is None:
        for cid in wanted:
            rep.skip(cid, "needs a real speech recording (--audio-diarize); none "
                          "found in recordings/ — PROFILE-STORE SEPARATION IS UNTESTED")
        return

    sc = Sidecar("diarize-service.py",
                 env_extra={"MT_PROFILE_DIR": str(profile_dir)}, text_stdin=True)
    try:
        ready = sc.wait_for(lambda m: m.get("type") in ("status", "error"),
                            timeout=ctx.load_timeout)
        if ready is None or ready.get("type") == "error":
            for cid in wanted:
                rep.fail(cid, f"sidecar never reported LOADED: {ready}")
            return

        # Start from a known-empty store so profile ids are predictable.
        sc.send_json({"cmd": "reset"})
        sc.wait_for(lambda m: m.get("text") == "RESET", timeout=60)

        # -- check 8: a job with no "stream" key is an office job and its reply
        #    has the pre-dual-stream shape — no "stream" echoed back.
        sc.send_json({"cmd": "final", "audio": audio_path})
        office = sc.wait_for(lambda m: m.get("type") in ("result", "error"),
                             timeout=ctx.job_timeout)
        if ctx.wants("diarize/absent-stream-means-office"):
            cid = "diarize/absent-stream-means-office"
            if office is None or office.get("type") == "error":
                rep.fail(cid, f"office job failed: {office}")
            else:
                rep.expect(cid, set(office) == {"type", "segments"},
                           f'reply was exactly {{"type","segments"}} with '
                           f'{len(office["segments"])} turns, no "stream" key',
                           f"reply key set was {sorted(office)}")

        # -- check 9: the two stores cannot cross-contaminate. The SAME audio is
        #    sent again as a remote job: if the stores were shared it would match
        #    the office profiles and create nothing at all. It must instead build
        #    its own R-named identities in its own file.
        if ctx.wants("diarize/profile-stores-are-disjoint"):
            cid = "diarize/profile-stores-are-disjoint"
            sc.send_json({"cmd": "final", "audio": audio_path, "stream": "remote"})
            remote = sc.wait_for(lambda m: m.get("type") in ("result", "error"),
                                 timeout=ctx.job_timeout)
            if office is None or office.get("type") == "error":
                rep.fail(cid, f"office job failed, cannot compare: {office}")
            elif remote is None or remote.get("type") == "error":
                rep.fail(cid, f"remote job failed: {remote}")
            else:
                def read_json(name):
                    path = profile_dir / name
                    return json.loads(path.read_text()) if path.exists() else None

                office_ids = {s["id"] for s in office["segments"]}
                remote_ids = {s["id"] for s in remote["segments"]}
                remote_names = {s["name"] for s in remote["segments"]}
                office_file = read_json("profiles.json") or []
                remote_file = read_json("profiles-remote.json")

                problems = []
                if remote.get("stream") != "remote":
                    problems.append(f'reply did not echo stream=remote: {sorted(remote)}')
                if not office_ids:
                    problems.append("office job produced no identified speakers")
                if any(i >= REMOTE_ID_BASE for i in office_ids):
                    problems.append(f"office wire ids not below {REMOTE_ID_BASE}: {office_ids}")
                if not remote_ids:
                    problems.append("remote job produced no identified speakers")
                if any(i < REMOTE_ID_BASE for i in remote_ids):
                    problems.append(f"remote wire ids not >= {REMOTE_ID_BASE}: {remote_ids}")
                if not all(n.startswith("R") for n in remote_names):
                    problems.append(f"remote names are not R-named: {remote_names}")
                if remote_file is None:
                    problems.append("profiles-remote.json was never written — the "
                                    "remote job matched the office store instead of "
                                    "creating its own identities")
                else:
                    if not remote_file:
                        problems.append("profiles-remote.json is empty — remote created "
                                        "no profiles of its own")
                    if any(not p["name"].startswith("R") for p in remote_file):
                        problems.append(f"office-space entries leaked into "
                                        f"profiles-remote.json: {remote_file}")
                    local = {p["id"] for p in remote_file}
                    if local != {i - REMOTE_ID_BASE for i in remote_ids}:
                        problems.append(f"remote wire ids {remote_ids} do not map onto "
                                        f"stored local ids {local}")
                if any(p["name"].startswith("R") for p in office_file):
                    problems.append(f"remote-space entries leaked into profiles.json: "
                                    f"{office_file}")
                if any(p["id"] >= REMOTE_ID_BASE for p in office_file):
                    problems.append(f"profiles.json holds remote-range ids: {office_file}")

                rep.expect(cid, not problems,
                           f"office ids {sorted(office_ids)} in profiles.json, "
                           f"remote ids {sorted(remote_ids)} as {sorted(remote_names)} "
                           f"in profiles-remote.json, neither file holding the other",
                           "; ".join(problems))

        # -- check 10: reset wipes BOTH spaces. Leaving half of it populated
        #    would make remote numbering carry over into a "fresh" session.
        # -- the sample-rate regression. WeSpeaker is a 16 kHz model and does not
        #    error on another rate — it silently embeds pitch- and tempo-shifted
        #    speech. Live chunks arrive as 16 kHz temp WAVs, but a stop-time pass
        #    reads the RECORDING, which is at the capture device's native rate
        #    (44.1 kHz here). On the owner's audio the same voice scored 0.98 on a
        #    chunk and 0.11 on the final, fell under SIM_THRESHOLD and minted a
        #    duplicate profile — one person shown as two speakers.
        #
        #    So: enrol from 16 kHz chunks, then run a final over a 44.1 kHz file
        #    of the SAME speech. It must land on the SAME profile. Without the
        #    resample in resolve_speakers this yields a new id, which is exactly
        #    the shipped bug.
        if ctx.wants("diarize/native-rate-final-matches-16k-chunks"):
            cid = "diarize/native-rate-final-matches-16k-chunks"
            try:
                import numpy as np
                import torch
                import torchaudio

                sc.send_json({"cmd": "reset"})
                sc.wait_for(lambda m: m.get("text") == "RESET", timeout=60)

                # Build the 44.1 kHz file by genuinely resampling the 16 kHz clip,
                # so the check holds even if the fixture is already 16 kHz.
                native_sr = 44_100
                up = torchaudio.functional.resample(
                    torch.from_numpy(np.asarray(clip, dtype=np.float32)).unsqueeze(0),
                    SR, native_sr).squeeze(0).numpy()
                native_path = write_wav(pathlib.Path(ctx.tmp) / "diarize-native.wav",
                                        up, sr=native_sr)

                # Enrol from 16 kHz chunks, the live path.
                half = clip.size // 2
                chunk_ids = []
                for i, piece in enumerate((clip[:half], clip[half:])):
                    cp = write_wav(pathlib.Path(ctx.tmp) / f"diarize-16k-{i}.wav", piece)
                    sc.send_json({"cmd": "chunk", "audio": cp,
                                  "window_start": 0.0, "stream": "remote"})
                    m = sc.wait_for(lambda m: m.get("type") in ("chunk_result", "error"),
                                    timeout=ctx.job_timeout)
                    if m and m.get("type") == "chunk_result":
                        chunk_ids += [s["id"] for s in m.get("segments", [])]

                sc.send_json({"cmd": "final", "audio": native_path, "stream": "remote"})
                fin = sc.wait_for(lambda m: m.get("type") in ("result", "error"),
                                  timeout=ctx.final_timeout)
                final_ids = ([s["id"] for s in fin.get("segments", [])]
                             if fin and fin.get("type") == "result" else [])

                if not chunk_ids:
                    rep.fail(cid, "16 kHz chunks enrolled no speaker — nothing to compare")
                elif not final_ids:
                    rep.fail(cid, f"the {native_sr} Hz final produced no labelled turns: {fin}")
                else:
                    known = set(chunk_ids)
                    unknown = sorted(set(final_ids) - known)
                    rep.expect(cid, not unknown,
                               f"{native_sr} Hz final reused the 16 kHz profile "
                               f"{sorted(known)}",
                               f"final minted {unknown} instead of reusing "
                               f"{sorted(known)} — the same voice embedded at "
                               f"{native_sr} Hz did not match its own 16 kHz profile")
            except Exception as exc:  # noqa: BLE001
                rep.fail(cid, f"check could not run: {exc!r}")

        if ctx.wants("diarize/reset-wipes-both-stores"):
            cid = "diarize/reset-wipes-both-stores"
            before = sorted(p.name for p in profile_dir.iterdir())
            sc.send_json({"cmd": "reset"})
            done = sc.wait_for(lambda m: m.get("text") == "RESET", timeout=60)
            after = sorted(p.name for p in profile_dir.iterdir())
            if done is None:
                rep.fail(cid, "reset never acknowledged")
            elif not before:
                rep.fail(cid, "no profile files existed before reset — nothing proven")
            else:
                rep.expect(cid, after == [],
                           f"reset removed all of {before}",
                           f"still present after reset: {after} (was {before})")
    finally:
        sc.close()


# ===================================================================== main
GROUPS = [
    ("nemotron", NEMOTRON_CHECKS, run_nemotron),
    ("chunked", CHUNKED_CHECKS, run_chunked),
    ("diarize", DIARIZE_CHECKS, run_diarize),
]


class Context:
    def __init__(self, args, tmp):
        self.only = args.only
        self.tmp = tmp
        self.chunked_model = args.chunked_model
        self.clip_sec = args.clip_sec
        self.clip_start = args.clip_start
        self.diarize_sec = args.diarize_sec
        self.load_timeout = args.load_timeout
        self.job_timeout = args.job_timeout
        self.final_timeout = args.final_timeout
        self.audio_a = args.audio_a
        self.audio_b = args.audio_b
        self.audio_diarize = args.audio_diarize

    def wants(self, check_id: str) -> bool:
        if not self.only:
            return True
        return any(pattern in check_id for pattern in self.only)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", action="append", default=None,
                   help="substring filter on check id or group (repeatable)")
    p.add_argument("--list", action="store_true", help="list check ids and exit")
    p.add_argument("--audio-a", default=None, help="real speech WAV (office lane)")
    p.add_argument("--audio-b", default=None,
                   help="a DIFFERENT real speech WAV (remote lane)")
    p.add_argument("--audio-diarize", default=None, help="real speech WAV for diarization")
    p.add_argument("--clip-start", type=float, default=0.0)
    p.add_argument("--clip-sec", type=float, default=6.0,
                   help="seconds of each fixture fed to the realtime lanes")
    p.add_argument("--diarize-sec", type=float, default=25.0)
    p.add_argument("--chunked-model", default=DEFAULT_CHUNKED_MODEL)
    p.add_argument("--load-timeout", type=float, default=300.0)
    p.add_argument("--job-timeout", type=float, default=600.0)
    p.add_argument("--final-timeout", type=float, default=90.0,
                   help="how long one realtime FLUSH may take before the lane "
                        "counts as broken (a 6 s clip finalises in ~1 s)")
    args = p.parse_args()

    if args.list:
        for name, checks, _ in GROUPS:
            print(name)
            for cid in checks:
                print(f"    {cid}")
        return 0

    if not VENV_PY.exists():
        print(f"!! {VENV_PY} not found — run download-best-models.sh", file=sys.stderr)
        return 2

    # Fill in any audio fixture the caller did not name.
    if not (args.audio_a and args.audio_b and args.audio_diarize):
        found = find_fixtures()
        if len(found) >= 2:
            args.audio_a = args.audio_a or str(found[0])
            args.audio_b = args.audio_b or str(found[1])
        args.audio_diarize = args.audio_diarize or (str(found[0]) if found else None)

    print("=" * 72)
    print("MeetingTranscriber sidecar integration tests")
    print(f"  python        {VENV_PY}")
    print(f"  HF_HOME       {os.environ['HF_HOME']}  (HF_HUB_OFFLINE="
          f"{os.environ.get('HF_HUB_OFFLINE')})")
    print(f"  audio A       {args.audio_a or '(none — real-speech checks will SKIP)'}")
    print(f"  audio B       {args.audio_b or '(none — real-speech checks will SKIP)'}")
    print(f"  audio diarize {args.audio_diarize or '(none — diarize checks will SKIP)'}")
    print("=" * 72)

    # SAFETY: fingerprint the owner's real profiles before anything runs.
    before = profile_hashes()
    print("\nreal speaker-profile fingerprints BEFORE:")
    for name, digest in before.items():
        print(f"  {name:26} {digest or '(absent)'}")
    print()

    tmp = tempfile.mkdtemp(prefix="mt-sidecar-tests-")
    rep = Report()
    try:
        ctx = Context(args, tmp)
        for name, checks, runner in GROUPS:
            if not any(ctx.wants(cid) for cid in checks):
                continue
            print(f"\n--- {name} " + "-" * (68 - len(name)))
            try:
                runner(rep, ctx)
            except Exception as exc:  # noqa: BLE001 — one bad group must not
                import traceback       # hide the results of the other two
                traceback.print_exc()
                rep.fail(f"{name}/<group>", f"group raised {exc!r}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    after = profile_hashes()
    print("\nreal speaker-profile fingerprints AFTER:")
    for name, digest in after.items():
        print(f"  {name:26} {digest or '(absent)'}")
    if before == after:
        rep.ok("safety/real-profiles-untouched",
               "all four real profile files byte-identical before and after")
    else:
        changed = [n for n in before if before[n] != after[n]]
        rep.fail("safety/real-profiles-untouched",
                 f"REAL SPEAKER PROFILES WERE MODIFIED: {changed}")

    return rep.summarize()


if __name__ == "__main__":
    sys.exit(main())
