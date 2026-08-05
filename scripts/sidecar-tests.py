#!/usr/bin/env python3
"""
Integration test suite for the Python sidecars.

    .venv/bin/python3 scripts/sidecar-tests.py            # everything
    .venv/bin/python3 scripts/sidecar-tests.py --list
    .venv/bin/python3 scripts/sidecar-tests.py --only wespeaker
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
The WeSpeaker identity sidecar persists speaker profiles to models/speaker-profiles/.
Those are the OWNER'S real voices and a test that corrupts them would be worse
than no test at all, so:
  * every diarization/identity subprocess gets MT_PROFILE_DIR pointed at a temp
    dir — including the pyannote one, which cannot write a profile at all now,
    because a fence that is only applied where it is currently needed stops being
    applied at all the moment the code moves,
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
import re
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
SWIFT_SOURCES = PROJECT / "MeetingTranscriber" / "Sources"

# Every service lives in its OWN folder under scripts/ (owner, 2026-07-29:
# "pisahin folder foldernya jangan di satuin"), and each service's vendored
# third-party code sits next to its owner rather than in a shared scripts/vendor.
# Named here once so a future move is one edit, not fifteen. Both `Sidecar(...)`
# and `load_sidecar_module(...)` resolve these with `SCRIPTS / name`, which takes
# a relative path with slashes unchanged. This file itself stays at scripts/
# root — it tests everything, so it belongs to no single service.
NEMOTRON_SERVICE = "nemotron/nemotron-service.py"
# ONE SERVICE PER ASR MODEL, finished 2026-07-30: the shared
# chunked/chunked-asr-service.py is DELETED and each model has its own standalone
# file. The three below are the mlx-audio ones — byte-identical extractions of that
# file's mlx-audio branch, proven on real audio before it was removed. They are
# listed individually rather than behind one name precisely because the protocol
# now lives in four copies, and `whisper/protocol-matches-chunked` has to drive
# every one of them.
QWEN3_SERVICE = "qwen3/qwen3-service.py"
GRANITE_SERVICE = "granite/granite-service.py"
VOXTRAL_SERVICE = "voxtral/voxtral-service.py"
MLX_SERVICES = [("qwen3", QWEN3_SERVICE), ("granite", GRANITE_SERVICE),
                ("voxtral", VOXTRAL_SERVICE)]
WHISPER_SERVICE = "whisper/whisper-service.py"
ALIGNER_SERVICE = "aligner/aligner-service.py"
# MOSS is TWO services since 2026-07-31 (ONE SERVICE PER ROLE): the same model is
# selectable as the chunked ASR model AND as the diarization engine, and in "MOSS
# as ASR + MOSS as diarizer" mode BOTH run at once. One script for two live
# processes meant two writers on one log, so each role got its own service.
# MOSS_ASR_SERVICE is the one `chunked.model = moss` starts; MOSS_SERVICE is the
# DIARIZATION-role file that `diarization.engine = moss` starts.
MOSS_ASR_SERVICE = "moss-asr/moss-asr-service.py"
MOSS_SERVICE = "moss-diar/moss-diar-service.py"
MOSS_ASR_VENDOR = SCRIPTS / "moss-asr" / "vendor"
MOSS_VENDOR = SCRIPTS / "moss-diar" / "vendor"
# Every MOSS copy, in the order the checks report them. Each entry is
# (label, service, vendor dir) — the four rule checks below run against ALL of
# them, the canned-gate-across-copies precedent, so a gate fixed in one file and
# not the other fails here instead of mid-meeting in whichever role got missed.
MOSS_COPIES = [("moss-asr", MOSS_ASR_SERVICE, MOSS_ASR_VENDOR),
               ("moss-diar", MOSS_SERVICE, MOSS_VENDOR)]
# Diarization is TWO services since 2026-07-30 (ONE SERVICE PER MODEL). The
# pipeline half answers who-spoke-when with run-local labels and owns no profile
# store at all; the WeSpeaker half owns the embedder and BOTH stores. That split is
# why the identity checks below need no pipeline load and run in seconds.
PYANNOTE_SERVICE = "pyannote/pyannote-service.py"
WESPEAKER_SERVICE = "wespeaker/wespeaker-service.py"
# The THIRD diarizer (2026-08-03): the vendored, Apache-2.0 `diarize` engine
# (Silero VAD → WeSpeaker windows → GMM-BIC counting → spectral clustering). It
# answers the SAME Swift caller as the pyannote service, with the same
# turns-only wire, so the two must not drift apart — see
# `spectral/protocol-matches-pyannote`. Its vendored tree lives under it, the
# MOSS rule.
SPECTRAL_SERVICE = "spectral/spectral-service.py"
SPECTRAL_VENDOR = SCRIPTS / "spectral" / "vendor"

# Pin the model cache the same way every sidecar does, so the in-process
# white-box checks (which import mlx_audio directly) resolve offline too.
os.environ.setdefault("HF_HOME", str(PROJECT / "models"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SR = 16_000

DEFAULT_CHUNKED_MODEL = "mlx-community/Qwen3-ASR-1.7B-bf16"
DEFAULT_ALIGN_MODEL = "mlx-community/Qwen3-ForcedAligner-0.6B-bf16"

# Mirrors of the sidecars' own constants. Deliberately RE-DECLARED rather than
# force a conscious decision here, not silently follow along.
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


def first_speech_offset(audio, block_sec=1.0, min_rms=0.01):
    """Seconds until the first block of real speech, or None if there is none.

    THE FIXTURE TRAP THIS EXISTS FOR, which has now cost two debugging rounds.
    The owner's recordings routinely open with a long stretch of DIGITAL silence
    (the array/loopback is live before anyone talks): `meeting-2026-07-28T04-10-59Z`
    is silent for its first 40 s, `meeting-2026-07-30T04-53-29Z` for its first 22 s.
    Slicing a fixture from offset 0 therefore yields a clip of pure zeros, and the
    failure is never an obvious "no audio" — it is a *plausible-looking wrong
    answer*: an embedding of silence is a real, finite, meaningless vector, so
    `wespeaker/native-rate-final-matches-16k-chunks` minted a third profile, and
    `nemotron/lane-isolation` skipped itself with an empty baseline. Both were the
    same silent-fixture bug wearing two costumes.

    `find_fixtures`'s own RMS gate does NOT catch it, and cannot: it averages over
    30 s, so 20 s of silence plus 10 s of speech scores 0.044 and sails past the
    0.01 bar while 20 of the 25 seconds a check actually uses are zeros.

    Returning None (nothing anywhere clears the bar) is meaningful, not an error —
    it is how a wholly silent file, like the owner's dead BlackHole `-remote`
    capture, still gets rejected by that same gate.
    """
    import numpy as np
    block = max(1, int(block_sec * SR))
    for i in range(0, max(0, audio.size - block) + 1, block):
        segment = audio[i:i + block]
        if segment.size and float(np.sqrt(np.mean(segment * segment))) >= min_rms:
            return i / SR
    return None


def load_clip(path, start=None, seconds=None):
    """Read a fixture as float32 16 kHz mono, same loader the aligner probe uses.

    `start=None` means AUTO-SEEK to the first speech — see `first_speech_offset`
    for why that is the default rather than 0.0. Pass an explicit number (including
    `--clip-start 0`) to force a position.
    """
    import numpy as np
    from mlx_audio.stt.utils import load_audio
    audio = np.asarray(load_audio(str(path), sr=SR), dtype=np.float32)
    if start is None:
        # No speech at all -> fall back to 0.0 so the caller's RMS gate sees the
        # silence and rejects the file, instead of us hiding it behind an offset.
        start = first_speech_offset(audio) or 0.0
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
            # Judge the region the CHECKS will use, i.e. auto-seeked past any
            # leading silence — not blindly the first 30 s. Same call, same
            # default, so the gate and the checks can never disagree about
            # whether a file has speech.
            clip = load_clip(path, None, 30.0)
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


def extract_flush_branch(source: str):
    """Compile the frame loop's `if n == 0:` (FLUSH) branch into a callable.

    Same idea as extract_nested, one level deeper: the FLUSH branch is a
    STATEMENT BLOCK inside `main()`'s `while True:` loop, so there is no function
    to lift and no way to reach it from stdin without loading a model. Lifting it
    by AST runs the shipped code against injected stubs, which is what lets the
    protocol-conformance check compare two sidecars' FLUSH handling in
    milliseconds instead of two model loads.

    The trailing `continue` is dropped (it would be a SyntaxError outside a loop);
    nothing else is touched.
    """
    import ast
    tree = ast.parse(source)
    main_fn = next(n for n in tree.body
                   if isinstance(n, ast.FunctionDef) and n.name == "main")
    loop = next(n for n in ast.walk(main_fn) if isinstance(n, ast.While))
    branch = next(s for s in loop.body
                  if isinstance(s, ast.If) and ast.unparse(s.test) == "n == 0")
    body = [s for s in branch.body if not isinstance(s, ast.Continue)]
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    code = compile(module, "<main.flush-branch>", "exec")

    def run(namespace: dict) -> dict:
        exec(code, namespace)  # noqa: S102 — the point is to run the real code
        return namespace

    return run


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
    "nemotron/partial-full-buffer-keeps-up",
    "nemotron/silence-gate",
]


def run_nemotron(rep: Report, ctx):
    # -- check 1: single-stream output is byte-identical to the pre-dual-stream
    #    contract. Its own process, fed ONLY office frames, so the assertion
    #    covers every byte the sidecar produced rather than a filtered subset.
    if ctx.wants("nemotron/single-stream-bytes"):
        sc = Sidecar(NEMOTRON_SERVICE, ["--language", "en-US"])
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

    sc = Sidecar(NEMOTRON_SERVICE, ["--language", "en-US"])
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

        # -- check 3: a partial transcribes the WHOLE buffer (so the live caption
        #    shows the full utterance, not just its tail), and the realtime loop
        #    stays caught up regardless, because the cadence stretches as the
        #    buffer grows. An earlier fixed 10 s window kept cost flat but froze
        #    the caption mid-sentence; a naive full-buffer partial at a fixed 1.5 s
        #    cadence fell behind (`wait` drifted to seconds). This asserts both:
        #    the buffer is fully transcribed AND the loop never falls far behind.
        #    Observed through the sidecar's `buf=.. took=.. wait=..` stderr line.
        if ctx.wants("nemotron/partial-full-buffer-keeps-up"):
            cid = "nemotron/partial-full-buffer-keeps-up"
            mark = len(sc.stderr_lines)
            audio = tone(24.0, base_hz=200.0, seed=3)
            for i in range(0, audio.size, SR // 2):
                sc.write(frame_office(audio[i:i + SR // 2]))
            sc.write(FLUSH_OFFICE)
            sc.wait_for(lambda m: m.get("type") == "final", timeout=300)
            time.sleep(0.5)

            import re
            pattern = re.compile(
                r"(office|remote) (partial|final) buf=([\d.]+)s took=[\d.]+s "
                r"\([\d.]+x\) wait=([\d.]+)s")
            partials = []   # (buf_sec, wait_sec)
            bufs = []
            for line in sc.stderr_lines[mark:]:
                m = pattern.search(line)
                if not m:
                    continue
                if m.group(2) == "partial":
                    partials.append((float(m.group(3)), float(m.group(4))))
                    bufs.append(float(m.group(3)))
            if not partials:
                rep.fail(cid, "no partials were logged at all")
            elif max(bufs) < 15.0:
                rep.fail(cid, f"buffer never grew past 15 s (max {max(bufs):.1f}s) — "
                              "long-utterance behaviour was not exercised")
            else:
                # Caught up: the loop never sat idle waiting a long time between
                # generate() calls (the drift bug pushed wait to seconds).
                max_wait = max(w for _, w in partials)
                # Cadence stretched: the buffer-size step between consecutive
                # partials near the end is larger than near the start.
                steps = [bufs[i + 1] - bufs[i] for i in range(len(bufs) - 1)]
                stretched = len(steps) >= 2 and steps[-1] > steps[0] + 0.4
                rep.expect(cid, max_wait < 1.5 and stretched,
                           f"{len(partials)} partials, buffer reached {max(bufs):.1f}s, "
                           f"cadence step {steps[0]:.1f}s→{steps[-1]:.1f}s, "
                           f"max wait {max_wait:.2f}s",
                           f"max_wait={max_wait:.2f}s (want <1.5), "
                           f"stretched={stretched} (steps {steps})")
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
# The three mlx-audio chunked sidecars — scripts/qwen3/, scripts/granite/,
# scripts/voxtral/. Whisper was split out first (2026-07-29) and its checks live in
# the `whisper` group below; the last shared file, chunked/chunked-asr-service.py,
# was deleted on 2026-07-30 once these three were proven byte-identical to it.
#
# The canned-gate check now runs against ALL THREE modules, not one: with the gate
# copied into three files, a check that only read one of them would pass while
# another file's copy had drifted — and this gate deletes transcript, so drift in
# it is the expensive kind.
CHUNKED_CHECKS = [
    "chunked/canned-gate-spares-real-short-replies",
    "chunked/final-shape",
]

# Qwen3's decoding options, exposed 2026-08-03 on the Whisper precedent. Pure
# calls to qwen3_generate_kwargs — no model load — because the failure they guard
# is INVISIBLE in the output: an option whose default quietly differs from what
# the call used to pass changes every transcript, and nothing anywhere says so.
QWEN3_CHECKS = [
    "qwen3/option-defaults-are-todays-behaviour",
    "qwen3/sentinels-become-none",
    "qwen3/context-size-never-travels-alone",
]

# What model.generate() was called with BEFORE the options existed. The
# pre-options sidecar built it with the literal expression
#     kwargs = {"language": language} if language else {}
# so there are exactly two shapes and NEITHER carries anything else. Written out
# literally rather than derived, so a change to qwen3_generate_kwargs cannot also
# change what it is compared against — the whole point of the check.
QWEN3_TODAYS_KWARGS = {"language": "en"}
QWEN3_TODAYS_KWARGS_AUTO = {}


def chunked_service_for(model: str) -> str:
    """Which sidecar serves this HF repo. Mirrors the Swift `ChunkedASRModel`s.

    Exhaustive on purpose, and it RAISES rather than falling back: a default would
    have quietly kept pointing at the deleted shared file.
    """
    lower = model.lower()
    for needle, service in (("qwen3-asr", QWEN3_SERVICE),
                            ("granite", GRANITE_SERVICE),
                            ("voxtral", VOXTRAL_SERVICE),
                            ("whisper", WHISPER_SERVICE),
                            # The ASR-role service: this function answers
                            # "--chunked-model X runs which sidecar", and that is
                            # the chunked ASR role. The diarization role is never
                            # selected this way.
                            ("moss", MOSS_ASR_SERVICE)):
        if needle in lower:
            return service
    raise ValueError(f"no sidecar known for --chunked-model {model!r}")

# Segments captured from the owner's real meetings, with the verdict each one
# must get. The dropped rows are Whisper's silence hallucinations; the kept rows
# are real sentences that an earlier no_speech-only gate deleted from the
# transcript. Both failure directions are represented on purpose — this rule has
# been wrong in each of them, so a test that only proved hallucinations die
# would have passed while real speech was being thrown away.
WHISPER_GATE_CASES = [
    # (text, no_speech, compression, duration, keep?)
    ("Thank you.", 0.89, 1.0, 30.0, False),
    ("Thank you.", 0.67, 1.0, 5.0, False),
    ("you", 0.72, 1.0, 10.0, False),
    ("If we have a one-minute speech to deliver,", 0.86, 1.6, 3.0, True),
    ("into the picture this framework will not only help us speak in a concise manner",
     0.55, 1.8, 5.0, True),
    ("while making sure all the important points are in place but the", 0.55, 1.8, 4.0, True),
    ("words for their um if you're a prime membership you get delivery faster i guess",
     0.55, 1.8, 5.0, True),
    ("yeah amazon prime also serves into that needs based um or their partnership with whole foods",
     0.55, 1.8, 6.0, True),
    ("four or five or six or seven or eight or 10 or 20 or 20 or 20 or 20", 0.2, 13.12, 10.0, False),
    ("Okay.", 0.1, 1.0, 1.0, True),
]


def run_chunked(rep: Report, ctx):
    # -- the model-agnostic canned-phrase gate. Granite/Qwen3 expose no
    #    confidence numbers, so this is their only hallucination check — and the
    #    thing it must never do is eat a real short reply.
    if ctx.wants("chunked/canned-gate-spares-real-short-replies"):
        cid = "chunked/canned-gate-spares-real-short-replies"
        try:
            modules = [(label, load_sidecar_module(service,
                                                   f"mt_{label}_canned")[0])
                       for label, service in MLX_SERVICES]
            cases = [
                # (text, duration, keep?)
                ("Thank you.", 30.0, False),          # observed on Granite
                ("thank you", 30.0, False),           # Granite writes lowercase
                ("you", 20.0, False),
                ("Okay.", 30.0, True),                # real reply, must survive
                ("Yes.", 30.0, True),
                ("Thank you.", 2.0, True),            # short chunk: plausibly real
                ("Thank you for joining the meeting today.", 30.0, True),
                ("thanks, I agree with that point", 30.0, True),
            ]
            wrong = []
            for label, mod in modules:
                for text, dur, want_keep in cases:
                    reason = mod.canned_drop_reason(text, dur)
                    if (reason is None) != want_keep:
                        verdict = "kept" if reason is None else f"dropped as {reason}"
                        wrong.append(f"{label}: {text!r} ({dur:.0f}s) was {verdict}")
                # The thresholds were RENAMED in the split (they carried a
                # WHISPER_ prefix in the shared file although the rule is the
                # mlx-audio one). Values pinned here so the rename cannot have
                # moved them, and the old names must be gone — a file defining
                # both would mean one is dead and nobody knows which.
                if getattr(mod, "CANNED_MIN_DURATION_SEC", None) != 4.0:
                    wrong.append(f"{label}: CANNED_MIN_DURATION_SEC is "
                                 f"{getattr(mod, 'CANNED_MIN_DURATION_SEC', None)!r}, not 4.0")
                if getattr(mod, "CANNED_MIN_WORDS_PER_SEC", None) != 0.5:
                    wrong.append(f"{label}: CANNED_MIN_WORDS_PER_SEC is "
                                 f"{getattr(mod, 'CANNED_MIN_WORDS_PER_SEC', None)!r}, not 0.5")
                for stale in ("WHISPER_MIN_DENSITY_DURATION", "WHISPER_MIN_WORDS_PER_SEC",
                              "WHISPER_NO_SPEECH_MAX", "WHISPER_COMPRESSION_MAX"):
                    if hasattr(mod, stale):
                        wrong.append(f"{label}: still defines {stale} — this file has "
                                     "no Whisper in it")
            rep.expect(cid, not wrong,
                       f"all {len(cases)} canned/real cases judged correctly by all "
                       f"{len(modules)} mlx-audio services, on the renamed thresholds",
                       "; ".join(wrong))
        except Exception as exc:  # noqa: BLE001
            rep.fail(cid, f"could not evaluate the canned gate: {exc}")

    # -- check 5: a final carries EXACTLY type and text (plus "conf" on
    #    Whisper) — NEVER "words"/"dur". It used to carry them whenever
    #    --align-model was given; alignment moved to its own sidecar on
    #    2026-07-29, so this now pins that an ASR final can never carry word
    #    timestamps at all, whatever the settings say.
    if ctx.wants("chunked/final-shape"):
        cid = "chunked/final-shape"
        # Whichever service owns --chunked-model (default Qwen3). Resolved rather
        # than hard-coded because the shared sidecar that used to serve every MLX
        # model no longer exists.
        sc = Sidecar(chunked_service_for(ctx.chunked_model),
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
                    # "conf" is Whisper-only. With the DEFAULT model (Qwen3) the
                    # allowance below is empty, so this check doubles as the pin
                    # that an mlx-audio model NEVER fabricates a confidence.
                    is_whisper = "whisper" in ctx.chunked_model.lower()
                    allowed_extra = {"conf"} if is_whisper else set()
                    extra = set(final) - {"type", "text"}
                    conf_ok = True
                    if "conf" in final:
                        conf = final["conf"]
                        conf_ok = isinstance(conf, (int, float)) and 0 < conf <= 1.0
                    rep.expect(cid, extra <= allowed_extra and conf_ok,
                               'final carried exactly {"type","text"}'
                               + (f' + conf={final.get("conf")}' if "conf" in final
                                  else " (no conf — mlx-audio reports none)"),
                               f"final key set was {sorted(final)} with conf="
                               f"{final.get('conf')!r} (expected no words/dur — "
                               f"alignment is a separate sidecar; conf allowed "
                               f"only on Whisper, and only as 0 < x <= 1)")
        finally:
            sc.close()


# ============================================================= whisper group
# scripts/whisper/whisper-service.py — the standalone Whisper sidecar, extracted
# VERBATIM out of the old shared chunked-asr-service.py on 2026-07-29 (owner: one
# sidecar per ASR model, FULLY standalone, no shared protocol module). That shared
# file is gone since 2026-07-30; Whisper's gate and confidence code lives only here.
#
# The first three checks are the SAME checks that used to run as `chunked/*`;
# they simply followed the code when it moved here. All four are pure imports — no
# model load, milliseconds.
#
# The fourth is the price of the standalone choice: with the wire protocol defined
# in more than one file, disciplined editing CANNOT prevent drift, but a test can
# detect it. It drives EVERY ASR sidecar's payload builders and real FLUSH branch
# with the same inputs and fails loudly if a protocol edit lands in one file and
# not the others. It compared two files when only Whisper was split; since
# 2026-07-30 it compares Whisper against all three mlx-audio services, because a
# protocol edit missed in ONE of four files is exactly the failure the standalone
# choice was known to risk (and it fails silently in the app — the number simply
# never appears).
#
# The last four cover the decoding options exposed on 2026-07-29
# (--initial-prompt, --best-of, …). They are pure calls to
# whisper_transcribe_kwargs — no model load — because the failure they guard is
# invisible in the output: an option whose default quietly differs from what the
# call used to pass changes every transcript, and nothing anywhere says so.
WHISPER_CHECKS = [
    "whisper/gate-keeps-real-speech",
    "whisper/conf-pooling",
    "whisper/conf-only-from-kept-segments",
    "whisper/protocol-matches-chunked",
    "whisper/option-defaults-are-todays-behaviour",
    "whisper/sentinels-become-none",
    "whisper/hallucination-implies-word-timestamps",
    "whisper/autodetect-clears-language",
]

# What mlx_whisper.transcribe() was called with BEFORE the options existed, plus
# the library's own defaults for everything the call did not name. Written out
# literally rather than derived, so a change to whisper_transcribe_kwargs cannot
# also change what it is compared against — the whole point of the check.
WHISPER_TODAYS_KWARGS = {
    "path_or_hf_repo": "mlx-community/whisper-large-v3-mlx",
    "language": "en",
    "task": "transcribe",                  # DecodingOptions default
    "no_speech_threshold": 0.6,            # mlx_whisper default
    "logprob_threshold": -1.0,             # mlx_whisper default
    "compression_ratio_threshold": 2.4,    # mlx_whisper default
    "word_timestamps": False,              # mlx_whisper default
}

# Every wire message the two sidecars can produce, as (builder, args). Compared
# BYTE FOR BYTE rather than by key set, so key ORDER and the conf rounding are
# covered too — a re-ordered payload is still a protocol change.
PROTOCOL_EMIT_CASES = [
    ("emit", ("status", "LOADED")),
    ("emit", ("error", "Chunk transcription failed: boom")),
    # No words/dur cases any more: alignment left these sidecars on 2026-07-29
    # and a `final` can no longer carry word timestamps at all.
    ("emit_final", ("hello world", None)),
    ("emit_final", ("hello world", 0.876543)),
    ("emit_final", ("", None)),
    ("emit_file", ("file_result", 7, "some text", 0.5)),
    ("emit_file", ("file_result", 7, "some text", None)),
    ("emit_file", ("file_error", 7, "file not found: /nope.wav")),
    ("emit_file", ("file_error", None, "file transcription failed: boom")),
]

# Constants that are part of the wire behaviour, not implementation detail: a
# different MIN_CHUNK_SEC in one file means the two services silently disagree
# about which flushes produce a final at all.
PROTOCOL_CONSTANTS = ["SR", "MIN_CHUNK_SEC", "MAX_BUFFER_SEC"]


def capture_stdout(fn, *args):
    """Run fn(*args) and return the raw stdout lines it wrote."""
    import contextlib
    import io
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        fn(*args)
    return buffer.getvalue().splitlines()


def run_flush_branch(module, source, buffered_sec: float, transcribed):
    """Execute a sidecar's REAL `n == 0` FLUSH branch over a synthetic buffer.

    Returns (stdout_lines, log_lines, remaining_buffer_size). The ASR is stubbed
    (`transcribed` is the (text, conf) it returns), so the only thing under test
    is the branch's own decisions: the MIN_CHUNK_SEC floor, whether a final is
    emitted at all, and the buffer reset. There is no alignment stub any more —
    the aligner is its own sidecar and this branch never calls it.
    """
    import numpy as np
    logs = []
    namespace = dict(vars(module))
    namespace.update({
        "buffer": np.zeros(int(buffered_sec * SR), dtype=np.float32),
        "np": np,
        "log": logs.append,
        "transcribe": lambda audio: transcribed,
    })
    lines = capture_stdout(extract_flush_branch(source), namespace)
    return lines, logs, int(namespace["buffer"].size)


def run_whisper(rep: Report, ctx):
    module = source = None
    try:
        module, source = load_sidecar_module(WHISPER_SERVICE, "mt_whisper_service")
    except Exception as exc:  # noqa: BLE001
        for cid in WHISPER_CHECKS:
            if ctx.wants(cid):
                rep.fail(cid, f"could not import whisper-service.py: {exc!r}")
        return

    # -- the hallucination gate, as a pure rule. No model load: it imports the
    #    sidecar module and replays real captured segments, so it runs in
    #    milliseconds and cannot be skipped for want of a fixture.
    if ctx.wants("whisper/gate-keeps-real-speech"):
        cid = "whisper/gate-keeps-real-speech"
        try:
            wrong = []
            for text, ns, cr, dur, want_keep in WHISPER_GATE_CASES:
                reason = module.whisper_drop_reason(text, no_speech=ns,
                                                    compression=cr, duration=dur)
                if (reason is None) != want_keep:
                    verdict = "kept" if reason is None else f"dropped as {reason}"
                    wrong.append(f"{text[:45]!r} was {verdict}")
            # The extraction must NOT have dragged the mlx-audio canned-phrase
            # gate along. Whisper's path never called it, so wiring it in here
            # would ADD a gate — over-deletion, the dangerous direction.
            if hasattr(module, "canned_drop_reason") or hasattr(module, "CANNED_HALLUCINATIONS"):
                wrong.append("whisper-service.py defines the mlx-audio canned-phrase "
                             "gate — Whisper never had it; adding it would delete text "
                             "the shared sidecar keeps")
            rep.expect(cid, not wrong,
                       f"all {len(WHISPER_GATE_CASES)} real segments judged "
                       "correctly, and no canned-phrase gate came along",
                       "; ".join(wrong))
        except Exception as exc:  # noqa: BLE001
            rep.fail(cid, f"could not evaluate the gate: {exc}")

    # -- transcript confidence, as a pure pooling rule. No model load.
    #    The number goes in front of the user as "asr 0.92", so the way it is
    #    pooled is the part that can be quietly, unfalsifiably wrong: averaging
    #    the exponentiated per-segment values (instead of the log-probs) or
    #    weighting every segment equally (instead of by words) both produce a
    #    plausible-looking number that is not the geometric-mean per-token
    #    probability it is presented as. The last case is the important one:
    #    NOTHING KEPT MUST YIELD None, never a fabricated 0.0 or 1.0.
    if ctx.wants("whisper/conf-pooling"):
        cid = "whisper/conf-pooling"
        try:
            import math
            fn = module.whisper_chunk_confidence
            problems = []

            # (a) word-WEIGHTED pooling in the LOG domain. A 10-word confident
            #     segment and a 2-word unsure one: the long one must dominate.
            got = fn([(10, -0.1), (2, -2.0)])
            weighted_log = math.exp((10 * -0.1 + 2 * -2.0) / 12)      # 0.6592
            unweighted_log = math.exp((-0.1 + -2.0) / 2)              # 0.3499
            mean_of_probs = (math.exp(-0.1) + math.exp(-2.0)) / 2     # 0.5201
            if got is None or abs(got - weighted_log) > 1e-9:
                problems.append(f"pooled {got} != word-weighted log mean "
                                f"{weighted_log:.4f} (unweighted would be "
                                f"{unweighted_log:.4f}, mean-of-probs "
                                f"{mean_of_probs:.4f})")

            # (b) exp() mapping of a single segment is just its per-token prob.
            single = fn([(7, -0.3)])
            if single is None or abs(single - math.exp(-0.3)) > 1e-9:
                problems.append(f"single segment gave {single}, expected "
                                f"{math.exp(-0.3):.4f}")

            # (c) clamp: avg_logprob can come back a hair above 0, and "1.22"
            #     shown as a confidence is a visible bug.
            clamped = fn([(5, 0.2)])
            if clamped != 1.0:
                problems.append(f"positive logprob was not clamped to 1.0: {clamped}")

            # (d) NOTHING to pool ⇒ None, never a number. Three ways to get there.
            for label, arg in (("empty list", []),
                               ("zero words", [(0, -0.5)]),
                               ("no avg_logprob", [(5, None)])):
                if fn(arg) is not None:
                    problems.append(f"{label} fabricated a value: {fn(arg)}")

            # (e) a missing avg_logprob is skipped, not counted as 0.0.
            mixed = fn([(5, None), (5, -0.2)])
            if mixed is None or abs(mixed - math.exp(-0.2)) > 1e-9:
                problems.append(f"segment with no logprob polluted the pool: {mixed}")

            rep.expect(cid, not problems,
                       "word-weighted log-domain pooling, exp mapping, clamp at "
                       "1.0, and None (never 0) when there is nothing to pool",
                       "; ".join(problems))
        except Exception as exc:  # noqa: BLE001
            rep.fail(cid, f"could not evaluate the pooling rule: {exc!r}")

    # -- the confidence describes the transcript the USER SEES. A dropped
    #    hallucination must therefore contribute nothing to it — otherwise a
    #    30 s silence read as a very confident "Thank you." would inflate the
    #    number for the real sentence beside it. This runs the REAL
    #    transcribe_path with mlx_whisper stubbed out, so it covers the actual
    #    ordering of the drop decision and the score collection — the thing a
    #    pure-function test of the pooling rule cannot see.
    if ctx.wants("whisper/conf-only-from-kept-segments"):
        cid = "whisper/conf-only-from-kept-segments"
        try:
            import math
            real = ("If we have a one-minute speech to deliver,", -0.2)   # 8 words
            fake = ("Thank you.", -0.01)   # 2 words, 30 s of silence → dropped
            segments = [
                {"text": " " + real[0], "start": 0.0, "end": 3.0,
                 "no_speech_prob": 0.86, "compression_ratio": 1.6,
                 "avg_logprob": real[1]},
                {"text": " " + fake[0], "start": 3.0, "end": 33.0,
                 "no_speech_prob": 0.89, "compression_ratio": 1.0,
                 "avg_logprob": fake[1]},
            ]
            # **kwargs, not a fixed signature: the real call now passes the dict
            # built by whisper_transcribe_kwargs, and this check is about the
            # drop/score ORDERING, not about which decoding options are in it.
            fake_whisper = type("FakeWhisper", (), {
                "transcribe": staticmethod(
                    lambda audio, **kwargs: {
                        "text": real[0] + " " + fake[0], "segments": segments}),
            })()
            logs = []
            fn = extract_nested(
                module, source, "main", "transcribe_path",
                {"mlx_whisper": fake_whisper,
                 "args": type("Args", (), {"model": "mlx-community/whisper-large-v3-mlx"})(),
                 "language": "en",
                 "transcribe_kwargs": module.whisper_transcribe_kwargs(
                     "mlx-community/whisper-large-v3-mlx", "en"),
                 "prompt_words": 0,
                 "load_audio_16k": lambda path: None,
                 "log": logs.append},
            )
            text, conf = fn("/dev/null")

            kept_only = math.exp(real[1])                                  # 0.8187
            both = math.exp((8 * real[1] + 2 * fake[1]) / 10)              # 0.8504
            problems = []
            if fake[0] in text:
                problems.append(f"the dropped hallucination is still in the text: {text!r}")
            if real[0] not in text:
                problems.append(f"the real sentence was lost: {text!r}")
            if not any("drop" in l for l in logs):
                problems.append(f"the drop was not logged (logs: {logs})")
            if conf is None:
                problems.append("no confidence at all for a chunk with a kept segment")
            elif abs(conf - kept_only) > 1e-9:
                problems.append(f"conf {conf:.4f} != kept-only {kept_only:.4f} "
                                f"(counting the dropped segment gives {both:.4f})")
            rep.expect(cid, not problems,
                       f"1 kept + 1 dropped segment ⇒ conf {conf} from the kept "
                       f"segment alone (both would be {both:.4f})",
                       "; ".join(problems))
        except Exception as exc:  # noqa: BLE001
            rep.fail(cid, f"could not run the Whisper branch: {exc!r}")

    # -- THE DRIFT DETECTOR. The owner chose fully standalone services over a
    #    shared protocol module, accepting that the wire format now lives in
    #    several copies. This is what makes that safe: it drives whisper-service.py
    #    and EVERY mlx-audio service (qwen3, granite, voxtral) through the SAME
    #    payload builders and the SAME real FLUSH branch, and fails if their replies
    #    stop agreeing. A protocol edit applied to one file and not the others dies
    #    HERE rather than as a Swift decode that silently drops a key mid-meeting.
    #    Validated once with a negative control: an extra "engine" key and a changed
    #    MIN_CHUNK_SEC in one file made it fail loudly on every drift.
    #
    #    No model load: the builders are pure, and the FLUSH branch is lifted out
    #    by AST with the ASR stubbed (see extract_flush_branch).
    if ctx.wants("whisper/protocol-matches-chunked"):
        cid = "whisper/protocol-matches-chunked"
        try:
            import re
            others = [(label, *load_sidecar_module(service,
                                                   f"mt_{label}_protocol"))
                      for label, service in MLX_SERVICES]
            problems = []

            # (1) the constants that decide WHICH flushes produce a message.
            for name in PROTOCOL_CONSTANTS:
                a = getattr(module, name, None)
                for label, other, _ in others:
                    b = getattr(other, name, None)
                    if a is None or b is None or a != b:
                        problems.append(f"{name}: whisper={a!r} {label}={b!r}")

            # (2) every wire message, byte for byte (key set, key ORDER, rounding).
            key_sets = {}
            for builder, cargs in PROTOCOL_EMIT_CASES:
                mine = capture_stdout(getattr(module, builder), *cargs)
                agreed = True
                for label, other, _ in others:
                    theirs = capture_stdout(getattr(other, builder), *cargs)
                    if mine != theirs:
                        problems.append(f"{builder}{cargs!r}: whisper wrote {mine} "
                                        f"but {label} wrote {theirs}")
                        agreed = False
                if not agreed:
                    continue
                for line in mine:
                    payload = json.loads(line)
                    key_sets.setdefault(payload["type"], set()).update(payload)

            # (3) the top-level key sets, stated explicitly rather than implied by
            #     (2), because these are the exact sets the Swift decoder reads.
            expected_keys = {
                "status": {"type", "text"},
                "error": {"type", "text"},
                # No "words"/"dur": word timestamps left these sidecars for
                # scripts/aligner/aligner-service.py on 2026-07-29. Either key
                # reappearing here means an ASR service has grown an aligner
                # again — which is exactly the drift this check exists to catch.
                "final": {"type", "text", "conf"},
                "file_result": {"type", "id", "text", "conf"},
                "file_error": {"type", "id", "text"},
            }
            for kind, want in expected_keys.items():
                got = key_sets.get(kind)
                if got != want:
                    problems.append(f"{kind} key union was {sorted(got or [])}, "
                                    f"expected {sorted(want)}")

            # (4) the REAL `n == 0` branch of both sidecars, over the same buffer.
            #     The under-MIN_CHUNK_SEC case is the one that must agree most:
            #     a service that emitted a final there (or one that stopped
            #     emitting one above the floor) would desync the app's
            #     pendingChunkWindows FIFO for the rest of the meeting.
            def mask(lines):
                # Log lines carry timings; compare their SHAPE, not the clock.
                return [re.sub(r"[\d.]+", "#", l) for l in lines]

            for case, secs in (("under MIN_CHUNK_SEC", 0.1),
                               ("above MIN_CHUNK_SEC", 1.0)):
                mine = run_flush_branch(module, source, secs, ("hello there", 0.9))
                for label, other, other_source in others:
                    theirs = run_flush_branch(other, other_source, secs,
                                              ("hello there", 0.9))
                    if mine[0] != theirs[0]:
                        problems.append(f"FLUSH {case}: whisper emitted {mine[0]} "
                                        f"but {label} emitted {theirs[0]}")
                    if mask(mine[1]) != mask(theirs[1]):
                        problems.append(f"FLUSH {case}: log shapes differ — "
                                        f"whisper {mask(mine[1])} vs {label} "
                                        f"{mask(theirs[1])}")
                    if theirs[2] != 0:
                        problems.append(f"FLUSH {case}: {label} did not reset its "
                                        f"buffer ({theirs[2]} samples left)")
                if mine[2] != 0:
                    problems.append(f"FLUSH {case}: whisper did not reset its "
                                    f"buffer ({mine[2]} samples left)")
                # And the behaviour itself, not merely that they agree on it.
                emitted = bool(mine[0])
                if emitted != (secs >= module.MIN_CHUNK_SEC):
                    problems.append(f"FLUSH {case}: emitted={emitted} for a "
                                    f"{secs}s buffer")

            rep.expect(cid, not problems,
                       f"{len(PROTOCOL_CONSTANTS)} constants, "
                       f"{len(PROTOCOL_EMIT_CASES)} wire messages byte-identical, "
                       f"{len(expected_keys)} key sets exact, and all "
                       f"{len(others) + 1} FLUSH branches agree above and below "
                       "MIN_CHUNK_SEC",
                       "; ".join(problems))
        except Exception as exc:  # noqa: BLE001
            rep.fail(cid, f"could not compare the sidecars: {exc!r}")

    # -- THE GOVERNING RULE of the decoding-options change: adding the options
    #    must not change a single transcript until a knob is deliberately moved.
    #    At default settings the kwargs must therefore be VALUE-identical to the
    #    old three-argument call — the four optional knobs absent (mlx_whisper's
    #    own None), the thresholds at the library's defaults, word_timestamps
    #    off. The failure this guards is silent: a default that is a hair off
    #    changes every meeting and nothing in the output says why.
    if ctx.wants("whisper/option-defaults-are-todays-behaviour"):
        cid = "whisper/option-defaults-are-todays-behaviour"
        try:
            got = module.whisper_transcribe_kwargs(
                WHISPER_TODAYS_KWARGS["path_or_hf_repo"], "en")
            problems = []
            if got != WHISPER_TODAYS_KWARGS:
                extra = {k: v for k, v in got.items()
                         if k not in WHISPER_TODAYS_KWARGS}
                missing = {k: v for k, v in WHISPER_TODAYS_KWARGS.items()
                           if k not in got}
                differing = {k: (got[k], v) for k, v in WHISPER_TODAYS_KWARGS.items()
                             if k in got and got[k] != v}
                problems.append(f"extra={extra} missing={missing} "
                                f"differing(got, want)={differing}")
            # Stated separately from the dict compare so the message names the
            # option when one of these ever appears at its sentinel.
            for key in ("initial_prompt", "best_of",
                        "hallucination_silence_threshold"):
                if key in got:
                    problems.append(f"{key} is present at defaults ({got[key]!r}) "
                                    "— it must be omitted so mlx_whisper uses None")
            rep.expect(cid, not problems,
                       f"default kwargs are exactly the pre-options call "
                       f"({len(WHISPER_TODAYS_KWARGS)} keys, no optional knobs)",
                       "; ".join(problems))
        except Exception as exc:  # noqa: BLE001
            rep.fail(cid, f"could not build the default kwargs: {exc!r}")

    # -- 0 means OFF and must reach mlx_whisper as None, never as 0. They are not
    #    the same setting: best_of=0 is not "sample nothing", and
    #    hallucination_silence_threshold=0.0 is not "no skipping". Omitting the
    #    key is how None is expressed here (mlx_whisper defaults all three to
    #    None), so the check is that the key is ABSENT and, explicitly, not 0.
    if ctx.wants("whisper/sentinels-become-none"):
        cid = "whisper/sentinels-become-none"
        try:
            got = module.whisper_transcribe_kwargs(
                "repo", "en", best_of=0,
                hallucination_silence_sec=0, initial_prompt="")
            problems = []
            for key in ("best_of", "hallucination_silence_threshold",
                        "initial_prompt"):
                if key in got:
                    problems.append(f"{key}={got[key]!r} was passed through; "
                                    "the sentinel must become None (absent)")
                if got.get(key, None) is not None:
                    problems.append(f"{key} resolves to {got.get(key)!r}, not None")
            if got.get("word_timestamps") is not False:
                problems.append("a 0 hallucination threshold turned word "
                                "timestamps on")
            # And the positive control: non-zero values DO come through, so the
            # check above is not passing merely because everything is dropped.
            on = module.whisper_transcribe_kwargs(
                "repo", "en", best_of=5,
                hallucination_silence_sec=2.0, initial_prompt=" Aggia ")
            for key, want in (("best_of", 5),
                              ("hallucination_silence_threshold", 2.0),
                              ("initial_prompt", "Aggia")):
                if on.get(key) != want:
                    problems.append(f"{key} was {on.get(key)!r}, expected {want!r}")
            # beam_size is NOT a knob here and must stay unrepresentable:
            # mlx_whisper's decoding.py:437 raises NotImplementedError for any
            # beam_size, so a "beam 2" setting could only ever kill the sidecar
            # at load (measured 2026-07-29 — `mlx_whisper.transcribe(...,
            # beam_size=2)` raises immediately, so the warmup fails and the
            # sidecar exits with a load error instead of starting). Pinned
            # so re-adding it has to be a conscious edit to this check too.
            try:
                module.whisper_transcribe_kwargs("repo", "en", beam_size=2)
                problems.append("whisper_transcribe_kwargs accepted beam_size — "
                                "mlx_whisper cannot do beam search at all")
            except TypeError:
                pass
            rep.expect(cid, not problems,
                       "0/\"\" are dropped (⇒ None), real values pass through, "
                       "and beam_size is refused outright",
                       "; ".join(problems))
        except Exception as exc:  # noqa: BLE001
            rep.fail(cid, f"could not evaluate the sentinels: {exc!r}")

    # -- THE COUPLING. mlx_whisper reads hallucination_silence_threshold only
    #    inside its `if word_timestamps:` block, so setting the threshold alone
    #    is accepted and does NOTHING. A setting that appears to work but does
    #    nothing is worse than one that refuses, so the sidecar forces word
    #    timestamps on (and says so in the log and the UI, because that reverses
    #    a measured decision — CLAUDE.md, "word_timestamps MEASURED and
    #    REJECTED"). If this check ever fails, the option has gone silent.
    if ctx.wants("whisper/hallucination-implies-word-timestamps"):
        cid = "whisper/hallucination-implies-word-timestamps"
        try:
            got = module.whisper_transcribe_kwargs("repo", "en",
                                                   hallucination_silence_sec=2.0)
            problems = []
            if got.get("hallucination_silence_threshold") != 2.0:
                problems.append(f"threshold was {got.get('hallucination_silence_threshold')!r}")
            if got.get("word_timestamps") is not True:
                problems.append("word_timestamps was NOT forced on — the "
                                "threshold would be silently ignored by "
                                "mlx_whisper")
            # It must not leak the other way: word timestamps stay off unless
            # the threshold asked for them.
            if module.whisper_transcribe_kwargs("repo", "en")["word_timestamps"]:
                problems.append("word_timestamps is on at default settings")
            rep.expect(cid, not problems,
                       "threshold 2.0s carries word_timestamps=True with it; "
                       "off by default",
                       "; ".join(problems))
        except Exception as exc:  # noqa: BLE001
            rep.fail(cid, f"could not evaluate the coupling: {exc!r}")

    # -- auto-detect must CLEAR the language, not merely be recorded. The flag is
    #    resolved in main() (the app sends --language AND the flag), so this
    #    lifts main()'s own two lines rather than restating the rule.
    if ctx.wants("whisper/autodetect-clears-language"):
        cid = "whisper/autodetect-clears-language"
        try:
            import ast
            problems = []

            def resolve(language_arg: str, auto: bool):
                """Run main()'s real language-resolution statements."""
                tree = ast.parse(source)
                main_fn = next(n for n in tree.body
                               if isinstance(n, ast.FunctionDef) and n.name == "main")
                start = next(i for i, s in enumerate(main_fn.body)
                             if isinstance(s, ast.Assign)
                             and getattr(s.targets[0], "id", "") == "language")
                block = ast.Module(body=main_fn.body[start:start + 2],
                                   type_ignores=[])
                ast.fix_missing_locations(block)
                namespace = {"args": type("Args", (), {
                    "language": language_arg, "auto_detect_language": auto})()}
                exec(compile(block, "<main.language>", "exec"), namespace)  # noqa: S102
                return namespace["language"]

            if resolve("en", True) is not None:
                problems.append("auto-detect did not clear an explicit language")
            if resolve("en", False) != "en":
                problems.append(f"explicit code was lost: {resolve('en', False)!r}")
            if resolve("auto", False) is not None:
                problems.append("language 'auto' did not become None")
            # …and the resolved value is what lands in the kwargs.
            if module.whisper_transcribe_kwargs("repo", None)["language"] is not None:
                problems.append("a None language did not reach the kwargs")
            rep.expect(cid, not problems,
                       "auto-detect ⇒ language=None; off ⇒ the explicit code "
                       "survives",
                       "; ".join(problems))
        except Exception as exc:  # noqa: BLE001
            rep.fail(cid, f"could not evaluate language resolution: {exc!r}")


# ============================================================= aligner group
# scripts/aligner/aligner-service.py — the forced aligner, extracted out of the
# ASR sidecars on 2026-07-29 so that word timestamps became ASYNCHRONOUS (the
# transcript no longer waits on them).
#
# The first two checks are the SAME two that used to run as `chunked/*`; they
# follow the code, because the functions they test now live here. They also got
# CHEAPER and more honest in the move: `pair_source_indices` and `align_chunk`
# are module-level functions taking `(align_proc, log)` / `(aligner, align_proc,
# log)` explicitly, so the checks are plain imports instead of AST extractions
# out of a nested closure.
#
# The third is new and covers the one thing this service's wire format must get
# right: a REJECTED alignment is an `align_result` WITHOUT words, not an
# `align_error` and not an empty list. Rejection is a normal outcome (the gates
# exist to produce it) and the app must not treat it as a failure.
ALIGNER_CHECKS = [
    "aligner/src-indices-skip-punctuation",
    "aligner/words-never-past-chunk",
    "aligner/reply-shape",
]


def run_aligner(rep: Report, ctx):
    try:
        module, _ = load_sidecar_module(ALIGNER_SERVICE, "mt_aligner_service")
    except Exception as exc:  # noqa: BLE001
        for cid in ALIGNER_CHECKS:
            if ctx.wants(cid):
                rep.fail(cid, f"could not import aligner-service.py: {exc!r}")
        return

    # -- "src" indices point into the ORIGINAL text.split(), skipping
    #    punctuation-only source words. The aligner's tokenizer drops a
    #    standalone em-dash, so a naive item->word index shifts every later word
    #    one speaker to the left. That is the exact bug "src" exists to prevent.
    if ctx.wants("aligner/src-indices-skip-punctuation"):
        cid = "aligner/src-indices-skip-punctuation"
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
            text = "we shipped it — and then we tested it"
            words = text.split()
            assert words[3] == "—", "fixture must contain a standalone em-dash"

            class Item:  # minimal stand-in for the aligner's result items
                def __init__(self, t):
                    self.text = t

            tokens, _ = proc.encode_timestamp(text, "English")
            indices = module.pair_source_indices(proc, logs.append, text,
                                                 [Item(t) for t in tokens])

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

    # -- alignment items are never allowed to run past the chunk. Forced
    #    alignment force-fits whatever text it is handed, so a real Whisper
    #    hallucination got stamped after the end of the audio: a few stragglers
    #    are dropped, but a bulk overrun rejects the whole alignment rather than
    #    emitting a truncated guess.
    if ctx.wants("aligner/words-never-past-chunk"):
        cid = "aligner/words-never-past-chunk"
        import numpy as np

        class Item:
            def __init__(self, text, start, end):
                self.text, self.start_time, self.end_time = text, start, end

        def run_align(items):
            """The REAL align_chunk with a fake aligner and an identity src
            mapping, so the only thing under test is the past-the-end gate."""
            fake = type("FakeAligner", (),
                        {"generate": lambda self, a, t, language=None: items})()
            logs = []
            audio = np.zeros(int(10.0 * SR), dtype=np.float32)  # a 10 s chunk
            text = " ".join(it.text for it in items)
            # pair_source_indices is a module global here, so replacing it
            # replaces the very name align_chunk calls. Restored below.
            original = module.pair_source_indices
            module.pair_source_indices = \
                lambda proc, log, t, its: list(range(len(its)))
            try:
                return module.align_chunk(fake, None, logs.append, audio, text), logs
            finally:
                module.pair_source_indices = original

        # (a) one straggler out of twenty: dropped, the rest survive.
        good = [Item(f"w{i}", i * 0.4, i * 0.4 + 0.3) for i in range(19)]
        good.append(Item("hallucination", 10.9, 11.4))
        kept, logs_a = run_align(good)

        # (b) a bulk overrun: no timestamps at all rather than a truncated guess.
        bad = [Item(f"w{i}", i * 0.4, i * 0.4 + 0.3) for i in range(10)]
        bad += [Item(f"h{i}", 11.0 + i, 11.5 + i) for i in range(6)]
        rejected, _ = run_align(bad)

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

    # -- the reply shapes, byte for byte. No model load: the emitters are pure.
    #    The distinction under test is the one the app depends on — rejection is
    #    an align_result with NO words (absent, never []), an error is a
    #    different message type entirely — because the app treats the first as
    #    "keep the estimate" and the second as a failure worth logging.
    if ctx.wants("aligner/reply-shape"):
        cid = "aligner/reply-shape"
        try:
            problems = []
            words = [{"text": "hello", "start": 0.0, "end": 0.4, "src": 0}]

            ok_line = capture_stdout(module.emit_align, 7, words, 30.0)
            none_line = capture_stdout(module.emit_align, 7, None, None)
            err_line = capture_stdout(module.emit_align_error, 7,
                                      "file not found: /nope.wav")
            for label, lines in (("align_result", ok_line),
                                 ("rejected", none_line), ("align_error", err_line)):
                if len(lines) != 1:
                    problems.append(f"{label} wrote {len(lines)} lines")
            ok = json.loads(ok_line[0])
            none = json.loads(none_line[0])
            err = json.loads(err_line[0])

            if set(ok) != {"type", "id", "words", "dur"}:
                problems.append(f"success key set was {sorted(ok)}")
            if ok.get("type") != "align_result" or ok.get("id") != 7:
                problems.append(f"success type/id were {ok.get('type')!r}/{ok.get('id')!r}")
            if ok.get("words") != words or ok.get("dur") != 30.0:
                problems.append("success payload did not carry the words/dur verbatim")

            if set(none) != {"type", "id"}:
                problems.append(f"rejected key set was {sorted(none)} — "
                                "words/dur must be ABSENT, not [] or null")
            if none.get("type") != "align_result":
                problems.append(f"a rejected alignment was sent as "
                                f"{none.get('type')!r}, not align_result")

            if set(err) != {"type", "id", "text"} or err.get("type") != "align_error":
                problems.append(f"error key set was {sorted(err)} "
                                f"(type {err.get('type')!r})")
            rep.expect(cid, not problems,
                       "align_result carries id+words+dur, a rejection carries "
                       "id alone, and an error is align_error",
                       "; ".join(problems))
        except Exception as exc:  # noqa: BLE001
            rep.fail(cid, f"could not evaluate the reply shapes: {exc!r}")


# ================================================================ moss group
# MOSS-Transcribe-Diarize (speaker-attributed ASR). Every check is a pure import
# of the MOSS sidecars — no 3.6 GB model load, milliseconds — for the same reason
# the chunked gate checks are: the parts that can be silently wrong here are
# RULES (a gate's verdict, a wire shape), and a check that needs a model load is
# a check nobody runs.
#
# The first five run against BOTH copies (see MOSS_COPIES); the last three exist
# only because there ARE two copies.
MOSS_CHECKS = [
    "moss/silence-gate-skips-allsilent",
    "moss/refusal-gate-is-narrow",
    "moss/truncation-warning-fires-at-cap",
    "moss/parse-to-wire-shape",
    "moss/final-shape-empty-on-skip",
    "moss/asr-matches-moss",
    "moss/diar-has-no-file-branch",
    "moss/asr-vendor-is-own-and-identical",
]

# Every wire message BOTH MOSS sidecars can produce, as (builder, args). MOSS's
# own shapes, NOT the mlx/Whisper ones: its `final` carries `segments` and never
# `conf`. Compared BYTE FOR BYTE across the copies so key ORDER and the segment
# rounding are covered too.
#
# MOSS deliberately stays OUT of `whisper/protocol-matches-chunked`: that check
# asserts the final key set is {"type","text","conf"}, which is exactly what a
# MOSS final must NOT be. Loosening it to admit MOSS would destroy the thing it
# proves about the other four services.
MOSS_SHARED_EMIT_CASES = [
    ("emit", ("status", "LOADED")),
    ("emit", ("error", "Chunk transcription failed: boom")),
    ("emit_final", ("hello there. hi.",
                    [{"start": 0.0, "end": 4.1, "speaker": "S01", "text": "Hello there."},
                     {"start": 4.4, "end": 5.2, "speaker": "S02", "text": "Hi."}])),
    ("emit_final", ("", [])),
]

# The `-2` FILE-TRANSCRIBE replies, which ONLY the ASR role has (phase 2,
# 2026-07-31 — the diar process is never sent a `-2` frame, see
# `moss/diar-has-no-file-branch`). Having lost its second participant, this half
# can no longer be pinned by cross-copy comparison, so each case carries the
# EXACT LINE it must produce. That keeps the byte-level pin — key set AND key
# ORDER — alive, and it is worth keeping: the `-2` protocol is load-bearing for
# Remote chunks and for overlap-repair re-ASR.
MOSS_ASR_ONLY_EMIT_CASES = [
    (("file_result", 7, "some text"),
     '{"type": "file_result", "id": 7, "text": "some text"}'),
    (("file_result", 7, ""),
     '{"type": "file_result", "id": 7, "text": ""}'),
    (("file_error", 7, "file not found: /nope.wav"),
     '{"type": "file_error", "id": 7, "text": "file not found: /nope.wav"}'),
    (("file_error", None, "file transcription failed: boom"),
     '{"type": "file_error", "id": null, "text": "file transcription failed: boom"}'),
]

# Constants that are part of MOSS's wire/gate behaviour, not implementation
# detail. The two gate constants matter as much as the framing ones: a different
# SILENCE_RMS in one copy means one role calls the model on silence and answers
# as a chatbot while the other does not.
MOSS_PROTOCOL_CONSTANTS = ["SR", "MIN_CHUNK_SEC", "MAX_BUFFER_SEC", "SILENCE_RMS",
                           "MAX_NEW_TOKENS", "REFUSAL_MARKERS", "REFUSAL_PREFIXES",
                           # A per-copy MPS cap would let one role survive a long
                           # chunk while the other died on the same audio — and in
                           # MOSS+MOSS mode both are live at once, so the two caps
                           # ADD against one machine. They must move together.
                           "MPS_MEMORY_CAP_GB"]

# The exact refusal captured on the owner's M4 when the model was handed 30 s of
# digital silence — it answered as an LLM instead of transcribing. The gate that
# must catch it is a BACKSTOP (the silence gate stops the call happening at all),
# so its false-positive risk matters more than its recall: everything in the
# `True` rows below is ordinary meeting speech that must survive, including the
# apology that opens the refusal but is not it.
MOSS_REFUSAL_CASES = [
    # (text, keep?)
    ("I'm sorry, I can't assist with that request. I'm a Qwen-1 model developed "
     "by Qwen-Omni, and I can't process audio.", False),
    ("I can't assist with that request.", False),
    ("I'm a Qwen model.", False),
    ("Okay.", True),
    ("I'm sorry, I missed that.", True),
    ("I'm sorry, could you repeat the question about the quarterly numbers?", True),
    ("So the main challenge that we face is how do we really consolidate so much "
     "important information into just one minute?", True),
    ("He said he couldn't assist with that request, which I thought was odd.", True),
]


def run_moss(rep: Report, ctx):
    # Both copies, loaded once. A missing/broken file fails EVERY moss check
    # rather than silently reducing the loop to one participant — a drift check
    # that quietly stops comparing is worse than no drift check.
    copies = []
    try:
        for label, service, vendor in MOSS_COPIES:
            mod, src = load_sidecar_module(service, f"mt_{label.replace('-', '_')}_service")
            copies.append((label, mod, src, vendor))
    except Exception as exc:  # noqa: BLE001
        for cid in MOSS_CHECKS:
            if ctx.wants(cid):
                rep.fail(cid, f"could not import a MOSS sidecar: {exc!r}")
        return

    # -- check 1: the silence gate. MEASURED failure it guards: 30 s of digital
    #    silence made the model reply "I'm sorry, I can't assist with that
    #    request. I'm a Qwen-1 model…", which would have entered the transcript
    #    as speech. The gate must fire on silence and must NOT fire on audio the
    #    rest of the suite already treats as speech-like.
    if ctx.wants("moss/silence-gate-skips-allsilent"):
        cid = "moss/silence-gate-skips-allsilent"
        try:
            import numpy as np
            problems = []
            digital_silence = np.zeros(30 * SR, dtype="float32")
            speech_like = tone(30.0)
            for label, module, _, _ in copies:
                if module.silence_skip_reason(digital_silence) is None:
                    problems.append(f"{label}: 30 s of digital silence was NOT skipped")
                if module.silence_skip_reason(near_silence(30.0)) is None:
                    problems.append(f"{label}: an idle (near-silent) channel was NOT skipped")
                reason = module.silence_skip_reason(speech_like)
                if reason is not None:
                    problems.append(f"{label}: the tone fixture was skipped as {reason!r}")
                if module.silence_skip_reason(np.zeros(0, dtype="float32")) is None:
                    problems.append(f"{label}: an empty buffer was NOT skipped")
            rep.expect(cid, not problems,
                       f"silence/near-silence skipped, tone kept, in all "
                       f"{len(copies)} copies (threshold {copies[0][1].SILENCE_RMS})",
                       "; ".join(problems))
        except Exception as exc:  # noqa: BLE001
            rep.fail(cid, f"could not evaluate the silence gate: {exc!r}")

    # -- check 2: the refusal backstop, in BOTH directions. Over-deletion is the
    #    dangerous direction (a dropped chunk leaves no trace in the transcript,
    #    only in each role's own log), so the surviving cases are the point of this
    #    check as much as the dropped ones.
    if ctx.wants("moss/refusal-gate-is-narrow"):
        cid = "moss/refusal-gate-is-narrow"
        try:
            wrong = []
            for label, module, _, _ in copies:
                for text, want_keep in MOSS_REFUSAL_CASES:
                    reason = module.refusal_drop_reason(text)
                    if (reason is None) != want_keep:
                        verdict = "kept" if reason is None else f"dropped as {reason}"
                        wrong.append(f"{label}: {text[:50]!r} was {verdict}")
            rep.expect(cid, not wrong,
                       f"all {len(MOSS_REFUSAL_CASES)} refusal/real cases judged "
                       f"correctly in all {len(copies)} copies (real speech survives)",
                       "; ".join(wrong))
        except Exception as exc:  # noqa: BLE001
            rep.fail(cid, f"could not evaluate the refusal gate: {exc!r}")

    # -- check 2b: the truncation warning (added by the 2026-07-31 upstream audit).
    #    Upstream `generate_transcription` returns `generated_tokens`; we discarded
    #    it, and hitting `max_new_tokens` cuts the transcript MID-SEGMENT while
    #    `parse_transcript` drops the incomplete tail WITHOUT raising (measured: a
    #    3-segment transcript cut mid-text parses to 2). So the log line this gate
    #    produces is the ONLY trace such a loss would ever leave — the same reason
    #    every hallucination-gate drop is logged with its reason.
    #
    #    The boundary is `>=`, not `>`: transformers stops AT the cap, so a run
    #    that reached exactly `cap` is the truncated case, not the last good one.
    #    Both directions are asserted, because a gate that cried truncation on
    #    every chunk would train the reader to ignore the one that matters.
    if ctx.wants("moss/truncation-warning-fires-at-cap"):
        cid = "moss/truncation-warning-fires-at-cap"
        try:
            wrong = []
            caps = {}
            for label, module, _, _ in copies:
                cap = module.MAX_NEW_TOKENS
                caps[label] = cap
                cases = [(0, False), (1, False), (cap // 2, False), (cap - 1, False),
                         (cap, True), (cap + 250, True)]
                for generated, want_warning in cases:
                    warning = module.truncation_warning(generated)
                    if (warning is not None) != want_warning:
                        verdict = "silent" if warning is None else "warned"
                        wrong.append(f"{label}: {generated} tokens vs cap {cap} → {verdict}")
                    if warning is not None and str(generated) not in warning:
                        wrong.append(f"{label}: the {generated}-token warning does not "
                                     "name the actual count, so the log cannot show "
                                     "how far past the cap it went")
            # Same cap in both copies. `asr-matches-moss` compares the constant
            # too; here it matters for a different reason — a per-copy cap would
            # make one role truncate while the other did not, on the SAME audio.
            if len(set(caps.values())) > 1:
                wrong.append(f"the copies disagree on MAX_NEW_TOKENS: {caps}")
            rep.expect(cid, not wrong,
                       f"silent below the {sorted(set(caps.values()))[0]}-token cap and "
                       f"warns at or above it, in all {len(copies)} copies",
                       "; ".join(wrong))
        except Exception as exc:  # noqa: BLE001
            rep.fail(cid, f"could not evaluate the truncation gate: {exc!r}")

    # -- check 3: model output → the wire shape the Swift decoder reads. The
    #    vendored parser is exercised too, so a broken/absent
    #    scripts/<service>/vendor/moss_transcribe_diarize fails HERE rather than at
    #    the first chunk of a real meeting (the build.sh `git+`-strip trap).
    if ctx.wants("moss/parse-to-wire-shape"):
        cid = "moss/parse-to-wire-shape"
        try:
            raw = "[00.00][S01]Hello there.[04.10] [04.40][S02]Hi.[05.20]"
            expected = [
                {"start": 0.0, "end": 4.1, "speaker": "S01", "text": "Hello there."},
                {"start": 4.4, "end": 5.2, "speaker": "S02", "text": "Hi."},
            ]
            problems = []
            for label, module, _, vendor in copies:
                # Each copy's OWN vendor tree is exercised, not whichever one
                # happened to be imported first — the point of the check is that
                # a service's parser is really there, so a cached module would
                # let a missing/broken tree pass.
                for name in [m for m in sys.modules
                             if m == "moss_transcribe_diarize"
                             or m.startswith("moss_transcribe_diarize.")]:
                    del sys.modules[name]
                sys.path.insert(0, str(vendor))
                try:
                    from moss_transcribe_diarize import parse_transcript
                    import moss_transcribe_diarize as _mtd
                    if str(vendor) not in _mtd.__file__:
                        problems.append(f"{label}: parser came from {_mtd.__file__}, "
                                        f"not {vendor}")
                    segments = module.wire_segments(parse_transcript(raw))
                finally:
                    sys.path.remove(str(vendor))
                if segments != expected:
                    problems.append(f"{label}: segments were {segments}, expected {expected}")
                for seg in segments:
                    if set(seg) != {"start", "end", "speaker", "text"}:
                        problems.append(f"{label}: unexpected keys {sorted(seg)}")
                # Times are CHUNK-LOCAL — the app adds the window start. A parser
                # that ever returned absolute times would double-offset every row.
                if segments and segments[0]["start"] != 0.0:
                    problems.append(f"{label}: first segment does not start at 0 "
                                    "(not chunk-local)")
                joined = module.joined_text(segments)
                if joined != "Hello there. Hi.":
                    problems.append(f"{label}: joined text was {joined!r}")
                if module.speaker_index("S01") != 1 or module.speaker_index("S12") != 12:
                    problems.append(f"{label}: speaker_index does not map S01→1 / S12→12")
            rep.expect(cid, not problems,
                       f"2 segments, chunk-local floats, exact keys, text joins in "
                       f"order — from each of {len(copies)} services' own vendor tree",
                       "; ".join(problems))
        except Exception as exc:  # noqa: BLE001
            rep.fail(cid, f"could not evaluate the wire shape: {exc!r}")

    # -- check 4: a SKIPPED chunk still emits a final. The app pops one entry off
    #    pendingChunkWindows per final; a silent chunk that emitted nothing would
    #    leave that FIFO one deep forever and misalign every later chunk's window
    #    against the diarization turns.
    if ctx.wants("moss/final-shape-empty-on-skip"):
        cid = "moss/final-shape-empty-on-skip"
        try:
            import io
            import contextlib
            problems = []
            for label, module, _, _ in copies:
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    module.emit_final("", [])
                payload = json.loads(buffer.getvalue().strip())
                if payload != {"type": "final", "text": "", "segments": []}:
                    problems.append(f"{label}: payload was {payload}")
                # Never invent a confidence: MOSS reports none, and absent means
                # "not measured" everywhere in this app.
                if "conf" in payload or "words" in payload:
                    problems.append(f"{label}: a MOSS final carried conf/words")
            rep.expect(cid, not problems,
                       'skipped chunk emits {"type":"final","text":"","segments":[]} '
                       f"in all {len(copies)} copies",
                       "; ".join(problems))
        except Exception as exc:  # noqa: BLE001
            rep.fail(cid, f"could not evaluate the skipped-chunk final: {exc!r}")

    # -- check 5: THE DRIFT DETECTOR for the two MOSS copies. Same job as
    #    `whisper/protocol-matches-chunked` does for the five ASR services, and
    #    the same justification: the owner chose standalone services over a
    #    shared module, so the wire format and the gates now live in two copies
    #    of MOSS as well. A `-2` reply reshaped in one file, or a gate threshold
    #    moved in one file, dies HERE — not as one role behaving differently from
    #    the other in a meeting, which nothing in the output would explain.
    #
    #    No model load: the builders are pure and the FLUSH branch is lifted by
    #    AST with `transcribe` stubbed (extract_flush_branch).
    if ctx.wants("moss/asr-matches-moss"):
        cid = "moss/asr-matches-moss"
        try:
            import re
            import numpy as np
            base_label, base, base_src, _ = copies[0]
            others = copies[1:]
            problems = []

            # (1) the constants that decide what reaches the model and what a
            #     FLUSH produces.
            for name in MOSS_PROTOCOL_CONSTANTS:
                a = getattr(base, name, None)
                for label, other, _, _ in others:
                    b = getattr(other, name, None)
                    if a is None or b is None or a != b:
                        problems.append(f"{name}: {base_label}={a!r} {label}={b!r}")

            # (2) every SHARED wire message, byte for byte across the copies (key
            #     set, key ORDER, the segment rounding).
            key_sets = {}
            for builder, cargs in MOSS_SHARED_EMIT_CASES:
                mine = capture_stdout(getattr(base, builder), *cargs)
                agreed = True
                for label, other, _, _ in others:
                    theirs = capture_stdout(getattr(other, builder), *cargs)
                    if mine != theirs:
                        problems.append(f"{builder}{cargs!r}: {base_label} wrote {mine} "
                                        f"but {label} wrote {theirs}")
                        agreed = False
                if not agreed:
                    continue
                for line in mine:
                    payload = json.loads(line)
                    key_sets.setdefault(payload["type"], set()).update(payload)

            # (2b) the ASR-ONLY `-2` replies. `emit_file` exists in the ASR file
            #      alone, so there is nothing to compare it against — each case is
            #      pinned to its exact line instead, which holds key ORDER as
            #      firmly as the cross-copy comparison did. Deliberately run on
            #      the ASR copy BY NAME rather than on `base`: if MOSS_COPIES is
            #      ever reordered this must still land on the file that has it.
            asr_label, asr_module, _, _ = next(
                (c for c in copies if c[0] == "moss-asr"), (None, None, None, None))
            if asr_module is None:
                problems.append("the moss-asr copy is not in MOSS_COPIES — the "
                                "`-2` protocol has no owner to check")
            elif not hasattr(asr_module, "emit_file"):
                # Reported as a DRIFT, not left to raise inside the try: an
                # AttributeError here would be a crash report, and the thing that
                # actually happened is that the ASR role lost the `-2` protocol
                # Remote and overlap repair ride on.
                problems.append(f"{asr_label} no longer defines emit_file — the `-2` "
                                "FILE-TRANSCRIBE protocol has been deleted from the "
                                "role that needs it")
            elif any(hasattr(m, "emit_file") for l, m, _, _ in copies if l != "moss-asr"):
                problems.append("a non-ASR MOSS copy defines emit_file — the `-2` "
                                "frame belongs to the ASR role alone")
            else:
                for cargs, want in MOSS_ASR_ONLY_EMIT_CASES:
                    got = capture_stdout(asr_module.emit_file, *cargs)
                    if got != [want]:
                        problems.append(f"emit_file{cargs!r}: {asr_label} wrote {got}, "
                                        f"expected [{want!r}]")
                        continue
                    payload = json.loads(got[0])
                    key_sets.setdefault(payload["type"], set()).update(payload)

            # (3) the top-level key sets, stated explicitly rather than implied
            #     by (2) — these are the exact sets the Swift decoder reads, and
            #     they are MOSS's, not the other services'. `segments` on a final
            #     is load-bearing: in MOSS+MOSS mode this one process is the only
            #     thing feeding the diarization rows. `conf`/`words` must never
            #     appear — MOSS measures neither, and absent means absent.
            #     UNCHANGED by the phase-2 split: all five types are still
            #     required, the file_* two now sourced from the ASR copy alone
            #     (2b) rather than from the cross-copy comparison.
            expected_keys = {
                "status": {"type", "text"},
                "error": {"type", "text"},
                "final": {"type", "text", "segments"},
                "file_result": {"type", "id", "text"},
                "file_error": {"type", "id", "text"},
            }
            for kind, want in expected_keys.items():
                got = key_sets.get(kind)
                if got != want:
                    problems.append(f"{kind} key union was {sorted(got or [])}, "
                                    f"expected {sorted(want)}")

            # (4) the REAL `n == 0` branch of both copies, over the same buffers.
            #     Three cases, because MOSS's branch has three outcomes and each
            #     must agree: below the floor, above it but silent (the gate that
            #     stops the model answering as a chatbot), and a real transcribe.
            def mask(lines):
                return [re.sub(r"[\d.]+", "#", l) for l in lines]

            stub = ("hello there. hi.",
                    [{"start": 0.0, "end": 4.1, "speaker": "S01", "text": "Hello there."},
                     {"start": 4.4, "end": 5.2, "speaker": "S02", "text": "Hi."}])

            def flush(module, source, buf):
                """The sidecar's own FLUSH branch over an EXPLICIT buffer.

                `run_flush_branch` builds a zeros buffer, which for MOSS can only
                ever exercise the silence gate — so the transcribing case needs
                the buffer passed in. Same machinery (extract_flush_branch), same
                stubbing rule: only the branch's own decisions are under test.
                """
                logs = []
                namespace = dict(vars(module))
                namespace.update({"buffer": buf.copy(), "np": np, "log": logs.append,
                                  "transcribe": lambda audio: stub})
                lines = capture_stdout(extract_flush_branch(source), namespace)
                return lines, logs, int(namespace["buffer"].size)

            cases = [
                ("under MIN_CHUNK_SEC", np.zeros(int(0.1 * SR), dtype=np.float32), False),
                ("silent above MIN_CHUNK_SEC", np.zeros(SR, dtype=np.float32), False),
                ("real speech", tone(1.0).astype(np.float32), True),
            ]
            for case, buf, wants_text in cases:
                mine = flush(base, base_src, buf)
                for label, other, other_src, _ in others:
                    theirs = flush(other, other_src, buf)
                    if mine[0] != theirs[0]:
                        problems.append(f"FLUSH {case}: {base_label} emitted {mine[0]} "
                                        f"but {label} emitted {theirs[0]}")
                    if mask(mine[1]) != mask(theirs[1]):
                        problems.append(f"FLUSH {case}: log shapes differ — "
                                        f"{base_label} {mask(mine[1])} vs {label} "
                                        f"{mask(theirs[1])}")
                    if theirs[2] != 0:
                        problems.append(f"FLUSH {case}: {label} did not reset its "
                                        f"buffer ({theirs[2]} samples left)")
                if mine[2] != 0:
                    problems.append(f"FLUSH {case}: {base_label} did not reset its "
                                    f"buffer ({mine[2]} samples left)")
                # And the behaviour itself, not merely that they agree on it: a
                # final is ALWAYS emitted (the pendingChunkWindows FIFO must
                # drain), but only the real-speech case may carry text.
                if len(mine[0]) != 1:
                    problems.append(f"FLUSH {case}: emitted {len(mine[0])} lines, "
                                    "expected exactly one final")
                else:
                    payload = json.loads(mine[0][0])
                    if bool(payload.get("text")) != wants_text:
                        problems.append(f"FLUSH {case}: text was {payload.get('text')!r}")
                    if bool(payload.get("segments")) != wants_text:
                        problems.append(f"FLUSH {case}: segments were "
                                        f"{payload.get('segments')!r}")

            rep.expect(cid, not problems,
                       f"{len(MOSS_PROTOCOL_CONSTANTS)} constants, "
                       f"{len(MOSS_SHARED_EMIT_CASES)} wire messages byte-identical "
                       f"across copies + {len(MOSS_ASR_ONLY_EMIT_CASES)} ASR-only "
                       f"`-2` replies pinned to exact lines, "
                       f"{len(expected_keys)} key sets exact, and all "
                       f"{len(copies)} FLUSH branches agree over {len(cases)} cases",
                       "; ".join(problems))
        except Exception as exc:  # noqa: BLE001
            rep.fail(cid, f"could not compare the MOSS copies: {exc!r}")

    # -- check 6: the ONE deliberate behavioural difference between the copies —
    #    the DIARIZATION role has no `-2` FILE-TRANSCRIBE frame and no emit_file.
    #
    #    Why it is pinned rather than left to notice (the
    #    `pyannote/no-path-to-torchcodec` precedent — an invariant that is
    #    invisible in normal use):
    #      * ADDING it back is the tidy-minded mistake. `transcribeFile` is only
    #        ever reached through `modelLoader.chunkedASR`, so the diar process
    #        cannot be sent a `-2` frame; the branch would never run, and dead
    #        branches rot silently.
    #      * REMOVING it from the ASR file is the dangerous mistake, and it is the
    #        over-deletion direction: Remote chunks and overlap-repair re-ASR both
    #        ride `-2`, and a missing branch there stalls a request forever rather
    #        than failing. That file's docstring says nothing may be dropped from
    #        it; this is the check that enforces it.
    #    Both halves are asserted, so the check is PROVEN ABLE TO FAIL in each
    #    direction rather than merely passing on an absence.
    #
    #    By AST, not grep: both files DOCUMENT the `-2` asymmetry at length, so a
    #    textual search matches its own explanation (build.sh's tokenizer gate and
    #    `pyannote/no-path-to-torchcodec` learned this the hard way).
    if ctx.wants("moss/diar-has-no-file-branch"):
        cid = "moss/diar-has-no-file-branch"
        try:
            import ast

            def shape(source):
                """(emit_file defs, `-2` compares in the read loop, `-2` anywhere)."""
                tree = ast.parse(source)
                defs = [n for n in ast.walk(tree)
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and n.name == "emit_file"]

                def is_minus_two(node):
                    return (isinstance(node, ast.UnaryOp)
                            and isinstance(node.op, ast.USub)
                            and isinstance(node.operand, ast.Constant)
                            and node.operand.value == 2)

                def compares(scope):
                    return [n for n in ast.walk(scope) if isinstance(n, ast.Compare)
                            and any(is_minus_two(c) for c in n.comparators)]

                main = next((n for n in ast.walk(tree)
                             if isinstance(n, ast.FunctionDef) and n.name == "main"), None)
                loops = [n for n in ast.walk(main)
                         if isinstance(n, ast.While)] if main is not None else []
                in_loop = [c for loop in loops for c in compares(loop)]
                return len(defs), len(in_loop), len(compares(tree)), main is not None

            shapes = {label: shape(source) for label, _, source, _ in copies}
            problems = []
            for label in ("moss-asr", "moss-diar"):
                if label not in shapes:
                    problems.append(f"{label} is not in MOSS_COPIES — this check "
                                    "cannot compare the roles")
            if not problems:
                # The NEGATIVE half: the diar role must have neither.
                d_defs, d_loop, d_any, d_main = shapes["moss-diar"]
                if not d_main:
                    problems.append("moss-diar has no main() — the read loop could "
                                    "not be located")
                if d_defs:
                    problems.append(f"moss-diar defines emit_file ({d_defs} times) — "
                                    "it is never sent a `-2` frame")
                if d_loop or d_any:
                    problems.append(f"moss-diar compares against -2 ({d_loop} in its "
                                    f"read loop, {d_any} in the file)")
                # The POSITIVE half: the ASR role must have BOTH, which is what
                # makes the assertion above capable of failing at all — and what
                # catches an over-deletion from the file that needs them.
                a_defs, a_loop, a_any, a_main = shapes["moss-asr"]
                if not a_main:
                    problems.append("moss-asr has no main() — the read loop could "
                                    "not be located")
                if a_defs != 1:
                    problems.append(f"moss-asr defines emit_file {a_defs} times, "
                                    "expected exactly 1 (Remote + overlap repair "
                                    "depend on it)")
                if a_loop < 1:
                    problems.append("moss-asr's read loop no longer handles the `-2` "
                                    "FILE-TRANSCRIBE frame")
            rep.expect(cid, not problems,
                       "moss-diar has no emit_file and no `-2` branch, while "
                       "moss-asr has both (so this check can fail in either "
                       "direction)",
                       "; ".join(problems))
        except Exception as exc:  # noqa: BLE001
            rep.fail(cid, f"could not compare the MOSS read loops: {exc!r}")

    # -- check 7: each MOSS service must put its OWN vendor tree on sys.path,
    #    and the trees must be identical.
    #
    #    THE SILENT TRAP this exists for: the two trees are byte-identical and
    #    both present, so either service pointing at the OTHER's vendor folder
    #    WORKS TODAY and breaks only when that folder moves — a failure that would
    #    appear months later, in a release, as one MOSS role refusing to load. It
    #    already caught the phase-2 rename exactly as predicted: moss/vendor became
    #    moss-diar/vendor. Read by AST, NOT by grep: both files DOCUMENT this at length,
    #    so a textual search matches its own explanation (the lesson from
    #    `pyannote/no-path-to-torchcodec` and build.sh's tokenizer gate).
    if ctx.wants("moss/asr-vendor-is-own-and-identical"):
        cid = "moss/asr-vendor-is-own-and-identical"
        try:
            import ast
            import hashlib
            problems = []
            for label, _, source, vendor in copies:
                inserted = []
                for node in ast.walk(ast.parse(source)):
                    if (isinstance(node, ast.Call)
                            and isinstance(node.func, ast.Attribute)
                            and node.func.attr == "insert"
                            and ast.unparse(node.func.value) == "sys.path"):
                        inserted.append([c.value for c in ast.walk(node)
                                         if isinstance(c, ast.Constant)
                                         and isinstance(c.value, str)])
                if len(inserted) != 1:
                    problems.append(f"{label}: {len(inserted)} sys.path.insert calls, "
                                    "expected exactly 1")
                    continue
                parts = inserted[0]
                # The folder is the service's OWN, and it is that folder's vendor.
                if vendor.name not in parts or vendor.parent.name not in parts:
                    problems.append(f"{label}: sys.path.insert names {parts}, "
                                    f"not {vendor.parent.name}/{vendor.name}")
                # The POSITIVE half: the path it names actually exists and holds
                # the package. "No wrong literal" alone would pass a typo.
                if not (vendor / "moss_transcribe_diarize" / "__init__.py").exists():
                    problems.append(f"{label}: {vendor}/moss_transcribe_diarize is missing")

            def tree_hash(vendor):
                pkg = vendor / "moss_transcribe_diarize"
                out = {}
                for path in sorted(pkg.rglob("*.py")):
                    if "__pycache__" in path.parts:
                        continue
                    out[str(path.relative_to(pkg))] = hashlib.sha256(
                        path.read_bytes()).hexdigest()
                return out

            hashes = [(label, tree_hash(vendor)) for label, _, _, vendor in copies]
            base_label, base_hash = hashes[0]
            if not base_hash:
                problems.append(f"{base_label}: vendor tree is empty")
            for label, other in hashes[1:]:
                if other != base_hash:
                    differing = sorted(set(base_hash) ^ set(other)) or \
                        sorted(k for k in base_hash if base_hash[k] != other.get(k))
                    problems.append(f"{base_label} and {label} vendor trees differ: "
                                    f"{differing}")
            rep.expect(cid, not problems,
                       f"each of {len(copies)} services inserts its OWN vendor dir, "
                       f"and the trees are byte-identical ({len(base_hash)} files)",
                       "; ".join(problems))
        except Exception as exc:  # noqa: BLE001
            rep.fail(cid, f"could not check the vendor trees: {exc!r}")


# ============================================================ pyannote group
#
# The PIPELINE half of the 2026-07-30 split. Everything here is about what the
# pipeline may and may not say: spans and run-local labels, never identity.
PYANNOTE_CHECKS = [
    "pyannote/telemetry-is-off",
    "pyannote/no-path-to-torchcodec",
    "pyannote/local-labels-only",
    "pyannote/absent-stream-means-office",
]


def run_offline_guards(rep: Report, ctx):
    """Both pyannote-side sidecars must stay OFFLINE and ffmpeg-free.

    Two real, shipped defects, found 2026-07-30 by auditing pyannote.audio 4.0.7:

    1. pyannote 4.x ships OpenTelemetry tracing ON BY DEFAULT
       (`telemetry/config.yaml: metrics_enabled: true`) and posts to
       `https://otel.pyannote.ai/v1/traces` on every model init, pipeline init and
       every single `apply()` — carrying the AUDIO DURATION of a client meeting.
       `HF_HUB_OFFLINE` does not cover it. That breaks hard requirement #1.
    2. pyannote hands a file PATH to torchcodec, whose dylibs resolve ffmpeg
       through one LC_RPATH of `/opt/homebrew/opt/ffmpeg/lib`. No client Mac has
       that, and there is no fallback — diarization died at the first chunk in the
       packaged `.app`. `torchaudio.load` is the same trap (a torchcodec wrapper
       since TorchAudio 2.9).

    Both are invisible in normal use ON THIS MACHINE — #1 succeeds silently, #2
    only fails where Homebrew ffmpeg is absent — which is exactly why they need
    pinning here rather than being left to notice.

    Pure: module import only, no model load, no audio. Milliseconds.
    """
    import ast

    cid = "pyannote/telemetry-is-off"
    if ctx.wants(cid):
        # BOTH sidecars in ONE check, reported once: the id is the invariant
        # ("nothing pyannote-side phones home"), and one id per invariant is this
        # suite's whole bookkeeping discipline.
        results, problems = [], []
        for service, mod_name in ((PYANNOTE_SERVICE, "mt_pyannote_offline"),
                                  (WESPEAKER_SERVICE, "mt_wespeaker_offline")):
            previous = os.environ.get("PYANNOTE_METRICS_ENABLED")
            # Force the DANGEROUS value FIRST, so a sidecar that merely
            # `setdefault`s inherits "true" and gets caught. That is precisely the
            # failure "assign, don't setdefault" exists to prevent, and a check
            # starting from a clean environment could not see it.
            os.environ["PYANNOTE_METRICS_ENABLED"] = "true"
            try:
                load_sidecar_module(service, mod_name)
                value = os.environ.get("PYANNOTE_METRICS_ENABLED")
                from pyannote.audio.telemetry.metrics import is_metrics_enabled
                live = is_metrics_enabled()
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{service} would not import: {exc!r}")
                continue
            finally:
                if previous is None:
                    os.environ.pop("PYANNOTE_METRICS_ENABLED", None)
                else:
                    os.environ["PYANNOTE_METRICS_ENABLED"] = previous
            results.append(f"{pathlib.Path(service).name}={value!r}")
            if value != "false" or live:
                problems.append(f"{service} left PYANNOTE_METRICS_ENABLED={value!r} "
                                f"(is_metrics_enabled()={live})")
        rep.expect(cid, not problems,
                   f"both sidecars forced metrics off over an inherited 'true' "
                   f"({', '.join(results)})",
                   "; ".join(problems) + " — pyannote would post meeting durations "
                   "to otel.pyannote.ai, breaking the 100%-offline requirement")

    cid = "pyannote/no-path-to-torchcodec"
    if ctx.wants(cid):
        # AST, not grep: both files DOCUMENT this trap at length, so a textual
        # search matches its own explanation. Only real calls count.
        offenders = []
        for service in (PYANNOTE_SERVICE, WESPEAKER_SERVICE):
            source = (SCRIPTS / service).read_text()
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Attribute):
                    continue
                if (node.attr in ("load", "info")
                        and isinstance(node.value, ast.Name)
                        and node.value.id == "torchaudio"):
                    offenders.append(f"{service}:{node.lineno} torchaudio.{node.attr}")
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id == "AudioDecoder":
                    offenders.append(f"{service}:{node.lineno} AudioDecoder")
        # And the positive half: pyannote must be CALLED with a waveform mapping.
        pyannote_src = (SCRIPTS / PYANNOTE_SERVICE).read_text()
        hands_waveform = '"waveform"' in pyannote_src and "sf.read(" in pyannote_src
        rep.expect(cid, not offenders and hands_waveform,
                   "neither sidecar calls torchaudio.load/info or AudioDecoder, and "
                   "pyannote-service decodes with soundfile into a waveform mapping",
                   f"offenders={offenders or 'none'}, "
                   f"pyannote-service hands over a waveform: {hands_waveform} — "
                   f"a path-based decode needs Homebrew ffmpeg, which no client Mac has")


def run_pyannote(rep: Report, ctx):
    run_offline_guards(rep, ctx)

    wanted = [c for c in PYANNOTE_CHECKS
              if ctx.wants(c) and c not in ("pyannote/telemetry-is-off",
                                            "pyannote/no-path-to-torchcodec")]
    if not wanted:
        return

    audio_path = None
    if ctx.audio_diarize:
        clip = load_clip(ctx.audio_diarize, ctx.clip_start, ctx.diarize_sec)
        audio_path = write_wav(pathlib.Path(ctx.tmp) / "pyannote-fixture.wav", clip)
    if audio_path is None:
        for cid in wanted:
            rep.skip(cid, "needs a real speech recording (--audio-diarize); none "
                          "found in recordings/")
        return

    # MT_PROFILE_DIR is pointed at a temp dir even though this process has NO
    # profile store — see SAFETY in the module docstring. If it ever grows one,
    # the fence is already there.
    profile_dir = pathlib.Path(tempfile.mkdtemp(prefix="mt-pyannote-", dir=ctx.tmp))
    sc = Sidecar(PYANNOTE_SERVICE,
                 env_extra={"MT_PROFILE_DIR": str(profile_dir)}, text_stdin=True)
    try:
        ready = sc.wait_for(lambda m: m.get("type") in ("status", "error"),
                            timeout=ctx.load_timeout)
        if ready is None or ready.get("type") == "error":
            for cid in wanted:
                rep.fail(cid, f"sidecar never reported LOADED: {ready}")
            return

        # One real diarization run answers both checks.
        sc.send_json({"cmd": "final", "audio": audio_path})
        office = sc.wait_for(lambda m: m.get("type") in ("result", "error"),
                            timeout=ctx.job_timeout)

        # -- IDENTITY CANNOT LEAK BACK OUT OF THE PIPELINE STAGE. This is the
        #    structural half of the split: the process that decides WHERE the
        #    speech is has no embedder and no profile store, so it cannot be the
        #    thing that writes a voice into one. A segment carrying "id"/"name"/
        #    "conf" would mean that separation had quietly been undone.
        if ctx.wants("pyannote/local-labels-only"):
            cid = "pyannote/local-labels-only"
            if office is None or office.get("type") == "error":
                rep.fail(cid, f"office job failed: {office}")
            else:
                segs = office["segments"]
                bad_keys = sorted({k for s in segs for k in s} - {"start", "end", "label"})
                unlabelled = [s for s in segs if not str(s.get("label", "")).strip()]
                # Unrounded on purpose: the identity stage slices the waveform with
                # these numbers, so rounding here would move each slice by up to
                # ~8 samples and change the embedding. Rounding happens once, in
                # AudioRecorder.composeTurns.
                rounded = [s for s in segs
                           if s["start"] == round(s["start"], 3)
                           and s["end"] == round(s["end"], 3)]
                problems = []
                if not segs:
                    problems.append("the pipeline produced no turns at all")
                if bad_keys:
                    problems.append(f"segments carried non-pipeline keys {bad_keys} "
                                    f"— identity leaked back into the pipeline stage")
                if unlabelled:
                    problems.append(f"{len(unlabelled)} segments had no label")
                if segs and len(rounded) == len(segs):
                    problems.append("every time was already at 3 dp — the wire looks "
                                    "rounded, which would change the identity "
                                    "stage's waveform slices")
                rep.expect(cid, not problems,
                           f"{len(segs)} segments, keys exactly "
                           f"{{start,end,label}}, "
                           f"{len(segs) - len(rounded)}/{len(segs)} times at full "
                           f"precision, labels like "
                           f"{sorted({s['label'] for s in segs})[:3]}",
                           "; ".join(problems))

        # -- a job with no "stream" key is an office job and its reply has the
        #    pre-dual-stream shape — no "stream" echoed back. The key set now
        #    includes "audio", which is LOAD-BEARING rather than incidental: it is
        #    how the app knows which file to hand the identity stage, with no
        #    second bookkeeping map. Re-pinned against the new shape; the intent
        #    (office echoes nothing, the key set is fixed) is what carried over
        #    from the merged sidecar's version of this check.
        if ctx.wants("pyannote/absent-stream-means-office"):
            cid = "pyannote/absent-stream-means-office"
            if office is None or office.get("type") == "error":
                rep.fail(cid, f"office job failed: {office}")
            else:
                sc.send_json({"cmd": "final", "audio": audio_path, "stream": "remote"})
                remote = sc.wait_for(lambda m: m.get("type") in ("result", "error"),
                                     timeout=ctx.job_timeout)
                problems = []
                if set(office) != {"type", "segments", "audio"}:
                    problems.append(f"office reply key set was {sorted(office)}")
                if office.get("audio") != audio_path:
                    problems.append(f"office reply did not echo the audio path: "
                                    f"{office.get('audio')!r}")
                if remote is None or remote.get("type") == "error":
                    problems.append(f"remote job failed: {remote}")
                elif set(remote) != {"type", "segments", "audio", "stream"}:
                    problems.append(f"remote reply key set was {sorted(remote)}")
                elif remote.get("stream") != "remote":
                    problems.append("remote reply did not echo stream=remote")
                rep.expect(cid, not problems,
                           'office reply was exactly {"type","segments","audio"} '
                           f'with {len(office["segments"])} turns and no "stream" '
                           'key; the remote job added exactly "stream"',
                           "; ".join(problems))
    finally:
        sc.close()


# ============================================================ spectral group
#
# The THIRD diarization engine (vendored `diarize`, 2026-08-03). Everything here
# is PURE — AST, module import and hashes, no model load and no audio, the
# `pyannote/telemetry-is-off` and `layout/*` precedent. Milliseconds.
SPECTRAL_CHECKS = [
    "spectral/protocol-matches-pyannote",
    "spectral/telemetry-is-off",
    "spectral/no-live-chunk-branch",
    "spectral/vendor-is-own",
    "spectral/vad-reader-is-shimmed",
    "spectral/rejects-zero-frame-audio",
]

# ============================================================== tools group
#
# Utilities under scripts/tools/ that the app does not launch but that recover
# data when something has already gone wrong. They get checks for the same
# reason the sidecars do: nobody runs them until a bad day, and a repair tool
# that has quietly stopped working is worse than none — it is reached exactly
# when the alternative has already failed.
TOOLS_CHECKS = [
    "tools/wav-header-repair",
]

# The vendored tree, file by file, as SHA-256.
#
# PROVENANCE, and what these hashes are: the four upstream-verbatim files and the
# LICENSE were fetched from github.com/FoxNoseTech/diarize at commit
# `4f25d27dee54f7e8264a914e705f7cee182151e2` (`src/diarize/`) and hashed when this
# check was written — they are UPSTREAM's bytes, not merely a fingerprint of
# whatever happened to be on disk. `embeddings.py` is the ONE modified file
# (Apache-2.0 §4(b): the ONNX `wespeakerruntime` backend swapped for this
# project's PyTorch WeSpeaker), so BOTH hashes are recorded: upstream's, which the
# vendored copy must NOT equal, and ours, which it must.
#
# The hashes live HERE, in the check, deliberately: the suite must run air-gapped
# (client hard requirement #1), so re-fetching upstream at check time is not an
# option. What this pins is therefore drift SINCE vendoring — an "upgrade" that
# silently edits an upstream file, a second local modification slipped into a file
# whose header still claims to be verbatim, or our shim quietly disappearing.
SPECTRAL_VENDOR_UPSTREAM = {
    "__init__.py": "ad8dc4bd4e3fc9cb36ba4abf184744d62a6c9f5eee995797b52d3de8bd077835",
    "clustering.py": "5a6c55ae182aa2858b6a94284660836caa9d3a258623bf6c0d1fa6e8fda1c29d",
    "utils.py": "5c98305606c6473b73306314f80e78119c90d0a8f2958791fa1311259f9f2b01",
    "vad.py": "9281e28a6e97a8c51e7961ca094d872ba48c91ed5d2b59fa7cbe9d45b7334a4b",
    # Apache-2.0 §4(a) requires the licence travel with the copy. It is checked
    # like any other upstream file so "tidying" it away fails loudly.
    "LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
}
SPECTRAL_VENDOR_MODIFIED = {
    # (upstream, ours). The pair is what makes the claim checkable in both
    # directions: equalling upstream would mean the ONNX backend is back and the
    # engine would try to download its own weights at runtime; equalling neither
    # means a second, undocumented edit.
    #
    # OURS updated 2026-08-05 (was d47cd324…) for the SECOND §4(b) modification:
    # the full-file read is `dtype="float32"` instead of soundfile's float64
    # default. Both modifications are listed in that file's module docstring, and
    # both are documented at their own line. The decode was measured
    # BIT-IDENTICAL either way and whole-pipeline output byte-identical on four
    # recordings before the hash was moved — a pin is only worth updating with the
    # evidence that the change was safe, otherwise it just records whatever
    # happened.
    "embeddings.py": ("564c72615dad725b0afbfe1101c4ae787a518d1f77e34c02a290b02f80da571d",
                      "85ca47467379d01388d499746a06c9b0195d2fd7d4bd1d3d4568836ce9dcdbb2"),
}
# Ours, not upstream's. Presence is pinned but NOT its hash: it is this project's
# own file and may legitimately change (a device, a resample), whereas an upstream
# file changing is by definition a defect.
SPECTRAL_VENDOR_OURS = "torch_embedder.py"


def _emit_shapes(source: str):
    """Every `emit({...})` call in a diarization sidecar, by AST.

    Returns a list of (type, key tuple IN ORDER, splices **echo, inside the
    stdin read loop). The replies of these two services are built as DICT
    LITERALS at the emit site rather than by a named builder — there is no
    `emit_final` to call — so the literal is where the wire shape actually lives
    and the literal is what gets read.
    """
    import ast
    tree = ast.parse(source)
    main = next((n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef) and n.name == "main"), None)
    loop = next((n for n in ast.walk(main) if isinstance(n, ast.For)
                 and ast.unparse(n.iter) == "sys.stdin"), None) if main else None
    in_loop = {id(n) for n in ast.walk(loop)} if loop is not None else set()

    shapes = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "emit" and len(node.args) == 1
                and isinstance(node.args[0], ast.Dict)):
            continue
        d = node.args[0]
        keys = tuple(k.value for k in d.keys
                     if isinstance(k, ast.Constant) and isinstance(k.value, str))
        echo = any(k is None and ast.unparse(v) == "echo"
                   for k, v in zip(d.keys, d.values))
        kind = next((ast.literal_eval(v) for k, v in zip(d.keys, d.values)
                     if isinstance(k, ast.Constant) and k.value == "type"
                     and isinstance(v, ast.Constant)), None)
        shapes.append((kind, keys, echo, id(node) in in_loop))
    return shapes


def _fn_source(source: str, name: str):
    """One top-level function's normalised source, for a verbatim comparison."""
    import ast
    tree = ast.parse(source)
    fn = next((n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name == name), None)
    return None if fn is None else ast.unparse(fn)


def run_spectral(rep: Report, ctx):
    import ast

    # -- check 1: THE DRIFT DETECTOR, `whisper/protocol-matches-chunked` and
    #    `moss/asr-matches-moss` applied to the two turns-only diarizers.
    #
    #    Both sidecars answer the SAME Swift caller (`DiarizationService`), so a
    #    reply key added, renamed or reordered in one and not the other is a
    #    decode that silently drops a field for whichever engine was missed —
    #    exactly the failure the standalone-services decision accepted and this
    #    kind of check is the mitigation for. `spectral` is the third copy of a
    #    protocol that already lives in two.
    #
    #    TIMING PRECISION IS DELIBERATELY OUT OF SCOPE, and this is the one place
    #    where this check must NOT copy `pyannote/local-labels-only`. That check
    #    asserts pyannote's times are NOT all at 3 dp, i.e. that nothing rounded
    #    the wire (the identity stage slices the waveform with those floats).
    #    Spectral's segment times really are 1-decimal — because Silero's
    #    `get_speech_timestamps` defaults to `time_resolution=1`, UPSTREAM, not
    #    because anything here rounds them — so a "looks rounded" heuristic would
    #    false-positive on a correct sidecar. What is shared and therefore pinned
    #    here is SHAPE: the key sets, the key order, the label field, the
    #    absent-stream-means-office rule, the `audio` echo, and the fact that a
    #    segment can never carry identity.
    if ctx.wants("spectral/protocol-matches-pyannote"):
        cid = "spectral/protocol-matches-pyannote"
        try:
            spec_mod, spec_src = load_sidecar_module(SPECTRAL_SERVICE, "mt_spectral_protocol")
            pyan_mod, pyan_src = load_sidecar_module(PYANNOTE_SERVICE, "mt_pyannote_protocol")
            problems = []

            # (1) the shared plumbing, verbatim. These four are the same functions
            #     in both files; a divergence here is how one engine starts
            #     writing a different JSON line or a differently shaped log.
            for name in ("emit", "fail", "brief_traceback", "log"):
                a, b = _fn_source(pyan_src, name), _fn_source(spec_src, name)
                if a is None or b is None or a != b:
                    problems.append(f"{name}(): pyannote={'absent' if a is None else 'differs'}"
                                    f" vs spectral={'absent' if b is None else 'differs'}")

            # (2) the wire bytes themselves. `emit` is the only builder these
            #     services have, so the payloads are handed to BOTH modules and
            #     compared byte for byte — key order and json separators included.
            wire_cases = [
                {"type": "status", "text": "LOADED"},
                {"type": "error", "text": "Audio not found: /nope.wav"},
                {"type": "result", "audio": "/tmp/m.wav",
                 "segments": [{"start": 0.5, "end": 1.25, "label": "SPEAKER_00"}]},
                {"type": "result", "audio": "/tmp/m.wav", "segments": [],
                 "stream": "remote"},
            ]
            for payload in wire_cases:
                mine = capture_stdout(pyan_mod.emit, payload)
                theirs = capture_stdout(spec_mod.emit, payload)
                if mine != theirs:
                    problems.append(f"emit({payload['type']}): pyannote wrote {mine} "
                                    f"but spectral wrote {theirs}")

            # (3) the REAL `wire_segments` of each, lifted by AST and driven with
            #     the input its own engine produces (pyannote: (start, end, label)
            #     tuples; spectral: the vendored `DiarizeResult`). Different
            #     inputs, and the output must be the SAME JSON — which is what
            #     pins {start,end,label}, the key ORDER, and that neither can emit
            #     id/name/conf. Identity leaking out of a pipeline stage is the
            #     structural failure the pyannote/wespeaker split exists to make
            #     impossible, and it must be impossible for this engine too.
            import types
            pyan_wire = extract_nested(pyan_mod, pyan_src, "main", "wire_segments", {})
            spec_wire = extract_nested(spec_mod, spec_src, "main", "wire_segments", {})
            turns = [(0.5, 1.25, "SPEAKER_00"), (1.25, 3.0, "SPEAKER_01")]
            result = types.SimpleNamespace(segments=[
                types.SimpleNamespace(start=s, end=e, speaker=l) for s, e, l in turns])
            a_json = json.dumps(pyan_wire(turns))
            b_json = json.dumps(spec_wire(result))
            if a_json != b_json:
                problems.append(f"wire_segments: pyannote produced {a_json} but "
                                f"spectral produced {b_json}")
            for label, produced in (("pyannote", a_json), ("spectral", b_json)):
                for banned in ('"id"', '"name"', '"conf"'):
                    if banned in produced:
                        problems.append(f"{label}'s segments carry {banned} — identity "
                                        f"leaked back into a pipeline stage")

            # (4) the reply SHAPES, read out of the emit sites by AST. Only the
            #     types both services can produce are compared: pyannote also has
            #     `chunk_result`, which spectral must NOT have — that asymmetry is
            #     `spectral/no-live-chunk-branch`'s subject, not a drift.
            pyan_shapes, spec_shapes = _emit_shapes(pyan_src), _emit_shapes(spec_src)
            for label, shapes in (("pyannote", pyan_shapes), ("spectral", spec_shapes)):
                result_shapes = {(k, e) for kind, k, e, _ in shapes if kind == "result"}
                if result_shapes != {(("type", "audio", "segments"), True)}:
                    problems.append(f"{label}: `result` emit shapes were {result_shapes}, "
                                    f'expected exactly ("type","audio","segments") in '
                                    f"that order with **echo — `audio` is load-bearing "
                                    f"(it is how the app knows which file to hand the "
                                    f"identity stage)")
                status_shapes = {k for kind, k, _, _ in shapes if kind == "status"}
                if status_shapes != {("type", "text")}:
                    problems.append(f"{label}: `status` emit shapes were {status_shapes}")
                errors = {k for kind, k, _, _ in shapes if kind == "error"}
                if errors != {("type", "text")}:
                    problems.append(f"{label}: `error` emit shapes were {errors}, "
                                    f'expected only ("type","text")')
                # ABSENT MEANS OFFICE: every reply raised once the job's stream is
                # known must splice `**echo`, so an office reply carries no
                # "stream" key at all and a remote one carries exactly that. The
                # sole exception in both files is the bad-job-line reply, which is
                # emitted BEFORE `stream` has been read.
                naked = [(kind, k) for kind, k, echo, loop in shapes if loop and not echo]
                if len(naked) != 1:
                    problems.append(f"{label}: {len(naked)} replies inside the read loop "
                                    f"do not splice **echo ({naked}) — expected exactly "
                                    f"one (the bad-job-line reply, emitted before the "
                                    f"job's stream is known)")

            # (5) and the office/remote rule itself, verbatim across the files.
            #     The three statements that implement "absent means office" are
            #     one line each; comparing their source is what stops one engine
            #     defaulting to remote, or echoing "office" back explicitly and
            #     changing the single-stream wire bytes.
            def echo_rule(source):
                tree = ast.parse(source)
                return [ast.unparse(n) for n in ast.walk(tree)
                        if isinstance(n, ast.Assign)
                        and ast.unparse(n.targets[0]) in ("stream", "is_remote", "echo")]

            a_rule, b_rule = echo_rule(pyan_src), echo_rule(spec_src)
            if a_rule != b_rule or len(a_rule) != 3:
                problems.append(f"the stream/echo rule differs: pyannote {a_rule} vs "
                                f"spectral {b_rule}")

            rep.expect(cid, not problems,
                       f"4 shared functions verbatim, {len(wire_cases)} wire messages "
                       f"byte-identical, wire_segments agrees from either engine's own "
                       f"input, the shared reply shapes are exact, and the "
                       f"absent-stream-means-office rule is the same 3 statements "
                       f"(segment TIMES are deliberately not compared — spectral's "
                       f"1 dp is Silero's time_resolution, upstream)",
                       "; ".join(problems))
        except Exception as exc:  # noqa: BLE001
            rep.fail(cid, f"could not compare spectral with pyannote: {exc!r}")

    # -- check 2: the engine's embedder IS a pyannote model, so pyannote 4.x's
    #    default-ON OpenTelemetry applies here exactly as it does to
    #    pyannote-service.py — a network call describing a client meeting, which
    #    hard requirement #1 forbids. `HF_HUB_OFFLINE` does not cover it.
    #    The `pyannote/telemetry-is-off` precedent, including its most important
    #    detail: the DANGEROUS value is forced into the environment FIRST, so a
    #    sidecar that merely `setdefault`s inherits "true" and is caught.
    if ctx.wants("spectral/telemetry-is-off"):
        cid = "spectral/telemetry-is-off"
        problems, results = [], []
        previous = os.environ.get("PYANNOTE_METRICS_ENABLED")
        os.environ["PYANNOTE_METRICS_ENABLED"] = "true"
        try:
            load_sidecar_module(SPECTRAL_SERVICE, "mt_spectral_offline")
            value = os.environ.get("PYANNOTE_METRICS_ENABLED")
            from pyannote.audio.telemetry.metrics import is_metrics_enabled
            live = is_metrics_enabled()
            results.append(f"spectral-service.py={value!r}")
            if value != "false" or live:
                problems.append(f"spectral-service left PYANNOTE_METRICS_ENABLED="
                                f"{value!r} (is_metrics_enabled()={live})")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"spectral-service would not import: {exc!r}")
        finally:
            if previous is None:
                os.environ.pop("PYANNOTE_METRICS_ENABLED", None)
            else:
                os.environ["PYANNOTE_METRICS_ENABLED"] = previous

        # The vendored shim is asserted TOO, and this is a judgement call worth
        # stating. The sidecar's own line is the guarantee for the shipped app: it
        # runs first, unconditionally, before anything imports the engine. But
        # `vendor/diarize/torch_embedder.py` is the file that actually imports
        # pyannote, and it is OURS, so it is the last line of defence for every
        # OTHER entry point — this suite, `scripts/tools/`, a future service
        # reusing the embedder — none of which run the sidecar's module body. It
        # is checked by AST rather than by importing it: an import here would pull
        # in numpy/soundfile for nothing and could not tell `setdefault` from an
        # assignment anyway, which is the whole distinction being pinned.
        shim = SPECTRAL_VENDOR / "diarize" / SPECTRAL_VENDOR_OURS
        try:
            assigns, setdefaults = [], []
            for node in ast.walk(ast.parse(shim.read_text())):
                if (isinstance(node, ast.Assign)
                        and ast.unparse(node.targets[0])
                        == "os.environ['PYANNOTE_METRICS_ENABLED']"):
                    assigns.append(ast.literal_eval(node.value))
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "setdefault"
                        and ast.unparse(node.func.value) == "os.environ"
                        and node.args
                        and getattr(node.args[0], "value", None)
                        == "PYANNOTE_METRICS_ENABLED"):
                    setdefaults.append(ast.unparse(node))
            if assigns != ["false"]:
                problems.append(f"{SPECTRAL_VENDOR_OURS} assigns "
                                f"PYANNOTE_METRICS_ENABLED = {assigns} — expected "
                                f"exactly one assignment of 'false'")
            if setdefaults:
                problems.append(f"{SPECTRAL_VENDOR_OURS} uses setdefault for "
                                f"PYANNOTE_METRICS_ENABLED ({setdefaults}) — an "
                                f"inherited 'true' would then win")
            else:
                results.append(f"{SPECTRAL_VENDOR_OURS}: assigned, not setdefault")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"could not read {shim}: {exc!r}")

        rep.expect(cid, not problems,
                   f"the sidecar forced metrics off over an inherited 'true', and the "
                   f"vendored embedder shim assigns it too ({', '.join(results)})",
                   "; ".join(problems) + " — pyannote would post meeting durations to "
                   "otel.pyannote.ai, breaking the 100%-offline requirement")

    # -- check 3: v1 is FINAL-ONLY, and pyannote is not. The
    #    `moss/diar-has-no-file-branch` precedent, for the same reason and with
    #    the same both-directions discipline.
    #
    #      * ADDING a live/chunk branch here is the symmetry-minded mistake, and
    #        it is the DANGEROUS one: this engine counts speakers globally
    #        (GMM-BIC over every embedding in the file) and then clusters
    #        globally, so a 30 s window would be counted and clustered on its own
    #        and its labels would mean nothing across windows. That failure is
    #        already documented on MOSS — two different people both numbered S01
    #        because the chunk boundary fell on the speaker change — and it is
    #        SILENT: the rows look continuous and are not.
    #      * REMOVING pyannote's is the other direction, and the positive half
    #        below is what makes this check able to fail at all. Without it, a
    #        pyannote-service that had stopped being a live diarizer entirely
    #        would sail through.
    #
    #    By AST, not grep: BOTH files explain this asymmetry at length in prose
    #    (spectral's module docstring is largely about it), so a textual search
    #    matches its own explanation. Note also that spectral's refusal message
    #    does not contain the word "chunk" as a compared constant — the refusal is
    #    `mode != "final"`, so only Compare nodes and dict keys are read, never
    #    message text.
    if ctx.wants("spectral/no-live-chunk-branch"):
        cid = "spectral/no-live-chunk-branch"
        try:
            def live_shape(source):
                """(job.get keys, constants compared against, emitted types, dict keys)."""
                tree = ast.parse(source)
                gets, compared, dict_keys = set(), set(), set()
                for node in ast.walk(tree):
                    if (isinstance(node, ast.Call)
                            and isinstance(node.func, ast.Attribute)
                            and node.func.attr == "get"
                            and ast.unparse(node.func.value) == "job"
                            and node.args
                            and isinstance(node.args[0], ast.Constant)):
                        gets.add(node.args[0].value)
                    if isinstance(node, ast.Compare):
                        for c in node.comparators:
                            if isinstance(c, ast.Constant) and isinstance(c.value, str):
                                compared.add(c.value)
                    if isinstance(node, ast.Dict):
                        dict_keys |= {k.value for k in node.keys
                                      if isinstance(k, ast.Constant)
                                      and isinstance(k.value, str)}
                kinds = {kind for kind, _, _, _ in _emit_shapes(source)}
                return gets, compared, kinds, dict_keys

            s_gets, s_cmp, s_kinds, s_keys = live_shape(
                (SCRIPTS / SPECTRAL_SERVICE).read_text())
            p_gets, p_cmp, p_kinds, p_keys = live_shape(
                (SCRIPTS / PYANNOTE_SERVICE).read_text())
            problems = []

            # The NEGATIVE half: no live path anywhere in spectral.
            if "chunk" in s_cmp:
                problems.append('spectral compares a command against "chunk" — a '
                                "windowed pass through a globally-clustering engine "
                                "returns labels that look continuous and are not")
            if "chunk_result" in s_kinds or "chunk_result" in s_keys:
                problems.append("spectral emits a chunk_result reply")
            if "window_start" in s_gets or "window_start" in s_keys:
                problems.append("spectral reads or emits window_start — the only "
                                "reason to know a window's offset is to serve one")
            # …and that the refusal really is there, so "no chunk branch" is a
            # deliberate refusal rather than an unhandled command falling through
            # to the whole-file pass.
            if "final" not in s_cmp:
                problems.append("spectral never compares the command against "
                                '"final" — a non-final job would fall through to '
                                "the whole-file pass instead of being refused")

            # The POSITIVE half: pyannote really does have the live path this
            # engine is declining to copy.
            if "chunk" not in p_cmp:
                problems.append('pyannote no longer compares a command against '
                                '"chunk" — its live per-30s path is the thing '
                                "spectral is being contrasted with")
            if "chunk_result" not in p_kinds:
                problems.append("pyannote no longer emits chunk_result")
            if "window_start" not in p_gets:
                problems.append("pyannote no longer reads window_start")

            rep.expect(cid, not problems,
                       "spectral has no chunk comparison, no chunk_result and no "
                       "window_start, and refuses any cmd != final — while pyannote "
                       "has all three (so this check can fail in either direction)",
                       "; ".join(problems))
        except Exception as exc:  # noqa: BLE001
            rep.fail(cid, f"could not compare the read loops: {exc!r}")

    # -- check 4: the service inserts its OWN vendor dir, and that tree is still
    #    the upstream one it claims to be.
    #
    #    The `moss/asr-vendor-is-own-and-identical` precedent for the first half:
    #    a `sys.path.insert` pointing at ANOTHER service's vendor folder works
    #    today and breaks months later, in a release, the moment that folder
    #    moves. Read by AST — and, better than reading the literal, EVALUATED, so
    #    the assertion is about the directory that will really be on sys.path
    #    rather than about how it was spelled.
    #
    #    The second half is provenance, which MOSS did not need (its two trees
    #    check each other). There is only one spectral tree, so it is checked
    #    against recorded UPSTREAM hashes instead, with `embeddings.py` pinned as
    #    the ONE deviation in BOTH directions.
    if ctx.wants("spectral/vendor-is-own"):
        cid = "spectral/vendor-is-own"
        try:
            source = (SCRIPTS / SPECTRAL_SERVICE).read_text()
            problems = []
            inserts = [n for n in ast.walk(ast.parse(source))
                       if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                       and n.func.attr == "insert"
                       and ast.unparse(n.func.value) == "sys.path"]
            if len(inserts) != 1:
                problems.append(f"{len(inserts)} sys.path.insert calls, expected exactly 1")
            else:
                expr = ast.unparse(inserts[0].args[1])
                try:
                    # `BASE` is bound the way the sidecar itself computes it, so a
                    # BASE-rooted spelling (what the MOSS services use) resolves
                    # here too and is reported as the WRONG DIRECTORY rather than
                    # as an unresolvable expression. A negative control pointing
                    # this at moss-asr/vendor should name that folder.
                    resolved = pathlib.Path(eval(  # noqa: S307 — our own source, no input
                        expr, {"os": os, "BASE": str(PROJECT),
                               "__file__": str(SCRIPTS / SPECTRAL_SERVICE)}))
                except Exception as exc:  # noqa: BLE001
                    resolved = None
                    problems.append(f"sys.path.insert argument {expr!r} could not be "
                                    f"resolved: {exc!r}")
                if resolved is not None and resolved.resolve() != SPECTRAL_VENDOR.resolve():
                    problems.append(f"sys.path.insert resolves to {resolved} — it must "
                                    f"be this service's OWN {SPECTRAL_VENDOR}; pointing "
                                    f"at another service's vendor tree works until that "
                                    f"folder moves")
                # Any other service's folder named in the literal is wrong even if
                # the resolution somehow agreed.
                named = {c.value for c in ast.walk(inserts[0])
                         if isinstance(c, ast.Constant) and isinstance(c.value, str)}
                strays = sorted(named - {"vendor", "spectral", "scripts"})
                if strays:
                    problems.append(f"sys.path.insert names {strays}")

            pkg = SPECTRAL_VENDOR / "diarize"
            expected = (set(SPECTRAL_VENDOR_UPSTREAM) | set(SPECTRAL_VENDOR_MODIFIED)
                        | {SPECTRAL_VENDOR_OURS})
            found = {p.name for p in pkg.iterdir()
                     if p.is_file() and "__pycache__" not in p.parts}
            if found != expected:
                problems.append(f"vendor tree file set is {sorted(found)}, expected "
                                f"{sorted(expected)}")

            def digest(name):
                path = pkg / name
                return (hashlib.sha256(path.read_bytes()).hexdigest()
                        if path.exists() else None)

            for name, want in sorted(SPECTRAL_VENDOR_UPSTREAM.items()):
                got = digest(name)
                if got != want:
                    problems.append(f"{name} is NOT upstream's file "
                                    f"({got} != {want}) — the tree is vendored "
                                    f"verbatim apart from embeddings.py")
            for name, (upstream, ours) in sorted(SPECTRAL_VENDOR_MODIFIED.items()):
                got = digest(name)
                if got == upstream:
                    problems.append(f"{name} is back to UPSTREAM's version — the ONNX "
                                    f"wespeakerruntime backend downloads its own "
                                    f"weights at runtime, which the offline "
                                    f"requirement forbids")
                elif got != ours:
                    problems.append(f"{name} matches neither upstream nor the recorded "
                                    f"modification ({got}) — a second, undocumented "
                                    f"edit to a vendored file")
            if not (pkg / SPECTRAL_VENDOR_OURS).exists():
                problems.append(f"{SPECTRAL_VENDOR_OURS} is missing — it is the PyTorch "
                                f"WeSpeaker shim embeddings.py imports")

            # And the shape of the one modification, so "only embeddings.py
            # differs" is a statement about WHAT differs, not only about how many
            # bytes moved: it must import our shim and must not import the ONNX
            # runtime under any name.
            emb = ast.parse((pkg / "embeddings.py").read_text())
            imports = set()
            for node in ast.walk(emb):
                if isinstance(node, ast.ImportFrom):
                    imports.add(f"{'.' * (node.level or 0)}{node.module or ''}")
                elif isinstance(node, ast.Import):
                    imports |= {a.name for a in node.names}
            if ".torch_embedder" not in imports:
                problems.append("embeddings.py no longer imports .torch_embedder")
            if any("wespeakerruntime" in i for i in imports):
                problems.append("embeddings.py imports wespeakerruntime again — that is "
                                "the ONNX backend the modification exists to remove")

            rep.expect(cid, not problems,
                       f"the service puts its OWN {SPECTRAL_VENDOR.parent.name}/vendor "
                       f"on sys.path, and the tree is upstream 4f25d27d "
                       f"({len(SPECTRAL_VENDOR_UPSTREAM)} files byte-identical incl. "
                       f"LICENSE) with embeddings.py the ONE deviation, importing our "
                       f"shim and no ONNX runtime",
                       "; ".join(problems))
        except Exception as exc:  # noqa: BLE001
            rep.fail(cid, f"could not check the vendored tree: {exc!r}")

    # -- check 5: the VAD reader is still shimmed onto soundfile.
    #
    #    THE FFMPEG LESSON, and the one copy of it that NO existing gate can see.
    #    `vendor/diarize/vad.py` calls `silero_vad.read_audio(path)`; on the
    #    installed torchaudio (2.11) that resolves to `torchaudio.load`, i.e. a
    #    **torchcodec** wrapper, whose dylibs reach libavcodec/libavformat only
    #    through a single `LC_RPATH` of `/opt/homebrew/opt/ffmpeg/lib` — and the
    #    2026-07-30 pyannote audit PRUNES torchcodec from both bundled
    #    interpreters. So `spectral-service.py` rebinds `silero_vad.read_audio` to
    #    a soundfile+scipy reader, and without that rebinding the packaged `.app`
    #    has no decoder for VAD at all.
    #
    #    Why this needs its own pin: `build.sh`'s `assert_no_torchcodec_use` reads
    #    the sidecars' TOKENS for `torchaudio.load` / `torchcodec` / `AudioDecoder`.
    #    `silero_vad.read_audio(path)` contains none of them, so deleting the shim
    #    passes that gate cleanly. It also passes on the owner's Mac, which has
    #    Homebrew ffmpeg — the invisible-until-shipped class
    #    `pyannote/no-path-to-torchcodec` exists for. Proved both directions in a
    #    bundle-shaped tree with `torchcodec` shadowed by an unimportable stub:
    #    WITH the shim, 9 turns / 5 speakers in 3.6 s; WITHOUT it, importing the
    #    vendored `run_vad` raised `RuntimeError: torchaudio version 2.11.0
    #    requires torchcodec for audio I/O`.
    #
    #    ORDERING — what is actually load-bearing, and what deliberately is NOT.
    #    The requirement is only that the rebinding happen before the first
    #    `run_diarize` CALL. It is NOT "before the `diarize` package is imported",
    #    because `vad.py` does its `from silero_vad import … read_audio` INSIDE
    #    `run_vad`, so the attribute is looked up on the package at call time and
    #    a late rebinding is picked up. Asserting import-order would therefore
    #    forbid a rearrangement that is perfectly correct. But that looseness is a
    #    property of the VENDORED file, not of ours — so the function-local import
    #    is itself asserted below. If a future re-vendor hoists it to module
    #    scope, the ordering requirement TIGHTENS silently, and this check is what
    #    says so.
    #
    #    By AST, not grep: `install_soundfile_vad_reader`'s docstring explains this
    #    trap at length (torchcodec, ffmpeg and `torchaudio.load` all appear in
    #    prose), so a textual search matches its own explanation — the
    #    `pyannote/no-path-to-torchcodec` and build-gate rule.
    if ctx.wants("spectral/vad-reader-is-shimmed"):
        cid = "spectral/vad-reader-is-shimmed"
        try:
            source = (SCRIPTS / SPECTRAL_SERVICE).read_text()
            tree = ast.parse(source)
            problems = []

            # (a) the rebinding itself: an assignment to `silero_vad.read_audio`.
            binds = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                     for t in n.targets
                     if isinstance(t, ast.Attribute) and t.attr == "read_audio"
                     and isinstance(t.value, ast.Name) and t.value.id == "silero_vad"]
            installer = None
            if not binds:
                problems.append("spectral-service.py never assigns "
                                "silero_vad.read_audio — the vendored VAD would "
                                "decode through torchaudio.load → torchcodec, which "
                                "is pruned from both bundled interpreters")
            else:
                # Which function holds it, so the call site can be located.
                for fn in ast.walk(tree):
                    if isinstance(fn, ast.FunctionDef) and any(
                            id(b) in {id(x) for x in ast.walk(fn)} for b in binds):
                        if fn.name != "main":
                            installer = fn
                            break
                if installer is None:
                    problems.append("the silero_vad.read_audio assignment is not "
                                    "inside a named installer function, so its "
                                    "ordering against run_diarize cannot be read")

            # (b) POSITIVE half — the replacement really decodes with soundfile.
            if installer is not None:
                reads_sf = any(isinstance(n, ast.Call)
                               and isinstance(n.func, ast.Attribute)
                               and n.func.attr == "read"
                               and isinstance(n.func.value, ast.Name)
                               and n.func.value.id == "sf"
                               for n in ast.walk(installer))
                imports_sf = any(isinstance(n, ast.Import)
                                 and any(a.name == "soundfile" for a in n.names)
                                 for n in ast.walk(installer))
                if not (reads_sf and imports_sf):
                    problems.append(f"{installer.name} does not decode with soundfile "
                                    f"(imports soundfile: {imports_sf}, calls sf.read: "
                                    f"{reads_sf}) — a shim that swaps one "
                                    f"ffmpeg-dependent reader for another fixes nothing")

            # (c) NEGATIVE half — no banned decode anywhere in the sidecar. Whole
            #     file, not just the shim: a banned call is wrong wherever it sits.
            banned = []
            for node in ast.walk(tree):
                if (isinstance(node, ast.Attribute) and node.attr in ("load", "info")
                        and isinstance(node.value, ast.Name)
                        and node.value.id == "torchaudio"):
                    banned.append(f"line {node.lineno}: torchaudio.{node.attr}")
                if isinstance(node, ast.Name) and node.id == "AudioDecoder":
                    banned.append(f"line {node.lineno}: AudioDecoder")
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "read_audio"
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "silero_vad"):
                    banned.append(f"line {node.lineno}: calls silero_vad.read_audio "
                                  "(the original, torchcodec-backed reader)")
            if banned:
                problems.append("banned decode path in spectral-service.py — "
                                + "; ".join(banned))

            # (d) ORDERING: installed before the first whole-file pass. The engine
            #     entry point is read from its import alias rather than assumed, so
            #     renaming it does not silently make this half vacuous.
            engine = next((a.asname or a.name for n in ast.walk(tree)
                           if isinstance(n, ast.ImportFrom) and n.module == "diarize"
                           for a in n.names if a.name == "diarize"), None)
            if engine is None:
                problems.append("no `from diarize import diarize` — the engine entry "
                                "point this ordering is measured against is gone")
            elif installer is not None:
                calls = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Name) and n.func.id == installer.name]
                runs = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Name) and n.func.id == engine]
                if not calls:
                    problems.append(f"{installer.name} is defined but never called — "
                                    f"the shim exists and does nothing")
                elif runs and min(calls) > min(runs):
                    problems.append(f"{installer.name} is first called at line "
                                    f"{min(calls)}, AFTER {engine}() at line "
                                    f"{min(runs)} — the first job would decode "
                                    f"through torchcodec")

            # (e) the vendored side of the contract, which is what makes (d)'s
            #     looseness legitimate AND makes this check self-invalidating: if
            #     upstream stops calling `silero_vad.read_audio`, the shim is no
            #     longer the right fix and should be re-derived, not kept passing.
            vad_tree = ast.parse((SPECTRAL_VENDOR / "diarize" / "vad.py").read_text())
            run_vad = next((n for n in ast.walk(vad_tree)
                            if isinstance(n, ast.FunctionDef) and n.name == "run_vad"),
                           None)
            if run_vad is None:
                problems.append("vendor/diarize/vad.py has no run_vad")
            else:
                local_import = any(isinstance(n, ast.ImportFrom)
                                   and n.module == "silero_vad"
                                   and any(a.name == "read_audio" for a in n.names)
                                   for n in ast.walk(run_vad))
                calls_it = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                               and n.func.id == "read_audio"
                               for n in ast.walk(run_vad))
                if not calls_it:
                    problems.append("vendor/diarize/vad.py no longer calls read_audio — "
                                    "the shim rebinds something nothing uses; re-derive "
                                    "the fix against the new upstream instead of "
                                    "keeping this check green")
                if not local_import:
                    problems.append("vendor/diarize/vad.py imports read_audio at module "
                                    "scope, not inside run_vad — late rebinding is no "
                                    "longer picked up, so the shim must now run BEFORE "
                                    "`diarize` is imported and this check's ordering "
                                    "half is too loose")

            rep.expect(cid, not problems,
                       "spectral-service rebinds silero_vad.read_audio to a soundfile "
                       "reader before the engine runs, calls no torchaudio.load / "
                       "AudioDecoder / original read_audio, and the vendored run_vad "
                       "still imports read_audio inside the function (which is why "
                       "late rebinding works)",
                       "; ".join(problems))
        except Exception as exc:  # noqa: BLE001
            rep.fail(cid, f"could not read the VAD shim: {exc!r}")

    # A recording whose WAV header declares ZERO FRAMES must be REFUSED, not
    # answered with an empty result.
    #
    # THE FAILURE (2026-08-05). `AVAudioFile` writes the `data` chunk size only on
    # release, so an app killed mid-recording leaves it at 0 over a file full of
    # audio. VAD then finds no speech in an empty array and this engine returned
    # `segments: []` — a whole meeting reported as having no speakers, silently.
    # pyannote already fails loudly on the same file (its pipeline raises), so
    # spectral was the ONE engine that was quiet, and quiet is the direction this
    # project treats as dangerous.
    #
    # By AST, and the ordering half is the point: a guard that runs AFTER
    # `run_diarize` would never fire. The message is also required to name the
    # repair tool — an error that does not say the audio is recoverable would send
    # the user to delete the file.
    if ctx.wants("spectral/rejects-zero-frame-audio"):
        cid = "spectral/rejects-zero-frame-audio"
        try:
            tree = ast.parse((SCRIPTS / SPECTRAL_SERVICE).read_text())
            problems = []

            zero_tests = [n for n in ast.walk(tree) if isinstance(n, ast.Compare)
                          and isinstance(n.left, ast.Name) and n.left.id == "frames"
                          and any(isinstance(c, ast.Constant) and c.value == 0
                                  for c in n.comparators)]
            if not zero_tests:
                problems.append("nothing compares a frame count against 0 — an "
                                "unreadable recording would come back as `segments: []`")

            emits = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Name) and n.func.id == "emit"]
            guard_emits = [n for n in emits
                           if any(isinstance(k, ast.Constant)
                                  and isinstance(k.value, str)
                                  and "repair-wav-header" in k.value
                                  for a in n.args if isinstance(a, ast.Dict)
                                  for k in a.values)]
            if not guard_emits:
                problems.append("no error names scripts/tools/repair-wav-header.py — "
                                "the user is told the recording is empty but not that "
                                "it is recoverable")

            runs = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id == "run_diarize"]
            if zero_tests and runs and min(n.lineno for n in zero_tests) > min(runs):
                problems.append("the zero-frame guard sits AFTER run_diarize — it "
                                "could never fire before the pass returns nothing")

            rep.expect(cid, not problems,
                       "a 0-frame recording is refused with an error that names the "
                       "repair tool, and the guard runs before run_diarize",
                       "; ".join(problems))
        except Exception as exc:  # noqa: BLE001
            rep.fail(cid, f"could not read the zero-frame guard: {exc!r}")


# =========================================================== wespeaker group
#
# The IDENTITY half of the split: the WeSpeaker embedder and BOTH profile stores.
# None of these checks loads the pyannote pipeline — the spans are FABRICATED,
# which is the whole point of the split being testable this way. They used to cost
# a pipeline load each.
WESPEAKER_CHECKS = [
    "wespeaker/assign-conf-matched-vs-new",
    "wespeaker/absent-stream-means-office",
    "wespeaker/profile-stores-are-disjoint",
    "wespeaker/native-rate-final-matches-16k-chunks",
    "wespeaker/reset-wipes-both-stores",
]


def run_assign_conf(rep: Report, ctx):
    """`ProfileStore.assign` reports the winning cosine — and NOTHING for a new
    voice.

    The speaker confidence shown in the transcript is that cosine. The failure
    this pins is the tempting one: making a brand-new profile report 0.0 (or
    1.0) so every turn has a number. A first appearance was never scored against
    anything, and "0.00" in front of a correctly-identified speaker reads as
    "this is definitely the wrong person".

    Pure: no pipeline, no embedder, no audio — the real ProfileStore over a
    throwaway PROFILE_DIR and hand-built vectors, so it runs in milliseconds.
    """
    cid = "wespeaker/assign-conf-matched-vs-new"
    tmp_profiles = pathlib.Path(tempfile.mkdtemp(prefix="mt-assign-conf-", dir=ctx.tmp))
    try:
        import numpy as np
        # PROFILE_DIR is read from MT_PROFILE_DIR at IMPORT time, so point the
        # env at the temp dir for exactly that moment and put it back after —
        # nothing this process spawns later should inherit it.
        previous = os.environ.get("MT_PROFILE_DIR")
        os.environ["MT_PROFILE_DIR"] = str(tmp_profiles)
        try:
            module, _ = load_sidecar_module(WESPEAKER_SERVICE, "mt_wespeaker_conf")
        finally:
            if previous is None:
                os.environ.pop("MT_PROFILE_DIR", None)
            else:
                os.environ["MT_PROFILE_DIR"] = previous
        if pathlib.Path(module.PROFILE_DIR) != tmp_profiles:
            rep.fail(cid, f"module PROFILE_DIR is {module.PROFILE_DIR}, not the "
                          f"temp dir — REFUSING to touch the real profiles")
            return

        rng = np.random.default_rng(7)
        base = rng.standard_normal(192).astype("float32")
        store = module.ProfileStore(np)
        problems = []

        # (1) first voice ever: a new profile, and NO similarity to report.
        first = store.assign({"SPEAKER_00": base})
        entry = first.get("SPEAKER_00")
        if not (isinstance(entry, tuple) and len(entry) == 2):
            problems.append(f"assign did not return a (pid, conf) pair: {first!r}")
        else:
            new_pid, new_conf = entry
            if new_conf is not None:
                problems.append(f"a brand-new profile reported conf={new_conf!r} — "
                                "it was never scored against anything")
            if new_pid != 1:
                problems.append(f"first profile id was {new_pid}, expected 1")

        # (2) the same voice again (nudged): matched, conf = that cosine.
        similar = (base + 0.15 * rng.standard_normal(192)).astype("float32")
        expected = store._cosine(similar, store.centroids[1])
        second = store.assign({"SPEAKER_00": similar})
        pid2, conf2 = second["SPEAKER_00"]
        if pid2 != 1:
            problems.append(f"the same voice minted profile {pid2} instead of matching 1")
        elif conf2 is None:
            problems.append("a MATCHED voice reported no confidence at all")
        elif abs(conf2 - expected) > 1e-6:
            problems.append(f"conf {conf2:.4f} is not the winning cosine {expected:.4f}")
        elif not (module.SIM_THRESHOLD <= conf2 <= 1.0):
            problems.append(f"matched conf {conf2:.4f} outside "
                            f"[{module.SIM_THRESHOLD}, 1.0]")

        # (3) a genuinely different voice: new profile, again no conf.
        other = rng.standard_normal(192).astype("float32")
        third = store.assign({"SPEAKER_01": other})
        pid3, conf3 = third["SPEAKER_01"]
        if pid3 == 1:
            problems.append("an unrelated vector matched the existing profile — "
                            "fixture is too weak to prove anything")
        if conf3 is not None:
            problems.append(f"a second new profile reported conf={conf3!r}")

        # (4) a local id may never reach REMOTE_ID_BASE. Nothing enforced this
        #     before the split; it was merely true because nobody has had 10 000
        #     speakers. The offset is applied at the process boundary, so a local
        #     id of 10 000 would leave here indistinguishable from a remote id and
        #     the app would start writing an office voice into profiles-remote.json
        #     — the silent, permanent corruption the two stores exist to prevent.
        store.profiles.append({"id": module.REMOTE_ID_BASE - 1,
                               "name": "edge", "count": 1})
        store.centroids[module.REMOTE_ID_BASE - 1] = rng.standard_normal(192).astype("float32")
        try:
            store.assign({"SPEAKER_02": rng.standard_normal(192).astype("float32")})
            problems.append(f"minting local id {module.REMOTE_ID_BASE} was allowed — "
                            "the office and remote id spaces would collide")
        except RuntimeError:
            pass  # refused, as it must be

        rep.expect(cid, not problems,
                   f"matched voice → (1, {conf2:.3f}); both new profiles → conf "
                   f"None; local id {module.REMOTE_ID_BASE} refused",
                   "; ".join(problems))
    except Exception as exc:  # noqa: BLE001
        rep.fail(cid, f"could not evaluate assign(): {exc!r}")


def fabricated_turns(seconds: float, labels=("SPEAKER_00", "SPEAKER_01")):
    """Spans over a real clip, WITHOUT loading the pipeline.

    The identity service is handed spans by somebody else — that is the contract
    the split created — so these checks supply them directly. Each label gets a
    contiguous block comfortably over MIN_EMBED_SEC, plus one deliberate sliver
    that must come back unidentifiable.
    """
    block = (seconds - 1.0) / len(labels)
    turns = []
    for i, label in enumerate(labels):
        start = 0.5 + i * block
        turns.append({"start": round(start, 3),
                      "end": round(start + block - 0.2, 3),
                      "label": label})
    turns.append({"start": round(seconds - 0.45, 3),
                  "end": round(seconds - 0.05, 3),
                  "label": "SPEAKER_SLIVER"})
    return turns


def run_wespeaker(rep: Report, ctx):
    wanted = [c for c in WESPEAKER_CHECKS if ctx.wants(c)]
    if not wanted:
        return

    # No model, no audio — run it before the fixture gate below so a machine with
    # no recordings still proves the conf contract.
    if ctx.wants("wespeaker/assign-conf-matched-vs-new"):
        run_assign_conf(rep, ctx)
        wanted = [c for c in wanted if c != "wespeaker/assign-conf-matched-vs-new"]
        if not wanted:
            return

    # EVERY identity run is pointed at a throwaway profile dir. The real one
    # holds the owner's voices; see SAFETY in the module docstring.
    profile_dir = pathlib.Path(tempfile.mkdtemp(prefix="mt-profiles-", dir=ctx.tmp))
    audio_path = None
    if ctx.audio_diarize:
        clip = load_clip(ctx.audio_diarize, ctx.clip_start, ctx.diarize_sec)
        audio_path = write_wav(pathlib.Path(ctx.tmp) / "wespeaker-fixture.wav", clip)

    if audio_path is None:
        for cid in wanted:
            rep.skip(cid, "needs a real speech recording (--audio-diarize); none "
                          "found in recordings/ — PROFILE-STORE SEPARATION IS UNTESTED")
        return

    seconds = clip.size / SR
    turns = fabricated_turns(seconds)

    sc = Sidecar(WESPEAKER_SERVICE,
                 env_extra={"MT_PROFILE_DIR": str(profile_dir)}, text_stdin=True)
    next_id = [0]

    def identify(audio, spans, stream=None, timeout=None):
        next_id[0] += 1
        job = {"cmd": "identify", "id": next_id[0], "audio": audio, "turns": spans}
        if stream:
            job["stream"] = stream
        sc.send_json(job)
        return sc.wait_for(lambda m: m.get("type") in ("identify_result", "error"),
                           timeout=timeout or ctx.job_timeout)

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

        office = identify(audio_path, turns)

        # -- an identify job with no "stream" key is an office job, and its reply
        #    echoes NOTHING back: the request `id` is what correlates it. The
        #    merged sidecar's version of this check pinned "no stream echo on an
        #    office job" plus "conf is a real cosine where present, never a bare 0
        #    standing in for not-measured". Both are re-pinned here against the
        #    identify reply shape.
        if ctx.wants("wespeaker/absent-stream-means-office"):
            cid = "wespeaker/absent-stream-means-office"
            if office is None or office.get("type") == "error":
                rep.fail(cid, f"office identify failed: {office}")
            else:
                speakers = office["speakers"]
                named = {k: v for k, v in speakers.items() if v is not None}
                bad_conf = [v for v in named.values()
                            if "conf" in v and not (isinstance(v["conf"], (int, float))
                                                    and not isinstance(v["conf"], bool)
                                                    and -1.0 <= v["conf"] <= 1.0)]
                bad_keys = sorted({k for v in named.values() for k in v}
                                  - {"id", "name", "conf"})
                problems = []
                if set(office) != {"type", "id", "speakers"}:
                    problems.append(f"reply key set was {sorted(office)}")
                if office.get("id") != 1:
                    problems.append(f"the request id was not correlated back: "
                                    f"{office.get('id')!r}")
                if not named:
                    problems.append("no voice was identified at all")
                if bad_keys:
                    problems.append(f"an identity carried unexpected keys {bad_keys}")
                if bad_conf:
                    problems.append(f"invalid conf values: "
                                    f"{[v.get('conf') for v in bad_conf][:3]}")
                # A first appearance must carry NO conf key — this is the empty
                # store, so every identity here is brand new.
                with_conf = [k for k, v in named.items() if "conf" in v]
                if with_conf:
                    problems.append(f"brand-new profiles reported a confidence: "
                                    f"{with_conf}")
                if speakers.get("SPEAKER_SLIVER") is not None:
                    problems.append("a 0.4 s span was identified — MIN_EMBED_SEC "
                                    "did not gate it")
                rep.expect(cid, not problems,
                           'reply was exactly {"type","id","speakers"} with '
                           f'{len(named)} identified voices, no "stream" key, no '
                           'conf on a first appearance, and the sliver null',
                           "; ".join(problems))

        # -- the two stores cannot cross-contaminate. The SAME audio and the SAME
        #    spans are sent again as a remote job: if the stores were shared it
        #    would match the office profiles and create nothing at all. It must
        #    instead build its own R-named identities in its own file.
        if ctx.wants("wespeaker/profile-stores-are-disjoint"):
            cid = "wespeaker/profile-stores-are-disjoint"
            remote = identify(audio_path, turns, stream="remote")
            if office is None or office.get("type") == "error":
                rep.fail(cid, f"office identify failed, cannot compare: {office}")
            elif remote is None or remote.get("type") == "error":
                rep.fail(cid, f"remote identify failed: {remote}")
            else:
                def read_json(name):
                    path = profile_dir / name
                    return json.loads(path.read_text()) if path.exists() else None

                office_ids = {v["id"] for v in office["speakers"].values() if v}
                remote_ids = {v["id"] for v in remote["speakers"].values() if v}
                remote_names = {v["name"] for v in remote["speakers"].values() if v}
                office_file = read_json("profiles.json") or []
                remote_file = read_json("profiles-remote.json")

                problems = []
                if "stream" in remote:
                    problems.append("a SUCCESS reply echoed stream — the id is what "
                                    "correlates it")
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

        # -- the sample-rate regression. WeSpeaker is a 16 kHz model and does not
        #    error on another rate — it silently embeds pitch- and tempo-shifted
        #    speech. Live chunks arrive as 16 kHz temp WAVs, but a stop-time pass
        #    reads the RECORDING, which is at the capture device's native rate
        #    (44.1 kHz here). On the owner's audio the same voice scored 0.98 on a
        #    chunk and 0.11 on the final, fell under SIM_THRESHOLD and minted a
        #    duplicate profile — one person shown as two speakers.
        #
        #    So: enrol from 16 kHz audio, then identify the SAME speech from a
        #    44.1 kHz file with the SAME spans. It must land on the SAME profiles.
        #    Without the resample in resolve_speakers this yields new ids, which is
        #    exactly the shipped bug. No pipeline is involved either way, which is
        #    why this check now costs seconds instead of two diarization runs.
        if ctx.wants("wespeaker/native-rate-final-matches-16k-chunks"):
            cid = "wespeaker/native-rate-final-matches-16k-chunks"
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
                native_path = write_wav(pathlib.Path(ctx.tmp) / "wespeaker-native.wav",
                                        up, sr=native_sr)

                enrol = identify(audio_path, turns, stream="remote")
                enrolled = ({v["id"] for v in enrol["speakers"].values() if v}
                            if enrol and enrol.get("type") == "identify_result" else set())

                fin = identify(native_path, turns, stream="remote",
                               timeout=ctx.final_timeout)
                final_ids = ({v["id"] for v in fin["speakers"].values() if v}
                             if fin and fin.get("type") == "identify_result" else set())

                if not enrolled:
                    rep.fail(cid, "the 16 kHz pass enrolled no speaker — nothing to compare")
                elif not final_ids:
                    rep.fail(cid, f"the {native_sr} Hz pass identified nobody: {fin}")
                else:
                    unknown = sorted(final_ids - enrolled)
                    rep.expect(cid, not unknown,
                               f"{native_sr} Hz pass reused the 16 kHz profiles "
                               f"{sorted(enrolled)}",
                               f"it minted {unknown} instead of reusing "
                               f"{sorted(enrolled)} — the same voice embedded at "
                               f"{native_sr} Hz did not match its own 16 kHz profile")
            except Exception as exc:  # noqa: BLE001
                rep.fail(cid, f"check could not run: {exc!r}")

        if ctx.wants("wespeaker/reset-wipes-both-stores"):
            cid = "wespeaker/reset-wipes-both-stores"
            # Repopulate the OFFICE store first. The native-rate check above resets
            # and then works purely on the remote stream, so without this the
            # "before" list holds only profiles-remote.* — and a check that never
            # saw an office file cannot prove reset wipes BOTH spaces, which is the
            # one thing it exists to prove (leaving half populated would make
            # remote numbering carry over into a "fresh" session).
            identify(audio_path, turns)
            before = sorted(p.name for p in profile_dir.iterdir())
            if not {"profiles.json", "profiles-remote.json"} <= set(before):
                rep.fail(cid, f"both stores were meant to be populated before the "
                              f"reset; only {before} existed")
                return
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


# ============================================================== layout group
#
# NAMING CONSISTENCY, 2026-07-31. One rule, and it now holds 13/13 on BOTH sides:
#
#     scripts/<name>/<name>-service.py   writes   logs/<name>.log
#
# Before this pass the tree agreed on the folder but not on the file or the log:
# `vad/silero-vad-service.py`, `nemotron/nemotron-asr-service.py`, and eight logs
# still carrying a ROLE suffix (`whisper-asr`, `overlap-repair-sidecar`,
# `dicow-sidecar`, `silero-vad`, …) left over from the shared chunked sidecar and
# from the days when a service's log was named for the job it did.
#
# WHY THIS IS PINNED RATHER THAN LEFT TO CARE. The 2026-07-28 one-service-per-model
# split established that a missed edit in one of N copies FAILS SILENTLY — the
# number simply never appears — and `whisper/protocol-matches-chunked` exists for
# exactly that reason on the wire side. This is the same failure on the *naming*
# side, and it is worse in one respect: a log written to the wrong basename still
# works. Nothing breaks, nothing is empty, and the evidence for the next bug just
# quietly lands in a file nobody greps.
#
# Both defects it guards are invisible on this machine in normal use:
#   * a stale SCRIPT path is loud for Nemotron (a session-start error) but SILENT
#     for the VAD — `SileroVADService.init?` returns nil and callers fall back to
#     the heuristic VAD, so the app runs and ATND beam gating quietly degrades,
#   * a stale LOG name is silent for every service.
#
# Pure: directory listing + regex/scan over Swift source. No model load, no audio,
# milliseconds — the `pyannote/telemetry-is-off` precedent.
LAYOUT_CHECKS = [
    "layout/one-service-per-folder",
    "layout/log-name-matches-folder",
]

# Direct children of scripts/ that legitimately hold no *-service.py. Each carries
# its reason HERE, so an exemption is a decision on the record rather than a name
# someone added to make a check go green.
LAYOUT_EXEMPT = {
    "atnd": "ATND1061 TCP/multicast simulators and the diagnostic receiver — "
            "developer tools driven by hand, never launched by the app",
    "tools": "one-off utilities (align-test, make-icon-variants), not services",
    "__pycache__": "CPython bytecode, not source",
}

# The three shapes a log basename is written in on the Swift side. All three are
# LITERALS on purpose: an indirection (`logHandle(name: Self.logName)`) is caught
# at its declaration instead, so nothing is counted twice.
SWIFT_LOG_PATTERNS = (
    re.compile(r'logHandle\(name:\s*"([^"]+)"\s*\)'),      # direct call site
    re.compile(r'\blogName\s*=\s*"([^"]+)"'),              # static let logName
    re.compile(r'\blogName:\s*String\s*\{\s*"([^"]+)"\s*\}'),  # protocol witness
    re.compile(r'\blogName:\s*"([^"]+)"'),                 # Config(...) argument
)

SERVICE_REF = re.compile(r'([A-Za-z0-9_.+-]+)/([A-Za-z0-9_.+-]+-service\.py)')


def _swift_scan(source: str):
    """Split Swift source into (string literals, code with comments removed).

    Hand-rolled rather than regex'd because a regex cannot tell the `//` in
    "https://…" from a real comment, and this codebase is full of both.

    The split matters: COMMENTS ARE DELIBERATELY EXEMPT from the
    "referenced script must exist" rule below. This project records deleted
    files as history — `ChunkedASRService.swift` names `chunked/chunked-asr-service.py`
    and `moss/moss-service.py` precisely to say they are GONE — and house style
    keeps history as history. A string literal is a live path the app really uses;
    a comment is a note about the past. Only the former is checked.
    """
    literals, code, i, n = [], [], 0, len(source)
    while i < n:
        c = source[i]
        if c == '"':
            j, buf, escaped = i + 1, [], False
            while j < n:
                d = source[j]
                if escaped:
                    escaped = False
                elif d == "\\":
                    escaped = True
                elif d == '"':
                    break
                buf.append(d)
                j += 1
            literals.append("".join(buf))
            code.append(source[i:j + 1])
            i = j + 1
        elif c == "/" and i + 1 < n and source[i + 1] == "/":
            while i < n and source[i] != "\n":
                i += 1
        elif c == "/" and i + 1 < n and source[i + 1] == "*":
            end = source.find("*/", i + 2)
            i = n if end < 0 else end + 2
        else:
            code.append(c)
            i += 1
    return literals, "".join(code)


def service_layout(scripts_dir: pathlib.Path, swift_sources: pathlib.Path):
    """The whole rule, against ANY tree. Returns (services, script_problems, log_problems).

    Takes its roots as arguments so the negative control can drive it against a
    FAKE tree — a check that has only ever been run against a healthy tree has
    not been shown to fail.
    """
    script_problems, log_problems = [], []

    # ---- (a)(b)(c) the SCRIPT side -------------------------------------
    services = {}       # folder -> "<folder>/<folder>-service.py"
    discovered = set()  # folders that hold ANY *-service.py (valid name or not)
    for child in sorted(p for p in scripts_dir.iterdir() if p.is_dir()):
        # DIRECT children only: a service must not hide in a vendor/ subtree.
        found = sorted(p.name for p in child.glob("*-service.py"))
        if child.name in LAYOUT_EXEMPT:
            # (b) An exemption means "this folder has no service", NOT "this
            # folder is unchecked" — otherwise the cheapest way to dodge the rule
            # is to move a service into an exempt folder.
            if found:
                script_problems.append(
                    f"exempt folder {child.name}/ contains {found} — the exemption "
                    f"says it has NO service ({LAYOUT_EXEMPT[child.name]}); a "
                    f"service there would escape the naming rule entirely")
            continue
        discovered.add(child.name)
        expected = f"{child.name}-service.py"
        if not found:
            discovered.discard(child.name)
            script_problems.append(
                f"{child.name}/ is not exempt but holds no *-service.py — either it "
                f"is a service folder missing its service, or it needs an entry in "
                f"LAYOUT_EXEMPT with a reason")
        elif len(found) > 1:
            script_problems.append(
                f"{child.name}/ holds {len(found)} service files {found} — one "
                f"service per folder, because one service per LOG is what keeps two "
                f"writers off one file (the 2026-07-15 mistake)")
        elif found[0] != expected:
            script_problems.append(
                f"scripts/{child.name}/{found[0]} breaks the rule — it must be "
                f"scripts/{child.name}/{expected} so that it writes logs/{child.name}.log")
        else:
            services[child.name] = f"{child.name}/{expected}"

    # (c) The pre-2026-07-28 flat layout must not creep back. sidecar-tests.py is
    # not a service and is deliberately not matched by *-service.py.
    stray = sorted(p.name for p in scripts_dir.glob("*-service.py"))
    if stray:
        script_problems.append(
            f"*-service.py directly under scripts/: {stray} — the flat pre-2026-07-28 "
            f"layout is gone, and a sidecar one folder shallower silently resolves "
            f"BASE to scripts/ (the three-dirname trap)")

    # ---- (d) the LOG side, derived from the SWIFT SOURCE ----------------
    # Deliberately NOT a hard-coded list here: a list would be a copy of the thing
    # this check exists to verify, and would agree with itself after a bad rename.
    logs, refs, per_file = set(), [], {}
    swift_files = sorted(swift_sources.rglob("*.swift"))
    if not swift_files:
        log_problems.append(f"no .swift files under {swift_sources} — the log side "
                            f"of the rule could not be checked at all")
    for path in swift_files:
        source = path.read_text()
        literals, code = _swift_scan(source)
        file_logs = set()
        for pattern in SWIFT_LOG_PATTERNS:
            file_logs.update(pattern.findall(code))
        logs |= file_logs
        file_refs = {f"{m.group(1)}/{m.group(2)}"
                     for lit in literals for m in SERVICE_REF.finditer(lit)}
        for ref in sorted(file_refs):
            refs.append((path.name, ref))
        if file_logs or file_refs:
            per_file[path.name] = (file_refs, file_logs)

    # Every script path the app can actually reach must be a real file. This is
    # the half that catches a rename applied to disk but not to Swift.
    for name, ref in refs:
        if not (scripts_dir / ref).exists():
            log_problems.append(
                f"{name} names \"{ref}\" but no such file exists — for the VAD this "
                f"failure is SILENT (init? returns nil, callers fall back to the "
                f"heuristic VAD)")

    # Set equality: the log basenames Swift writes ARE the service folder names.
    if logs != discovered:
        only_swift = sorted(logs - discovered)
        only_disk = sorted(discovered - logs)
        log_problems.append(
            f"log names and service folders disagree — Swift writes "
            f"{only_swift or 'nothing extra'} with no matching scripts/<name>/, and "
            f"{only_disk or 'nothing'} has a service but no log name in Swift")

    # And the PAIRING, wherever one file names exactly one script and one log.
    # (ChunkedASRService/ChunkedASRModels name many of each and are covered by
    # ChunkedASRSidecarRoutingTests on the Swift side instead.)
    for name, (file_refs, file_logs) in sorted(per_file.items()):
        if len(file_refs) != 1 or len(file_logs) != 1:
            continue
        ref, log = next(iter(file_refs)), next(iter(file_logs))
        folder = ref.split("/", 1)[0]
        if folder != log:
            log_problems.append(
                f"{name} runs {ref} but writes logs/{log}.log — the log is named for "
                f"the service folder, not for the role it plays")

    return services, script_problems, log_problems


def service_layout_problems(scripts_dir: pathlib.Path, swift_sources: pathlib.Path):
    """Flat problem list — what the negative control drives."""
    _, script_problems, log_problems = service_layout(scripts_dir, swift_sources)
    return script_problems + log_problems


def run_tools(rep: Report, ctx):
    """`scripts/tools/repair-wav-header.py`, driven for real on synthetic files.

    WHAT IT RECOVERS. `AVAudioFile` writes the WAV `data` chunk size only when it
    is released, so an app killed mid-recording leaves that field at 0 over a file
    full of audio — and every stage here then reads the recording as EMPTY.
    Measured 2026-08-05: 13 such recordings, ~1.7 GB, one holding 34.1 minutes of
    speech. The app side now closes the files on ⌘Q and on SIGINT/SIGTERM, but
    SIGKILL cannot be caught by anyone, so this tool stays the last resort.

    Driven on files this check BUILDS, never on the owner's recordings: a repair
    tool's test must not need the damage to still exist.
    """
    cid = "tools/wav-header-repair"
    if not ctx.wants(cid):
        return
    try:
        import importlib.util
        import struct

        import numpy as np
        import soundfile as sf

        tool = SCRIPTS / "tools" / "repair-wav-header.py"
        spec = importlib.util.spec_from_file_location("repair_wav_header", tool)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        work = pathlib.Path(ctx.tmp) / "wavrepair"
        work.mkdir(parents=True, exist_ok=True)
        rate, frames = 16_000, 8_000
        tone = (np.sin(np.arange(frames) * 0.01) * 0.5).astype(np.float32)

        def write_healthy(name):
            path = work / name
            sf.write(path, tone, rate, subtype="FLOAT")
            return path

        def break_header(path):
            """Zero the `data` size field, exactly as an unreleased file leaves it."""
            info = mod.scan(path)
            with path.open("r+b") as f:
                f.seek(info["size_field_at"])
                f.write(struct.pack("<I", 0))
            return path

        problems = []

        # (1) A broken file is repaired and becomes readable again — and the
        #     SAMPLES come back unchanged, which is the whole claim: nothing was
        #     ever missing except the number.
        broken = break_header(write_healthy("broken.wav"))
        if sf.info(broken).frames != 0:
            problems.append("the fixture did not actually break — "
                            "scan()/size_field_at is wrong")
        mod.repair(mod.scan(broken), apply=True)
        back, back_rate = sf.read(broken, dtype="float32")
        if back_rate != rate or len(back) != frames:
            problems.append(f"after repair got {len(back)} frames at {back_rate} Hz, "
                            f"expected {frames} at {rate}")
        elif not np.array_equal(back, tone):
            problems.append("repaired samples differ from the ones written — "
                            "the tool must touch NO audio byte")

        # (2) A HEALTHY file is refused. This is the only way the tool could
        #     destroy something, so the refusal matters more than the repair.
        healthy = write_healthy("healthy.wav")
        before = healthy.read_bytes()
        try:
            mod.repair(mod.scan(healthy), apply=True)
            problems.append("a healthy file was repaired instead of refused")
        except mod.NotRepairable:
            pass
        if healthy.read_bytes() != before:
            problems.append("a healthy file was modified despite the refusal")

        # (3) A TORN final frame is dropped, not handed on as a partial sample.
        #     A kill can land mid-sample; rounding up would be a second, subtler
        #     corruption than the one being repaired.
        torn = write_healthy("torn.wav")
        with torn.open("ab") as f:
            f.write(b"\x00\x00")          # half a float32 frame
        break_header(torn)
        note = mod.repair(mod.scan(torn), apply=True)
        if sf.info(torn).frames != frames:
            problems.append(f"torn frame not dropped: {sf.info(torn).frames} frames, "
                            f"expected {frames}")
        if "torn frame" not in note:
            problems.append("the torn-frame case is repaired silently — the note "
                            "must say bytes were dropped")

        # (4) Dry run writes NOTHING. The tool defaults to it, so a broken default
        #     would mean every "report" run silently edited the owner's files.
        dry = break_header(write_healthy("dry.wav"))
        snapshot = dry.read_bytes()
        mod.repair(mod.scan(dry), apply=False)
        if dry.read_bytes() != snapshot:
            problems.append("a dry run modified the file")

        rep.expect(cid, not problems,
                   "repairs a zeroed data chunk with samples byte-identical, "
                   "refuses a healthy file, drops a torn final frame and says so, "
                   "and writes nothing without --apply",
                   "; ".join(problems))
    except Exception as exc:  # noqa: BLE001
        rep.fail(cid, f"could not drive the repair tool: {exc!r}")


def run_layout(rep: Report, ctx):
    services, script_problems, log_problems = service_layout(SCRIPTS, SWIFT_SOURCES)

    cid = "layout/one-service-per-folder"
    if ctx.wants(cid):
        rep.expect(cid, not script_problems,
                   f"{len(services)} folders, each with exactly one "
                   f"<folder>-service.py as a direct child; "
                   f"{len(LAYOUT_EXEMPT)} exemptions hold none; none at scripts/ root",
                   "; ".join(script_problems))

    cid = "layout/log-name-matches-folder"
    if ctx.wants(cid):
        rep.expect(cid, not log_problems,
                   f"every service's log basename equals its folder name "
                   f"({len(services)}/{len(services)}), read out of the Swift source "
                   f"rather than listed here",
                   "; ".join(log_problems))


def run_qwen3(rep: Report, ctx):
    """Qwen3's decoding options — pure, no model load, milliseconds.

    The Whisper precedent, for the same reason it exists there: this is the one
    place where a settings value becomes a decoding decision, and every way it
    can go wrong is silent. A default that drifts changes every transcript with
    nobody having touched a setting; a sentinel passed through reaches the
    decoder as a real value; and a flag that the model reads nowhere is a control
    that does nothing (the Granite language picker, shipped 2026-08-01).
    """
    module = source = None
    try:
        module, source = load_sidecar_module(QWEN3_SERVICE, "mt_qwen3_service")
    except Exception as exc:  # noqa: BLE001
        for cid in QWEN3_CHECKS:
            if ctx.wants(cid):
                rep.fail(cid, f"could not import qwen3-service.py: {exc!r}")
        return

    if ctx.wants("qwen3/option-defaults-are-todays-behaviour"):
        cid = "qwen3/option-defaults-are-todays-behaviour"
        try:
            problems = []
            # BOTH pre-options shapes, because the old expression had two: a
            # forced language, and auto (where it was the empty dict).
            for label, language, want in (
                    ("forced language", "en", QWEN3_TODAYS_KWARGS),
                    ("auto", None, QWEN3_TODAYS_KWARGS_AUTO)):
                got = module.qwen3_generate_kwargs(language)
                if got != want:
                    extra = {k: v for k, v in got.items() if k not in want}
                    missing = {k: v for k, v in want.items() if k not in got}
                    differing = {k: (got[k], v) for k, v in want.items()
                                 if k in got and got[k] != v}
                    problems.append(f"{label}: extra={extra} missing={missing} "
                                    f"differing(got, want)={differing}")
            # Stated separately from the dict compare so the message names the
            # option when one of these ever appears at its sentinel.
            got = module.qwen3_generate_kwargs("en")
            for key in ("system_prompt", "repetition_penalty",
                        "repetition_context_size"):
                if key in got:
                    problems.append(f"{key} is present at defaults ({got[key]!r}) "
                                    "— it must be omitted so mlx-audio uses its own")
            rep.expect(cid, not problems,
                       "default kwargs are exactly the pre-options expression "
                       "({\"language\": language} if language else {}), no "
                       "optional knobs",
                       "; ".join(problems))
        except Exception as exc:  # noqa: BLE001
            rep.fail(cid, f"could not build the default kwargs: {exc!r}")

    if ctx.wants("qwen3/sentinels-become-none"):
        cid = "qwen3/sentinels-become-none"
        try:
            got = module.qwen3_generate_kwargs(
                "en", system_prompt="", repetition_penalty=0.0)
            problems = []
            for key in ("system_prompt", "repetition_penalty",
                        "repetition_context_size"):
                if key in got:
                    problems.append(f"{key}={got[key]!r} was passed through; the "
                                    "sentinel must become None (absent)")
            # And the positive control: real values DO come through, so the check
            # above is not passing merely because everything is dropped.
            on = module.qwen3_generate_kwargs(
                "en", system_prompt="  Aggia PREP  ", repetition_penalty=1.2,
                repetition_context_size=20)
            for key, want in (("system_prompt", "Aggia PREP"),
                              ("repetition_penalty", 1.2),
                              ("repetition_context_size", 20)):
                if on.get(key) != want:
                    problems.append(f"{key} was {on.get(key)!r}, expected {want!r}")
            rep.expect(cid, not problems,
                       "0/\"\" are dropped (⇒ mlx-audio's own default) and real "
                       "values pass through",
                       "; ".join(problems))
        except Exception as exc:  # noqa: BLE001
            rep.fail(cid, f"could not evaluate the sentinels: {exc!r}")

    # -- THE COUPLING, and the reason this is a check rather than a comment.
    #    mlx-audio reads repetition_context_size ONLY where it builds the
    #    penalty (`if repetition_penalty`, qwen3_asr.py:1292), so sending it
    #    alone is accepted and does NOTHING — the exact shape of the Granite
    #    language picker that shipped inert. Measured 2026-08-03 on real audio:
    #    ctx 5 vs 100 at penalty 1.2 gave different text, so WITH a penalty it
    #    genuinely matters; the positive half asserts that too, because "never
    #    sent" alone would pass a build that had dropped the option entirely.
    if ctx.wants("qwen3/context-size-never-travels-alone"):
        cid = "qwen3/context-size-never-travels-alone"
        try:
            problems = []
            alone = module.qwen3_generate_kwargs("en", repetition_context_size=20)
            if "repetition_context_size" in alone:
                problems.append(
                    f"repetition_context_size={alone['repetition_context_size']!r} "
                    "was sent with no penalty — mlx-audio never reads it there, "
                    "so it is a flag that cannot change the transcript")
            paired = module.qwen3_generate_kwargs(
                "en", repetition_penalty=1.2, repetition_context_size=20)
            if paired.get("repetition_context_size") != 20:
                problems.append("with a penalty active the context size did not "
                                f"travel: {paired.get('repetition_context_size')!r}")
            if paired.get("repetition_penalty") != 1.2:
                problems.append("the penalty itself did not travel: "
                                f"{paired.get('repetition_penalty')!r}")
            rep.expect(cid, not problems,
                       "the context size is sent only alongside a penalty, and "
                       "always when one is active",
                       "; ".join(problems))
        except Exception as exc:  # noqa: BLE001
            rep.fail(cid, f"could not evaluate the coupling: {exc!r}")


# ===================================================================== main
GROUPS = [
    ("layout", LAYOUT_CHECKS, run_layout),
    ("tools", TOOLS_CHECKS, run_tools),
    ("nemotron", NEMOTRON_CHECKS, run_nemotron),
    ("chunked", CHUNKED_CHECKS, run_chunked),
    ("whisper", WHISPER_CHECKS, run_whisper),
    ("qwen3", QWEN3_CHECKS, run_qwen3),
    ("aligner", ALIGNER_CHECKS, run_aligner),
    ("moss", MOSS_CHECKS, run_moss),
    ("pyannote", PYANNOTE_CHECKS, run_pyannote),
    ("spectral", SPECTRAL_CHECKS, run_spectral),
    ("wespeaker", WESPEAKER_CHECKS, run_wespeaker),
]


def resolve_fixtures(args):
    """Pick the audio fixtures and PRINT what was chosen, once, on first need.

    Split out of `main` so it can be deferred — see `Context.audio_a`. The
    printing moved with it deliberately: the resolved offset must stay on screen,
    because a run over a fixture whose speech starts at 22 s used to look
    identical to one over speech at 0 s and produced confident nonsense (the
    silent-fixture trap, 2026-07-30). Better placed mid-run than dropped.
    """
    if not (args.audio_a and args.audio_b and args.audio_diarize):
        found = find_fixtures()
        if len(found) >= 2:
            args.audio_a = args.audio_a or str(found[0])
            args.audio_b = args.audio_b or str(found[1])
        args.audio_diarize = args.audio_diarize or (str(found[0]) if found else None)

    print("-" * 72)
    print(f"  audio A       {args.audio_a or '(none — real-speech checks will SKIP)'}")
    print(f"  audio B       {args.audio_b or '(none — real-speech checks will SKIP)'}")
    print(f"  audio diarize {args.audio_diarize or '(none — pyannote+wespeaker checks will SKIP)'}")
    if args.clip_start is None:
        for label, path in (("A", args.audio_a), ("B", args.audio_b),
                            ("diarize", args.audio_diarize)):
            if not path:
                continue
            try:
                import numpy as np
                from mlx_audio.stt.utils import load_audio
                whole = np.asarray(load_audio(str(path), sr=SR), dtype=np.float32)
                found = first_speech_offset(whole)
            except Exception:  # noqa: BLE001
                found = None
            where = (f"auto {found:.1f}s" if found is not None
                     else "auto -> 0.0s (NO SPEECH FOUND IN THIS FILE)")
            print(f"  clip start {label:8s} {where}")
    else:
        print(f"  clip start    {args.clip_start:.1f}s (explicit)")
    print("-" * 72)


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
        self._args = args
        self._fixtures_resolved = False

    # ------------------------------------------------------------------ audio
    #
    # LAZY, and that is the whole design. Resolving fixtures means picking the
    # largest recordings, auto-seeking each past its leading silence, and then
    # decoding three WHOLE files just to print where the speech starts — on the
    # owner's machine that is ~3.5 minutes before a single check runs, and it ran
    # even for `--only layout`, whose checks are AST and a directory listing.
    #
    # It is not merely slow, it has cost real work: three back-to-back negative
    # controls timed out at 10 minutes on 2026-08-05 and the third left a source
    # file mid-experiment in a deliberately broken state, because each run paid
    # the discovery again.
    #
    # NO "pure checks" LIST, deliberately. A list would be a second copy of the
    # truth, and its failure direction is the bad one: a new audio-using check
    # accidentally listed as pure would SKIP silently. Asking for a fixture is
    # what a check that needs audio does anyway, so touching the property IS the
    # declaration — self-maintaining, and a new check cannot get it wrong.
    def _resolve_fixtures(self):
        if self._fixtures_resolved:
            return
        self._fixtures_resolved = True
        resolve_fixtures(self._args)

    @property
    def audio_a(self):
        self._resolve_fixtures()
        return self._args.audio_a

    @property
    def audio_b(self):
        self._resolve_fixtures()
        return self._args.audio_b

    @property
    def audio_diarize(self):
        self._resolve_fixtures()
        return self._args.audio_diarize

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
    # Default None = auto-seek past leading digital silence (see load_clip /
    # first_speech_offset). It is NOT 0.0: the owner's recordings routinely open
    # with 20-40 s of zeros, and slicing from 0 gave checks an embedding of
    # silence rather than an obvious empty-audio failure. `--clip-start 0` still
    # forces offset 0 explicitly.
    p.add_argument("--clip-start", type=float, default=None,
                   help="seconds into each fixture (default: first speech)")
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

    print("=" * 72)
    print("MeetingTranscriber sidecar integration tests")
    print(f"  python        {VENV_PY}")
    print(f"  HF_HOME       {os.environ['HF_HOME']}  (HF_HUB_OFFLINE="
          f"{os.environ.get('HF_HUB_OFFLINE')})")
    print("  audio         resolved on demand — the block below appears only if a "
          "selected check needs it")
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
