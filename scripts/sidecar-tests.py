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
import types
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
# THE SECOND AND THIRD REALTIME ENGINES (2026-08-11). Each has its own sidecar,
# per the one-service-per-model rule, and all three speak the SAME frames because
# a single Swift client (`RealtimeASRService`) drives them — so they must not
# drift apart; see `realtime/protocol-matches-nemotron`.
PARAKEET_SERVICE = "parakeet/parakeet-service.py"
FUNASR_SERVICE = "funasr/funasr-service.py"
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
# THE SIXTH DIARIZATION ENGINE (2026-08-11). Speaks the SAME wire as pyannote and
# spectral, so one Swift caller drives any of them; see `campplus/*`.
CAMPPLUS_SERVICE = "campplus/campplus-service.py"
# THE FIFTH DIARIZATION ENGINE (2026-08-10). Runs in `.venv-diarizen`, its own
# Python 3.11 interpreter, so every check below must be PURE — this suite runs
# under the main `.venv`, where `import diarizen` does not exist.
DIARIZEN_SERVICE = "diarizen/diarizen-service.py"
SPECTRAL_VENDOR = SCRIPTS / "spectral" / "vendor"
# The FOURTH diarizer (2026-08-07): NVIDIA NeMo's ClusteringDiarizer (MarbleNet
# VAD -> multi-scale TitaNet-Large -> NME-SC spectral clustering). Like spectral
# it answers the same Swift caller with the same turns-only wire, so the same
# drift check applies. It runs in `.venv-nemo`, its OWN interpreter — the second
# sidecar after DiCoW to need one — which is why EVERY check below is pure: this
# suite runs under the main `.venv`, where `import nemo` does not exist. Nothing
# here loads a model, and nothing here needs to.
NEMO_SERVICE = "nemo/nemo-service.py"
NEMO_VENDOR = SCRIPTS / "nemo" / "vendor"

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


def dead_channel(seconds: float, seed=2):
    """Below MOSS's SILENCE_RMS (0.0001 = -80 dBFS) — a channel with no signal.

    DELIBERATELY NOT near_silence(). That fixture is -68 dBFS, and on 2026-08-20
    MOSS was measured transcribing real speech perfectly all the way down to
    -70 dBFS — so -68 dBFS is a level a real, quiet meeting genuinely occupies,
    and skipping it is what the lowered threshold exists to stop. The realtime
    engines still gate their PARTIALS at 0.004 and still use near_silence(),
    because there the question is "is this lane idle enough to skip the compute",
    not "is there speech here to lose".
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(seconds * SR)) * 0.00002).astype("float32")


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
    """Import a sidecar for its module-level constants WITHOUT running main().

    COMPILED FROM THE SOURCE TEXT THIS FUNCTION RETURNS, never through the
    normal loader, and that is a correctness fix rather than a style choice.

    Callers use both halves together: the AST checks parse the returned SOURCE
    while the drift checks read the returned MODULE's constants. Going through
    `spec.loader.exec_module` let the module half come from a stale
    `__pycache__/*.pyc` while the source half was read fresh — so the two halves
    could describe different files. Observed 2026-08-11 during a negative
    control: a one-character edit (`1.5` → `2.0`, same byte length) inside
    CPython's mtime granularity left the cached bytecode valid, and the suite
    reported a `PARTIAL_EVERY` drift that no longer existed in the source.

    The direction that matters is the other one: the same staleness would let a
    real drift keep passing, which is the silent failure this whole group exists
    to catch.
    """
    path = SCRIPTS / filename
    source = path.read_text()
    module = types.ModuleType(mod_name)
    module.__file__ = str(path)
    # __name__ is mod_name, not "__main__", so main() stays inert.
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module, source


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


# ============================================================ realtime group
# THE SECOND REALTIME ENGINE (2026-08-11). `scripts/parakeet/parakeet-service.py`
# speaks the SAME frames as the Nemotron sidecar, because ONE Swift client
# (`RealtimeASRService`) drives both — the chunked side's shape, where five
# scripts answer one client.
#
# That is exactly the arrangement whose failure mode this project has already
# written down: a protocol edit applied to one copy and missed in the other
# fails SILENTLY — the number, or the lane, simply never appears. Hence the same
# mitigation as `whisper/protocol-matches-chunked`: detection, not discipline.
#
# Both checks are PURE — module import and AST, no model load, milliseconds.
# The sidecars' heavy imports live inside `main()`, which is what makes that
# possible; the `nemotron/*` checks above are the ones that drive a real model.
REALTIME_CHECKS = [
    "realtime/protocol-matches-nemotron",
    "realtime/engine-divergences",
    "realtime/funasr-signal-gate",
]

# The constants that decide WHAT the two engines put on the wire and WHEN. Every
# one of them is shared on purpose:
#   SR / MAX_BUFFER      — the app feeds both the same 16 kHz mono stream and
#                          relies on the same 60 s utterance cap.
#   PARTIAL_EVERY        — the caption cadence the app's overlay is tuned to.
#   PARTIAL_SILENCE_RMS  — the idle-lane saving, which is engine-INDEPENDENT
#                          (people take turns whatever model is loaded).
# `PARTIAL_DUTY` is deliberately ABSENT from this list: it is Nemotron's
# cadence-stretch and Parakeet must not have one. That divergence is asserted by
# the second check, where it can be stated with its reason.
REALTIME_SHARED_CONSTANTS = ["SR", "MAX_BUFFER", "PARTIAL_EVERY",
                             "PARTIAL_SILENCE_RMS"]

# WHAT EACH NON-REFERENCE ENGINE MUST AND MUST NOT HAVE, with the measurement
# behind it. Stated per engine rather than as one "everything that is not
# Nemotron looks alike" rule, because the third engine broke that assumption the
# day it landed: Parakeet drops `PARTIAL_DUTY` and Fun-ASR keeps it.
#
#   ATT_CONTEXT / --chunk-ms — Nemotron's alone. Neither other model has an
#     attention-context setting, so the flag would be an argparse error at
#     session start. The Swift side makes it unrepresentable (`Config.chunkMs`
#     is nil for both); this is the other end of that guarantee.
#
#   PARTIAL_DUTY — the cadence STRETCH, and it tracks cost, not engine age.
#     A 30 s partial costs Nemotron 2.135 s, Fun-ASR 0.609 s and Parakeet
#     0.235 s. Two ACTIVE lanes at the flat 1.5 s cadence therefore sit at
#     285 % / 81 % / 31 % duty — so Parakeet genuinely does not need the stretch
#     and Fun-ASR genuinely does. Copying either decision to the other engine
#     would be wrong in opposite directions.
#
#   HALLUCINATION GATES — likewise measured per model, and asserted in BOTH
#     directions. Parakeet returned the EMPTY STRING for 30 s of silence, 30 s
#     of tiny noise and a 3 s flush-sized silent buffer, so a gate there would be
#     pure over-deletion risk. Fun-ASR returned '<gbg>', "I'm not sure if I can
#     do that." and 'Okay.' for the same three inputs, so a gate there is
#     mandatory. Deleting Fun-ASR's gates must fail this check just as loudly as
#     adding gates to Parakeet.
REALTIME_ENGINE_RULES = {
    "parakeet": {
        "forbidden_names": ("ATT_CONTEXT", "PARTIAL_DUTY"),
        "required_names": (),
        "gates": False,
    },
    "funasr": {
        "forbidden_names": ("ATT_CONTEXT",),
        "required_names": ("PARTIAL_DUTY",),
        "gates": True,
    },
}

# The gate machinery Fun-ASR must keep and Parakeet must never grow. Named here
# so both halves of the rule read off ONE list.
REALTIME_GATE_NAMES = ("GARBAGE_TAGS", "REFUSAL_PREFIXES", "drop_reason")

# Every wire message either sidecar can write, as (builder, args). The office
# cases pass `stream=None` EXPLICITLY rather than relying on the default, so a
# copy that quietly changed the default would still be compared on the value the
# app actually sees.
REALTIME_EMIT_CASES = [
    ("emit", ("status", "READY", None)),
    ("emit", ("partial", "hello there", None)),
    ("emit", ("final", "hello there", None)),
    ("emit", ("final", "", None)),
    ("emit", ("partial", "hello there", "remote")),
    ("emit", ("final", "hello there", "remote")),
    ("_emit_raw", ("error", "model load failed: boom", None)),
]


def _realtime_opcodes(source: str):
    """The integer opcodes a sidecar's read loop compares `n` against.

    AST, not grep — both files DOCUMENT their protocol at length in a docstring
    that lists every opcode, so a textual search matches its own explanation.
    That lesson is written down three times in this repo already
    (`pyannote/no-path-to-torchcodec`, `moss/asr-vendor-is-own-and-identical`,
    build.sh's tokenizer gate); this is the fourth place it applies.
    """
    import ast
    tree = ast.parse(source)
    main = next((n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef) and n.name == "main"), None)
    if main is None:
        return None
    loops = [n for n in ast.walk(main) if isinstance(n, ast.While)]
    found = set()
    for loop in loops:
        for node in ast.walk(loop):
            if not isinstance(node, ast.Compare):
                continue
            if not (isinstance(node.left, ast.Name) and node.left.id == "n"):
                continue
            for comparator in node.comparators:
                if isinstance(comparator, ast.Constant) and isinstance(comparator.value, int):
                    found.add(comparator.value)
                elif (isinstance(comparator, ast.UnaryOp)
                      and isinstance(comparator.op, ast.USub)
                      and isinstance(comparator.operand, ast.Constant)):
                    found.add(-comparator.operand.value)
    return found


def _realtime_names(source: str):
    """Every name the module DEFINES + every argparse flag string in the file.

    Functions and classes count, not just assignments: the hallucination gates
    this is asked about are a constant tuple AND a `drop_reason()` function, and
    a helper that could only see constants reported the function missing from a
    file that plainly defines it.
    """
    import ast
    tree = ast.parse(source)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
    flags = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            flags |= {a.value for a in node.args
                      if isinstance(a, ast.Constant) and isinstance(a.value, str)}
        # `ATT_CONTEXT` could also live inside main(); catch any assignment.
        if isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
    return names, flags


def run_realtime(rep: Report, ctx):
    nemo_mod, nemo_src = load_sidecar_module(NEMOTRON_SERVICE, "mt_nemotron_protocol")
    para_mod, para_src = load_sidecar_module(PARAKEET_SERVICE, "mt_parakeet_protocol")
    fun_mod, fun_src = load_sidecar_module(FUNASR_SERVICE, "mt_funasr_protocol")

    # Nemotron is the REFERENCE, not merely the first: it is the engine the app
    # falls back to for an absent or unknown `realtime.model`, so it is the one
    # copy that must never move to accommodate a newcomer.
    others = (("parakeet", para_mod, para_src), ("funasr", fun_mod, fun_src))
    everyone = (("nemotron", nemo_mod, nemo_src),) + others

    # -- check 1: every realtime sidecar speaks ONE protocol.
    if ctx.wants("realtime/protocol-matches-nemotron"):
        cid = "realtime/protocol-matches-nemotron"
        try:
            problems = []

            # (1) the shared constants.
            for name in REALTIME_SHARED_CONSTANTS:
                a = getattr(nemo_mod, name, None)
                for label, mod, _ in others:
                    b = getattr(mod, name, None)
                    if a is None or b is None or a != b:
                        problems.append(f"{name}: nemotron={a!r} {label}={b!r}")

            # (2) every wire message, byte for byte — key set, key ORDER and all.
            #     `json.dumps` of a re-parsed payload would hide an order change,
            #     so the raw lines are compared directly.
            key_sets = {}
            for builder, cargs in REALTIME_EMIT_CASES:
                mine = capture_stdout(getattr(nemo_mod, builder), *cargs)
                for label, mod, _ in others:
                    theirs = capture_stdout(getattr(mod, builder), *cargs)
                    if mine != theirs:
                        problems.append(f"{builder}{cargs!r}: nemotron wrote {mine} "
                                        f"but {label} wrote {theirs}")
                for line in mine:
                    payload = json.loads(line)
                    key_sets.setdefault(payload["type"], set()).update(payload)

            # (3) the top-level key sets, stated explicitly rather than implied by
            #     (2), because these are the exact sets the Swift decoder reads
            #     (`RealtimeASRService.Message`: type, text, stream?).
            #
            #     ABSENT MEANS OFFICE. A `"stream"` key appearing on an office
            #     line would route correctly today (the decoder tests for the
            #     literal "remote") and would silently change the bytes a
            #     single-stream session produces, which the Nemotron sidecar's own
            #     `nemotron/single-stream-bytes` check exists to protect.
            expected_keys = {
                "status": {"type", "text"},
                "error": {"type", "text"},
                "partial": {"type", "text", "stream"},
                "final": {"type", "text", "stream"},
            }
            for kind, want in expected_keys.items():
                got = key_sets.get(kind)
                if got != want:
                    problems.append(f"{kind} key union was {sorted(got or [])}, "
                                    f"expected {sorted(want)}")
            for kind in ("partial", "final"):
                for mod_name, mod, _ in everyone:
                    office = capture_stdout(mod.emit, kind, "x", None)
                    if any("stream" in json.loads(l) for l in office):
                        problems.append(f"{mod_name} put a \"stream\" key on an "
                                        f"office {kind} — absent means office")

            # (4) the stdin opcodes. The app writes these five and nothing else;
            #     a sidecar missing one does not error, it silently ignores that
            #     frame — a remote lane that never flushes, or an exit that hangs.
            opcodes = {label: _realtime_opcodes(src) for label, _, src in everyone}
            if any(v is None for v in opcodes.values()):
                problems.append("a sidecar has no main() — its read loop could not "
                                "be located")
            else:
                for label, ops in opcodes.items():
                    for op in (0, -1, -2, -3):
                        if op not in ops:
                            problems.append(f"{label}'s read loop no longer handles {op}")
                    if ops != opcodes["nemotron"]:
                        problems.append(
                            f"opcode sets differ: nemotron="
                            f"{sorted(opcodes['nemotron'])} {label}={sorted(ops)}")

            # (5) the READY handshake. `RealtimeASRService.waitUntilReady` blocks
            #     for 120 s on a status line containing READY and then FAILS THE
            #     SESSION; a sidecar that stopped emitting it would look exactly
            #     like a model that would not load.
            import ast
            for label, _, src in everyone:
                ready = [n for n in ast.walk(ast.parse(src))
                         if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Name) and n.func.id == "emit"
                         and len(n.args) >= 2
                         and all(isinstance(a, ast.Constant) for a in n.args[:2])
                         and n.args[0].value == "status"
                         and "READY" in str(n.args[1].value)]
                if len(ready) != 1:
                    problems.append(f"{label} emits the READY status {len(ready)} "
                                    "times, expected exactly 1")

            # (6) the Lane contract, driven for real: the 60 s trim, the partial
            #     cadence gate and the office/remote tag. Behaviour, not shape.
            import numpy as np
            for label, mod, _ in everyone:
                lane = mod.Lane()
                if lane.stream is not None:
                    problems.append(f"{label}: a default Lane is not the office lane")
                if mod.Lane("remote").stream != "remote":
                    problems.append(f"{label}: the remote Lane is not tagged")
                lane.append(np.zeros(mod.MAX_BUFFER + 5 * mod.SR, dtype=np.float32))
                if lane.buffer.size != mod.MAX_BUFFER:
                    problems.append(f"{label}: buffer not trimmed to MAX_BUFFER "
                                    f"({lane.buffer.size})")
                short = mod.Lane()
                short.append(np.zeros(mod.SR // 4, dtype=np.float32))
                if short.wants_partial():
                    problems.append(f"{label}: wants a partial on a sub-0.5 s buffer")
                short.reset()
                if short.buffer.size or short.samples_since_partial:
                    problems.append(f"{label}: reset() left state behind")

            rep.expect(cid, not problems,
                       f"{len(everyone)} engines agree: "
                       f"{len(REALTIME_SHARED_CONSTANTS)} constants, "
                       f"{len(REALTIME_EMIT_CASES)} wire messages byte-identical, "
                       f"{len(expected_keys)} key sets exact, the same 4 stdin "
                       "opcodes, one READY each, and every Lane behaves alike",
                       "; ".join(problems))
        except Exception as exc:  # noqa: BLE001
            rep.fail(cid, f"could not compare the realtime sidecars: {exc!r}")

    # -- check 2: THE DELIBERATE DIVERGENCES, asserted in BOTH directions.
    #
    #    `moss/diar-has-no-file-branch` precedent, for the same reason and with
    #    the same shape. Three things Nemotron has and Parakeet must not:
    #
    #    ATT_CONTEXT / --chunk-ms — Parakeet has no attention-context setting at
    #      all, so the flag would be an argparse error at session start; the Swift
    #      side makes it unrepresentable (`Config.chunkMs` is nil there) and this
    #      is the other end of that guarantee.
    #    PARTIAL_DUTY — Nemotron's cadence STRETCH, which exists solely because
    #      its 30 s partial costs 2.1 s. Parakeet's costs 0.235 s and its ~31 %
    #      two-lane duty was measured AT the flat cadence, so copying the stretch
    #      across would only make the caption slower.
    #
    #    THE POSITIVE HALF IS LOAD-BEARING: Nemotron must still have all three.
    #    Without it, a file that had simply stopped configuring attention context
    #    — the exact over-deletion this project treats as the dangerous direction
    #    — would keep this check green.
    # -- Fun-ASR's SIGNAL-shape gate. Pure numpy, no model load, milliseconds.
    #
    #    Pinned because it replaced a defence rather than adding one: the FLUSH
    #    level gate was lowered from 0.004 to 0.0001 on the strength of THIS
    #    rule, so losing it silently restores a hallucination that reaches the
    #    transcript at any level. Both directions — the keep cases are what stop
    #    a future tightening from deleting quiet or noisy real speech.
    if ctx.wants("realtime/funasr-signal-gate"):
        cid = "realtime/funasr-signal-gate"
        try:
            import numpy as np
            import scipy.signal as ss_
            module, source = load_sidecar_module(
                SCRIPTS / "funasr" / "funasr-service.py", "funasr_service")
            fn = getattr(module, "speech_peak_ratio", None)
            problems = []
            if fn is None:
                problems.append("funasr-service.py no longer defines "
                                "speech_peak_ratio — noise-driven hallucinations "
                                "return at every level, and FINAL_SILENCE_RMS was "
                                "lowered on the strength of this rule")
            elif getattr(module, "FINAL_SILENCE_RMS", None) != 0.0001:
                problems.append("FINAL_SILENCE_RMS moved; re-measure the pair "
                                "before changing either half")
            else:
                bar = module.MIN_SPEECH_PEAK_RATIO
                floor_sec = getattr(module, "MIN_SPEECH_JUDGE_SEC", None)
                pct = getattr(module, "SPEECH_FLOOR_PERCENTILE", None)
                if floor_sec is None or pct is None:
                    problems.append("the gate no longer declares "
                                    "MIN_SPEECH_JUDGE_SEC / SPEECH_FLOOR_PERCENTILE "
                                    "— both were added 2026-08-21 after the median "
                                    "form was measured deleting real sentences")
                    floor_sec, pct = floor_sec or 4.0, pct or 10
                rng = np.random.default_rng(4)
                b, a = ss_.butter(1, 0.08)

                def at(x, target):
                    return (x / np.sqrt((x * x).mean()) * target).astype("float32")

                # NO SPEECH — must all read below the bar, at any level.
                white = rng.standard_normal(30 * SR)
                pink = ss_.lfilter(b, a, rng.standard_normal(30 * SR))
                t = np.arange(30 * SR) / SR
                for label, sig in (("white", white), ("pink", pink),
                                   ("hum50", np.sin(2 * np.pi * 50 * t)),
                                   ("tone1k", np.sin(2 * np.pi * 1000 * t))):
                    for level in (0.0002, 0.02, 0.08):
                        got = fn(at(np.asarray(sig, dtype="float64"), level))
                        if got is None or got >= bar:
                            problems.append(f"{label} at rms {level} read {got} "
                                            f"(>= {bar}) — that is the input this "
                                            "model invents sentences over")

                # SPEECH — must all read above the bar. Synthetic on purpose, so
                # the check needs no fixture: bursts of noise separated by a
                # quiet floor is what "bursty" means, and it is the shape the
                # measurement on real recordings found (4.73 - 313).
                floor = ss_.lfilter(b, a, rng.standard_normal(30 * SR)) * 0.02
                bursty = np.array(floor, dtype="float32")
                for start in range(0, 30, 3):          # 1 s of speech every 3 s
                    i0 = int(start * SR)
                    bursty[i0:i0 + SR] += (rng.standard_normal(SR) * 0.2).astype("float32")
                lone = np.array(ss_.lfilter(b, a, rng.standard_normal(120 * SR)) * 0.0006,
                                dtype="float32")
                lone[:int(1.5 * SR)] += (rng.standard_normal(int(1.5 * SR)) * 0.05).astype("float32")

                # ⚠ DENSE CONTINUOUS SPEECH — THE CASE THAT BROKE THE FIRST
                # VERSION OF THIS RULE, and the reason the two fixtures above
                # were not enough. Both of them are SPARSE: mostly silence with
                # speech in it, which is the easiest thing this statistic sees.
                # A person talking without pauses is the opposite, and there the
                # median frame IS speech. Measured on real audio 2026-08-21,
                # confirmed by transcribing every offender with mlx_whisper:
                # "I think that's a great idea." read 1.86 and would have been
                # deleted by the shipped 2.0 bar.
                #
                # Carrier modulated at a syllable rate but NEVER reaching
                # silence, which is exactly that property. Verified to stand in
                # for the real thing: under the OLD max/median statistic these
                # read 1.75 and 1.88 — i.e. this fixture reproduces the failure
                # rather than merely resembling it — and the assertion below
                # that the ratio far exceeds the bar is what fails if anyone
                # reverts the denominator.
                dense = []
                for depth in (0.85, 0.95):
                    for secs in (4.0, 30.0):
                        n = int(secs * SR)
                        tt = np.arange(n) / SR
                        car = ss_.lfilter(b, a, rng.standard_normal(n))
                        car = car / np.abs(car).max()
                        env = 1.0 - depth * 0.5 * (1 + np.sin(2 * np.pi * 4.0 * tt))
                        dense.append((f"dense speech depth={depth} {secs:.0f}s",
                                      np.asarray(car * env, dtype="float32")))

                for label, sig in ([("bursty speech", bursty),
                                    ("ONE utterance in 120 s", lone)] + dense):
                    for level in (0.0002, 0.02, 0.08):
                        got = fn(at(sig.astype("float64"), level))
                        if got is None or got < bar:
                            problems.append(f"{label} at rms {level} read {got} "
                                            f"(< {bar}) — real speech would be deleted")

                # Level-independence is the property that stops this gate
                # reintroducing the -48 dBFS cliff it was written to avoid.
                # Relative, not exact: rescaling to each level round-trips
                # through float32, so the last couple of digits legitimately
                # move. A real level dependency would move it by orders of
                # magnitude, which 0.1 % catches with room to spare.
                ratios = [fn(at(bursty.astype("float64"), lv))
                          for lv in (0.0002, 0.02, 0.08)]
                if max(ratios) / min(ratios) - 1.0 > 1e-3:
                    problems.append(f"the ratio moved with LEVEL ({ratios}) — it must "
                                    "be dimensionless or it is a loudness gate again")

                # ⚠ THE ABSTAIN FLOOR, asserted at the boundary and in BOTH
                # directions. Below it the two populations genuinely overlap —
                # swept at 50 and 20 ms frames against p50/p25/p10/p5, the best
                # separation at 1 s is 1.21x — so a rule that cannot tell must
                # not answer. The `flush_lane` buffer is ONE UTTERANCE (the app
                # flushes on the VAD speech->silence edge), which is precisely
                # this range, so this half is not an edge case: it is the
                # commonest input the gate sees.
                quiet = np.zeros(int(SR // 2), dtype="float32")
                for secs in (0.2, 1.0, 2.0, floor_sec - 0.01):
                    if fn(np.zeros(int(secs * SR), dtype="float32")) is not None:
                        problems.append(f"a {secs:.2f}s buffer was judged — below "
                                        f"{floor_sec}s nothing separates speech from "
                                        "noise, and answering there deleted real "
                                        "sentences on 2026-08-20")
                if fn(np.zeros(int(floor_sec * SR), dtype="float32")) is None:
                    problems.append(f"a buffer of exactly {floor_sec}s was NOT judged "
                                    "— the floor must let the partial path's own "
                                    "4 s tail through or that gate is dead code")
                del quiet
            rep.expect(cid, not problems,
                       "noise/tones read below the bar and speech above it at every "
                       "level — bursty, lone AND dense-continuous, the last being "
                       "the case the median form deleted — the ratio is "
                       "level-independent, and the abstain floor holds on both sides",
                       "; ".join(problems))
        except Exception as exc:  # noqa: BLE001
            rep.fail(cid, f"could not evaluate the signal gate: {exc!r}")

    if ctx.wants("realtime/engine-divergences"):
        cid = "realtime/engine-divergences"
        try:
            nemo_names, nemo_flags = _realtime_names(nemo_src)
            problems = []

            # THE POSITIVE HALF, stated once for the reference engine. Without
            # it, a Nemotron file that had simply stopped configuring attention
            # context — the over-deletion direction — would keep this green.
            for name in ("ATT_CONTEXT", "PARTIAL_DUTY"):
                if name not in nemo_names:
                    problems.append(f"nemotron no longer defines {name} — this check "
                                    "has lost the half that lets it fail")
            if "--chunk-ms" not in nemo_flags:
                problems.append("nemotron no longer accepts --chunk-ms")

            for label, mod, src in others:
                names, flags = _realtime_names(src)
                rules = REALTIME_ENGINE_RULES[label]
                for name in rules["forbidden_names"]:
                    if name in names:
                        problems.append(f"{label} defines {name} — see "
                                        "REALTIME_ENGINE_RULES for why it must not")
                for name in rules["required_names"]:
                    if name not in names:
                        problems.append(f"{label} no longer defines {name} — its "
                                        "measured cost needs it")
                if "--chunk-ms" in flags:
                    problems.append(f"{label} accepts --chunk-ms, a flag its model "
                                    "has nothing to apply")

            # BOTH must still take --language. Parakeet's is INERT and measured so
            # (generate() has no such parameter; None/"fr"/"de"/"xx" all returned
            # byte-identical output) — it is accepted and LOGGED anyway, so the
            # user's choice dies where the truth is instead of vanishing at the
            # Swift boundary with no record.
            for label, _, src in everyone:
                if "--language" not in _realtime_names(src)[1]:
                    problems.append(f"{label} no longer accepts --language")

            # HALLUCINATION GATES, ASSERTED IN BOTH DIRECTIONS — the half of this
            # check most likely to be "tidied" wrongly. Each engine's rule is
            # written against what that model was MEASURED to emit over silence;
            # see REALTIME_ENGINE_RULES.
            for label, _, src in others:
                names, _ = _realtime_names(src)
                present = [g for g in REALTIME_GATE_NAMES if g in names]
                if REALTIME_ENGINE_RULES[label]["gates"]:
                    missing = [g for g in REALTIME_GATE_NAMES if g not in names]
                    if missing:
                        problems.append(
                            f"{label} lost its hallucination gates ({missing}) — it "
                            "was MEASURED emitting '<gbg>', a chatbot refusal and "
                            "'Okay.' over silence; without these that text reaches "
                            "the transcript")
                else:
                    if present:
                        problems.append(
                            f"{label} grew {present} — no hallucination has been "
                            "OBSERVED for this model (silence returned the empty "
                            "string); measure one first, then write the gate "
                            "against what was seen")
                    for banned in ("CANNED_HALLUCINATIONS", "canned_drop_reason",
                                   "whisper_drop_reason"):
                        if banned in names:
                            problems.append(f"{label} grew {banned} — same reason")

            rep.expect(cid, not problems,
                       "nemotron keeps ATT_CONTEXT/--chunk-ms/PARTIAL_DUTY; parakeet "
                       "has none of them and no gates; funasr keeps PARTIAL_DUTY and "
                       "its measured gates but no attention context — each half "
                       "asserted both ways",
                       "; ".join(problems))
        except Exception as exc:  # noqa: BLE001
            rep.fail(cid, f"could not inspect the realtime sidecars: {exc!r}")


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


# The WHOLE-CHUNK backstop added 2026-08-20, and it is a SEPARATE table from
# WHISPER_GATE_CASES on purpose: that one judges a segment against its own
# duration, this one judges the surviving transcript against the CHUNK's. The
# blind spot it covers is invisible to the other — measured over 62 speech-free
# inputs, TWO escaped (120 s of digital silence, and 120 s of a 1 kHz tone),
# both as four one-word 'you' segments of which three were individually short
# enough and confident enough to pass every per-segment rule.
#
# BOTH DIRECTIONS, because this rule deletes an entire chunk and over-deletion
# leaves no trace in a transcript. The keep cases are the load-bearing half.
WHISPER_CHUNK_GATE_CASES = [
    # (surviving text, chunk seconds, keep?)
    ("you you you", 120.0, False),                     # THE measured escape
    ("Thank you.", 30.0, False),                       # the classic, a full window
    ("Okay.", 30.0, True),   # ⚠ KEPT, and load-bearing: 'okay' is deliberately
                             # NOT in the vocabulary, so a genuine short reply
                             # survives. Adding it there fails the mlx-audio
                             # check too — see the note beside the constant.
    ("you", 120.0, False),
    ("Thank you. Bye.", 8.0, True),      # a real closing tail — under the floor
    ("Okay.", 2.0, True),                # genuine short reply
    ("Okay.", 18.0, True),               # still under CANNED_CHUNK_MIN_DURATION
    ("Thanks.", 120.0, False),           # 'thanks' IS canned, and 0.008 w/s
    ("", 120.0, True),                   # nothing kept is not a drop
    ("Let's try that next week.", 120.0, True),        # sparse REAL speech
    ("you know what I mean", 120.0, True),             # one non-canned word is enough
    ("thank you very much everyone", 120.0, True),
    (" ".join(["you"] * 13), 120.0, True),   # 0.108 w/s — above the density floor
    (" ".join(["you"] * 12), 120.0, True),   # 0.100 w/s — AT it, and `<` keeps it,
                                             # matching whisper_drop_reason's own
                                             # boundary convention. The safe side.
    (". . .", 120.0, False),   # no word characters at all — measured on granite,
                               # which returned '.' for 120 s of white noise.
                               # Dropping it loses no words, by definition.
    (". . .", 8.0, True),      # ...but the duration floor still comes first
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
    "whisper/canned-chunk-backstop",
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

    # -- the WHOLE-CHUNK backstop. Pure, no model load, milliseconds.
    #    Pinned rather than left to notice because its failure is silent in BOTH
    #    directions: losing it puts 'you you you' back into a two-minute chunk
    #    with nothing to say it happened, and loosening it deletes real speech
    #    that likewise leaves no trace outside logs/whisper.log.
    if ctx.wants("whisper/canned-chunk-backstop"):
        cid = "whisper/canned-chunk-backstop"
        try:
            fn = getattr(module, "canned_chunk_drop_reason", None)
            if fn is None:
                rep.fail(cid, "whisper-service.py no longer defines "
                              "canned_chunk_drop_reason — the per-segment gate "
                              "cannot see a whole-chunk hallucination, so the "
                              "measured 'you you you' escape is back")
            else:
                wrong = []
                for text, secs, want_keep in WHISPER_CHUNK_GATE_CASES:
                    reason = fn(text, secs)
                    if (reason is None) != want_keep:
                        verdict = "kept" if reason is None else f"dropped as {reason}"
                        wrong.append(f"{text[:34]!r} @{secs:g}s was {verdict}")
                # The constants must stay SEPARATE from the per-segment ones.
                # Reusing those was measured to delete a real 'Thank you. Bye.'
                # closing tail; a future tidy-up that folds them back would
                # reintroduce exactly that, and every case above would still pass.
                if getattr(module, "CANNED_CHUNK_MAX_WORDS_PER_SEC", None) == \
                        getattr(module, "WHISPER_MIN_WORDS_PER_SEC", None):
                    wrong.append("CANNED_CHUNK_MAX_WORDS_PER_SEC has been folded "
                                 "into WHISPER_MIN_WORDS_PER_SEC — measured to "
                                 "delete a real closing tail")
                rep.expect(cid, not wrong,
                           f"all {len(WHISPER_CHUNK_GATE_CASES)} chunk cases judged "
                           "correctly, on their own constants",
                           "; ".join(wrong))
        except Exception as exc:  # noqa: BLE001
            rep.fail(cid, f"could not evaluate the chunk backstop: {exc}")

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
                 # A LENGTH is now required, not merely an object: since the
                 # whole-chunk backstop landed (2026-08-20) transcribe_path
                 # measures the chunk from the decoded samples. 33 s matches the
                 # segments below, and the surviving 8-word sentence is far above
                 # the backstop's density floor, so it changes nothing this check
                 # is about. `None` here used to be enough and now raises.
                 "load_audio_16k": lambda path: [0.0] * (33 * 16000),
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
                hallucination_silence_sec=2.0, initial_prompt=" pyannote ")
            for key, want in (("best_of", 5),
                              ("hallucination_silence_threshold", 2.0),
                              ("initial_prompt", "pyannote")):
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
    "moss/refusal-stems-need-sparse-audio",
    "moss/duration-reader-accepts-the-apps-wav",
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
    # WITH segments — the stop-time full pass (2026-08-18). MOSS transcribes and
    # labels in one call, and this frame used to discard the labels, so a full
    # pass produced a re-transcribed meeting with nobody in it. Pinned LAST in
    # key order on purpose: appended after `text`, so the four cases above stay
    # byte-identical and an older app still decodes a newer sidecar's reply.
    (("file_result", 7, "hello", [{"start": 0.0, "end": 1.0, "speaker": "S01",
                                   "text": "hello"}]),
     '{"type": "file_result", "id": 7, "text": "hello", "segments": '
     '[{"start": 0.0, "end": 1.0, "speaker": "S01", "text": "hello"}]}'),
    # An empty LIST is not the same as absent: it says "this model reports
    # speakers and found none here", where absent says "this model has no
    # speakers to report". Both must survive on the wire.
    (("file_result", 7, "hello", []),
     '{"type": "file_result", "id": 7, "text": "hello", "segments": []}'),
]

# Constants that are part of MOSS's wire/gate behaviour, not implementation
# detail. The two gate constants matter as much as the framing ones: a different
# SILENCE_RMS in one copy means one role calls the model on silence and answers
# as a chatbot while the other does not.
# The reworded-refusal family (2026-08-20). BOTH DIRECTIONS: the keep cases are
# the load-bearing half, because a stem alone is a thing people really say.
MOSS_REFUSAL_STEM_CASES = [
    # (text, chunk seconds, keep?)
    ("I'm not gonna do that.", 120.0, False),        # the owner's reported wording
    ("I'm not sure if I can do it.", 120.0, False),  # the wording Fun-ASR produced
    ("I'm not sure if I can do that.", 30.0, False),
    ("I can't help with that.", 120.0, False),
    # ...and the other half. A real sentence lives in a chunk that carries the
    # rest of the meeting's words, so its density clears the bar.
    ("I'm not sure if I can make it on Friday, but let me check my calendar "
     "and come back to you before the end of today.", 30.0, True),
    ("I'm not gonna do that, honestly, because the roadmap is public and "
     "everyone can already see it.", 30.0, True),
    ("Let's try that next week.", 120.0, True),      # sparse, but no refusal
    ("", 120.0, True),
    ("I'm not gonna do that.", None, True),          # unknown duration fails OPEN

    # ⚠ THE SAME REAL SENTENCES AT 120 s, and they are the cases this table was
    # MISSING. Both keep-cases above sit at 30 s, where 24 words is 0.8 w/s and
    # clears the density bar on its own — so nothing here tested a real sentence
    # in a LONG window, which is exactly what `chunked.intervalSec` ships (120)
    # and what the office FULL PASS at Stop uses. At 120 s the bar means "fewer
    # than 60 words", which no ordinary sentence reaches, and the first of these
    # was measured on 2026-08-21 being DROPPED at 0.19 w/s before
    # REFUSAL_STEM_MAX_WORDS existed. Delete that ceiling and these three fail.
    ("I'm not going to be there on Thursday, but Sam can cover it and we "
     "should still ship the release by Friday afternoon.", 120.0, True),
    ("I'm not sure if I can make it on Friday, but let me check my calendar "
     "and come back to you before the end of today.", 120.0, True),
    ("I'm not gonna do that, honestly, because the roadmap is public and "
     "everyone can already see it.", 120.0, True),

    # The ceiling itself, at the boundary and on both sides of it. 10 words is
    # comfortably above both refusals ever OBSERVED (5 and 8 words), so the
    # measured cases stay caught while a sentence that carries a clause after
    # the opening survives.
    ("I'm not gonna do that because it is not ready.", 120.0, False),      # 10w
    ("I'm not gonna do that because it is not ready yet.", 120.0, True),   # 11w
]

MOSS_PROTOCOL_CONSTANTS = ["SR", "MIN_CHUNK_SEC", "MAX_BUFFER_SEC", "SILENCE_RMS",
                           "MAX_NEW_TOKENS", "REFUSAL_MARKERS", "REFUSAL_PREFIXES",
                           # The filler gate (2026-08-20). Listed for the same
                           # reason as everything else here: it DELETES a chunk,
                           # and in the diarization role its speaker turns with
                           # it, so the two copies drifting would mean one role
                           # keeping a hallucination the other dropped — on the
                           # same audio, in the same meeting, in MOSS+MOSS mode.
                           "FILLER_TOKENS", "FILLER_MIN_DURATION_SEC",
                           "FILLER_MAX_TOKENS_PER_SEC", "FILLER_STRIP",
                           # Same reasoning for the reworded-refusal family. The
                           # absolute ceiling joined them on 2026-08-21: without
                           # it the density bar is no bar at all on a 120 s
                           # window (0.5 w/s there means "fewer than 60 words"),
                           # and a copy that lost it would delete real sentences
                           # the other kept.
                           "REFUSAL_STEMS", "REFUSAL_STEM_MAX_WORDS_PER_SEC",
                           "REFUSAL_STEM_MAX_WORDS",
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
                if module.silence_skip_reason(dead_channel(30.0)) is None:
                    problems.append(f"{label}: a dead channel (-94 dBFS) was NOT skipped")
                # ...and the OTHER direction, which is the one the 2026-08-20
                # measurement added: a QUIET but live capture must reach the
                # model. The client Mac records at -47 dBFS and MOSS was
                # measured transcribing real speech down to -70; near_silence()
                # is -68 dBFS, so at the old 0.004 threshold this was skipped
                # and the meeting's text was silently lost.
                if module.silence_skip_reason(near_silence(30.0)) is not None:
                    problems.append(f"{label}: a quiet (-68 dBFS) capture was skipped — "
                                    "that is where real speech was being deleted")
                reason = module.silence_skip_reason(speech_like)
                if reason is not None:
                    problems.append(f"{label}: the tone fixture was skipped as {reason!r}")
                if module.silence_skip_reason(np.zeros(0, dtype="float32")) is None:
                    problems.append(f"{label}: an empty buffer was NOT skipped")
            rep.expect(cid, not problems,
                       f"digital silence and a dead channel skipped, a QUIET "
                       f"capture and a tone kept, in all "
                       f"{len(copies)} copies (threshold {copies[0][1].SILENCE_RMS})",
                       "; ".join(problems))
        except Exception as exc:  # noqa: BLE001
            rep.fail(cid, f"could not evaluate the silence gate: {exc!r}")

    # -- check 1b: the REWORDED-refusal family. Its own check rather than more
    #    MOSS_REFUSAL_CASES rows, because it takes a second argument and asks a
    #    different question: refusal_drop_reason drops on wording alone, this one
    #    needs wording AND sparse audio, and that pairing is the whole design.
    if ctx.wants("moss/refusal-stems-need-sparse-audio"):
        cid = "moss/refusal-stems-need-sparse-audio"
        try:
            problems = []
            for label, module, _, _ in copies:
                fn = getattr(module, "refusal_stem_drop_reason", None)
                if fn is None:
                    problems.append(f"{label}: refusal_stem_drop_reason is gone — a "
                                    "reworded refusal ('I'm not gonna do that') is "
                                    "back in the transcript")
                    continue
                for text, secs, want_keep in MOSS_REFUSAL_STEM_CASES:
                    reason = fn(text, secs)
                    if (reason is None) != want_keep:
                        verdict = "kept" if reason is None else f"dropped as {reason}"
                        problems.append(f"{label}: {text[:38]!r} @{secs}s was {verdict}")
            rep.expect(cid, not problems,
                       f"all {len(MOSS_REFUSAL_STEM_CASES)} cases judged correctly in "
                       f"all {len(copies)} copies — reworded refusals dropped only "
                       "over sparse audio, real sentences kept",
                       "; ".join(problems))
        except Exception as exc:  # noqa: BLE001
            rep.fail(cid, f"could not evaluate the refusal stems: {exc!r}")

    # -- check 1b: the duration reader accepts the WAV the APP really hands it.
    #
    # 🔴 THE FAILURE THIS GUARDS SHIPPED FOR A DAY AND WAS COMPLETELY SILENT.
    # `audio_seconds` first used CPython's `wave` module, on the strength of a
    # comment asserting that every path reaching this sidecar is "16 kHz mono
    # PCM16". Half of that was false, and it was the half that matters:
    #
    #     write_temp_wav (in the sidecar)     PCM16    -> `wave` reads it
    #     AudioRecorder.writeTempWAV (Swift)  FLOAT32  -> `wave` RAISES
    #                                                     "unknown format: 3"
    #
    # Swift builds it with AVAudioFormat(commonFormat: .pcmFormatFloat32) and
    # AVAudioFile, which writes IEEE-float WAV; `wave` supports integer PCM
    # only. Both gates that take a duration FAIL OPEN on None, by design — so
    # they were inert on every `-2` file-transcribe request, which is to say on
    # the office FULL PASS at Stop, on Remote chunks, and on overlap-repair
    # re-ASR. Nothing failed, nothing was logged, and the two gates added
    # against a hallucination the owner had actually SEEN could not fire on the
    # paths that carry the transcript.
    #
    # A unit test over the gates themselves cannot catch this — they were
    # correct; it was the value handed to them that was always None. That is
    # the project's own lesson ("a component test cannot see a call site"), so
    # the fixture here is the FORMAT, written the way the app writes it.
    if ctx.wants("moss/duration-reader-accepts-the-apps-wav"):
        cid = "moss/duration-reader-accepts-the-apps-wav"
        try:
            import numpy as np
            import soundfile as sf
            problems = []
            tmp = pathlib.Path(ctx.tmp)
            tone_ = (np.sin(2 * np.pi * 220 * np.arange(3 * SR) / SR)
                     * 0.1).astype("float32")
            # 'FLOAT' is exactly what AVAudioFile emits for .pcmFormatFloat32;
            # 'PCM_16' is what this sidecar's own write_temp_wav emits.
            fixtures = []
            for subtype in ("FLOAT", "PCM_16"):
                p = tmp / f"moss-duration-{subtype.lower()}.wav"
                sf.write(str(p), tone_, SR, subtype=subtype)
                fixtures.append((subtype, str(p)))
            missing = str(tmp / "moss-duration-absent.wav")

            for label, module, _, _ in copies:
                fn = getattr(module, "audio_seconds", None)
                if fn is None:
                    problems.append(f"{label}: audio_seconds is gone — the filler and "
                                    "refusal-stem gates have no duration and fail open "
                                    "everywhere")
                    continue
                for subtype, path in fixtures:
                    got = fn(path)
                    if got is None or abs(got - 3.0) > 0.01:
                        problems.append(
                            f"{label}: a {subtype} WAV read {got!r}, not 3.0 — "
                            + ("this is the format AudioRecorder.writeTempWAV "
                               "produces, so every -2 request (Stop full pass, "
                               "Remote chunks, overlap repair) loses both gates"
                               if subtype == "FLOAT" else
                               "this is what the sidecar writes for its own live "
                               "FLUSH, so even the office path loses them"))
                # ...and the fail-OPEN half, which must survive any fix to the
                # above: an unreadable file must never become a reason to delete
                # a transcript.
                if fn(missing) is not None:
                    problems.append(f"{label}: a missing file did not fail open")
            rep.expect(cid, not problems,
                       f"both copies read a FLOAT WAV (what the app writes) and a "
                       f"PCM_16 one (what the sidecar writes) as 3.0 s, and an "
                       f"unreadable path still fails open",
                       "; ".join(problems))
        except Exception as exc:  # noqa: BLE001
            rep.fail(cid, f"could not evaluate the duration reader: {exc!r}")

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
                # `segments` on a file_result is the 2026-08-18 addition, and it
                # is in the UNION rather than required on every reply: the frame
                # omits it entirely when the caller passes none, which is what
                # keeps Remote and overlap repair byte-identical and lets an older
                # app decode a newer sidecar. `file_error` must NEVER carry it —
                # a failed transcription has no speakers to report.
                "file_result": {"type", "id", "text", "segments"},
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

# ================================================================ nemo group
#
# The FOURTH diarization engine (NVIDIA NeMo ClusteringDiarizer, 2026-08-07).
#
# EVERY CHECK HERE IS PURE — AST, source scan and a file hash. That is not a
# style preference: NeMo runs in `.venv-nemo` and this suite runs under the main
# `.venv`, where `import nemo` raises. Anything needing the engine itself has to
# be a manual drive with the other interpreter, so what is pinned here is
# everything that can be established WITHOUT it, which turns out to be all four
# of the measured constants, the whole wire protocol and both refusals.
NEMO_CHECKS = [
    "nemo/protocol-matches-pyannote",
    "nemo/wire-is-isolated-from-nemo-logging",
    "nemo/no-live-chunk-branch",
    "nemo/rejects-zero-frame-audio",
    "nemo/offline-env-is-assigned",
    "nemo/measured-constants-are-pinned",
    "nemo/vendored-config-is-upstream",
]

# The vendored inference config, byte-identical to
# NVIDIA-NeMo/Speech @ 6c57e73e83de967eed4d334c493ac313b9afd147,
# examples/speaker_tasks/diarization/conf/inference/diar_infer_general.yaml.
# Recorded from the upstream clone, so this is a provenance pin rather than a
# self-fingerprint — the spectral/vendor-is-own precedent.
NEMO_VENDORED_CONFIG = "diar_infer_general.yaml"
NEMO_VENDORED_CONFIG_SHA = \
    "f7b10d79cbf5f481f24453363167232e81759571046adb63e03e0bccf8d03ca7"

# The four deviations from that config, each MEASURED on this M4 and each stated
# at its constant in the sidecar. Re-declared here rather than read from the
# sidecar for the reason every mirror in this file is: reading the value from the
# thing under test makes the check agree with whatever it finds.
NEMO_MEASURED_CONSTANTS = {
    # MANDATORY on macOS. With the stock 1, Lightning's DataLoader spawns
    # workers, NeMo's dynamically-created SpeechLabelEntity cannot be pickled,
    # and the job dies with PicklingError before reading a frame.
    "NUM_WORKERS": 0,
    # diar_infer_general.yaml ships 10; `meeting` and `telephonic` upstream both
    # already use 30. At 10, Meeting5People.wav (ground truth 5) returns FOUR
    # speakers, losing the one holding 7 % of the speech.
    "SPARSE_SEARCH_VOLUME": 30,
    # Forces ONE global clustering. The stock long-form path (selected when the
    # base-scale segment count exceeds this) COLLAPSED a real 48.2-minute
    # recording to 1 speaker / 73 turns; forced-global gives 8 speakers / 324
    # turns and is faster (170 s vs 227 s).
    "EMBEDDINGS_PER_CHUNK": 10_000_000,
    # Speaker count is AUTO on this engine and there is no Settings control, so
    # this bound is the only thing standing between a 12-person meeting and a
    # silent cap. The stock value is 8.
    "MAX_NUM_SPEAKERS": 20,
}


def run_nemo(rep: Report, ctx):
    import ast

    source = (SCRIPTS / NEMO_SERVICE).read_text()
    tree = ast.parse(source)

    def module_constants(t):
        """Module-level `NAME = <literal>` pairs, by AST — no import, no env."""
        out = {}
        for node in t.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                    and isinstance(node.targets[0], ast.Name):
                try:
                    out[node.targets[0].id] = ast.literal_eval(node.value)
                except Exception:  # noqa: BLE001 — not a literal; not our subject
                    pass
        return out

    def function(t, name):
        return next((n for n in ast.walk(t) if isinstance(n, ast.FunctionDef)
                     and n.name == name), None)

    # -- check 1: THE DRIFT DETECTOR, third application. Same argument as
    #    `spectral/protocol-matches-pyannote`: this sidecar answers the SAME
    #    Swift caller, so a reply key added, renamed or REORDERED in one file and
    #    not the other silently drops a field for whichever engine was missed.
    #
    #    ONE DELIBERATE DIFFERENCE FROM THE SPECTRAL CHECK, and it is the whole
    #    reason this is a separate check rather than a third arm of that one:
    #    `emit` is NOT compared verbatim here. It cannot be — NeMo logs to stdout
    #    (see check 2), so this service's `emit` writes to a private `WIRE` handle
    #    instead of `sys.stdout`. What is compared is the thing that actually
    #    matters, the BYTES: both modules' real `emit` are driven with the same
    #    payloads and the output must match character for character. The other
    #    three shared functions ARE compared verbatim, because nothing about the
    #    wire isolation touches them.
    if ctx.wants("nemo/protocol-matches-pyannote"):
        cid = "nemo/protocol-matches-pyannote"
        try:
            # Importing the sidecar ASSIGNS its offline env vars into this
            # process (that assignment is check 5's subject). Snapshot and
            # restore, so a test run cannot leave WANDB_MODE/SENTRY_DSN behind
            # for whatever runs next — the MT_PROFILE_DIR discipline applied to
            # the environment.
            touched = ("HF_HOME", "HF_HUB_OFFLINE", "WANDB_MODE", "WANDB_DISABLED",
                       "SENTRY_DSN", "NEMO_ONELOGGER_ENABLED", "ONE_LOGGER_ENABLED")
            before = {k: os.environ.get(k) for k in touched}
            try:
                nemo_mod, nemo_src = load_sidecar_module(NEMO_SERVICE, "mt_nemo_protocol")
            finally:
                for key, value in before.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
            pyan_mod, pyan_src = load_sidecar_module(PYANNOTE_SERVICE, "mt_pyannote_nemo")
            problems = []

            # (1) the shared plumbing, verbatim — emit deliberately excluded, see
            #     the header. If `emit` ever becomes identical again that is not a
            #     failure here; check 2 is what would notice the wire isolation
            #     going away.
            for name in ("fail", "brief_traceback", "log"):
                a, b = _fn_source(pyan_src, name), _fn_source(nemo_src, name)
                if a is None or b is None or a != b:
                    problems.append(f"{name}(): pyannote={'absent' if a is None else 'differs'}"
                                    f" vs nemo={'absent' if b is None else 'differs'}")

            # (2) the wire bytes. `WIRE` starts as `sys.stdout` precisely so an
            #     importer sees no fd surgery, which is what lets `capture_stdout`
            #     work on both modules identically; it is repointed only by
            #     `install_wire()`, which only `main()` calls.
            if nemo_mod.WIRE is not sys.stdout:
                problems.append("importing nemo-service.py already moved WIRE off "
                                "sys.stdout — the fd surgery must happen in "
                                "install_wire(), called from main(), or merely "
                                "importing this sidecar steals its importer's stdout")
            wire_cases = [
                {"type": "status", "text": "LOADED"},
                {"type": "error", "text": "Audio not found: /nope.wav"},
                {"type": "result", "audio": "/tmp/m.wav",
                 "segments": [{"start": 0.5, "end": 1.25, "label": "SPEAKER_00"}]},
                {"type": "result", "audio": "/tmp/m.wav", "segments": [],
                 "stream": "remote"},
            ]

            def capture_wire(payload):
                """Drive nemo's REAL emit and collect what it wrote.

                `contextlib.redirect_stdout` (what `capture_stdout` uses) cannot
                see this one: `WIRE` was bound to the true `sys.stdout` OBJECT at
                import, so rebinding the NAME `sys.stdout` afterwards does not
                reach it — the JSON would go straight to the terminal and the
                comparison would read as an empty reply. Swapping the module's
                own `WIRE` is the honest capture, and it also proves `emit` reads
                that global at call time rather than having captured a handle.
                """
                import io
                buffer = io.StringIO()
                saved = nemo_mod.WIRE
                nemo_mod.WIRE = buffer
                try:
                    nemo_mod.emit(payload)
                finally:
                    nemo_mod.WIRE = saved
                return buffer.getvalue().splitlines()

            for payload in wire_cases:
                mine = capture_stdout(pyan_mod.emit, payload)
                theirs = capture_wire(payload)
                if mine != theirs:
                    problems.append(f"emit({payload['type']}): pyannote wrote {mine} "
                                    f"but nemo wrote {theirs}")

            # (3) the REAL `wire_segments` of each, lifted by AST. Both take the
            #     same (start, end, label) tuple shape — NeMo's `read_rttm`
            #     produces exactly what pyannote's `itertracks` does — so this
            #     pins {start,end,label}, the key ORDER, and that neither can emit
            #     id/name/conf. Identity leaking out of a pipeline stage is the
            #     structural failure the pyannote/wespeaker split exists to make
            #     impossible, and it must be impossible for this engine too.
            pyan_wire = extract_nested(pyan_mod, pyan_src, "main", "wire_segments", {})
            nemo_wire = extract_nested(nemo_mod, nemo_src, "main", "wire_segments", {})
            turns = [(0.5, 1.25, "SPEAKER_00"), (1.25, 3.0, "SPEAKER_01")]
            a_json, b_json = json.dumps(pyan_wire(turns)), json.dumps(nemo_wire(turns))
            if a_json != b_json:
                problems.append(f"wire_segments: pyannote produced {a_json} but "
                                f"nemo produced {b_json}")
            for banned in ('"id"', '"name"', '"conf"'):
                if banned in b_json:
                    problems.append(f"nemo's segments carry {banned} — identity "
                                    f"leaked back into a pipeline stage")

            # (4) the reply SHAPES. Only the types both services can produce:
            #     pyannote also has `chunk_result`, which nemo must NOT have —
            #     that asymmetry is check 3's subject, not a drift.
            pyan_shapes, nemo_shapes = _emit_shapes(pyan_src), _emit_shapes(nemo_src)
            for label, shapes in (("pyannote", pyan_shapes), ("nemo", nemo_shapes)):
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
                # ONE additive shape is allowed, and only for nemo: the classified
                # `kind` key that tells a VERDICT from a FAULT ("no speech in this
                # recording" is not a failure — see
                # layout/no-speech-is-routed-by-kind-not-prose). Additive by
                # design, so a reader that ignores the key still gets the sentence.
                #
                # DELIBERATELY NOT widened to pyannote: it does not raise on silent
                # audio, so a `kind` appearing there would be a real drift and this
                # check must still say so. And the plain shape stays REQUIRED —
                # without that, a sidecar that classified EVERY error would pass,
                # and the app would be back to understanding every kind or reading
                # prose.
                allowed = {("type", "text")}
                if label == "nemo":
                    allowed.add(("type", "kind", "text"))
                if ("type", "text") not in errors or not errors <= allowed:
                    problems.append(f"{label}: `error` emit shapes were {errors}, "
                                    f"expected a subset of {allowed} with "
                                    f'("type","text") always present')
                # ABSENT MEANS OFFICE: every reply raised once the job's stream is
                # known must splice `**echo`. The sole exception in both files is
                # the bad-job-line reply, emitted BEFORE `stream` has been read.
                naked = [(kind, k) for kind, k, echo, loop in shapes if loop and not echo]
                if len(naked) != 1:
                    problems.append(f"{label}: {len(naked)} replies inside the read loop "
                                    f"do not splice **echo ({naked}) — expected exactly "
                                    f"one (the bad-job-line reply, emitted before the "
                                    f"job's stream is known)")

            # (5) and the office/remote rule itself, verbatim across the files.
            def echo_rule(src):
                return [ast.unparse(n) for n in ast.walk(ast.parse(src))
                        if isinstance(n, ast.Assign)
                        and ast.unparse(n.targets[0]) in ("stream", "is_remote", "echo")]

            a_rule, b_rule = echo_rule(pyan_src), echo_rule(nemo_src)
            if a_rule != b_rule or len(a_rule) != 3:
                problems.append(f"the stream/echo rule differs: pyannote {a_rule} vs "
                                f"nemo {b_rule}")

            rep.expect(cid, not problems,
                       f"3 shared functions verbatim, {len(wire_cases)} wire messages "
                       f"byte-identical out of a DIFFERENT emit (nemo writes to WIRE, "
                       f"not sys.stdout — see nemo/wire-is-isolated-from-nemo-logging), "
                       f"wire_segments agrees, the shared reply shapes are exact, and "
                       f"the absent-stream-means-office rule is the same 3 statements",
                       "; ".join(problems))
        except Exception as exc:  # noqa: BLE001
            rep.fail(cid, f"could not compare nemo with pyannote: {exc!r}")

    # -- check 2: THE WIRE IS ISOLATED FROM NEMO'S OWN LOGGING.
    #
    #    A REAL DEFECT, caught on the first drive of this sidecar (2026-08-07):
    #    one job put 28 `[NeMo I …]` lines on fd 1, interleaved with the JSON, and
    #    a line-by-line decoder died on the first of them. No other sidecar has
    #    this problem — pyannote, spectral and MOSS all log to stderr — so nothing
    #    that already exists would catch it coming back.
    #
    #    THREE things are asserted, and the third is the one a "cleanup" breaks:
    #      (a) fd 1 is duplicated to a private handle and fd 1 itself is pointed
    #          at fd 2, so C-level writes and tqdm move too — a `sys.stdout`
    #          rebinding alone would not,
    #      (b) `emit` writes to that handle and NOT to sys.stdout,
    #      (c) the installer runs BEFORE nemo is imported. Importing NeMo alone
    #          prints, and a logging handler bound to fd 1 at import time would
    #          hold the wire for the life of the process. This is an ordering
    #          requirement that nothing else can express — the spectral VAD shim
    #          precedent.
    #
    #    By AST, not grep: `install_wire`'s docstring explains the trap at length,
    #    so a textual search matches its own explanation.
    if ctx.wants("nemo/wire-is-isolated-from-nemo-logging"):
        cid = "nemo/wire-is-isolated-from-nemo-logging"
        try:
            problems = []
            installer = function(tree, "install_wire")
            if installer is None:
                problems.append("install_wire() is gone — NeMo's own logger writes to "
                                "stdout and would interleave with the JSON protocol")
            else:
                calls = [ast.unparse(n) for n in ast.walk(installer)
                         if isinstance(n, ast.Call)]
                if not any(c.startswith("os.dup(1)") or c == "os.dup(1)" for c in calls):
                    problems.append("install_wire never calls os.dup(1) — without a "
                                    "private duplicate of the original fd 1 there is "
                                    "nowhere left to write the protocol")
                if "os.dup2(2, 1)" not in calls:
                    problems.append("install_wire never calls os.dup2(2, 1) — rebinding "
                                    "sys.stdout alone leaves C-level writes, tqdm and "
                                    "Lightning's progress bars on the wire")
                if "global WIRE" not in ast.unparse(installer):
                    problems.append("install_wire does not rebind the module-level WIRE")

            # (b) emit writes to WIRE, and nowhere in the file does anything write
            #     to sys.stdout.
            emit_fn = function(tree, "emit")
            if emit_fn is None:
                problems.append("emit() is gone")
            else:
                body = ast.unparse(emit_fn)
                if "WIRE.write" not in body:
                    problems.append("emit() does not write to WIRE")
                if "sys.stdout.write" in body:
                    problems.append("emit() writes to sys.stdout — NeMo's log lines "
                                    "share that stream and would corrupt the protocol")

            # (c) ORDERING: installed before NeMo is imported. The import is found
            #     by AST rather than assumed to be on a particular line.
            nemo_imports = [n.lineno for n in ast.walk(tree)
                            if isinstance(n, ast.ImportFrom)
                            and (n.module or "").startswith("nemo")]
            install_calls = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
                             and isinstance(n.func, ast.Name)
                             and n.func.id == "install_wire"]
            if not nemo_imports:
                problems.append("nothing imports nemo — the ordering this check "
                                "measures no longer has a subject")
            elif not install_calls:
                problems.append("install_wire is defined but never called — the wire "
                                "is not actually isolated")
            elif min(install_calls) > min(nemo_imports):
                problems.append(f"install_wire is first called at line "
                                f"{min(install_calls)}, AFTER nemo is imported at line "
                                f"{min(nemo_imports)} — importing NeMo prints, and a "
                                f"handler bound to fd 1 at import time keeps the wire "
                                f"for the whole process")

            rep.expect(cid, not problems,
                       "install_wire dups fd 1 to a private handle and points fd 1 at "
                       "fd 2 before nemo is imported, and emit writes to that handle "
                       "rather than to sys.stdout",
                       "; ".join(problems))
        except Exception as exc:  # noqa: BLE001
            rep.fail(cid, f"could not read the wire isolation: {exc!r}")

    # -- check 3: FINAL-ONLY, and pyannote is not. The
    #    `moss/diar-has-no-file-branch` and `spectral/no-live-chunk-branch`
    #    precedent, for the same reason and with the same both-directions
    #    discipline.
    #
    #      * ADDING a live/chunk branch here is the symmetry-minded mistake and
    #        the DANGEROUS one: NME-SC runs its eigengap analysis over the whole
    #        file's affinity matrix, so a 30 s window would be counted and
    #        clustered on its own and its labels would mean nothing across
    #        windows. That failure is already documented on MOSS — two different
    #        people both numbered S01 — and it is SILENT.
    #      * REMOVING pyannote's is the other direction, and the positive half is
    #        what makes this check able to fail at all.
    if ctx.wants("nemo/no-live-chunk-branch"):
        cid = "nemo/no-live-chunk-branch"
        try:
            def live_shape(src):
                t = ast.parse(src)
                gets, compared, dict_keys = set(), set(), set()
                for node in ast.walk(t):
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
                return gets, compared, {kind for kind, _, _, _ in _emit_shapes(src)}, dict_keys

            n_gets, n_cmp, n_kinds, n_keys = live_shape(source)
            p_gets, p_cmp, p_kinds, p_keys = live_shape(
                (SCRIPTS / PYANNOTE_SERVICE).read_text())
            problems = []

            # The NEGATIVE half: no live path anywhere in nemo.
            if "chunk" in n_cmp:
                problems.append('nemo compares a command against "chunk" — a windowed '
                                "pass through a globally-clustering engine returns "
                                "labels that look continuous and are not")
            if "chunk_result" in n_kinds or "chunk_result" in n_keys:
                problems.append("nemo emits a chunk_result reply")
            if "window_start" in n_gets or "window_start" in n_keys:
                problems.append("nemo reads or emits window_start — the only reason to "
                                "know a window's offset is to serve one")
            # …and that the refusal really is there, so "no chunk branch" is a
            # deliberate refusal rather than an unhandled command falling through
            # to the whole-file pass.
            if "final" not in n_cmp:
                problems.append('nemo never compares the command against "final" — a '
                                "non-final job would fall through to the whole-file "
                                "pass instead of being refused")

            # The POSITIVE half: pyannote really does have the live path this
            # engine is declining to copy.
            if "chunk" not in p_cmp:
                problems.append('pyannote no longer compares a command against "chunk"')
            if "chunk_result" not in p_kinds:
                problems.append("pyannote no longer emits chunk_result")
            if "window_start" not in p_gets:
                problems.append("pyannote no longer reads window_start")

            rep.expect(cid, not problems,
                       "nemo has no chunk comparison, no chunk_result and no "
                       "window_start, and refuses any cmd != final — while pyannote "
                       "has all three (so this check can fail in either direction)",
                       "; ".join(problems))
        except Exception as exc:  # noqa: BLE001
            rep.fail(cid, f"could not compare the read loops: {exc!r}")

    # -- check 4: a recording whose WAV header declares ZERO FRAMES is REFUSED,
    #    not answered with an empty result. Verbatim spectral semantics, and the
    #    same reasoning: `AVAudioFile` writes the `data` chunk size only on
    #    release, so an app killed mid-recording leaves it at 0 over a file full
    #    of audio (13 such recordings, ~1.7 GB, found on the owner's machine
    #    2026-08-05). Without the guard, VAD finds no speech in an empty array
    #    and a whole meeting comes back as having no speakers, silently.
    #
    #    The ORDERING half is the point: a guard AFTER the diarize call could
    #    never fire. The message is also required to name the repair tool — an
    #    error that does not say the audio is recoverable sends the user to
    #    delete the file.
    if ctx.wants("nemo/rejects-zero-frame-audio"):
        cid = "nemo/rejects-zero-frame-audio"
        try:
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

            # The engine entry point, read from the source rather than assumed:
            # NeMo answers through `ClusteringDiarizer(...).diarize()`, so THAT
            # call is what the guard must precede.
            runs = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute) and n.func.attr == "diarize"]
            if not runs:
                problems.append("nothing calls .diarize() — the engine entry point this "
                                "guard is positioned against is gone")
            elif zero_tests and min(n.lineno for n in zero_tests) > min(runs):
                problems.append("the zero-frame guard sits AFTER the diarize call — it "
                                "could never fire before the pass returns nothing")

            rep.expect(cid, not problems,
                       "a 0-frame recording is refused with an error that names the "
                       "repair tool, and the guard runs before ClusteringDiarizer.diarize()",
                       "; ".join(problems))
        except Exception as exc:  # noqa: BLE001
            rep.fail(cid, f"could not read the zero-frame guard: {exc!r}")

    # -- check 5: the offline environment is ASSIGNED, never `setdefault`.
    #
    #    The `pyannote/telemetry-is-off` precedent, and its most important detail:
    #    hard requirement #1 is ABSOLUTE, so an inherited environment must not be
    #    able to switch any of these back on. `setdefault` would let a stray
    #    `WANDB_MODE=online` in the launching environment win silently.
    #
    #    HONESTY ABOUT WHAT THIS IS: unlike pyannote's, these variables did NOT
    #    fix a measured leak. A DNS-recording probe over a full diarization job
    #    (instrumentation positive-controlled against a real outbound request)
    #    recorded ZERO lookups both WITH the block and with a BARE environment.
    #    They are precautionary — nemo_toolkit installs wandb, sentry-sdk and
    #    nv-one-logger, and a later release wiring up an exporter is exactly the
    #    silent regression this project pins rather than notices. Read by AST, not
    #    by importing: an import cannot tell `setdefault` from an assignment,
    #    which is the whole distinction being pinned.
    if ctx.wants("nemo/offline-env-is-assigned"):
        cid = "nemo/offline-env-is-assigned"
        try:
            required = {"HF_HUB_OFFLINE", "WANDB_MODE", "SENTRY_DSN"}
            assigned, setdefaults = {}, []
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Subscript):
                    target = node.targets[0]
                    if ast.unparse(target.value) == "os.environ" \
                            and isinstance(target.slice, ast.Constant):
                        try:
                            assigned[target.slice.value] = ast.literal_eval(node.value)
                        except Exception:  # noqa: BLE001 — a computed path, e.g. HF_HOME
                            assigned[target.slice.value] = "<computed>"
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "setdefault"
                        and ast.unparse(node.func.value) == "os.environ"):
                    setdefaults.append(ast.unparse(node))

            problems = []
            missing = sorted(required - set(assigned))
            if missing:
                problems.append(f"never assigned: {missing} — nemo_toolkit installs "
                                f"wandb, sentry-sdk and nv-one-logger, and this is the "
                                f"only place they are switched off")
            if setdefaults:
                problems.append(f"os.environ.setdefault used ({setdefaults}) — an "
                                f"inherited value would then win, and the 100%-offline "
                                f"requirement is absolute rather than a default")
            if assigned.get("WANDB_MODE") not in (None, "disabled"):
                problems.append(f"WANDB_MODE is set to {assigned['WANDB_MODE']!r}, "
                                f"expected 'disabled'")
            if assigned.get("SENTRY_DSN") not in (None, ""):
                problems.append(f"SENTRY_DSN is set to {assigned['SENTRY_DSN']!r} — "
                                f"sentry sends nothing only with an EMPTY dsn")

            rep.expect(cid, not problems,
                       f"{len(assigned)} offline/telemetry variables are ASSIGNED "
                       f"({', '.join(sorted(assigned))}) and none uses setdefault, so an "
                       f"inherited value cannot switch any of them back on",
                       "; ".join(problems))
        except Exception as exc:  # noqa: BLE001
            rep.fail(cid, f"could not read the offline environment: {exc!r}")

    # -- check 6: the four MEASURED constants, and the LOCAL checkpoints.
    #
    #    Each constant is a deviation from the vendored upstream config that cost
    #    a real measurement, and each fails SILENTLY if it drifts back:
    #    `num_workers` is a crash, but the other three are wrong ANSWERS —
    #    a speaker lost, or a 48-minute meeting collapsed to one speaker — with
    #    nothing in the transcript saying so. Pinning the constant is not enough
    #    on its own, so the FORWARDING is asserted too: a constant that
    #    `build_config` has stopped reading is the `overlap.detect.model` defect.
    #
    #    The checkpoint half is the offline requirement. `ClusteringDiarizer`
    #    branches on `model_path.endswith('.nemo')`: a PATH is restored off disk,
    #    a bare pretrained NAME goes to NGC over the network. Both spellings work
    #    on this Mac (the checkpoints are in ~/.cache/torch/NeMo), so this is
    #    invisible-until-shipped — the class this project keeps being bitten by.
    if ctx.wants("nemo/measured-constants-are-pinned"):
        cid = "nemo/measured-constants-are-pinned"
        try:
            problems = []
            constants = module_constants(tree)
            for name, want in NEMO_MEASURED_CONSTANTS.items():
                got = constants.get(name)
                if got != want:
                    problems.append(f"{name} is {got!r}, expected {want!r} — see the "
                                    f"measurement recorded at its declaration")

            build = function(tree, "build_config")
            if build is None:
                problems.append("build_config() is gone")
            else:
                forwarded = {ast.unparse(n) for n in ast.walk(build)
                             if isinstance(n, ast.Assign)}
                wanted_assigns = {
                    "cfg.num_workers = NUM_WORKERS",
                    "params.sparse_search_volume = SPARSE_SEARCH_VOLUME",
                    "params.embeddings_per_chunk = EMBEDDINGS_PER_CHUNK",
                    "cfg.diarizer.speaker_embeddings.model_path = SPEAKER_MODEL_PATH",
                    "cfg.diarizer.vad.model_path = VAD_MODEL_PATH",
                    # MAX_NUM_SPEAKERS joined this set on 2026-08-10 and used to
                    # need a special case below. It reached the config through a
                    # per-job `max_speakers` argument, so it could only be checked
                    # as "set from the argument", never by name — and the audit
                    # that removed the argument found nothing had ever SENT it.
                    # Now the constant is assigned directly, which is what lets it
                    # be pinned the same way as the other five. The stock 8 would
                    # silently cap a larger meeting.
                    "params.max_num_speakers = MAX_NUM_SPEAKERS",
                }
                lost = sorted(wanted_assigns - forwarded)
                if lost:
                    problems.append(f"build_config no longer forwards {lost} — the "
                                    f"constant would still be pinned above while the "
                                    f"config quietly used upstream's value")

            # The checkpoints: local absolute .nemo paths under models/nemo, and
            # NOTHING assigns a bare pretrained name to a model_path.
            for name in ("SPEAKER_MODEL_PATH", "VAD_MODEL_PATH", "MODEL_DIR"):
                if name not in {t.id for n in tree.body
                                if isinstance(n, ast.Assign)
                                for t in n.targets if isinstance(t, ast.Name)}:
                    problems.append(f"{name} is gone")
            joins = {t.id: ast.unparse(n.value) for n in tree.body
                     if isinstance(n, ast.Assign)
                     for t in n.targets if isinstance(t, ast.Name)}
            if joins.get("MODEL_DIR") != "os.path.join(BASE, 'models', 'nemo')":
                problems.append(f"MODEL_DIR is {joins.get('MODEL_DIR')!r} — it must be "
                                f"rooted at BASE, which is three dirname calls up "
                                f"(one folder deeper resolves to scripts/models)")
            for name in ("SPEAKER_MODEL_PATH", "VAD_MODEL_PATH"):
                expr = joins.get(name, "")
                if "MODEL_DIR" not in expr or ".nemo" not in expr:
                    problems.append(f"{name} = {expr!r} — it must be a LOCAL .nemo path "
                                    f"under MODEL_DIR; a bare pretrained name sends "
                                    f"ClusteringDiarizer to api.ngc.nvidia.com at "
                                    f"runtime, which works here and fails on a client Mac")
            # THE THREE-DIRNAME TRAP, pinned directly: two would resolve BASE to
            # scripts/, MODEL_DIR to scripts/models, py_compile would pass and it
            # would fail at the first model load.
            base = joins.get("BASE", "")
            if base.count("os.path.dirname") != 3:
                problems.append(f"BASE has {base.count('os.path.dirname')} dirname "
                                f"calls, expected 3 — this file is one folder under "
                                f"scripts/, so two would point MODEL_DIR at "
                                f"scripts/models and fail only at the first model load")

            rep.expect(cid, not problems,
                       f"the {len(NEMO_MEASURED_CONSTANTS)} measured constants hold "
                       f"their values AND are still forwarded into the config, the two "
                       f"checkpoints are local .nemo paths under models/nemo, and BASE "
                       f"has its three dirname calls",
                       "; ".join(problems))
        except Exception as exc:  # noqa: BLE001
            rep.fail(cid, f"could not read the constants: {exc!r}")

    # -- check 7: the vendored config is still upstream's, byte for byte.
    #
    #    The `spectral/vendor-is-own` precedent. The whole design here is "vendor
    #    verbatim, override in CODE", so that every deviation sits next to the
    #    measurement justifying it. An edit to the YAML would move a value with no
    #    comment, no measurement and nothing pointing at it — and because the
    #    sidecar's own overrides are applied AFTERWARDS, an edit to a parameter the
    #    sidecar also sets would be silently invisible while an edit to any other
    #    parameter would silently take effect.
    if ctx.wants("nemo/vendored-config-is-upstream"):
        cid = "nemo/vendored-config-is-upstream"
        try:
            problems = []
            path = NEMO_VENDOR / NEMO_VENDORED_CONFIG
            if not path.exists():
                problems.append(f"{path} is missing — the sidecar loads it at every "
                                f"job and refuses to start without it")
            else:
                got = hashlib.sha256(path.read_bytes()).hexdigest()
                if got != NEMO_VENDORED_CONFIG_SHA:
                    problems.append(f"{NEMO_VENDORED_CONFIG} is NOT upstream's file "
                                    f"({got} != {NEMO_VENDORED_CONFIG_SHA}) — it is "
                                    f"vendored VERBATIM and every deviation belongs in "
                                    f"sidecar code, beside its measurement")

            # And that the sidecar loads THIS copy — its own vendor dir, not
            # another service's (the MOSS `asr-vendor-is-own-and-identical` rule)
            # and not a path that happens to resolve today.
            loads = {t.id: ast.unparse(n.value) for n in tree.body
                     if isinstance(n, ast.Assign)
                     for t in n.targets if isinstance(t, ast.Name)}
            expr = loads.get("VENDORED_CONFIG")
            if expr is None:
                problems.append("VENDORED_CONFIG is gone")
            else:
                try:
                    resolved = pathlib.Path(eval(  # noqa: S307 — our own source
                        expr, {"os": os, "BASE": str(PROJECT),
                               "__file__": str(SCRIPTS / NEMO_SERVICE)}))
                except Exception as exc:  # noqa: BLE001
                    resolved = None
                    problems.append(f"VENDORED_CONFIG {expr!r} could not be resolved: "
                                    f"{exc!r}")
                if resolved is not None and resolved.resolve() != path.resolve():
                    problems.append(f"VENDORED_CONFIG resolves to {resolved} — it must "
                                    f"be this service's OWN {path}; pointing at another "
                                    f"tree works until that folder moves")

            rep.expect(cid, not problems,
                       f"{NEMO_VENDORED_CONFIG} is byte-identical to upstream "
                       f"(NVIDIA-NeMo/Speech @ 6c57e73e) and the sidecar loads its own "
                       f"copy — every deviation lives in code, beside its measurement",
                       "; ".join(problems))
        except Exception as exc:  # noqa: BLE001
            rep.fail(cid, f"could not check the vendored config: {exc!r}")


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
    "layout/tail-window-start-is-recorded-not-derived",
    "layout/diarization-settings-locked-per-session",
    "layout/batch-engines-honour-final-pass",
    "layout/remote-passes-never-send-the-room-count",
    "layout/pdf-export-uses-fixed-ink",
    "layout/error-popup-marker-is-cleared-with-the-message",
    "layout/no-speech-is-routed-by-kind-not-prose",
    "layout/hub-downloads-fetch-their-model-card",
    "layout/inert-speaker-count-says-why-once",
    "layout/live-window-failures-leave-the-stop-gate-alone",
    "layout/no-banner-hard-codes-an-engine-name",
    "layout/the-startup-overlay-always-has-a-way-out",
    "layout/the-moss-full-pass-rebuilds-speakers",
    "layout/the-overwriting-migration-runs-once",
    "layout/shipped-defaults-have-one-source",
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

    cid = "layout/shipped-defaults-have-one-source"
    if ctx.wants(cid):
        # THE FAILURE THIS GUARDS IS SILENT AND USER-VISIBLE IN THE WORST WAY.
        # Before ShippedDefaults existed, the shipped default for
        # `overlap.detect.enabled` was written EIGHT times — three
        # `@AppStorage(...) = false` declarations in Settings and five
        # `d.object(forKey:) as? Bool ?? false` reads in ModelLoader and
        # AudioRecorder — and `chunked.model` was written TWELVE times. Change
        # one copy and miss another and the Settings toggle draws ON while the
        # loader, reading its own literal, never loads the model. Nothing fails,
        # nothing is logged, and the app quietly does not do what the UI says.
        #
        # This is the project's own recurring defect — two readers of one fact —
        # and the same reasoning as `layout/no-banner-hard-codes-an-engine-name`:
        # a hand-written copy of a value beside a derived one only ever drifts.
        #
        # Both halves are asserted. The POSITIVE half matters as much: a file
        # that had simply stopped declaring these constants would satisfy "no
        # literal defaults" trivially.
        keys = {
            "overlap.detect.enabled": "overlapDetect",
            "overlap.repair.enabled": "overlapRepair",
            "realtime.model": "realtimeModel",
            "diarization.engine": "diarizationEngine",
            "chunked.model": "chunkedModel",
        }
        # An APPROVED ALIAS is a name that is *defined as* one of these
        # constants, not a second copy of its value — `defaultModelID` reads far
        # better at a RealtimeASRService call site than the fully qualified name
        # would, and it cannot drift because it is derived. The definition itself
        # is asserted below, so the alias cannot quietly become a literal.
        aliases = {"realtime.model": "RealtimeASRService.defaultModelID"}
        problems = []
        decl = SWIFT_SOURCES / "MeetingTranscriber" / "App" / "ShippedDefaults.swift"
        if not decl.exists():
            problems.append("ShippedDefaults.swift is gone — every default is back "
                            "to being written once per reader")
        else:
            declared = decl.read_text()
            for key, name in keys.items():
                if f"static let {name}" not in declared:
                    problems.append(f"ShippedDefaults no longer declares {name} "
                                    f"(the default for {key})")

        # ...and no reader may state a default of its own. Comments are stripped
        # first: this file documents the literals it forbids, so a plain search
        # matches its own explanation — the same trap `assert_no_torchcodec_use`
        # and `layout/tail-window-start-is-recorded-not-derived` both record.
        for path in sorted(SWIFT_SOURCES.rglob("*.swift")):
            if path.name == "ShippedDefaults.swift":
                continue
            for lineno, line in enumerate(path.read_text().splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("//"):
                    continue
                for key in keys:
                    if f'"{key}"' not in line:
                        continue
                    # A default is expressed either as an @AppStorage initialiser
                    # or as a `??` fallback on the same line. Both must name
                    # ShippedDefaults; a bare literal is the drift.
                    is_default = ("@AppStorage" in line and "=" in line) or "??" in line
                    ok = "ShippedDefaults" in line
                    alias = aliases.get(key)
                    if alias and (alias in line or alias.split(".")[-1] in line):
                        ok = True
                    if is_default and not ok:
                        problems.append(
                            f"{path.name}:{lineno} states its own default for "
                            f"{key} — it must read ShippedDefaults.{keys[key]}")
        # ...and every approved alias really is DERIVED. Without this the alias
        # rule above would let `defaultModelID = "nemotron"` walk straight back
        # in — an allowlist that does not check what it allows is a hole.
        rt = (SWIFT_SOURCES / "MeetingTranscriber" / "Audio"
              / "RealtimeASRService.swift")
        if rt.exists() and ("static let defaultModelID = ShippedDefaults.realtimeModel"
                            not in rt.read_text()):
            problems.append("RealtimeASRService.defaultModelID is no longer defined "
                            "as ShippedDefaults.realtimeModel — the alias this check "
                            "accepts has become a value of its own")
        rep.expect(cid, not problems,
                   f"all {len(keys)} shipped defaults are declared once in "
                   f"ShippedDefaults, the one approved alias is derived from it, "
                   f"and no reader states its own",
                   "; ".join(problems))

    cid = "layout/the-overwriting-migration-runs-once"
    if ctx.wants(cid):
        # `adoptMeasuredEngineDefaults` is the ONLY migration in this app that
        # overwrites a value the user already chose. That is acceptable for
        # exactly one reason — it can never run twice — and the failure if the
        # marker guard is ever lost is severe AND silent: the two settings become
        # permanently unchangeable, because every launch would put them back.
        # The user would see their choice "not save" with nothing in the UI to
        # explain it.
        #
        # A Swift test cannot cover this: it would have to write UserDefaults,
        # and the suite runs in the owner's REAL preference domain (the standing
        # rule that also keeps `layout/diarization-settings-locked-per-session` a
        # source scan rather than a test).
        src = (SWIFT_SOURCES / "MeetingTranscriber" / "App"
               / "MeetingTranscriberApp.swift").read_text()
        body = "\n".join(l for l in src.splitlines()
                          if not l.strip().startswith("//") and not l.strip().startswith("///"))
        problems = []
        if "func adoptMeasuredEngineDefaults" not in body:
            problems.append("adoptMeasuredEngineDefaults is gone")
        else:
            fn = body.split("func adoptMeasuredEngineDefaults", 1)[1]
            if "guard !d.bool(forKey: marker)" not in fn:
                problems.append("the run-once guard is gone — the migration would "
                                "overwrite the engine and model on EVERY launch, "
                                "making both permanently unchangeable")
            # The marker must be written BEFORE the values, so a crash midway
            # leaves it done rather than pending: at-most-once beats at-least-once
            # for a migration that overwrites a deliberate choice.
            before = fn.find("d.set(true, forKey: marker)")
            first_write = fn.find('d.set(value, forKey: key)')
            if before < 0:
                problems.append("the marker is never set — the guard can never trip")
            elif first_write >= 0 and before > first_write:
                problems.append("the marker is set AFTER the values are written; a "
                                "crash in between would re-run an overwriting migration")
            if "adoptMeasuredEngineDefaults(d)" not in body:
                problems.append("nothing calls it — the migration is dead code")
        rep.expect(cid, not problems,
                   "the one overwriting migration is guarded by a marker written "
                   "before it changes anything, and is actually called",
                   "; ".join(problems))

    cid = "layout/the-moss-full-pass-rebuilds-speakers"
    if ctx.wants(cid):
        # A Swift test cannot see a CALL SITE (the 2026-08-06 MOSS lesson), and
        # this feature is three call sites that must all be present together. Any
        # one of them missing fails SILENTLY and in the worst direction:
        #
        #   * segments not forwarded  -> the full pass re-transcribes the meeting
        #                                and leaves NOBODY in it, which is worse
        #                                than the live chunks it just deleted
        #   * mossTurns not cleared   -> the live labels and the rebuilt ones both
        #                                stand, with different chunk indices, so
        #                                the meeting has two overlapping sets of
        #                                speaker spans (the 2026-08-05 duplication)
        #   * identify flag not reset -> `startMossIdentifyForOwnASR` short-circuits
        #                                on the live session's run and the rebuilt
        #                                windows are never stitched
        # Comments stripped first, for the reason the 2026-08-05 check learned the
        # hard way: this file documents the change at length, so a plain search
        # matches its own explanation.
        src = (SWIFT_SOURCES / "MeetingTranscriber" / "Audio"
               / "AudioRecorder+ChunkedStop.swift").read_text()
        body = "\n".join(l for l in src.splitlines()
                         if not l.strip().startswith("//"))
        missing = []
        if "mossSegments: result.segments" not in body:
            missing.append("the full pass does not forward `result.segments` to "
                           "replaceOfficeSegments — a MOSS full pass would produce "
                           "text with no speakers at all")
        for token, why in (("mossTurns = []", "the live labels are not cleared"),
                           ("mossChunkIndex = 0", "window numbering continues the live count, "
                                                  "so identify groups windows that no longer exist"),
                           ("mossIdentifyStarted = false", "identity cannot run again, so the "
                                                           "rebuilt windows are never stitched")):
            if token not in body:
                missing.append(f"`{token}` is absent — {why}")
        # The POSITIVE half, and it is what stops this passing on a file that had
        # simply stopped doing a full pass at all.
        if "mossPinnedSegments(from:" not in body:
            missing.append("replaceOfficeSegments no longer builds pinned MOSS rows")
        if "chunkedFullPassRefusalMessage" in body and '"moss"' in body.split(
                "chunkedFullPassRefusalMessage")[1][:1200]:
            missing.append("MOSS looks refused again in chunkedFullPassRefusalMessage")
        rep.expect(cid, not missing,
                   "the MOSS full pass forwards segments, clears the live labels, "
                   "restarts window numbering, re-arms identity and builds pinned rows",
                   "; ".join(missing))

    cid = "layout/tail-window-start-is-recorded-not-derived"
    if ctx.wants(cid):
        # `diarizeTailChunk` must NOT ask `diarization.live` where its buffer
        # started. It used to, and Settings is reachable WHILE RECORDING (the gear
        # button carries no `.disabled`), while `diarization.live` is re-read per
        # chunk — so turning live labels off mid-meeting with continue-on-stop on
        # makes the buffer start accumulating at that MOMENT, and the stop-time
        # read then declared it started at 0. Every tail turn came out shifted
        # earlier by however long the meeting had already run, silently.
        #
        # The fix records `chunkAudioStart` at each clear, so this check pins the
        # ABSENCE of the derivation AND the presence of the recorded fact — the
        # second half matters, since a file that had simply stopped computing a
        # window start would pass the first half alone.
        src = (SWIFT_SOURCES / "MeetingTranscriber" / "Audio" / "AudioRecorder+OfficeDiarization.swift").read_text()
        problems = []
        start = src.find("func diarizeTailChunk()")
        if start < 0:
            problems.append("diarizeTailChunk() is gone — re-derive this check")
        else:
            # COMMENTS STRIPPED FIRST, and this check learned that the hard way:
            # its first version failed on the very comment that explains why the
            # read is forbidden. Same lesson as `assert_no_torchcodec_use` and the
            # AST checks — a textual search matches its own documentation. Ended
            # at the next declaration rather than by a byte count, so the window is
            # the function and not "the function plus whatever follows".
            end = src.find("\n    func ", start + 1)
            body = src[start:end if end > 0 else len(src)]
            body = "\n".join(l for l in body.splitlines()
                              if not l.strip().startswith("//"))
            if "diarization.live" in body:
                problems.append("diarizeTailChunk reads `diarization.live` again — a "
                                "setting read at stop time cannot describe when the "
                                "buffer actually started")
            if "chunkAudioStart" not in body:
                problems.append("diarizeTailChunk no longer uses `chunkAudioStart`")
        if "var chunkAudioStart" not in (SWIFT_SOURCES / "MeetingTranscriber" / "Audio" / "AudioRecorder.swift").read_text():
            problems.append("`chunkAudioStart` is gone from AudioRecorder")
        rep.expect(cid, not problems,
                   "the tail pass takes its window start from the recorded "
                   "`chunkAudioStart`, never from a stop-time `diarization.live` read",
                   "; ".join(problems))

    cid = "layout/diarization-settings-locked-per-session"
    if ctx.wants(cid):
        # `diarization.live`, `.continueOnStop` and `.detectOverlap` may be read
        # in exactly ONE place: `lockDiarizationSettings`, at `beginCapture`.
        #
        # They used to be read BOTH per chunk and again at Stop, and Settings is
        # reachable while recording (the gear button carries no `.disabled`), so
        # changing one mid-meeting made the two reads DISAGREE about the same
        # recording: `live`+`continueOnStop` decide per chunk whether `chunkAudio`
        # is kept for a tail AND decide at Stop which pass runs, so audio was
        # dropped that the tail then needed; `detectOverlap` becomes `exclusive` on
        # the wire, so half a meeting could be diarized with overlap detection and
        # half without, with nothing marking the seam.
        #
        # Comments stripped first — the files that carry this fix explain it at
        # length, and a textual search matches its own documentation (the lesson
        # this check's sibling learned by failing on its own comment).
        locked = ("diarization.live", "diarization.continueOnStop",
                  "diarization.detectOverlap")
        problems = []
        audio_dir = SWIFT_SOURCES / "MeetingTranscriber" / "Audio"
        for path in sorted(audio_dir.glob("*.swift")):
            code = "\n".join(l for l in path.read_text().splitlines()
                              if not l.strip().startswith("//"))
            for key in locked:
                needle = f'forKey: "{key}"'
                if needle not in code:
                    continue
                if path.name != "AudioRecorder.swift":
                    problems.append(f"{path.name} reads {key} — it must use the "
                                    f"session-locked property instead")
                    continue
                # In AudioRecorder.swift the ONLY legal reader is the locker.
                start = code.find("func lockDiarizationSettings()")
                end = code.find("\n    }", start) if start >= 0 else -1
                body = code[start:end] if start >= 0 and end > 0 else ""
                if code.count(needle) != body.count(needle):
                    problems.append(f"{key} is read outside lockDiarizationSettings()")
        if "func lockDiarizationSettings()" not in (
                audio_dir / "AudioRecorder.swift").read_text():
            problems.append("lockDiarizationSettings() is gone")
        rep.expect(cid, not problems,
                   "the three session-scoped diarization settings are read once, in "
                   "lockDiarizationSettings(), and nowhere else",
                   "; ".join(problems))

    cid = "layout/batch-engines-honour-final-pass"
    if ctx.wants(cid):
        # ⚠ THIS CHECK WAS REVERSED ON 2026-08-13, and the old one was RIGHT for
        # the world it was written in. It asserted that no whole-file engine may
        # read `diarization.finalPass`, because the toggle was HIDDEN under those
        # engines and their stop pass WAS their only source of labels — so a
        # `false` left behind by a pyannote session deleted every label with
        # nothing in the UI able to undo it. That is a value outliving its control,
        # which the 2026-08-06 settings pass exists to forbid.
        #
        # Both halves of that premise are now gone. The owner asked for the toggle
        # on every engine, and those four gained a live per-interval path to fall
        # back to (`AudioRecorder+BatchLiveDiarization`), so the setting is
        # reachable AND switching it off no longer means "no labels". Honouring it
        # is what the control now means, and NOT honouring it would be the lying
        # control this project keeps having to fix.
        #
        # The POSITIVE half is unchanged and is what lets this fail at all:
        # pyannote's remote pass must still read the key too, so a file that had
        # simply stopped doing remote diarization cannot pass.
        needle = 'forKey: "diarization.finalPass"'
        audio_dir = SWIFT_SOURCES / "MeetingTranscriber" / "Audio"

        def _code(name):
            return "\n".join(l for l in (audio_dir / name).read_text().splitlines()
                             if not l.strip().startswith("//"))

        # ONE NAME PER BATCH ENGINE, and a new engine must be added here — the
        # DiariZen lesson: it landed while this list still named two files, so the
        # third engine's remote pass went unpinned.
        problems = []
        for name in ("AudioRecorder+Spectral.swift", "AudioRecorder+Nemo.swift",
                     "AudioRecorder+Diarizen.swift", "AudioRecorder+CamPlus.swift"):
            code = _code(name)
            if needle not in code:
                problems.append(f"{name} no longer reads diarization.finalPass — "
                                f"its Run-at-stop toggle is visible in the tab, so "
                                f"ignoring it would be a control that does nothing")
            if "finalPass: true" in code:
                problems.append(f"{name} still forces finalPass: true — that was "
                                f"correct only while the toggle was hidden")
        pyannote_remote = _code("AudioRecorder+RemoteDiarization.swift")
        if needle not in pyannote_remote:
            problems.append("AudioRecorder+RemoteDiarization.swift no longer reads "
                            "diarization.finalPass — pyannote's toggle must still be "
                            "honoured, so this check has lost the half that lets it "
                            "fail")
        rep.expect(cid, not problems,
                   "every engine's stop pass honours diarization.finalPass, and no "
                   "batch engine forces it true any more",
                   "; ".join(problems))

    cid = "layout/remote-passes-never-send-the-room-count"
    if ctx.wants(cid):
        # A REMOTE stop pass may not send `AudioRecorder.diarNumSpeakers`.
        #
        # That value is the ROOM's headcount — the SPK chip sits beside the room's
        # RMS meter and its own doc says "how many people are in the room". Nothing
        # anywhere asks how many people are on the far end of the call, so the
        # number is not a fact about the Remote WAV, and the two streams are
        # separate identity spaces precisely because they hold different people.
        #
        # Sending it was not inert: a pinned count is an EXACT constraint on three
        # of the four pipeline engines (pyannote and spectral pass `num_speakers=`,
        # NeMo turns it into `oracle_num_speakers`), so a 5-person room with one
        # caller split that caller into five remote profiles. Fabrication is the
        # direction this project treats as worse everywhere else.
        #
        # THE POSITIVE HALF IS LOAD-BEARING: the OFFICE passes must still send it.
        # Without that, a file that had simply stopped sending any count would pass
        # this check while quietly disabling the one control the owner asked for —
        # and it is spectral, the engine a count measurably fixes, that would lose
        # the most.
        audio_dir = SWIFT_SOURCES / "MeetingTranscriber" / "Audio"

        def _body(name, func):
            """The source of one method, comments stripped, up to the next `func`."""
            lines = (audio_dir / name).read_text().splitlines()
            out, inside = [], False
            for line in lines:
                if not inside:
                    if line.strip().startswith(f"func {func}"):
                        inside = True
                    continue
                if line.strip().startswith("func "):
                    break
                if line.strip().startswith("//"):
                    continue
                out.append(line)
            return "\n".join(out) if inside else None

        # ONE ENTRY PER ENGINE, remote and office side by side, so a new engine has
        # to be added to BOTH columns — the `batch-engines-ignore-final-pass` list
        # was left naming two engines when the third landed, and that is the exact
        # oversight this shape is meant to make visible.
        pairs = [
            ("AudioRecorder+RemoteDiarization.swift", "startRemoteFullDiarization",
             "AudioRecorder+OfficeDiarization.swift", "startDiarization"),
            ("AudioRecorder+Spectral.swift", "startRemoteSpectralDiarization",
             "AudioRecorder+Spectral.swift", "startSpectralDiarization"),
            ("AudioRecorder+Nemo.swift", "startRemoteNemoDiarization",
             "AudioRecorder+Nemo.swift", "startNemoDiarization"),
            ("AudioRecorder+Diarizen.swift", "startRemoteDiarizenDiarization",
             "AudioRecorder+Diarizen.swift", "startDiarizenDiarization"),
            ("AudioRecorder+CamPlus.swift", "startRemoteCamPlusDiarization",
             "AudioRecorder+CamPlus.swift", "startCamPlusDiarization"),
        ]
        problems = []
        for rfile, rfunc, ofile, ofunc in pairs:
            remote = _body(rfile, rfunc)
            office = _body(ofile, ofunc)
            if remote is None:
                problems.append(f"{rfile}: {rfunc} is gone")
            else:
                if "diarNumSpeakers" in remote:
                    problems.append(f"{rfunc} sends diarNumSpeakers — that is the "
                                    f"ROOM's count and the remote stream holds "
                                    f"different people; use remoteNumSpeakers")
                if "remoteNumSpeakers" not in remote:
                    problems.append(f"{rfunc} names no speaker count at all — it "
                                    f"must state the auto rule explicitly, not "
                                    f"omit the parameter")
            if office is None:
                problems.append(f"{ofile}: {ofunc} is gone")
            elif "diarNumSpeakers" not in office:
                problems.append(f"{ofunc} no longer sends diarNumSpeakers — the "
                                f"office pass is where the control is honoured, "
                                f"so this check has lost the half that lets it fail")
        rep.expect(cid, not problems,
                   "all four remote passes send remoteNumSpeakers (auto) while all "
                   "four office passes still honour the SPK control",
                   "; ".join(problems))

    cid = "layout/pdf-export-uses-fixed-ink"
    if ctx.wants(cid):
        # The PDF exporter may not draw with a DYNAMIC system colour.
        #
        # `NSColor.textColor`, `.labelColor`, `.secondaryLabelColor` and friends
        # resolve against the CURRENT APPEARANCE. The app runs dark, so on
        # 2026-08-10 the exported transcript was near-white ink on a page, and the
        # only visible element was the title — which alone carried no colour and
        # therefore defaulted to black. The owner found it by opening the file.
        #
        # PINNED HERE RATHER THAN IN A UNIT TEST because a unit test cannot see it:
        # a headless XCTest process has no application appearance, so the same
        # colour resolves to black there and the render looks perfect. Measured,
        # including with `performAsCurrentDrawingAppearance(.darkAqua)`, which does
        # not help. `ExportTests` covers the transparent-page half; this covers this
        # one.
        #
        # The POSITIVE half — that the file really does define fixed ink — is what
        # stops a file that had simply stopped drawing text from passing.
        path = SWIFT_SOURCES / "MeetingTranscriber" / "Export" / "TranscriptPDF.swift"
        problems = []
        if not path.exists():
            problems.append("TranscriptPDF.swift is gone")
        else:
            code = "\n".join(l for l in path.read_text().splitlines()
                             if not l.strip().startswith("//")
                             and not l.strip().startswith("///"))
            # AN ALLOWLIST, NOT A DENYLIST — the `assert_no_torchcodec_use` shape.
            #
            # This was six banned names, which is the special case: `.systemGray`,
            # `.headerTextColor`, `.textBackgroundColor`, a bridged SwiftUI
            # `Color(...)` or anything from `Theme` would all have passed while
            # being just as appearance-dependent. The RULE is "every colour in this
            # file is a literal component initialiser", so that is what is checked.
            # Anything else — a named system colour, a semantic colour, a Theme
            # token — fails by default, which is the right direction for a file
            # whose output is printed.
            allowed_init = re.compile(r"NSColor\((?:white|red|calibratedWhite|calibratedRed|"
                                      r"deviceWhite|deviceRed|displayP3Red|genericGamma22White):")
            for m in re.finditer(r"NSColor\s*(?:\.\s*(\w+)|\()", code):
                token = m.group(0)
                if token.rstrip().endswith("("):
                    if not allowed_init.match(code[m.start():m.start() + 60]):
                        problems.append(f"a non-component NSColor initialiser at "
                                        f"offset {m.start()} — printed ink must be "
                                        f"literal components")
                else:
                    problems.append(f"NSColor.{m.group(1)} is a NAMED system colour; "
                                    f"named colours resolve against the current "
                                    f"appearance and render near-white in dark mode. "
                                    f"A printed page has one appearance — use "
                                    f"NSColor(white:) or another component form")
            if "NSColor(white:" not in code:
                problems.append("no fixed NSColor(white:) ink is defined — this "
                                "check has lost the half that lets it fail")
        rep.expect(cid, not problems,
                   "the PDF exporter draws with fixed ink, never a dynamic "
                   "appearance colour",
                   "; ".join(problems))

    cid = "layout/error-popup-marker-is-cleared-with-the-message"
    if ctx.wants(cid):
        # `dismissedErrorMessage` holds the error the user has already closed, and
        # `showsErrorPopup` compares it against the CURRENT one. That comparison is
        # what lets the popup work without any of the nine `errorMessage = …` sites
        # remembering to reset a flag — but it makes the two CLEARS a pair: a
        # cleared message beside a surviving marker means the identical error
        # cannot raise the popup a second time.
        #
        # That case is the commonest one, not a corner: a denied microphone is
        # still denied when the user presses Start again, and `start()` clears
        # `errorMessage` on entry before re-assigning the same text. Left alone,
        # the second press would produce no visible outcome at all.
        #
        # PINNED HERE RATHER THAN IN A UNIT TEST because a unit test cannot reach
        # it — `start()` is private and hands off to `AVCaptureDevice.requestAccess`,
        # which would put a system permission prompt in the middle of the suite.
        # `FailurePanelTests` asserts the RULE; this asserts the CALL SITES obey
        # it, which is the 2026-08-06 MOSS lesson (a component test cannot see a
        # missing call site) applied to a two-line coupling.
        path = SWIFT_SOURCES / "MeetingTranscriber" / "Audio" / "AudioRecorder.swift"
        problems = []
        if not path.exists():
            problems.append("AudioRecorder.swift is gone")
        else:
            # Comments stripped first — this fix explains itself at length in the
            # very file it checks, and a textual search matches its own
            # documentation (the lesson this check's siblings learned by failing
            # on their own comments).
            lines = [l for l in path.read_text().splitlines()
                     if not l.strip().startswith("//")]
            # `dismissedErrorMessage` carries a capital E, so the lowercase needle
            # cannot match the marker's own clear. Verified below rather than
            # assumed — if that ever stops holding, this check silently passes.
            if "errorMessage = nil" in "dismissedErrorMessage = nil":
                problems.append("the needle now matches the marker's own clear, so "
                                "this check can no longer tell the two apart")
            clears = [i for i, l in enumerate(lines)
                      if re.search(r"(?<![A-Za-z])errorMessage\s*=\s*nil", l)]
            if not clears:
                problems.append("nothing clears `errorMessage` any more — this "
                                "check has lost the half that lets it fail")
            for i in clears:
                window = "\n".join(lines[i:i + 8])
                if "dismissedErrorMessage = nil" not in window:
                    problems.append(f"`errorMessage = nil` at line {i + 1} does not "
                                    f"clear `dismissedErrorMessage` with it — the "
                                    f"same error would then be unable to raise the "
                                    f"popup a second time")
            if "var dismissedErrorMessage" not in "\n".join(lines):
                problems.append("`dismissedErrorMessage` is gone from AudioRecorder")
        rep.expect(cid, not problems,
                   "every clear of `errorMessage` clears the already-read marker "
                   "with it, so an identical second failure is still shown",
                   "; ".join(problems))

    cid = "layout/the-startup-overlay-always-has-a-way-out"
    if ctx.wants(cid):
        # A client Mac showed the startup overlay with CAM++ marked failed, the
        # models below it never attempted, the header still reading "Loading
        # models" and NO Close button — the app unusable, Settings unreachable
        # (2026-08-18). `failureMessage` and the row's `.failed` state are set one
        # line apart in `loadAll`, so how they disagreed there is not established;
        # the fix makes the way out depend on the rows the user can SEE.
        #
        # PINNED HERE BECAUSE SWIFT CANNOT. `ModelLoader.items` is `private(set)`
        # — correctly, it is the loader's own record — so no test can pose "a
        # failed row with no message", which is exactly the reported state. The
        # negative control proved it: reverting `hasFailure` to the message alone
        # left the whole Swift suite green.
        #
        # Two halves, because either alone is satisfiable by a broken pair: the
        # rule must read the ROWS, and dismissing must CLEAR them — without the
        # second, Close clears the message, the red row still asserts a failure,
        # and the button is dead. That regression was written and caught inside
        # this same change.
        loader = (SWIFT_SOURCES / "MeetingTranscriber" / "Models" / "ModelLoader.swift")
        view = (SWIFT_SOURCES / "MeetingTranscriber" / "Views" / "Main"
                / "LoadingOverlayView.swift")
        problems = []
        for path, label in [(loader, "ModelLoader.swift"), (view, "LoadingOverlayView.swift")]:
            if not path.exists():
                problems.append(f"{label} is gone")
        if not problems:
            # Comments stripped — both files explain this trap at length, so a
            # textual search matches its own documentation.
            code = "\n".join(l for l in loader.read_text().splitlines()
                              if not l.strip().startswith("//"))
            vcode = "\n".join(l for l in view.read_text().splitlines()
                               if not l.strip().startswith("//"))
            body = code.split("var hasFailure")[-1].split("}")[0] if "var hasFailure" in code else ""
            if "var hasFailure" not in code:
                problems.append("ModelLoader has no `hasFailure` — the overlay's way "
                                "out is no longer a rule anything can pin")
            elif "isFailed" not in body:
                problems.append("`hasFailure` no longer reads the rows, so a failed "
                                "row with no failureMessage traps the user again — "
                                "the exact state reported from a client Mac")
            dismiss = code.split("func dismissFailure")[-1].split("}")[0] if "func dismissFailure" in code else ""
            if "items = []" not in dismiss:
                problems.append("`dismissFailure` no longer clears the rows, so Close "
                                "leaves a red row asserting a failure and the overlay "
                                "never goes — a dead button")
            if "loader.hasFailure" not in vcode:
                problems.append("LoadingOverlayView no longer asks the loader — it "
                                "has its own copy of the rule to drift from")
        rep.expect(cid, not problems,
                   "the startup overlay's Close depends on the rows the user can see, "
                   "and dismissing really clears them",
                   "; ".join(problems))

    cid = "layout/no-banner-hard-codes-an-engine-name"
    if ctx.wants(cid):
        # THE CLASS THAT WILL NOT DIE. Four separate instances now, all the same
        # shape — a hand-written engine name in user-facing text, going stale the
        # moment an engine is added:
        #
        #   2026-08-10  the settings rail printed "pyannote" for a DiariZen
        #               session, via a `default:` arm;
        #   2026-08-13  three engine LISTS in Views/, every one stale, found in
        #               one sweep;
        #   2026-08-14  the Overlap tab's banner said "The spectral engine" to a
        #               CAM++ user — a two-way ternary acting as a `default:`, in
        #               a block whose own comment claimed that had been fixed.
        #
        # The last one is why this check exists rather than another comment: the
        # comment WAS there, it asserted the fix, and the code under it was the
        # bug. `ModelCatalog.diarizationEngineShortName` returns nil for an unknown
        # engine instead of somebody else's name, so deriving is the only form that
        # a seventh engine cannot make wrong.
        #
        # Scoped to the DISPLAY NAMES, not to the ids: `ModelLoader.spectralEngineID`
        # and friends are comparisons, which is how a legitimately engine-specific
        # branch is written. It is the rendered noun that must never be typed.
        NAMES = ["The spectral engine", "The NeMo engine", "The CAM++ engine",
                 "The DiariZen engine", "The pyannote engine"]
        views = SWIFT_SOURCES / "MeetingTranscriber" / "Views"
        problems = []
        if not views.is_dir():
            problems.append("Views/ is gone")
        else:
            hits = 0
            for path in sorted(views.rglob("*.swift")):
                # Comments stripped — this check's own explanation names every one
                # of these strings, and so do the fixed sites' comments. A textual
                # search would match its own documentation, the lesson every
                # sibling check here learned by failing on itself.
                code = "\n".join(l for l in path.read_text().splitlines()
                                 if not l.strip().startswith("//"))
                for name in NAMES:
                    if f'"{name}' not in code:
                        continue
                    hits += 1
                    # ONE EXEMPTION, and it carries its reason: OverlapDetectTab's
                    # banner is shown only under pyannote (`redundantHere` is
                    # `diarOn && engine == pyannoteEngineID`), so naming pyannote
                    # there is a statement about the engine the user selected, not
                    # a fallback. Verified rather than assumed, below.
                    if path.name == "OverlapDetectTab.swift" and name == "The pyannote engine":
                        if "engine == ModelLoader.pyannoteEngineID" not in code:
                            problems.append(
                                "OverlapDetectTab names pyannote but no longer gates "
                                "on it — the exemption's premise is gone")
                        continue
                    problems.append(
                        f"{path.name} hard-codes {name!r} in user-facing text. Any "
                        f"engine reaching that branch and not being that engine is "
                        f"told about somebody else's — use "
                        f"ModelCatalog.diarizationEngineShortName")
            # The POSITIVE half: the derivation really is in use somewhere. Without
            # it, a Views/ that had stopped naming engines at all — or been deleted
            # — would pass while saying nothing.
            settings = views / "Settings" / "SettingsView.swift"
            if settings.exists():
                scode = "\n".join(l for l in settings.read_text().splitlines()
                                  if not l.strip().startswith("//"))
                if "diarizationEngineShortName" not in scode:
                    problems.append("SettingsView no longer derives an engine name — "
                                    "this check has lost the half that lets it fail")
        rep.expect(cid, not problems,
                   "no banner types an engine's display name except the one shown "
                   "only under that engine; the rest derive it",
                   "; ".join(problems))

    cid = "layout/live-window-failures-leave-the-stop-gate-alone"
    if ctx.wants(cid):
        # 🔴 THE 2026-08-14 DEFECT. All four whole-file engines routed EVERY
        # sidecar error into `handleDiarizationFailure` — the STOP-GATE handler,
        # which sets `diarizationError` (a red banner across the running
        # transcript) and `finalDiarDone`. Once those engines gained a live path,
        # that fired mid-recording for a single window, and NeMo returns an error
        # for 30 s of silence — an ordinary quiet stretch of a meeting.
        #
        # PINNED HERE RATHER THAN IN A SWIFT TEST because it is a CALL-SITE fact,
        # four times over. `handleBatchLiveFailure` is a pure-ish rule a unit test
        # can exercise, and exercising it proves nothing about whether the four
        # callbacks ask it first — the 2026-08-06 MOSS lesson exactly, which is
        # also how the bug got in: the rule was written and three of the four
        # call sites already existed.
        #
        # The 1 s floor is here for the same reason: it lives inside a `func` that
        # writes a temp WAV and dispatches, so nothing pure can see it, and the
        # negative control that removed it left the whole Swift suite green.
        engines = ["CamPlus", "Nemo", "Diarizen", "Spectral"]
        problems = []
        for e in engines:
            path = (SWIFT_SOURCES / "MeetingTranscriber" / "Audio"
                    / f"AudioRecorder+{e}.swift")
            if not path.exists():
                problems.append(f"AudioRecorder+{e}.swift is gone")
                continue
            # Comments stripped — every one of these files explains this trap at
            # length, so a textual search matches its own documentation. The
            # lesson each sibling check here learned by failing on its own comment.
            lines = [l for l in path.read_text().splitlines()
                     if not l.strip().startswith("//")]
            code = "\n".join(lines)
            calls = [i for i, l in enumerate(lines)
                     if "handleDiarizationFailure(" in l]
            if not calls:
                problems.append(f"{e} no longer calls handleDiarizationFailure — "
                                f"this check has lost the half that lets it fail")
                continue
            for i in calls:
                # Only the CALLBACK sites matter. The `guard let service` arm sets
                # the error itself and never reaches this helper, so it is
                # identified by the absence of a surrounding callback.
                #
                # ⚠ THE WINDOW IS BOUNDED AT THE NEAREST `Task { @MainActor in`,
                # NOT AT A FIXED LINE COUNT. It was `lines[i-10:i]` for one
                # revision, and the negative control caught it: `onError` and
                # `onNoSpeech` sit a few lines apart in NeMo, so a fixed lookback
                # spanned into the PREVIOUS callback and found ITS guard. Deleting
                # the onNoSpeech guard — the exact path a silent 30 s window takes
                # — left the check green. A callback's guard must be found inside
                # that callback.
                start = None
                for k in range(i - 1, -1, -1):
                    if "Task { @MainActor in" in lines[k]:
                        start = k
                        break
                if start is None:
                    continue
                window = "\n".join(lines[start:i])
                if "handleBatchLiveFailure(" not in window:
                    problems.append(f"{e}: a sidecar-callback call to "
                                    f"handleDiarizationFailure at line {i + 1} is "
                                    f"not preceded by handleBatchLiveFailure — a "
                                    f"live window's failure would paint the running "
                                    f"transcript red and settle the stop gate")
            if "handleBatchLiveFailure(" not in code:
                problems.append(f"{e} never consults handleBatchLiveFailure")

        bl = (SWIFT_SOURCES / "MeetingTranscriber" / "Audio"
              / "AudioRecorder+BatchLiveDiarization.swift")
        if not bl.exists():
            problems.append("AudioRecorder+BatchLiveDiarization.swift is gone")
        else:
            blcode = "\n".join(l for l in bl.read_text().splitlines()
                               if not l.strip().startswith("//"))
            # The SAME 1 s floor `diarizeTailChunk` applies. A sub-second window is
            # what makes the auto count fabricate and what NeMo raises on.
            if "guard samples.count > 16_000" not in blcode:
                problems.append("dispatchBatchLiveWindow lost its 1 s floor — a "
                                "sub-second window is the worst input these "
                                "engines take and NeMo raises on one")
            # And the release really empties the map, not just deletes files.
            if "liveDiarWindowByPath = [:]" not in blcode:
                problems.append("releaseBatchLiveWindows no longer empties the map")
        rep.expect(cid, not problems,
                   "all four engines route a live window's failure away from the "
                   "stop-gate handler, and the live dispatch keeps its 1 s floor",
                   "; ".join(problems))

    cid = "layout/inert-speaker-count-says-why-once"
    if ctx.wants(cid):
        # The SPK picker tells the user why a set number will not reach the
        # engine, in TWO places — the tooltip and a line under the pickers
        # (owner, 2026-08-14). Both must come from `inertReason`.
        #
        # Writing the on-screen warning as its own string is the obvious move and
        # is the mistake this project keeps paying for: two expressions of one
        # fact drift, and the drift is always toward under-reporting. On
        # 2026-08-13 a single sweep found THREE hand-written engine lists in
        # `Views/`, every one of them stale, each hiding a working feature from
        # the user. This is that shape with two copies instead of six.
        #
        # The failure would be silent in the worst way: the tooltip and the
        # warning would each be a plausible sentence, and only a user hovering
        # one while reading the other would ever see them disagree.
        path = SWIFT_SOURCES / "MeetingTranscriber" / "Views" / "Main" / "SpeakerCountView.swift"
        problems = []
        if not path.exists():
            problems.append("SpeakerCountView.swift is gone")
        else:
            # Comments stripped first — this file explains the rule at length, so
            # a textual search matches its own documentation. The lesson every
            # sibling check here learned by failing on its own comment.
            code = "\n".join(l for l in path.read_text().splitlines()
                             if not l.strip().startswith("//"))
            if "private var inertReason: String?" not in code:
                problems.append("`inertReason` is gone — the one source both the "
                                "tooltip and the on-screen warning must read")
            # Two READ sites: `help` returns it, and the body renders it. Fewer
            # than two means one of the surfaces has stopped using it, which is
            # exactly when a second hand-written copy appears.
            # The lookahead drops the DECLARATION (`inertReason: String?`), so this
            # counts readers only — verified rather than assumed, since counting
            # the declaration too would let a single reader satisfy a floor of 2.
            reads = len(re.findall(r"(?<![A-Za-z])inertReason(?!\s*:)", code))
            if reads < 2:
                problems.append(f"`inertReason` is READ {reads} time(s); the "
                                f"tooltip AND the on-screen warning must both read "
                                f"it, or one surface has grown its own copy")
            if "if let reason = inertReason" not in code:
                problems.append("nothing renders `inertReason` on screen — the "
                                "reason would be tooltip-only again, which is what "
                                "the owner asked to change")
            # The POSITIVE half: the rule really is the loader's, not restated
            # here. Without this a file that had stopped consulting the engine at
            # all would pass everything above while always claiming it works.
            if "ModelLoader.honoursSpeakerCount" not in code:
                problems.append("the reason no longer consults "
                                "`ModelLoader.honoursSpeakerCount` — a second copy "
                                "of which engines can be pinned is how this went "
                                "stale three times in Views/ on 2026-08-13")
        rep.expect(cid, not problems,
                   "the tooltip and the on-screen warning both read one "
                   "`inertReason`, which is keyed on the loader's own rule",
                   "; ".join(problems))

    cid = "layout/no-speech-is-routed-by-kind-not-prose"
    if ctx.wants(cid):
        # "No speech in the recording" is a VERDICT, not a fault, and the app
        # draws it as a skipped step rather than a red one. Telling the two
        # apart is a cross-process contract, and it is carried by a machine key
        # (`kind: "no_speech"`) precisely so it is not carried by the sentence.
        #
        # PROSE IS NOT A PROTOCOL. If Swift ever matched the wording instead,
        # rewording the sentence — the most innocent edit imaginable, and one a
        # copy editor would make — would silently paint the panel red again and
        # put an error banner back over the transcript. Nothing would fail to
        # build and no test elsewhere would notice.
        #
        # Both ends are pinned, because either alone is satisfiable by a broken
        # pair: a sidecar that stopped sending the key, or an app that stopped
        # reading it.
        py = SCRIPTS / "nemo" / "nemo-service.py"
        sw = SWIFT_SOURCES / "MeetingTranscriber" / "Transcription" / "NemoService.swift"
        problems = []
        for path, what in ((py, "nemo-service.py"), (sw, "NemoService.swift")):
            if not path.exists():
                problems.append(f"{what} is gone")
        if not problems:
            # Comments stripped on both sides — each file explains this contract
            # at length, so a textual search matches its own documentation. The
            # lesson `assert_no_torchcodec_use` and this check's siblings learned.
            py_code = "\n".join(l for l in py.read_text().splitlines()
                                if not l.lstrip().startswith("#"))
            sw_code = "\n".join(l for l in sw.read_text().splitlines()
                                if not l.strip().startswith("//"))
            if '"kind": "no_speech"' not in py_code:
                problems.append("nemo-service.py no longer emits "
                                "`\"kind\": \"no_speech\"` — the app can then only "
                                "tell a verdict from a fault by reading prose")
            if 'kind == "no_speech"' not in sw_code:
                problems.append("NemoService no longer routes on `kind` — a "
                                "reworded sentence would turn the panel red again")
            if "let kind: String?" not in sw_code:
                problems.append("NemoService.Message dropped `kind`, so the key is "
                                "decoded nowhere and the branch above is dead")
            # The POSITIVE half that stops a decoupled pair from passing: the
            # fallback exists, so an app that has not adopted `onNoSpeech` still
            # SETTLES the stop gate instead of hanging the blocking overlay until
            # the 600 s watchdog.
            if "onNoSpeech" not in sw_code:
                problems.append("`onNoSpeech` is gone — its fallback to onError is "
                                "what keeps an unadopted caller from hanging the "
                                "stop gate")
        rep.expect(cid, not problems,
                   "the no-speech verdict travels as a machine key on both sides "
                   "of the wire, never as a sentence to be matched",
                   "; ".join(problems))

    cid = "layout/hub-downloads-fetch-their-model-card"
    if ctx.wants(cid):
        # `build.sh` [B4] derives MODEL-LICENSES.txt from each checkpoint's own
        # card and FAILS the build when one cannot be read. So a checkpoint
        # fetched FILE BY FILE — rather than as a whole snapshot — must ask for
        # README.md explicitly, or it arrives without the card and the build
        # cannot ship it.
        #
        # THIS IS INVISIBLE ON A MACHINE THAT ALREADY HAS THE CARD, which is why
        # it is pinned rather than left to notice. CAM++ is fetched two files at a
        # time; the licence gate landed a day after that download was written; the
        # owner's main Mac happened to have README.md from an earlier fetch, so the
        # build passed here and failed on their SECOND machine at the first clean
        # checkout — "the snapshot has no README.md".
        #
        # Checked generically, not against a list of repos: any `hf_hub_download`
        # that does NOT pass `local_dir` lands in the hub cache and so needs a
        # card. The ones that DO pass it are flat layouts with no card to read,
        # covered by `model-licenses.py`'s own FLAT table.
        setup = PROJECT / "download-best-models.sh"
        problems = []
        if not setup.exists():
            problems.append("download-best-models.sh is gone")
        else:
            lines = [l for l in setup.read_text().splitlines()
                     if not l.lstrip().startswith("#")]
            for i, line in enumerate(lines):
                if "hf_hub_download(" not in line:
                    continue
                # The call and the loop that feeds it: a small window either side,
                # which is how both of these are written.
                window = "\n".join(lines[max(0, i - 6):i + 4])
                if "local_dir" in window:
                    continue                      # flat layout — no card to read
                if "README.md" not in window:
                    problems.append(
                        f"the hf_hub_download near line {i + 1} caches a hub "
                        f"checkpoint but never fetches README.md, so a clean "
                        f"machine gets weights with no model card and build.sh "
                        f"[B4] refuses to ship them")
        rep.expect(cid, not problems,
                   "every per-file hub download also fetches the model card the "
                   "licence gate reads",
                   "; ".join(problems))

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
                "en", system_prompt="  pyannote PREP  ", repetition_penalty=1.2,
                repetition_context_size=20)
            for key, want in (("system_prompt", "pyannote PREP"),
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


# ============================================================ diarizen group
#
# The FIFTH diarization engine's first sidecar check (2026-08-14). PURE — AST
# only, no model load and no audio, because this engine's interpreter is not the
# one running the suite.
DIARIZEN_CHECKS = [
    "diarizen/pinned-count-is-restored",
]


def run_diarizen(rep: Report, ctx):
    import ast

    # A PINNED SPEAKER COUNT MUST BE UNDONE BEFORE THE NEXT JOB, and it is pinned
    # here rather than left to notice because the failure is SILENT and lands on
    # the wrong stream.
    #
    # This engine takes no `num_speakers=` kwarg — it overrides pyannote's
    # `__call__` — so the count is applied by MUTATING `pipeline.min_speakers` /
    # `.max_speakers`, which outlive the job. Office and remote are separate
    # identity spaces holding different people and share this ONE process, so a
    # pinned office pass that never restored would silently force the room's
    # headcount onto the far end: the 2026-08-11 defect rebuilt inside the
    # sidecar, below every Swift guard that watches for it (`remoteNumSpeakers`,
    # `layout/remote-passes-never-send-the-room-count`). Nothing would fail — the
    # remote pass would simply return the wrong number of people.
    #
    # `finally` specifically, not a line after the call: `pipeline(audio)` is
    # wrapped in try/except and a job that RAISES must still restore.
    if ctx.wants("diarizen/pinned-count-is-restored"):
        cid = "diarizen/pinned-count-is-restored"
        try:
            tree = ast.parse((SCRIPTS / DIARIZEN_SERVICE).read_text())
            BOUNDS = {"min_speakers", "max_speakers"}

            def bounds_assigned(nodes, *, from_source=None):
                """Which of the two bounds are assigned on `pipeline` in here.

                `from_source` narrows it to assignments whose VALUE mentions that
                name. Without it the two halves of this check cannot tell each
                other apart: the restore assigns both bounds too, so a whole-tree
                walk is satisfied by the restore alone and the positive half
                silently stops discriminating. That is exactly what the second
                negative control caught on 2026-08-14 — it passed while the pin
                had been deleted.
                """
                found = set()
                for node in nodes:
                    for sub in ast.walk(node):
                        if not isinstance(sub, ast.Assign):
                            continue
                        if from_source and from_source not in ast.unparse(sub.value):
                            continue
                        for t in sub.targets:
                            for el in (t.elts if isinstance(t, ast.Tuple) else [t]):
                                if (isinstance(el, ast.Attribute)
                                        and el.attr in BOUNDS
                                        and ast.unparse(el.value) == "pipeline"):
                                    found.add(el.attr)
                return found

            problems = []
            restoring = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Try) and node.finalbody:
                    restoring.append(bounds_assigned(node.finalbody))

            # The POSITIVE half — the pin really is applied. Without it a sidecar
            # that had simply stopped honouring the count would pass this check
            # while the SPK picker stayed lit, which is the promise-with-nothing-
            # behind-it that `honoursSpeakerCount` exists to prevent.
            if bounds_assigned([tree], from_source="num_speakers") != BOUNDS:
                problems.append("the sidecar no longer assigns BOTH "
                                "pipeline.min_speakers and .max_speakers FROM "
                                "num_speakers — `ModelLoader.honoursSpeakerCount` "
                                "lists this engine, so the SPK picker would "
                                "promise a pin nothing applies")
            elif not any(b == BOUNDS for b in restoring):
                problems.append("no `finally` restores both bounds — a pinned "
                                "office pass would leak its count into the next "
                                "job, and the next job may be the REMOTE stream, "
                                "which is a different set of people")

            rep.expect(cid, not problems,
                       "a pinned speaker count is applied as instance bounds and "
                       "restored in a `finally`, so it cannot outlive its job and "
                       "reach the other stream",
                       "; ".join(problems))
        except Exception as exc:  # noqa: BLE001
            rep.fail(cid, f"could not read the diarizen sidecar: {exc!r}")


# ===================================================================== main
# ============================================================ campplus group
#
# The SIXTH diarization engine (CAM++, 2026-08-11). Both checks are PURE — AST
# and hashes, no model load and no audio — the `spectral/*` precedent.
CAMPPLUS_CHECKS = [
    "campplus/no-live-chunk-branch",
    "campplus/turns-never-intersect",
    "campplus/vendor-is-verbatim",
]

# The vendored WeSpeaker files, and the UPSTREAM sha256 each was fetched with.
#
# A REAL PROVENANCE PIN, not a self-fingerprint: these hashes were taken from
# github.com/wenet-e2e/wespeaker's raw files, and both vendored copies matched
# them byte for byte. So unlike `spectral/vendor-is-own` — which has to carve
# out `embeddings.py` as a deliberate one-file deviation — this tree is
# VERBATIM, and the check can say so without an exception.
CAMPPLUS_VENDOR_SHA = {
    "campplus.py":
        "b1d6a3c39cdb9d85db8d7a7fe1885c39f6644663fd465db2592d2a63ed3a0d08",
    "pooling_layers.py":
        "3874eb8b382bd7ed824ca6b3822ede02344a6b8cada20230502031a00bf5d72d",
}


def run_campplus(rep: Report, ctx):
    import ast
    import hashlib

    # -- check 1: no live/chunk path, asserted in BOTH directions.
    #
    #    The `spectral/no-live-chunk-branch` shape exactly, for the same reason:
    #    this engine counts and clusters GLOBALLY, so a 30 s window's labels
    #    would mean nothing across windows. The positive half — pyannote really
    #    does have the live path being declined — is what lets it fail at all.
    if ctx.wants("campplus/no-live-chunk-branch"):
        cid = "campplus/no-live-chunk-branch"
        try:
            def live_shape(source):
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

            c_gets, c_cmp, c_kinds, c_keys = live_shape(
                (SCRIPTS / CAMPPLUS_SERVICE).read_text())
            p_gets, p_cmp, p_kinds, p_keys = live_shape(
                (SCRIPTS / PYANNOTE_SERVICE).read_text())
            problems = []

            # The NEGATIVE half: no live path anywhere in campplus.
            if "chunk" in c_cmp:
                problems.append('campplus compares a command against "chunk" — a '
                                "windowed pass through a globally-clustering engine "
                                "returns labels that look continuous and are not")
            if "chunk_result" in c_kinds or "chunk_result" in c_keys:
                problems.append("campplus emits a chunk_result reply")
            if "window_start" in c_gets or "window_start" in c_keys:
                problems.append("campplus reads or emits window_start — the only "
                                "reason to know a window's offset is to serve one")
            # …and that the refusal really is there, so "no chunk branch" is a
            # deliberate refusal rather than an unhandled command falling through
            # to the whole-file pass.
            if "final" not in c_cmp:
                problems.append("campplus never compares the command against "
                                '"final" — a non-final job would fall through to '
                                "the whole-file pass instead of being refused")

            # The POSITIVE half: pyannote really does have the live path this
            # engine is declining to copy.
            if "chunk" not in p_cmp:
                problems.append('pyannote no longer compares a command against '
                                '"chunk" — its live per-30s path is the thing '
                                "campplus is being contrasted with")
            if "chunk_result" not in p_kinds:
                problems.append("pyannote no longer emits chunk_result")
            if "window_start" not in p_gets:
                problems.append("pyannote no longer reads window_start")

            rep.expect(cid, not problems,
                       "campplus has no chunk comparison, no chunk_result and no "
                       "window_start, and refuses any cmd != final — while pyannote "
                       "has all three (so this check can fail in either direction)",
                       "; ".join(problems))
        except Exception as exc:  # noqa: BLE001
            rep.fail(cid, f"could not compare the read loops: {exc!r}")

    # -- check 2: `merge_windows` NEVER returns two turns that intersect.
    #
    #    A SHIPPED BUG, found 2026-08-11 by driving the real sidecar over
    #    `Overlap123.wav`: four intersecting pairs, every one of them exactly
    #    1.000 s = WINDOW_SEC - HOP_SEC, one at each speaker change. Consecutive
    #    windows share that much audio, and emitting both spans verbatim across a
    #    label change made the turns overlap by precisely the shared amount.
    #
    #    WHY IT MATTERS ENOUGH TO PIN: this engine assigns ONE label per window,
    #    so it cannot detect overlap — but `overlapRegions()` does not know that.
    #    It infers overlap from any turns intersecting by >= 0.4 s, so a 1.0 s
    #    artefact reached the transcript as two people speaking at once. Invented
    #    speech is the failure direction this project ranks worst, and it was
    #    invisible: the speaker COUNT was right on every known-answer file.
    #
    #    Pure — `merge_windows` is module-level precisely so this needs no
    #    checkpoint. The synthetic input reproduces the exact geometry (2 s
    #    windows, 1 s hop) with a label change in the middle.
    if ctx.wants("campplus/turns-never-intersect"):
        cid = "campplus/turns-never-intersect"
        try:
            module, _ = load_sidecar_module(CAMPPLUS_SERVICE, "campplus_merge")
            spans = [(t, t + 2.0) for t in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)]
            problems = []
            for name, labels in [("one change", [0, 0, 0, 1, 1, 1]),
                                 ("alternating", [0, 1, 0, 1, 0, 1]),
                                 ("every window new", [0, 1, 2, 3, 4, 5]),
                                 ("no change", [0, 0, 0, 0, 0, 0])]:
                turns = module.merge_windows(spans, labels)
                for i in range(len(turns)):
                    for j in range(i + 1, len(turns)):
                        a, b = turns[i], turns[j]
                        if a["label"] == b["label"]:
                            continue
                        ov = min(a["end"], b["end"]) - max(a["start"], b["start"])
                        if ov > 1e-9:
                            problems.append(
                                f"{name}: {a['label']} [{a['start']:.3f},"
                                f"{a['end']:.3f}] intersects {b['label']} "
                                f"[{b['start']:.3f},{b['end']:.3f}] by {ov:.3f}s")
                for t in turns:
                    if t["end"] - t["start"] <= 0:
                        problems.append(f"{name}: non-positive turn {t}")
            # THE POSITIVE HALF, and it is what lets this check fail at all: a
            # `merge_windows` that returned [] — or that split every window into
            # its own turn — would satisfy "nothing intersects" trivially. The
            # single-label case must still collapse to ONE turn spanning the
            # whole span, and the one-change case to exactly two.
            one = module.merge_windows(spans, [0, 0, 0, 0, 0, 0])
            if len(one) != 1 or abs(one[0]["start"]) > 1e-9 or abs(one[0]["end"] - 7.0) > 1e-9:
                problems.append(f"same-label windows must collapse to one 0..7 turn, got {one}")
            two = module.merge_windows(spans, [0, 0, 0, 1, 1, 1])
            if len(two) != 2:
                problems.append(f"one label change must give exactly 2 turns, got {two}")
            elif abs(two[0]["end"] - two[1]["start"]) > 1e-9:
                problems.append(f"the two turns must meet at one boundary, got {two}")
            if problems:
                rep.fail(cid, "; ".join(problems))
            else:
                rep.ok(cid, "turns never intersect; boundaries meet at the midpoint")
        except Exception as exc:  # noqa: BLE001
            rep.fail(cid, f"could not drive merge_windows: {exc!r}")

    # -- check 3: the vendored WeSpeaker tree is upstream's, unmodified, and the
    #    service points its sys.path at ITS OWN copy.
    #
    #    The `moss/asr-vendor-is-own-and-identical` precedent for the second
    #    half: a `sys.path.insert` aimed at another service's vendor folder works
    #    today and breaks months later, in a release, the moment that folder
    #    moves. EVALUATED rather than string-matched, so the assertion is about
    #    the directory that will really be on sys.path.
    if ctx.wants("campplus/vendor-is-verbatim"):
        cid = "campplus/vendor-is-verbatim"
        try:
            problems = []
            vendor = SCRIPTS / "campplus" / "vendor"
            models = vendor / "wespeaker" / "models"

            for name, want in CAMPPLUS_VENDOR_SHA.items():
                path = models / name
                if not path.exists():
                    problems.append(f"{name} is missing from the vendored tree — "
                                    "the build gate ships what is here, and a pip "
                                    "install would be stripped from the .app")
                    continue
                got = hashlib.sha256(path.read_bytes()).hexdigest()
                if got != want:
                    problems.append(
                        f"{name} no longer matches the upstream hash it was "
                        f"vendored at (got {got[:12]}…, want {want[:12]}…). This "
                        "tree is VERBATIM upstream by design; if it was "
                        "deliberately re-vendored, update CAMPPLUS_VENDOR_SHA "
                        "with the new upstream hash rather than ours")
            if not (vendor / "LICENSE").exists():
                problems.append("the vendored tree has no LICENSE — this is "
                                "Apache-2.0 third-party code and shipping it "
                                "without its licence is the one thing that is not "
                                "merely untidy")

            # The service must insert ITS OWN vendor dir.
            source = (SCRIPTS / CAMPPLUS_SERVICE).read_text()
            namespace = {"__file__": str(SCRIPTS / CAMPPLUS_SERVICE),
                         "os": os, "sys": sys}
            resolved = None
            for node in ast.parse(source).body:
                if (isinstance(node, ast.Assign)
                        and any(getattr(t, "id", "") == "VENDOR" for t in node.targets)):
                    resolved = eval(compile(ast.Expression(node.value), "<v>", "eval"),
                                    namespace)
            if resolved is None:
                problems.append("campplus-service.py defines no VENDOR path — it "
                                "cannot import the CAM++ architecture at all")
            elif os.path.realpath(resolved) != os.path.realpath(str(vendor)):
                problems.append(
                    f"campplus-service.py's VENDOR resolves to {resolved!r}, not "
                    f"its own {str(vendor)!r} — pointing at another service's tree "
                    "works today and breaks when that folder moves")

            rep.expect(cid, not problems,
                       f"{len(CAMPPLUS_VENDOR_SHA)} vendored files byte-identical to "
                       "the upstream hashes they were fetched at, LICENSE present, "
                       "and the service inserts its own vendor dir",
                       "; ".join(problems))
        except Exception as exc:  # noqa: BLE001
            rep.fail(cid, f"could not verify the campplus vendor tree: {exc!r}")


GROUPS = [
    ("layout", LAYOUT_CHECKS, run_layout),
    ("tools", TOOLS_CHECKS, run_tools),
    ("nemotron", NEMOTRON_CHECKS, run_nemotron),
    ("realtime", REALTIME_CHECKS, run_realtime),
    ("chunked", CHUNKED_CHECKS, run_chunked),
    ("whisper", WHISPER_CHECKS, run_whisper),
    ("qwen3", QWEN3_CHECKS, run_qwen3),
    ("aligner", ALIGNER_CHECKS, run_aligner),
    ("moss", MOSS_CHECKS, run_moss),
    ("pyannote", PYANNOTE_CHECKS, run_pyannote),
    ("spectral", SPECTRAL_CHECKS, run_spectral),
    ("campplus", CAMPPLUS_CHECKS, run_campplus),
    ("diarizen", DIARIZEN_CHECKS, run_diarizen),
    ("nemo", NEMO_CHECKS, run_nemo),
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
