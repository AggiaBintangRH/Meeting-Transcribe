#!/usr/bin/env python3
"""Speaker DIARIZATION sidecar for MeetingTranscriber (persistent, stateless).

ONE SERVICE PER MODEL (owner, 2026-07-29: "whisper have a service, granite have
a service, all have a service"). This half of the former `diarize-service.py`
runs the pyannote community-1 pipeline and nothing else: it answers WHO-SPOKE-WHEN
with pipeline-local labels (`SPEAKER_00`, `SPEAKER_01`, …) and has no idea who
those people are. Identity — the WeSpeaker embeddings and the two persistent
profile stores — lives in `scripts/wespeaker/wespeaker-service.py`.

Two modes over one loaded pipeline (pyannote community-1 on MPS):
  • chunk  — diarize a ~30s chunk live
  • final  — batch diarization of the full recording at stop (best DER)

WHY THE SPLIT, and what it does NOT do
--------------------------------------
Identity used to be reachable only through this pipeline, because the embedder,
the profile stores and the clustering all lived in one process. They are now two
processes, so attaching saved identities to time spans is *structurally
available* to any diarizer that can produce spans. That is the whole claim.
Nothing new is wired: MOSS still labels speakers per chunk with no cross-chunk
stitching (the owner deferred that — "nanti saja"), and the ATND position layer
still owns its own cluster ids and has no embeddings at all. Neither goes through
this service or the WeSpeaker one.

THIS PROCESS IS STATELESS. There is deliberately no `reset` command: the only
state a session ever resets is the profile stores, and those belong entirely to
the WeSpeaker service now. Nothing here persists between jobs except the loaded
pipeline and its clustering threshold.

Protocol:
  stdout: {"type":"status","text":"LOADED"}
          {"type":"chunk_result","window_start":s,"audio":path,
           "segments":[{"start":…,"end":…,"label":"SPEAKER_00"}]}
          {"type":"result","audio":path,"segments":[…]}
          {"type":"error","text":...}
          A segment carries ONLY start/end/label — never id, name or conf. That
          is the split made structural: identity cannot leak back out of the
          pipeline stage, so the pipeline can never be the thing that writes a
          voice into a profile store.
          The "audio" ECHO IS LOAD-BEARING: it is how the app knows which file to
          hand the identity stage next, with no second bookkeeping map, and it
          works identically for a chunk's temp WAV and for the recording file.
          result/chunk_result/error carry "stream":"remote" for remote jobs; the
          field is OMITTED for office jobs, so the single-stream wire format is
          byte-identical to what it was before dual-stream existed. Here it is a
          PURE ROUTING ECHO — this process has no per-stream state whatsoever.
          START/END ARE NOT ROUNDED. The pipeline's own floats go out verbatim
          because the identity stage slices the waveform with them; rounding here
          would move each slice by up to ~8 samples and change the embedding.
          Rounding to 3 dp is done once, where it always was: at the point the
          app-visible turn is composed (`AudioRecorder.composeTurns`).
  stdin (one JSON per line):
          {"cmd":"chunk","audio":path,"window_start":seconds}
          {"cmd":"final","audio":path,"num_speakers":0}
          any job may carry "exclusive" and "cluster_threshold", and
          "stream":"office"|"remote" — ABSENT MEANS OFFICE.
          EOF exits.

Fully offline: HF_HOME points at the project models/ folder, PYANNOTE_METRICS_ENABLED
turns off pyannote 4.x's default-on telemetry, and the audio is decoded here and
handed over in memory so torchcodec (and its Homebrew-only ffmpeg) is never used.
The last two are not tidiness — see the comments on each.
"""
import json
import os
import sys
import time
import traceback

# This file lives at scripts/<service>/<file>.py (one folder per service,
# owner 2026-07-29), so the project root — which owns models/ — is THREE levels
# up, not two. Bundled, that root is Contents/Resources.
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HOME", os.path.join(BASE, "models"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# pyannote.audio 4.x ships OpenTelemetry tracing that is ON BY DEFAULT
# (pyannote/audio/telemetry/config.yaml: `metrics_enabled: true`) and posts to
# https://otel.pyannote.ai/v1/traces from three places: every Model init, every
# Pipeline init, and — via `core/pipeline.py` `track_pipeline_apply` — EVERY
# `apply()`, i.e. every 30 s chunk and every stop pass. The span carries the
# pyannote version, a random session uuid, and the AUDIO DURATION plus requested
# speaker counts. No audio and no text, but it is still a network call describing
# a client meeting, which client hard requirement #1 (100 % offline, nothing
# leaves the machine) forbids outright.
#
# `HF_HUB_OFFLINE` does NOT cover this — that flag only gates the Hugging Face
# hub. Verified with a DNS-recording probe before this line existed: importing
# the telemetry module and tracking one event resolved `otel.pyannote.ai:443`.
#
# Set BEFORE importing pyannote (below, inside main()) — `telemetry/metrics.py`
# only fills this in when it is absent, so setting it first wins. Assigned, not
# `setdefault`: the requirement is absolute, so an inherited environment must not
# be able to turn it back on. Upstream documents this exact opt-out.
#
# It also removes a per-chunk cost: `track_pipeline_apply` calls
# `Audio().get_duration(file)` synchronously before building its span.
os.environ["PYANNOTE_METRICS_ENABLED"] = "false"

MODEL = "pyannote/speaker-diarization-community-1"

# Hard ceiling on this process's GPU allocation, in GB — armed in main().
#
# A fuse, not a budget. macOS reports a "recommended max working set" (51.8 GB on
# the owner's 64 GB M4) and PyTorch's MPS allocator grows all the way to it
# WITHOUT EVER RAISING. MOSS proved that is not theoretical: on 2026-07-31 one
# oversized call reached ~50 GB and the Mac had to be rescued by hand. pyannote
# was the only other MPS sidecar still without a fuse, and its stop pass holds an
# entire recording in memory by design (~710 MB for 67 minutes at 44.1 kHz), so
# its appetite scales with meeting length too.
#
# HALF THE MACHINE, derived below rather than typed — the same rule both MOSS
# sidecars use. Normal passes
# sit far below it; exceeding it now fails ONE request loudly rather than
# swapping the whole machine.
def _mps_cap_gb(fallback: float = 32.0) -> float:
    """The MPS fuse for THIS machine, in GB: half the physical RAM, floored at 12.

    ⚠ WHY NOT THE LITERAL 32.0. That number WAS "half the machine", for the 64 GB
    M4 every figure in this project was measured on. On a 16 GB Mac it is TWICE
    the physical RAM, and because the cap is armed as
    `set_per_process_memory_fraction(min(CAP / ceiling, 1.0))` the `min` clamped
    to 1.0 — the whole allocator, uncapped. Captured in the wild on the client's
    Mac, which printed it three times:

        MPS memory capped at 32 GB of the 11.8 GB macOS reports as available

    ⚠ AND WHY THE FLOOR OF 12, which is the half of this that was LEARNED THE
    HARD WAY. "Half the physical RAM" alone is a rule calibrated on a 64 GB
    machine, and it does not survive being scaled down. It leaves 19.8 GB of
    macOS's 51.8 GB recommendation spare at 64 GB, but only 3.8 GB of 11.8 at
    16 GB — and the OTHER sidecars need a roughly FIXED ~4 GB, not a proportion.
    So on a small machine the proportional rule squeezes out work that fits.

    Measured, on the client's 16 GB Mac: with the cap at half (8.0 GB), NeMo
    raised `MPS backend out of memory (MPS allocated: 8.23 GiB ... max allowed:
    8.00 GiB)` — on a machine where macOS itself was willing to give 11.8 GB, and
    where NeMo had run FINE two days earlier, uncapped, in 5.7 s. The fuse was
    stricter than the operating system and broke working functionality. That is
    the over-deletion direction this project ranks worst, in a new costume.

    The floor is 12.0 rather than a proportion because the number that matters is
    absolute, and it was chosen for what it DOES rather than for what it is: macOS
    reports ~11.8 GB as the client Mac's ceiling, so a 12 GB cap makes
    `min(CAP / ceiling, 1.0)` clamp to 1.0 and NOTHING THERE IS EVER REFUSED. That
    is the owner's decision, 2026-08-26: "jangan ada yang di blokir harus bisa
    digunakan semuanya / untuk hang ngelag sekarang harus terlihat oleh Boss."
    Every engine stays selectable, and a slowdown is wanted as evidence for a
    hardware budget rather than hidden behind a refusal.

    It changes NOTHING on a machine of 24 GB or more, where half is already at or
    above the floor — this 64 GB M4 still gets exactly 32.0, fraction 0.62, a real
    fuse. The floor only binds between 8 and 24 GB.

    ⚠ THIS IS THE SAME ARITHMETIC AS THE DEFECT ABOVE, ON PURPOSE, AND ONE THING
    KEEPS IT HONEST. `32.0` was inert on a 16 GB Mac by ACCIDENT while logging
    "capped at 32 GB" — a claim it could not keep. `12.0` is inert there by
    DECISION, and the arming site prints `WARNING MPS fuse NOT ARMED` instead of
    claiming a cap it does not have. A silent inert fuse is a lie; a declared one
    is a trade-off, and `layout/mps-fuse-is-sized-to-the-machine` asserts that
    branch exists.

    ⚠ THE COST, owner-accepted with it stated: on a 16 GB machine nothing here
    refuses anything, so a long NeMo pass (peak 13.33 GB at 67 minutes) can drive
    that Mac into swap rather than failing fast. MOSS is still out of reach there,
    but by the HARDWARE ceiling of 11.8 GB rather than by this cap — the block
    moved, it did not disappear.

    `sysconf` rather than `torch.mps.recommended_max_memory()` because this is a
    module-level constant and torch is not imported yet in every service that
    carries one.

    Falls back to the historical constant if the query fails. A fuse that cannot
    size itself must not become a fuse that refuses everything.
    """
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        half = total / (1024 ** 3) / 2.0
        return round(max(half, 12.0), 1) if half > 0 else fallback
    except (ValueError, OSError, AttributeError):  # noqa: BLE001
        return fallback


MPS_MEMORY_CAP_GB = _mps_cap_gb()


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


# ---------------------------------------------------------------- main
def main() -> None:
    try:
        import soundfile as sf
        import torch
        from pyannote.audio import Pipeline
    except Exception:  # noqa: BLE001
        fail(f"pyannote.audio import failed: {brief_traceback()} — "
             "run download-best-models.sh")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Arm the fuse BEFORE the pipeline loads, so even a bad load cannot run away.
    # Wrapped because these APIs are version-dependent and a missing one must not
    # cost us diarization: an uncapped run is what we have always had, so failing
    # to cap is a warning, never fatal.
    if device.type == "mps":
        try:
            ceiling_gb = torch.mps.recommended_max_memory() / (1024 ** 3)
            if ceiling_gb > 0:
                fraction = min(MPS_MEMORY_CAP_GB / ceiling_gb, 1.0)
                torch.mps.set_per_process_memory_fraction(fraction)
                if fraction >= 1.0:
                    # THE FUSE IS INERT HERE, AND IT SAYS SO. Owner's decision,
                    # 2026-08-26: "jangan ada yang di blokir harus bisa digunakan
                    # semuanya / untuk hang ngelag sekarang harus terlihat oleh
                    # Boss." Every engine must be selectable on every machine, and
                    # slowness is the evidence for a hardware budget rather than
                    # something to hide behind a refusal.
                    #
                    # This line is that evidence. It is greppable on purpose --
                    # a machine too small to carry a fuse should not look the same
                    # in the log as one running comfortably under it.
                    log(f"WARNING MPS fuse NOT ARMED — this machine offers only "
                        f"{ceiling_gb:.1f} GB, under the {MPS_MEMORY_CAP_GB:.0f} GB "
                        "cap, so nothing is refused and one oversized allocation "
                        "can drive the Mac into swap. Deliberate: every engine "
                        "stays usable and the slowdown is left visible.")
                else:
                    log(f"MPS memory capped at {MPS_MEMORY_CAP_GB:.0f} GB "
                        f"of the {ceiling_gb:.1f} GB macOS reports as available")
        except Exception:  # noqa: BLE001
            log("WARNING could not cap MPS memory — a very long pass could "
                "allocate without bound")

    log(f"loading pipeline {MODEL}")
    try:
        pipeline = Pipeline.from_pretrained(MODEL)
        pipeline.to(device)
    except Exception:  # noqa: BLE001
        fail(f"Pipeline load failed: {brief_traceback()} — "
             "run download-best-models.sh (pyannote is a gated repo)")

    # pyannote's clustering threshold decides how readily two turns are called
    # the same speaker: LOWER merges more (fewer speakers), HIGHER splits more.
    # 0.6 is community-1's own default and is robust for distinct voices (a
    # sweep on real 1- and 2-person recordings gave the correct count across
    # 0.4-0.8), so it stays the default. It is exposed only so a specific room
    # with unusually similar voices can be calibrated, exactly like the ATND
    # tauDeg. Fa/Fb and min_duration_off are pyannote's own tuned constants and
    # are carried through unchanged; only the threshold is caller-controlled.
    DEFAULT_CLUSTER_THRESHOLD = 0.6
    current_threshold = [None]  # boxed so set_threshold can mutate it

    def set_threshold(th: float) -> None:
        if current_threshold[0] is not None and abs(th - current_threshold[0]) < 1e-9:
            return  # already instantiated at this value — instantiate is not free
        pipeline.instantiate({
            "segmentation": {"min_duration_off": 0.0},
            "clustering": {"threshold": th, "Fa": 0.07, "Fb": 0.8},
        })
        current_threshold[0] = th
        log(f"clustering threshold set to {th}")

    set_threshold(DEFAULT_CLUSTER_THRESHOLD)

    log(f"pipeline on {device} — turns only, no identity in this process")
    # In the packaged .app torchcodec is pruned (it needs Homebrew ffmpeg), so
    # pyannote emits a scary import-time UserWarning: "torchcodec is not installed
    # correctly so built-in audio decoding will fail". It is EXPECTED and harmless
    # here — we never use pyannote's built-in decoding — but a reader debugging a
    # real problem would chase it. Say so, right after it, in our own log.
    log("note: pyannote's torchcodec warning above is expected — audio is decoded "
        "in this process with soundfile and handed over in memory (load_waveform)")
    emit({"type": "status", "text": "LOADED"})

    def load_waveform(audio_path):
        """Decode a WAV to the in-memory shape pyannote accepts: a
        `{"waveform": (channel, time) float32 tensor, "sample_rate": int}` dict.

        WHY WE DECODE IT OURSELVES — this is the ffmpeg lesson a second time.
        pyannote 4.x hands a *path* to **torchcodec**, whose `libtorchcodec_core*`
        dylibs link `libavcodec`/`libavformat`/`libavutil` through a SINGLE
        `LC_RPATH` of `/opt/homebrew/opt/ffmpeg/lib`. Measured: the loaded library
        on the dev Mac really is `/opt/homebrew/Cellar/ffmpeg/8.1.1/lib/...`, there
        is no `libav*` anywhere in `.venv`, and there are ZERO in the packaged
        `.app`. On a client Mac without Homebrew ffmpeg all five variants fail
        (`_internally_replaced_utils.py` tries FFmpeg 8→4 then raises), pyannote's
        own `try: import torchcodec` swallows it into `TORCHCODEC_AVAILABLE=False`,
        and then `Audio.crop` raises — i.e. diarization dies at the first chunk.
        There is NO decoder fallback in pyannote 4.x; the waveform dict is the only
        documented escape (`core/io.py`), exactly as `whisper-service.py` passes a
        numpy array rather than a path for the same reason (37c16bac9).
        DO NOT "simplify" this back to `pipeline(audio_path)`.

        NATIVE SAMPLE RATE, deliberately. `Audio.crop` slices the in-memory
        waveform and then calls the very same `downmix_and_resample` the torchcodec
        path calls, so handing over untouched native-rate samples keeps behaviour
        bit-for-bit identical to reading the file. Resampling to 16 kHz here first
        would be cheaper (one pass instead of ~800 per-window resamples) but it is
        NOT the same signal — resampling a crop and cropping a resample differ at
        the window edges — so it is left alone. Verified before the switch:
        soundfile and torchcodec decode these recordings to bit-identical float32
        (max abs diff 0.000e+00 over 177 608 340 samples of a 67-minute file).

        `uri` is set to the file stem, which is what the path form produced, so
        `Annotation.uri` and pyannote's own error messages are unchanged.

        Cost: the whole recording is resident during a stop pass (~710 MB for
        67 minutes of 44.1 kHz mono float32) where the torchcodec path streamed it
        per window. Accepted — `wespeaker-service.py` already holds exactly this
        for the same file, and it is freed as soon as the job returns.
        """
        data, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        # soundfile gives (time, channel); pyannote validates (channel, time).
        waveform = torch.from_numpy(data.T.copy())
        return {"waveform": waveform, "sample_rate": int(sample_rate),
                "uri": os.path.splitext(os.path.basename(audio_path))[0]}

    def local_turns(audio_path, num_speakers=0, exclusive=False):
        kwargs = {"num_speakers": num_speakers} if num_speakers > 0 else {}
        output = pipeline(load_waveform(audio_path), **kwargs)
        # exclusive=True  → one speaker per instant (clean, no overlaps).
        # exclusive=False → overlap-aware: two speakers can be active at once,
        #                   so people talking over each other both show up.
        annotation = None
        if exclusive:
            annotation = getattr(output, "exclusive_speaker_diarization", None)
        if annotation is None:
            annotation = getattr(output, "speaker_diarization", output)
        return [(t.start, t.end, label)
                for t, _, label in annotation.itertracks(yield_label=True)]

    def wire_segments(turns):
        """Pipeline-LOCAL turns on the wire: start, end, label. Nothing else.

        No id, no name, no conf — this process has never seen a profile store and
        must not look as though it could. See the module docstring on why the
        times are not rounded.
        """
        return [{"start": start, "end": end, "label": label}
                for start, end, label in turns]

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
            cmd = job.get("cmd", "final")
        except Exception:  # noqa: BLE001
            emit({"type": "error", "text": f"Bad job line: {line[:120]}"})
            continue

        # Which identity space this job belongs to. ABSENT MEANS OFFICE, so a
        # single-stream app (or an older one) drives this sidecar exactly as it
        # always did. `echo` is spliced into every reply for this job and is
        # empty for office — keeping the single-stream wire bytes unchanged.
        # It is routing information ONLY: nothing in this process behaves
        # differently for a remote job, because nothing here is per-stream.
        stream = job.get("stream")
        is_remote = stream == "remote"
        echo = {"stream": stream} if stream else {}

        audio = job.get("audio", "")
        if not os.path.exists(audio):
            emit({"type": "error", "text": f"Audio not found: {audio}", **echo})
            continue

        exclusive = bool(job.get("exclusive", False))
        # Per-room calibration (absent = pyannote's default). Re-instantiating
        # only when the value actually changes keeps the common case free.
        set_threshold(float(job.get("cluster_threshold", DEFAULT_CLUSTER_THRESHOLD)))
        started = time.time()
        try:
            if cmd == "chunk":
                turns = local_turns(audio, exclusive=exclusive)
                segments = wire_segments(turns)
                log(f"chunk done in {time.time() - started:.1f}s — "
                    f"{len(segments)} turns, {len({s['label'] for s in segments})} voices"
                    f"{' [remote]' if is_remote else ''}")
                emit({"type": "chunk_result",
                      "window_start": job.get("window_start", 0.0),
                      "audio": audio,
                      "segments": segments, **echo})
            else:  # final
                turns = local_turns(audio, int(job.get("num_speakers", 0)), exclusive=exclusive)
                segments = wire_segments(turns)
                log(f"final done in {time.time() - started:.1f}s — "
                    f"{len(segments)} turns, {len({s['label'] for s in segments})} speakers"
                    f"{' [remote]' if is_remote else ''}")
                emit({"type": "result", "audio": audio,
                      "segments": segments, **echo})
        except Exception:  # noqa: BLE001 — job error: report, keep serving
            # The stream is echoed so the app fails only THAT stream's gate —
            # a remote failure must never abort the office transcript.
            emit({"type": "error", "text": f"Diarization failed: {brief_traceback()}", **echo})


if __name__ == "__main__":
    main()
