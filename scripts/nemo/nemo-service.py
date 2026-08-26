#!/usr/bin/env python3
"""Speaker DIARIZATION sidecar — the NeMo engine (persistent, stateless).

ONE SERVICE PER ROLE. This is the FOURTH diarizer in the project, alongside
`pyannote/pyannote-service.py` (pyannote community-1),
`spectral/spectral-service.py` (the vendored FoxNoseTech engine) and
`moss-diar/moss-diar-service.py` (MOSS speaker-attributed ASR). Like the first
two it answers WHO-SPOKE-WHEN with pipeline-LOCAL labels (`SPEAKER_00`,
`SPEAKER_01`, …) and has no idea who those people are. Identity — the WeSpeaker
embeddings and the two persistent profile stores — lives in
`scripts/wespeaker/wespeaker-service.py` and nowhere else.

THE ENGINE is NVIDIA NeMo's `ClusteringDiarizer` (nemo_toolkit 3.0.0, Apache
2.0): MarbleNet VAD → multi-scale TitaNet-Large embeddings → NME-SC
(normalised-maximum-eigengap spectral clustering, which estimates the speaker
count itself) → multi-scale label fusion.

IT RUNS IN ITS OWN INTERPRETER, `.venv-nemo` — the second sidecar to do so
after DiCoW. nemo_toolkit drags in lightning, hydra and a large pinned
dependency tree that conflicts with the main `.venv`'s MLX stack; a trial
install into `.venv` was reverted on 2026-08-07 for exactly that reason.

FINAL ONLY — THERE IS DELIBERATELY NO LIVE / PER-CHUNK PATH.
NME-SC counts and clusters GLOBALLY: the eigengap analysis runs over the whole
affinity matrix of the whole file. A 30 s window would therefore be counted and
clustered on its own and its labels would mean nothing across windows — the same
failure MOSS has when it is called per chunk (documented at length in CLAUDE.md:
two different people both numbered `S01` because the chunk boundary fell on the
speaker change), and the same reason `spectral-service.py` has no live branch.
Adding one "for symmetry" would ship labels that look continuous and are not.
A job carrying any `cmd` other than `final` is REFUSED loudly.

THIS PROCESS IS STATELESS: nothing persists between jobs except the loaded VAD
and embedding models. There is no `reset`, because the only session state — the
profile stores — belongs to the WeSpeaker service.

Protocol (byte-compatible with the pyannote and spectral services, so the Swift
identity path takes this engine unmodified):
  stdout: {"type":"status","text":"LOADED"}
          {"type":"result","audio":path,
           "segments":[{"start":…,"end":…,"label":"SPEAKER_00"}]}
          {"type":"error","text":...}
          A segment carries ONLY start/end/label — never id, name or conf. That
          is the pyannote/wespeaker split made structural for this engine too:
          identity cannot leak out of a pipeline stage, so a pipeline can never
          be the thing that writes a voice into a profile store.
          The "audio" ECHO IS LOAD-BEARING: it is how the app knows which file to
          hand the identity stage next, with no second bookkeeping map.
          result/error carry "stream":"remote" for remote jobs; the field is
          OMITTED for office jobs, so the office wire format is identical to the
          pyannote service's. It is a PURE ROUTING ECHO — nothing in this process
          is per-stream.
          START/END ARE NOT ROUNDED. The RTTM's own floats go out verbatim
          because the identity stage slices the waveform with them; rounding here
          would move each slice by up to ~8 samples and change the embedding.
          Rounding to 3 dp happens once, where it always has: at the point the
          app-visible turn is composed (`AudioRecorder.composeTurns`).
  stdin (one JSON per line):
          {"cmd":"final","audio":path,"num_speakers":0}
          optional: "stream":"office"|"remote" — ABSENT MEANS OFFICE.

          `max_speakers` USED TO BE ACCEPTED HERE AND IS NOT ANY MORE (2026-08-10
          audit). Nothing ever sent it: the count is automatic on this engine, and
          `MAX_NUM_SPEAKERS` is pinned below. A wire field that no caller writes is
          the `overlap.detect.model` defect wearing a protocol's clothes — harmless
          until someone reads the docstring, believes the knob exists, and tunes a
          value that goes nowhere. The bound is a property of this engine, so it
          lives at its declaration, not in a message.
          `exclusive` and `cluster_threshold` are pyannote's knobs and have no
          meaning here (NME-SC assigns exactly one label per segment, so turns
          never intersect, and its clustering has no threshold — it searches a
          p-value range instead). They are accepted and ignored rather than
          rejected, so one caller can drive any engine.
          EOF exits.

Fully offline: both checkpoints are LOCAL absolute paths under models/nemo/ (see
`SPEAKER_MODEL_PATH`), so NeMo's NGC resolver is never reached, and the three
telemetry SDKs nemo_toolkit pulls in are switched off by assignment below.

THE FFMPEG TRAP DOES NOT APPLY HERE, and that is structural rather than lucky:
`torchaudio` and `torchcodec` are NOT INSTALLED in `.venv-nemo` (verified
2026-08-07). NeMo's own audio path is soundfile/librosa. There is therefore no
`load_waveform` shim in this file, unlike pyannote's and spectral's — do not add
one, and do not add torchaudio to this venv's requirements without re-deriving
that whole argument.
"""
import json
import os
import shutil
import sys
import tempfile
import time
import traceback

# ---------------------------------------------------------------- THE WIRE
#
# The JSON protocol goes HERE and nothing else may. Rebound by `install_wire()`,
# which `main()` calls before it imports anything heavy. It starts as
# `sys.stdout` so that merely IMPORTING this module (which `sidecar-tests.py`
# must be able to do) changes no file descriptor and steals nobody's stdout.
WIRE = sys.stdout

# This file lives at scripts/<service>/<file>.py (one folder per service,
# owner 2026-07-29), so the project root — which owns models/ — is THREE levels
# up, not two. With two, MODEL_DIR silently resolves to `scripts/models`,
# py_compile passes, an import-only test passes, and it fails at the first model
# load. Bundled, that root is Contents/Resources.
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["HF_HOME"] = os.path.join(BASE, "models")
os.environ["HF_HUB_OFFLINE"] = "1"

# THE THREE TELEMETRY SDKs nemo_toolkit 3.0.0 installs, all switched off by
# ASSIGNMENT rather than `setdefault` — the `pyannote/telemetry-is-off`
# precedent. Client hard requirement #1 (100 % offline) is absolute, so an
# inherited environment must not be able to switch any of them back on.
#
# MEASURED, and the result is NOT the pyannote one — say so rather than imply a
# defect was caught. A DNS-recording probe (the 2026-07-30 audit's method,
# patching getaddrinfo/gethostbyname/socket.connect before any import) was run
# over a full diarization job TWICE: once with this block applied and once with a
# BARE environment, every one of these variables removed. BOTH runs performed
# ZERO lookups, loopback included. The probe's instrumentation was positive-
# controlled against a real outbound request first, so the zero is the engine's
# behaviour and not a blind harness.
#
# So unlike `PYANNOTE_METRICS_ENABLED`, these are PRECAUTIONARY, not a fix for a
# measured leak: with LOCAL checkpoints (see MODEL_DIR) nemo_toolkit 3.0.0 makes
# no network call of its own on the inference path. They are kept because three
# telemetry SDKs really are installed in this venv, a later NeMo release adding
# an exporter is exactly the silent regression this project pins rather than
# notices, and the cost is five lines. Notes on each, because they differ:
#   * WANDB_MODE=disabled  — wandb is imported by NeMo's exp_manager. "disabled"
#     is wandb's own documented no-network mode.
#   * SENTRY_DSN=""        — sentry-sdk sends nothing without a DSN; the empty
#     string is the documented off switch, and assigning it beats an inherited
#     one from some other tool's environment.
#   * ONE_LOGGER_* / NEMO_ONELOGGER_ENABLED — nv-one-logger is NVIDIA's training
#     telemetry. On this install it already reports "No exporters were provided",
#     i.e. it collects nothing by default, and NOTHING IN nemo_toolkit 3.0.0 WAS
#     FOUND TO READ `NEMO_ONELOGGER_ENABLED` — it is kept anyway because it costs
#     nothing and an exporter appearing in a later version is exactly the silent
#     regression this project pins rather than notices.
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"
os.environ["SENTRY_DSN"] = ""
os.environ["NEMO_ONELOGGER_ENABLED"] = "false"
os.environ["ONE_LOGGER_ENABLED"] = "false"

# Hard ceiling on this process's GPU allocation, in GB — armed in main().
#
# A fuse, not a budget. macOS reports a "recommended max working set" (51.8 GB on
# the owner's 64 GB M4) and PyTorch's MPS allocator grows all the way to it
# WITHOUT EVER RAISING; MOSS proved that is not theoretical (one oversized call
# reached ~50 GB and the Mac had to be rescued by hand).
#
# This engine is NOT inert against it, unlike spectral's: it really does run on
# MPS, and its peak scales with audio length — measured 1.15 GB for 98 s,
# 7.02 GB for 48 min, 13.33 GB for 67 min. A four-hour recording is the case this
# fuse exists for.
#
# HALF THE MACHINE, derived below rather than typed — the same rule the other
# six MPS sidecars use. The literal 32.0 that used to sit here was correct for
# this 64 GB M4 and became NO CAP AT ALL on the client's 16 GB Mac; see the
# docstring, which carries the arithmetic.
def _mps_cap_gb(fallback: float = 32.0) -> float:
    """The MPS fuse for THIS machine, in GB: half the physical RAM, floored at 10.

    ⚠ WHY NOT THE LITERAL 32.0. That number WAS "half the machine", for the 64 GB
    M4 every figure in this project was measured on. On a 16 GB Mac it is TWICE
    the physical RAM, and because the cap is armed as
    `set_per_process_memory_fraction(min(CAP / ceiling, 1.0))` the `min` clamped
    to 1.0 — the whole allocator, uncapped. Captured in the wild on the client's
    Mac, which printed it three times:

        MPS memory capped at 32 GB of the 11.8 GB macOS reports as available

    ⚠ AND WHY THE FLOOR OF 10, which is the half of this that was LEARNED THE
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

    The floor is 10.0 rather than a proportion because the number that matters is
    absolute: it must clear NeMo's measured 8.23 GB with margin, and stay under
    MOSS's measured 18.35 GB pool so the engine that actually took a Mac down on
    2026-08-21 is still refused. It changes NOTHING on a machine of 20 GB or more
    — this 64 GB M4 still gets exactly 32.0.

    ⚠ THE COST, owner-accepted with it stated (2026-08-26): 10 GB of MPS pool plus
    ~4 GB of other sidecars plus macOS is tight on a 16 GB machine, so a long NeMo
    meeting may still fail — its peak is 13.33 GB at 67 minutes — or the machine
    may slow down. The owner asked for exactly this: let it run, and let the lag be
    the evidence for a hardware upgrade rather than an invisible refusal. The
    failure stays BOUNDED either way, which is the whole point: PyTorch raises at
    the cap instead of growing without limit.

    `sysconf` rather than `torch.mps.recommended_max_memory()` because this is a
    module-level constant and torch is not imported yet in every service that
    carries one.

    Falls back to the historical constant if the query fails. A fuse that cannot
    size itself must not become a fuse that refuses everything.
    """
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        half = total / (1024 ** 3) / 2.0
        return round(max(half, 10.0), 1) if half > 0 else fallback
    except (ValueError, OSError, AttributeError):  # noqa: BLE001
        return fallback


MPS_MEMORY_CAP_GB = _mps_cap_gb()

# ---------------------------------------------------------------- checkpoints
#
# LOCAL ABSOLUTE PATHS, never NeMo's pretrained NAMES. `ClusteringDiarizer`
# branches on `model_path.endswith('.nemo')`: a path is `restore_from`'d off
# disk, while a bare name goes to `from_pretrained` → `maybe_download_from_cloud`
# → `https://api.ngc.nvidia.com/...`. The names would WORK on the owner's Mac —
# the checkpoints are already in `~/.cache/torch/NeMo` — and reach for the
# network on a client machine, which is the invisible-until-shipped shape this
# project keeps being bitten by. `download-best-models.sh` section 3c puts both
# files here.
MODEL_DIR = os.path.join(BASE, "models", "nemo")
SPEAKER_MODEL_PATH = os.path.join(MODEL_DIR, "titanet_large.nemo")
VAD_MODEL_PATH = os.path.join(MODEL_DIR, "vad_multilingual_marblenet.nemo")

# The upstream inference config, VENDORED VERBATIM (byte-identical to
# NVIDIA-NeMo/Speech @ 6c57e73e, examples/speaker_tasks/diarization/conf/
# inference/diar_infer_general.yaml). It is not edited: every deviation is
# applied in `build_config()` below so each one carries its own measured
# justification in a comment, the same rule that keeps MOSS's Chinese prompt and
# spectral's vendored tree verbatim.
VENDORED_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "vendor", "diar_infer_general.yaml")

# ---------------------------------------------------------------- the overrides
#
# FOUR deviations from the vendored config, each measured on this M4. Named as
# constants rather than written inline so `sidecar-tests.py` can pin the values
# without re-deriving them, and so a future edit has to argue with the comment.

# 1. MANDATORY ON macOS — NOT a tuning choice.
#    The stock value is 1, which makes PyTorch Lightning's `DataLoader` spawn
#    worker processes. NeMo's diarization dataset builds a `SpeechLabelEntity`
#    class DYNAMICALLY (collections.namedtuple-style, at import time), and a
#    dynamically-created class cannot be pickled — so on macOS's spawn start
#    method the workers die with `PicklingError` before a single frame of audio
#    is read. It is a hard crash at job start, not a slowdown.
NUM_WORKERS = 0

# 2. The speaker-count fix. `diar_infer_general.yaml` uses 10; the `meeting` and
#    `telephonic` configs upstream both already use 30, and only `general` is low.
#    Measured on recordings/Meeting5People.wav (98.5 s, ground truth 5 speakers):
#    stock general returns FOUR speakers, losing the one holding 7 % of the
#    speech; 8 of 8 sweep variants that perturbed anything at all returned five.
#    No measurable time cost — the p-value search is cheap next to embedding.
SPARSE_SEARCH_VOLUME = 30

# 3. ONE GLOBAL CLUSTERING, always.
#    NeMo switches to `long_forward_infer` when the base-scale segment count
#    exceeds `embeddings_per_chunk` (stock 10000, roughly 40 minutes of audio):
#    it over-clusters each chunk to `chunk_cluster_count` and merges the results.
#    Measured on a real 48.2-minute recording, that path COLLAPSES the meeting —
#    1 speaker, 73 turns, longest turn 275 s. Forcing the global path on the same
#    file gives 8 speakers, 324 turns, longest 73.8 s, median 3.50 s, and it is
#    FASTER (170 s vs 227 s) at 7.02 GB peak. The ceiling is set absurdly high
#    rather than removed because the chunked path is selected by a `>` on this
#    number; there is no flag for "never chunk".
EMBEDDINGS_PER_CHUNK = 10_000_000

# 4. Speaker count is AUTO by DEFAULT, and `max_num_speakers` bounds the eigengap
#    search either way. The stock 8 would silently cap a 12-person meeting at 8
#    speakers; 20 matches the ceiling the rest of the app assumes.
#
#    A COUNT CAN NOW ARRIVE (2026-08-10): the app gained a SPK control in its main
#    window, so `num_speakers > 0` sets `oracle_num_speakers` and the number
#    travels in the manifest. This note used to say the count was always automatic
#    and `oracle_num_speakers` stayed False — true when written, false since that
#    control landed. `MAX_NUM_SPEAKERS` still applies as the upper bound in both
#    modes, which is why it is not conditional on the pin.
MAX_NUM_SPEAKERS = 20

# NOT EXPOSED AND NOT SET, deliberately: `maj_vote_spk_count` and `nme_mat_size`.
# Only SIX config parameters ever reach the clustering call
# (`nemo/collections/asr/parts/utils/speaker_utils.py`, ~line 505):
# oracle_num_speakers, max_num_speakers, max_rp_threshold, sparse_search_volume,
# chunk_cluster_count, embeddings_per_chunk. The other two are declared in the
# YAML and in NMESC and never forwarded — setting them would be a knob nothing
# reads, which is precisely the `overlap.detect.model` defect this project
# shipped once already.


def install_wire() -> str:
    """Take fd 1 for the JSON protocol and send every other writer to stderr.

    NEMO LOGS TO STDOUT, AND THAT DESTROYS THIS PROTOCOL. Measured on the very
    first drive of this sidecar (2026-08-07): one job put 28 `[NeMo I …]` lines
    on fd 1 interleaved with our JSON, and a decoder reading it line by line died
    on the first of them. No other sidecar has this problem — pyannote, spectral
    and MOSS all log to stderr — so the mitigation lives here rather than in a
    shared rule.

    THE FIX IS AT THE FILE-DESCRIPTOR LEVEL, not `sys.stdout`, on purpose. NeMo's
    logger, PyTorch Lightning's progress bars, tqdm, a stray `print()` in a
    dependency and C-level writes from libsndfile are five different writers, and
    only one of them can be reached by rebinding a Python name. Duplicating fd 1
    to a private handle and then pointing fd 1 itself at fd 2 moves ALL of them —
    including anything a future NeMo release adds — into stderr, which is
    `logs/nemo.log`, where they are wanted.

    MUST RUN BEFORE NeMo IS IMPORTED: importing it alone prints banner lines, and
    a logging handler bound to fd 1 at import time would hold the wire for the
    life of the process.
    """
    global WIRE
    WIRE = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return "wire isolated on a private fd; every other stdout writer -> stderr"


def emit(payload: dict) -> None:
    # `WIRE`, not `sys.stdout` — see `install_wire`. The bytes written here are
    # byte-identical to what the pyannote and spectral services write for the
    # same payload, which is what `nemo/protocol-matches-pyannote` compares.
    try:
        WIRE.write(json.dumps(payload) + "\n")
        WIRE.flush()
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


def build_config(manifest_path: str, out_dir: str, device: str,
                 num_speakers: int):
    """The vendored upstream config plus this project's four measured overrides.

    Every override is applied HERE rather than in the YAML, so the vendored file
    stays byte-identical to upstream and each deviation sits next to the
    measurement that justifies it (see the constants above).
    """
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(VENDORED_CONFIG)
    cfg.diarizer.manifest_filepath = manifest_path
    cfg.diarizer.out_dir = out_dir
    cfg.device = device
    cfg.num_workers = NUM_WORKERS
    cfg.verbose = False

    # Local checkpoints, so the NGC resolver is never reached. See MODEL_DIR.
    cfg.diarizer.speaker_embeddings.model_path = SPEAKER_MODEL_PATH
    cfg.diarizer.vad.model_path = VAD_MODEL_PATH

    # The embedding tensors are a debugging artefact worth tens of MB per pass
    # and nothing here reads them back — identity is the WeSpeaker service's job,
    # from the recording, not from NeMo's pickles.
    cfg.diarizer.speaker_embeddings.parameters.save_embeddings = False

    params = cfg.diarizer.clustering.parameters
    params.sparse_search_volume = SPARSE_SEARCH_VOLUME
    params.embeddings_per_chunk = EMBEDDINGS_PER_CHUNK
    params.max_num_speakers = MAX_NUM_SPEAKERS
    # A pinned count is honoured through the MANIFEST (`num_speakers`), which is
    # the only thing `oracle_num_speakers` reads. Absent/0 leaves it False and
    # NME-SC counts for itself, which is this engine's normal mode.
    params.oracle_num_speakers = num_speakers > 0
    return cfg


def write_manifest(path: str, audio: str, num_speakers: int) -> None:
    """NeMo's one-line-per-recording input format.

    The keys are upstream's own example line; `duration: null` means "the whole
    file" and the three null paths mean "no oracle VAD, no reference RTTM".
    """
    with open(path, "w") as handle:
        handle.write(json.dumps({
            "audio_filepath": audio,
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "num_speakers": num_speakers if num_speakers > 0 else None,
            "rttm_filepath": None,
            "uem_filepath": None,
        }) + "\n")


def read_rttm(path: str):
    """RTTM → [(start, end, label)] — the same tuple shape pyannote's turns have.

    NeMo answers on disk, not in memory: `ClusteringDiarizer.diarize()` writes
    `<out_dir>/pred_rttms/<stem>.rttm`. Fields are
    `SPEAKER <file> <chan> <start> <dur> <NA> <NA> <speaker> …`, so end is
    start + duration. Times go out at full precision — see the module docstring.
    """
    turns = []
    with open(path) as handle:
        for line in handle:
            parts = line.split()
            if len(parts) > 7 and parts[0] == "SPEAKER":
                start, duration = float(parts[3]), float(parts[4])
                turns.append((start, start + duration, parts[7]))
    turns.sort(key=lambda t: (t[0], t[1]))
    return turns


# ---------------------------------------------------------------- main
def main() -> None:
    # FIRST, before any heavy import — see `install_wire`. NeMo prints on import.
    wire_note = install_wire()
    try:
        import torch
        from nemo.collections.asr.models import ClusteringDiarizer
    except Exception:  # noqa: BLE001
        fail(f"NeMo engine import failed: {brief_traceback()} — "
             "run download-best-models.sh (this engine needs .venv-nemo)")

    # Arm the fuse BEFORE anything loads weights, so even a bad load cannot run
    # away. Wrapped because these APIs are version-dependent and a missing one
    # must not cost us diarization: an uncapped run is what we have always had,
    # so failing to cap is a warning, never fatal.
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
        try:
            ceiling_gb = torch.mps.recommended_max_memory() / (1024 ** 3)
            if ceiling_gb > 0:
                torch.mps.set_per_process_memory_fraction(
                    min(MPS_MEMORY_CAP_GB / ceiling_gb, 1.0))
                log(f"MPS memory capped at {MPS_MEMORY_CAP_GB:.0f} GB "
                    f"of the {ceiling_gb:.1f} GB macOS reports as available")
        except Exception:  # noqa: BLE001
            log("WARNING could not cap MPS memory")

    # MPS is VERIFIED for this engine, and that is a measurement rather than an
    # assumption inherited from another model: output was identical to CPU to two
    # decimals, 12x faster (8.7x realtime vs 0.7x), and lighter (1.60 GB RSS vs
    # 8.22 GB). DiCoW's "MPS unverified, not worth the risk" precedent does NOT
    # bind here.
    log(f"device={device}")
    log(wire_note)

    for label, path in (("speaker", SPEAKER_MODEL_PATH), ("VAD", VAD_MODEL_PATH)):
        if not os.path.exists(path):
            fail(f"NeMo {label} checkpoint missing at {path} — "
                 "run download-best-models.sh")

    if not os.path.exists(VENDORED_CONFIG):
        fail(f"the vendored NeMo config is missing at {VENDORED_CONFIG}")

    log(f"models: {MODEL_DIR} (local .nemo files — the NGC resolver is never reached)")
    log(f"overrides: num_workers={NUM_WORKERS} sparse_search_volume="
        f"{SPARSE_SEARCH_VOLUME} embeddings_per_chunk={EMBEDDINGS_PER_CHUNK} "
        f"max_num_speakers={MAX_NUM_SPEAKERS}")

    # RESTORE BOTH CHECKPOINTS NOW, so that LOADED means loaded.
    #
    # Added by the 2026-08-10 audit. The existence checks above catch a MISSING
    # file, but a present-and-unreadable one — truncated by a full disk, half
    # copied, corrupted — was only discovered when `ClusteringDiarizer` built its
    # own models INSIDE the first job. On this engine the first job is the stop
    # pass, so the failure landed after the meeting was over, on the one pass that
    # produces every label, with nothing left to re-run. `spectral-service.py`
    # warms its two models for the same reason and says so in the same words.
    #
    # The instances are DISCARDED: `ClusteringDiarizer` constructs its own from
    # the same paths and there is no supported way to hand it ours. Do not
    # "optimise" this away on the grounds that the objects are unused — being
    # unused is the point, and the return value is not what is being bought.
    #
    # Measured cost 0.39 s (titanet 0.34, marblenet 0.05), and it also turns out
    # to PAY FOR ITSELF: the first job fell from 6.5 s to 4.0 s on the same audio,
    # because the lazy initialisation and the page cache carry over even though
    # the objects do not. Steady-state jobs are 2.5 s either way. That was not the
    # reason for the change and must not become the justification for reverting
    # it — if a future NeMo release stops sharing that state, the validation is
    # still worth its 0.39 s.
    try:
        from nemo.collections.asr.models import (EncDecClassificationModel,
                                                 EncDecSpeakerLabelModel)

        EncDecSpeakerLabelModel.restore_from(SPEAKER_MODEL_PATH, map_location="cpu")
        EncDecClassificationModel.restore_from(VAD_MODEL_PATH, map_location="cpu")
    except Exception:  # noqa: BLE001
        fail(f"a NeMo checkpoint is present but will not load: {brief_traceback()} "
             f"— re-fetch it with download-best-models.sh (section 3c)")
    log("checkpoints verified loadable (titanet_large + vad_multilingual_marblenet)")

    log("NeMo engine ready — turns only, no identity in this process, "
        "final pass only")
    emit({"type": "status", "text": "LOADED"})

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
        except Exception:  # noqa: BLE001
            emit({"type": "error", "text": f"Bad job line: {line[:120]}"})
            continue

        # Which identity space this job belongs to. ABSENT MEANS OFFICE, so a
        # single-stream app drives this sidecar exactly as it drives pyannote.
        # Routing information ONLY: nothing here behaves differently for remote.
        stream = job.get("stream")
        is_remote = stream == "remote"
        echo = {"stream": stream} if stream else {}

        # Whole-file passes and nothing else — see the module docstring. Refused
        # loudly, because a globally-clustering engine answering a windowed
        # request would return labels that look continuous and are not.
        mode = job.get("cmd", "final")
        if mode != "final":
            emit({"type": "error",
                  "text": f"NeMo has only a whole-file pass (got cmd={mode!r})",
                  **echo})
            continue

        audio = job.get("audio", "")
        if not os.path.exists(audio):
            emit({"type": "error", "text": f"Audio not found: {audio}", **echo})
            continue

        # A WAV whose header says ZERO FRAMES is not "a meeting where nobody
        # spoke" — it is a recording whose `data` chunk size was never written,
        # because the app went away before `AVAudioFile` was released. The samples
        # are still in the file; only the count is missing. Measured 2026-08-05:
        # 13 such recordings (~1.7 GB) on the owner's machine, one holding 34.1
        # minutes of speech.
        #
        # Without this, VAD finds no speech in an empty array and the pass returns
        # `segments: []` — a whole meeting reported as having no speakers, with no
        # message anywhere. Verbatim spectral semantics, for the same reason:
        # silence is the direction this project treats as dangerous.
        try:
            import soundfile as sf

            frames = sf.info(audio).frames
        except Exception:  # noqa: BLE001 — an unreadable header is its own answer
            frames = -1
        if frames == 0:
            emit({"type": "error",
                  "text": "This recording's WAV header says 0 frames, so nothing "
                          "can read it — the audio is almost certainly still in the "
                          "file and the size field was never written (the app was "
                          "quit or killed mid-recording). Repair it with "
                          "scripts/tools/repair-wav-header.py, then run the pass "
                          "again.",
                  **echo})
            continue

        # 0 = auto, which is this engine's normal mode: NME-SC estimates the
        # count from the eigengap and there is no Settings control for it.
        num_speakers = int(job.get("num_speakers", 0))

        started = time.time()
        # A WRITABLE PER-JOB DIRECTORY, never beside this script. NeMo writes a
        # manifest, per-scale segment lists, VAD outputs and the predicted RTTM,
        # and in the packaged `.app` this file lives under Contents/Resources,
        # which is READ-ONLY. Per-job rather than per-process so two passes can
        # never read each other's pred_rttms, and removed in `finally` so a long
        # meeting does not leave hundreds of MB behind.
        work = tempfile.mkdtemp(prefix="mt-nemo-")
        try:
            manifest = os.path.join(work, "manifest.json")
            write_manifest(manifest, audio, num_speakers)
            cfg = build_config(manifest, work, device, num_speakers)
            ClusteringDiarizer(cfg=cfg).to(device).diarize()

            stem = os.path.splitext(os.path.basename(audio))[0]
            rttm = os.path.join(work, "pred_rttms", f"{stem}.rttm")
            if not os.path.exists(rttm):
                emit({"type": "error",
                      "text": f"NeMo wrote no RTTM for {stem} — the pass produced "
                              f"no result at all", **echo})
                continue

            segments = wire_segments(read_rttm(rttm))
            log(f"final done in {time.time() - started:.1f}s — "
                f"{len(segments)} turns, {len({s['label'] for s in segments})} speakers"
                f"{' [remote]' if is_remote else ''}")
            emit({"type": "result", "audio": audio, "segments": segments, **echo})
        except Exception as exc:  # noqa: BLE001 — job error: report, keep serving
            # The stream is echoed so the app fails only THAT stream's gate —
            # a remote failure must never abort the office transcript.
            #
            # ONE failure is translated, because it is not a fault: NeMo's VAD
            # raises when it finds no speech at all, which is the honest verdict
            # on a recording of silence. The owner met it on 2026-08-12 with a
            # 1.3 s test recording and a Remote channel capturing nothing, and
            # what the app showed was three lines of `speaker_utils.py`
            # traceback — a correct refusal wearing the costume of a crash.
            #
            # The precedent is `spectral/rejects-zero-frame-audio`: a sidecar
            # that cannot proceed must say so in a sentence that names the
            # cause. Matched on the upstream text rather than the exception
            # type, since `ValueError` is raised for many other reasons here;
            # anything unmatched keeps the traceback, which is the right
            # default for a failure nobody has diagnosed yet.
            if "contains silence" in str(exc):
                where = "remote audio" if is_remote else "recording"
                # `kind` is what makes this MACHINE-readable. The app has to tell
                # a correct verdict from a fault, and it must not do that by
                # matching this sentence — a wording change here would silently
                # turn the panel red again, and prose is not a protocol. An older
                # app ignores the extra key and still shows the sentence.
                emit({"type": "error", "kind": "no_speech",
                      "text": f"No speech found in the {where}, so there were "
                              f"no speakers to identify. The transcript is "
                              f"unaffected.", **echo})
            else:
                emit({"type": "error",
                      "text": f"Diarization failed: {brief_traceback()}", **echo})
        finally:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
