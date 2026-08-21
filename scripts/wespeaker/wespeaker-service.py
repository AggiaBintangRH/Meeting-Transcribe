#!/usr/bin/env python3
"""Speaker IDENTITY sidecar for MeetingTranscriber (persistent).

ONE SERVICE PER MODEL (owner, 2026-07-29). This half of the former
`diarize-service.py` owns the WeSpeaker embedder and BOTH persistent profile
stores. It is named for the model, not the role — `wespeaker`, like every other
service in the split.

It never diarizes. It is handed time spans that somebody else found — today the
pyannote pipeline in `scripts/pyannote/pyannote-service.py` — embeds each span's
speech, and answers WHO those people are against the saved profiles.

Speaker profiles persist in models/speaker-profiles/:
  profiles.json   [{"id":1,"name":"Speaker 1","count":12}]  (names editable by app)
  embeddings.npz  {"1": centroid vector}

WHY THIS IS ITS OWN PROCESS, and what that does NOT mean
-------------------------------------------------------
Identity used to be reachable only through the pyannote pipeline, because both
lived in one process. Split, attaching saved identities to time spans is
*structurally available* to any diarizer that can produce spans. That is the
whole claim, and it is deliberately not cashed in anywhere yet: MOSS still labels
speakers per chunk with no cross-chunk stitching (the owner deferred that — "nanti
saja"), and the ATND position layer still owns its own cluster ids and has no
embeddings. Neither reaches this service.

It also starts FAST for what it does: this process needs `pyannote.audio` only
for `PretrainedSpeakerEmbedding` — ~26 MB of weights — and never instantiates the
Pipeline at all. It runs in the main `.venv`.

Dual-stream (Office + Remote): TWO profile stores in one process.
  The Office array and the conferencing (Remote) channel are different rooms of
  people, so their identities must not be comparable. `ProfileStore.assign()`
  scores an incoming embedding against EVERY stored centroid, so a single shared
  store would let a remote voice cosine-match onto an office profile — silent,
  permanent identity corruption. Two stores make that structurally impossible
  rather than merely filtered:
      office  profiles.json         embeddings.npz         names "Speaker N"
      remote  profiles-remote.json  embeddings-remote.npz  names "RN"
  A SECOND PROCESS would also be structurally safe, but would mean a second
  WeSpeaker embedder resident and a second startup for nothing: job ordering on a
  single stdin gives deterministic interleaving, and the app's client serializes
  its requests so store-mutation order matches the order the jobs were made.
  Remote ids go on the wire as REMOTE_ID_BASE + local_id, mirroring the app's
  proven position-id split (pyannote < 10_000 <= remote < 100_000 <= position),
  so a downstream consumer tells the spaces apart with one integer comparison.
  The remote files do not exist until a dual-stream session runs, and an office
  job never touches them — so there is nothing to migrate.

Protocol:
  stdout: {"type":"status","text":"LOADED"}
          {"type":"identify_result","id":7,"speakers":{
              "SPEAKER_00":{"id":1,"name":"Speaker 1","conf":0.874},
              "SPEAKER_01":{"id":2,"name":"Speaker 2"},
              "SPEAKER_02":null}}
          {"type":"status","text":"RESET"}
          {"type":"error","id":7,"text":...}   (+ "stream" echoed if sent)
          THREE encodings of "no value", and they mean different things:
            "conf" PRESENT — the winning cosine similarity that matched this voice
                             to its saved profile.
            "conf" ABSENT  — this call minted a brand-new profile. It was never
                             scored against anything, so there is no number. NOT
                             zero: a displayed 0.00 would read as "we are sure
                             this is the wrong person".
            null identity  — the voice could not be identified at all (under
                             MIN_EMBED_SEC of speech, or a degenerate embedding).
                             The caller DROPS those spans, exactly as the old
                             single service dropped them before emitting.
          A success reply carries no "stream": the request `id` correlates it, and
          the wire ids already have REMOTE_ID_BASE applied. Errors echo the stream
          so the app can fail only that stream's gate.
  stdin (one JSON per line):
          {"cmd":"identify","id":7,"audio":path,
           "turns":[{"start":…,"end":…,"label":"SPEAKER_00"},…]}
          {"cmd":"reset"}
          an identify job may carry "stream":"office"|"remote" — ABSENT MEANS
          OFFICE. EOF exits.

  ONE `identify` PER DIARIZATION RUN, carrying that whole run's turns.
  `assign()`'s one-to-one guarantee (two distinct voices in the same run can
  never collapse onto one profile) is defined PER RUN, so splitting a run across
  several calls would silently weaken it. This is the shape that preserves it.

Fully offline: HF_HOME points at the project models/ folder, PYANNOTE_METRICS_ENABLED
turns off pyannote 4.x's default-on telemetry, and audio is decoded with soundfile
rather than `torchaudio.load` (a torchcodec wrapper needing Homebrew ffmpeg).
"""
import json
import os
import sys
import time
import traceback

# This file lives at scripts/<service>/<file>.py (one folder per service,
# owner 2026-07-29), so the project root — which owns models/ — is THREE levels
# up, not two. Bundled, that root is Contents/Resources. Getting this wrong is
# silent: PROFILE_DIR would become scripts/models/speaker-profiles and a SECOND,
# empty profile store would appear while the real one went untouched.
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HOME", os.path.join(BASE, "models"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# pyannote.audio 4.x ships OpenTelemetry tracing that is ON BY DEFAULT and posts
# to https://otel.pyannote.ai/v1/traces. This process never calls a Pipeline, so
# only `track_model_init` fires here (once, at embedder load) rather than the
# per-`apply()` span `pyannote-service.py` gets — but one call still breaks client
# hard requirement #1 (100 % offline, nothing leaves the machine), and
# `HF_HUB_OFFLINE` does not cover it. See the longer note in
# `scripts/pyannote/pyannote-service.py` for the measurement that found it.
# Assigned, not `setdefault`: an inherited environment must not turn it back on.
os.environ["PYANNOTE_METRICS_ENABLED"] = "false"

EMBEDDING_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"
PROFILE_DIR = os.environ.get("MT_PROFILE_DIR", os.path.join(BASE, "models", "speaker-profiles"))
EMBED_SR = 16_000            # WeSpeaker's training rate — see resolve_speakers
SIM_THRESHOLD = 0.5          # cosine similarity to accept a profile match

# Hard ceiling on this process's GPU allocation, in GB — armed in main().
#
# The same fuse the two MOSS sidecars and pyannote carry, and for the same
# measured reason: PyTorch's MPS allocator grows to macOS's "recommended max"
# (51.8 GB on the owner's 64 GB M4) WITHOUT EVER RAISING, and MOSS proved on
# 2026-07-31 that one oversized call really can take ~50 GB and the machine with
# it. This service holds a whole recording in memory for a final pass (~710 MB
# for 67 minutes at 44.1 kHz) and embeds every speaker's spliced spans, so its
# appetite scales with meeting length too.
#
# HALF THE MACHINE, derived below rather than typed. Normal passes sit far
# below it; exceeding it fails
# ONE request loudly instead of swapping the Mac.
def _half_the_physical_ram_gb(fallback: float = 32.0) -> float:
    """Half this machine's physical RAM, in GB — the rule stated above, computed.

    ⚠ WHY THIS IS NO LONGER THE LITERAL 32.0. That number WAS "half the machine",
    for the 64 GB M4 every figure in this project was measured on. On a 16 GB Mac
    it is TWICE the physical RAM, so the fuse could never blow before macOS began
    swapping — which is exactly the failure it exists to prevent.

    OBSERVED on the client's 16 GB Mac, 2026-08-21: six Python sidecars resident
    at ~14.8 GB, 6.67 GB of swap in use, memory pressure in the red, and a MOSS
    stop pass that looked HUNG rather than slow. Every cap in the process tree
    was 32 GB — twice the machine — so nothing anywhere could fire.

    Failing one chunk loudly beats dragging the machine into swap for twenty
    minutes: the first is a message the user can act on, the second is
    indistinguishable from a bug.

    `sysconf` rather than `torch.mps.recommended_max_memory()` because this is a
    module-level constant and torch is not imported yet in every service that
    carries one. The two agree by construction on the development Mac: 64 GB
    physical -> 32.0, byte-for-byte the value that shipped for weeks, so this
    change is a verified no-op there and only acts on smaller machines.

    Falls back to the historical constant if the query fails. A fuse that cannot
    size itself must not become a fuse that refuses everything.
    """
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        half = total / (1024 ** 3) / 2.0
        return round(half, 1) if half > 0 else fallback
    except (ValueError, OSError, AttributeError):  # noqa: BLE001
        return fallback


MPS_MEMORY_CAP_GB = _half_the_physical_ram_gb()
MIN_EMBED_SEC = 1.5          # need this much speech to embed a voice
MAX_CENTROID_COUNT = 50      # cap running-mean weight so voices can drift
# Offset added to a REMOTE profile id before it leaves this process. Mirrors the
# app's PositionDiarizer.positionIDBase = 100_000 split; must stay below it.
REMOTE_ID_BASE = 10_000


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


# ---------------------------------------------------------------- store
class ProfileStore:
    """Voice profiles: centroid embeddings on disk, names editable by the app.

    One instance per identity SPACE (see the module docstring): the file names
    and the generated-name template are constructor arguments precisely so a
    second, disjoint space can exist without any cross-store code path.
    """

    def __init__(self, np, json_name="profiles.json", npz_name="embeddings.npz",
                 name_template="Speaker {n}"):
        self.np = np
        os.makedirs(PROFILE_DIR, exist_ok=True)
        self.json_path = os.path.join(PROFILE_DIR, json_name)
        self.npz_path = os.path.join(PROFILE_DIR, npz_name)
        self.name_template = name_template
        self.profiles = []   # [{"id","name","count"}]
        self.centroids = {}  # id(int) -> np.ndarray
        self.load()

    def load(self):
        if os.path.exists(self.json_path):
            with open(self.json_path) as f:
                self.profiles = json.load(f)
        if os.path.exists(self.npz_path):
            data = self.np.load(self.npz_path)
            self.centroids = {int(k): data[k] for k in data.files}

    def reset(self):
        """Wipe all saved voices — a fresh start for a new recording."""
        self.profiles = []
        self.centroids = {}
        for path in (self.json_path, self.npz_path):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:  # noqa: BLE001
                pass

    def reload_names(self):
        """Pick up renames done by the app in profiles.json."""
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path) as f:
                    on_disk = {p["id"]: p for p in json.load(f)}
                for p in self.profiles:
                    if p["id"] in on_disk:
                        p["name"] = on_disk[p["id"]]["name"]
            except Exception:  # noqa: BLE001
                pass

    def save(self):
        with open(self.json_path, "w") as f:
            json.dump(self.profiles, f, ensure_ascii=False, indent=1)
        self.np.savez(self.npz_path, **{str(k): v for k, v in self.centroids.items()})

    def name(self, pid: int) -> str:
        for p in self.profiles:
            if p["id"] == pid:
                return p["name"]
        return self.name_template.format(n=pid)

    def _cosine(self, a, b) -> float:
        np = self.np
        return float(np.dot(a, b)
                     / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def _create(self, embedding) -> int:
        new_id = max([p["id"] for p in self.profiles], default=0) + 1
        # A LOCAL id must stay below REMOTE_ID_BASE. The offset is applied once,
        # at the process boundary (`speaker_identities`), so a local id that
        # reached 10 000 would leave here indistinguishable from a remote id —
        # and the app routes renames and store writes on that comparison alone,
        # so an office voice would start being written into profiles-remote.json.
        # Nothing enforced this before; it was merely true because nobody has had
        # 10 000 speakers. RAISED, not asserted: `assert` is stripped under -O,
        # and this is exactly the failure that must never be optimised away. The
        # job handler turns it into an error reply and does NOT save the store, so
        # the bad id never reaches disk.
        if new_id >= REMOTE_ID_BASE:
            raise RuntimeError(
                f"refusing to mint local profile id {new_id}: ids must stay below "
                f"REMOTE_ID_BASE ({REMOTE_ID_BASE}) or the office and remote "
                f"identity spaces collide on the wire")
        self.profiles.append({"id": new_id,
                              "name": self.name_template.format(n=new_id),
                              "count": 1})
        self.centroids[new_id] = embedding
        return new_id

    def _update(self, pid: int, embedding) -> None:
        """Running-mean centroid update (capped so old voices can drift)."""
        for p in self.profiles:
            if p["id"] == pid:
                n = min(p["count"], MAX_CENTROID_COUNT)
                self.centroids[pid] = (self.centroids[pid] * n + embedding) / (n + 1)
                p["count"] += 1
                break

    def assign(self, embeddings: dict) -> dict:
        """Map several local speakers (one diarization run) to profiles at once,
        one-to-one: two distinct voices in the SAME run can never collapse onto
        the same profile. Unmatched voices become new distinct profiles.

        embeddings: {local_label: vector}  ->  {local_label: (profile_id, conf)}

        `conf` is the WINNING COSINE SIMILARITY when the voice matched a stored
        centroid, and None when this call minted a brand-new profile. None means
        "no measurement", NOT low confidence: a first appearance was never scored
        against anything, so any number here would be invented. The caller omits
        the field entirely in that case rather than sending 0 — a displayed 0.00
        would read as "we are sure this is the wrong person".
        """
        # Score every (local voice, existing profile) pair, best first.
        existing = list(self.centroids.items())          # snapshot (don't mutate mid-run)
        pairs = []
        for label, emb in embeddings.items():
            for pid, centroid in existing:
                pairs.append((self._cosine(emb, centroid), label, pid))
        pairs.sort(key=lambda t: t[0], reverse=True)

        mapping = {}
        used_profiles = set()
        for sim, label, pid in pairs:
            if label in mapping or pid in used_profiles:
                continue
            if sim >= SIM_THRESHOLD:
                mapping[label] = (pid, sim)
                used_profiles.add(pid)
                log(f"matched {label} -> profile {pid} (sim={sim:.2f})")

        # Why each unmatched voice lost, so a "new profile" line can be read.
        # 0.49 (the threshold is a hair too high) and 0.12 (genuinely a different
        # voice) call for opposite fixes and used to look identical — a real bug
        # hid behind exactly that ambiguity.
        #
        # There are TWO ways to lose, and they must not be conflated: scoring
        # below the threshold against everything, or scoring ABOVE it on a
        # profile another voice in this same run already claimed. The one-to-one
        # rule above makes the second case normal, not an anomaly — but reporting
        # it as "best sim=0.56 < 0.5" is a self-contradiction that sends the
        # reader hunting for a threshold bug that isn't there.
        best_free = {}   # label -> (sim, pid) over profiles still unclaimed
        best_any = {}    # label -> (sim, pid) over every profile
        for sim, label, pid in pairs:
            if sim > best_any.get(label, (-2.0, None))[0]:
                best_any[label] = (sim, pid)
            if pid not in used_profiles and sim > best_free.get(label, (-2.0, None))[0]:
                best_free[label] = (sim, pid)

        # Any voice with no confident, still-free match becomes a new profile.
        for label, emb in embeddings.items():
            if label not in mapping:
                new_id = self._create(emb)
                # No conf: nothing was matched, so there is no similarity to report.
                mapping[label] = (new_id, None)
                used_profiles.add(new_id)
                if label not in best_any:
                    log(f"new profile {new_id} for {label} (no existing profiles)")
                    continue
                any_sim, any_pid = best_any[label]
                free = best_free.get(label)
                if any_sim >= SIM_THRESHOLD:
                    # Lost to the one-to-one rule, not to the threshold.
                    free_note = (f"best free was profile {free[1]} at {free[0]:.2f}"
                                 if free else "no free profile left")
                    log(f"new profile {new_id} for {label} — profile {any_pid} "
                        f"fit at {any_sim:.2f} but was taken by another voice "
                        f"in this run; {free_note}")
                else:
                    log(f"new profile {new_id} for {label} "
                        f"(best sim={any_sim:.2f} on profile {any_pid}, "
                        f"below {SIM_THRESHOLD})")

        # Now fold each voice into its profile's running-mean centroid.
        for label, (pid, _conf) in mapping.items():
            self._update(pid, embeddings[label])
        return mapping


# ---------------------------------------------------------------- main
def main() -> None:
    try:
        import numpy as np
        import soundfile as sf
        import torch
        import torchaudio          # for functional.resample only — see resolve_speakers
        from pyannote.audio.pipelines.speaker_verification import (
            PretrainedSpeakerEmbedding,
        )
    except Exception:  # noqa: BLE001
        fail(f"pyannote.audio import failed: {brief_traceback()} — "
             "run download-best-models.sh")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Armed before any weights load, so even a bad load cannot run away. Wrapped
    # because these APIs are version-dependent and a missing one must not cost us
    # speaker identity: uncapped is what we always had, so failing to cap is a
    # warning, never fatal.
    if device.type == "mps":
        try:
            ceiling_gb = torch.mps.recommended_max_memory() / (1024 ** 3)
            if ceiling_gb > 0:
                torch.mps.set_per_process_memory_fraction(
                    min(MPS_MEMORY_CAP_GB / ceiling_gb, 1.0))
                log(f"MPS memory capped at {MPS_MEMORY_CAP_GB:.0f} GB "
                    f"of the {ceiling_gb:.1f} GB macOS reports as available")
        except Exception:  # noqa: BLE001
            log("WARNING could not cap MPS memory — a very long pass could "
                "allocate without bound")

    log(f"loading embedding model {EMBEDDING_MODEL}")
    try:
        embedder = PretrainedSpeakerEmbedding(EMBEDDING_MODEL, device=device)
    except Exception:  # noqa: BLE001
        fail(f"Embedding model load failed: {brief_traceback()} — "
             "run download-best-models.sh")

    # Two disjoint identity spaces in one process — see the module docstring.
    # Creating the remote store does NOT create its files (load() only reads what
    # exists, save() only runs for the store a job actually used), so a
    # single-stream session leaves models/speaker-profiles/ exactly as it was.
    office_store = ProfileStore(np)
    remote_store = ProfileStore(np, "profiles-remote.json", "embeddings-remote.npz", "R{n}")
    stores = (office_store, remote_store)
    log(f"embedder on {device}, {len(office_store.profiles)} saved profiles "
        f"({len(remote_store.profiles)} remote)")
    # See the same note in pyannote-service.py: torchcodec is pruned from the
    # packaged .app, so pyannote warns at import that built-in decoding will fail.
    # Expected — this process decodes with soundfile in resolve_speakers.
    log("note: pyannote's torchcodec warning above is expected — audio is decoded "
        "in this process with soundfile (resolve_speakers)")
    emit({"type": "status", "text": "LOADED"})

    def resolve_speakers(audio_path, turns, store):
        """Embed each local speaker's speech, map local label → (profile id, conf).

        `store` is the ONLY place this stream's voices are ever compared: the
        caller picks it from the job's stream, so an office embedding is never
        scored against a remote centroid or vice versa.

        Two distinct "no value" cases share this mapping and must not be
        conflated: a value of None means the voice could not be identified AT ALL
        (too little speech, degenerate embedding) and its turns are dropped
        downstream; a `(pid, None)` tuple means the voice WAS identified, as a
        brand-new profile that had nothing to be scored against.
        """
        # soundfile, NOT `torchaudio.load` — the same ffmpeg trap as Whisper
        # (37c16bac9) and as `pyannote-service.py`'s `load_waveform`. From
        # TorchAudio 2.9 `torchaudio.load` is documented as a thin wrapper over
        # **torchcodec's AudioDecoder** (its own docstring says so, and `normalize`
        # / `backend` are ignored), and torchcodec's dylibs resolve
        # `libavcodec`/`libavformat`/`libavutil` through one `LC_RPATH` of
        # `/opt/homebrew/opt/ffmpeg/lib`. There are ZERO `libav*` in the packaged
        # `.app`, so on a client Mac without Homebrew ffmpeg this line raised and
        # took speaker identity — and therefore the whole session, since both
        # sidecars must load — down with it. soundfile carries its own libsndfile.
        # Verified bit-identical to torchcodec on these recordings (max abs sample
        # diff 0.000e+00 over 177 608 340 samples). Do not "restore" torchaudio here.
        #
        # `torchaudio.functional.resample` below is pure torch — no codec, no
        # ffmpeg — so it stays.
        data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T.copy())  # (time, channel) -> (channel, time)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # WeSpeaker is a 16 kHz model. Feeding it audio at another rate does not
        # error — it silently embeds speech that is pitch- and tempo-shifted, and
        # the result is not the same voice: measured on a real 44.1 kHz recording,
        # the SAME 10 s of speech embedded at 44.1 kHz vs 16 kHz scored cosine
        # 0.036 against itself.
        #
        # This was live. Live chunks arrive as 16 kHz temp WAVs and matched at
        # 0.81–0.98, while a stop-time pass over the recording file (44.1 kHz,
        # the capture device's native rate) scored 0.11 against the very profile
        # it had just built and minted a duplicate — one person, two profiles.
        # pyannote's own pipeline resamples internally, so the TURNS were always
        # right; only the identity attached to them was wrong.
        if sr != EMBED_SR:
            waveform = torchaudio.functional.resample(waveform, sr, EMBED_SR)
            sr = EMBED_SR

        by_label = {}
        for start, end, label in turns:
            by_label.setdefault(label, []).append((start, end))

        # First embed every local speaker; too-short voices are left unlabelled.
        embeddings = {}
        mapping = {}
        for label, spans in by_label.items():
            pieces = []
            total = 0.0
            for start, end in spans:
                a, b = int(start * sr), int(end * sr)
                pieces.append(waveform[:, a:b])
                total += end - start
                if total >= 10.0:  # 10s of speech is plenty for an embedding
                    break
            if total < MIN_EMBED_SEC:
                mapping[label] = None  # too little speech to identify reliably
                continue
            speech = torch.cat(pieces, dim=1)
            embedding = np.asarray(embedder(speech.unsqueeze(0))).reshape(-1)
            # Reject a degenerate embedding rather than let it mint a profile.
            # MIN_EMBED_SEC gates on the turns' CLAIMED duration, but the vector
            # comes from waveform[:, a:b] slices; if a turn's timestamps run past
            # the actual audio the slice is empty, the embedder returns NaN (numpy
            # logs "Mean of empty slice"), and a NaN vector scores ~0 against
            # everything, so it always looks like a brand-new speaker. That was
            # the "best sim=-0.01" spurious profile in the log.
            norm = float(np.linalg.norm(embedding))
            if not np.all(np.isfinite(embedding)) or norm < 1e-6:
                log(f"skip {label}: degenerate embedding (norm={norm:.3g})")
                mapping[label] = None
                continue
            embeddings[label] = embedding

        # Assign all voices at once so distinct speakers get distinct profiles.
        if embeddings:
            mapping.update(store.assign(embeddings))
        return mapping

    def speaker_identities(mapping, store, id_base):
        """Local label → identity, with WIRE ids: local id + id_base. The offset is
        applied here, at the process boundary, so every id inside this file stays
        local to its store and the two spaces can never be indexed by the same int.

        A label whose voice could not be identified maps to `null`, and the caller
        drops those spans — the same drop the single service used to perform
        itself, just moved to the side that owns the spans.

        "conf" (the winning cosine similarity) is present only when the voice
        matched an existing profile. A brand-new profile carries no key at all —
        an absent field is the only honest encoding of "not measured", and a
        consumer that reads a missing key as 0 would be showing certainty about a
        number nobody computed.
        """
        identities = {}
        for label, entry in mapping.items():
            if entry is None:
                identities[label] = None   # unidentifiable blip
                continue
            pid, conf = entry
            identity = {"id": pid + id_base, "name": store.name(pid)}
            if conf is not None:
                identity["conf"] = round(float(conf), 3)
            identities[label] = identity
        return identities

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
            cmd = job.get("cmd", "identify")
        except Exception:  # noqa: BLE001
            emit({"type": "error", "text": f"Bad job line: {line[:120]}"})
            continue

        # Which identity space this job belongs to. ABSENT MEANS OFFICE.
        stream = job.get("stream")
        is_remote = stream == "remote"
        store = remote_store if is_remote else office_store
        id_base = REMOTE_ID_BASE if is_remote else 0
        echo = {"stream": stream} if stream else {}
        request_id = job.get("id")

        if cmd == "reset":
            # A fresh session resets BOTH spaces: the setting is "start this
            # recording with a clean speaker store", and leaving half of it
            # populated would make remote numbering carry over on its own.
            for s in stores:
                s.reset()
            log("speaker profiles reset (office + remote) — fresh session")
            emit({"type": "status", "text": "RESET"})
            continue

        audio = job.get("audio", "")
        if not os.path.exists(audio):
            emit({"type": "error", "id": request_id,
                  "text": f"Audio not found: {audio}", **echo})
            continue

        # Renames land in either file, and a job of one stream may follow a
        # rename of the other, so refresh both (a missing file is a no-op).
        for s in stores:
            s.reload_names()
        # The spans come off the wire as objects; resolve_speakers takes the same
        # (start, end, label) tuples the pipeline used to hand it directly.
        turns = [(float(t["start"]), float(t["end"]), t["label"])
                 for t in job.get("turns", [])]
        started = time.time()
        try:
            mapping = resolve_speakers(audio, turns, store)
            store.save()
            identities = speaker_identities(mapping, store, id_base)
            named = sum(1 for v in identities.values() if v is not None)
            log(f"identify done in {time.time() - started:.1f}s — "
                f"{len(turns)} turns, {named}/{len(identities)} voices identified"
                f"{' [remote]' if is_remote else ''}")
            emit({"type": "identify_result", "id": request_id,
                  "speakers": identities})
        except Exception:  # noqa: BLE001 — job error: report, keep serving
            # The stream is echoed so the app fails only THAT stream's gate —
            # a remote failure must never abort the office transcript. Note the
            # store was NOT saved on this path: a job that raised leaves disk
            # exactly as it was.
            emit({"type": "error", "id": request_id,
                  "text": f"Speaker identification failed: {brief_traceback()}",
                  **echo})


if __name__ == "__main__":
    main()
