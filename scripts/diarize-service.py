#!/usr/bin/env python3
"""
Hybrid speaker diarization sidecar for MeetingTranscriber (persistent).

Two modes over one loaded pipeline (pyannote community-1 on MPS):
  • chunk  — diarize a ~30s chunk live, embed each voice (WeSpeaker) and
             match against the speaker-profile store → stable names now
  • final  — batch diarization of the full recording at stop (best DER),
             clusters matched against the same store → refined labels

Speaker profiles persist in models/speaker-profiles/:
  profiles.json   [{"id":1,"name":"Speaker 1","count":12}]  (names editable by app)
  embeddings.npz  {"1": centroid vector}

Protocol:
  stdout: {"type":"status","text":"LOADED"}
          {"type":"chunk_result","window_start":s,"segments":[{start,end,id,name}]}
          {"type":"result","segments":[{start,end,id,name}]}
          {"type":"error","text":...}
  stdin (one JSON per line):
          {"cmd":"chunk","audio":path,"window_start":seconds}
          {"cmd":"final","audio":path,"num_speakers":0}
          EOF exits.

Fully offline: HF_HOME points at the project models/ folder.
"""
import json
import os
import sys
import time
import traceback

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(BASE, "models"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

MODEL = "pyannote/speaker-diarization-community-1"
EMBEDDING_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"
PROFILE_DIR = os.environ.get("MT_PROFILE_DIR", os.path.join(BASE, "models", "speaker-profiles"))
SIM_THRESHOLD = 0.5          # cosine similarity to accept a profile match
MIN_EMBED_SEC = 1.5          # need this much speech to embed a voice
MAX_CENTROID_COUNT = 50      # cap running-mean weight so voices can drift


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
    """Voice profiles: centroid embeddings on disk, names editable by the app."""

    def __init__(self, np):
        self.np = np
        os.makedirs(PROFILE_DIR, exist_ok=True)
        self.json_path = os.path.join(PROFILE_DIR, "profiles.json")
        self.npz_path = os.path.join(PROFILE_DIR, "embeddings.npz")
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
        return f"Speaker {pid}"

    def _cosine(self, a, b) -> float:
        np = self.np
        return float(np.dot(a, b)
                     / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def _create(self, embedding) -> int:
        new_id = max([p["id"] for p in self.profiles], default=0) + 1
        self.profiles.append({"id": new_id, "name": f"Speaker {new_id}", "count": 1})
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

        embeddings: {local_label: vector}  ->  {local_label: profile_id}
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
                mapping[label] = pid
                used_profiles.add(pid)
                log(f"matched {label} -> profile {pid} (sim={sim:.2f})")

        # Any voice with no confident, still-free match becomes a new profile.
        for label, emb in embeddings.items():
            if label not in mapping:
                new_id = self._create(emb)
                mapping[label] = new_id
                used_profiles.add(new_id)
                log(f"new profile {new_id} for {label}")

        # Now fold each voice into its profile's running-mean centroid.
        for label, pid in mapping.items():
            self._update(pid, embeddings[label])
        return mapping


# ---------------------------------------------------------------- main
def main() -> None:
    try:
        import numpy as np
        import torch
        import torchaudio
        from pyannote.audio import Pipeline
        from pyannote.audio.pipelines.speaker_verification import (
            PretrainedSpeakerEmbedding,
        )
    except Exception:  # noqa: BLE001
        fail(f"pyannote.audio import failed: {brief_traceback()} — "
             "run download-best-models.sh")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    log(f"loading pipeline {MODEL}")
    try:
        pipeline = Pipeline.from_pretrained(MODEL)
        pipeline.to(device)
    except Exception:  # noqa: BLE001
        fail(f"Pipeline load failed: {brief_traceback()} — "
             "run download-best-models.sh (pyannote is a gated repo)")

    log(f"loading embedding model {EMBEDDING_MODEL}")
    try:
        embedder = PretrainedSpeakerEmbedding(EMBEDDING_MODEL, device=device)
    except Exception:  # noqa: BLE001
        fail(f"Embedding model load failed: {brief_traceback()} — "
             "run download-best-models.sh")

    store = ProfileStore(np)
    log(f"pipeline + embedder on {device}, {len(store.profiles)} saved profiles")
    emit({"type": "status", "text": "LOADED"})

    def local_turns(audio_path, num_speakers=0, exclusive=False):
        kwargs = {"num_speakers": num_speakers} if num_speakers > 0 else {}
        output = pipeline(audio_path, **kwargs)
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

    def resolve_speakers(audio_path, turns):
        """Embed each local speaker's speech, map local label → profile id."""
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

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
            embedding = embedder(speech.unsqueeze(0))  # (1, ch, samples) → (1, dim)
            embeddings[label] = np.asarray(embedding).reshape(-1)

        # Assign all voices at once so distinct speakers get distinct profiles.
        if embeddings:
            mapping.update(store.assign(embeddings))
        return mapping

    def labeled_segments(turns, mapping):
        result = []
        for start, end, label in turns:
            pid = mapping.get(label)
            if pid is None:
                continue  # unidentifiable blip
            result.append({"start": round(start, 3), "end": round(end, 3),
                           "id": pid, "name": store.name(pid)})
        return result

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

        if cmd == "reset":
            store.reset()
            log("speaker profiles reset — fresh session")
            emit({"type": "status", "text": "RESET"})
            continue

        audio = job.get("audio", "")
        if not os.path.exists(audio):
            emit({"type": "error", "text": f"Audio not found: {audio}"})
            continue

        store.reload_names()  # pick up renames from the app
        exclusive = bool(job.get("exclusive", False))
        started = time.time()
        try:
            if cmd == "chunk":
                turns = local_turns(audio, exclusive=exclusive)
                mapping = resolve_speakers(audio, turns)
                store.save()
                segments = labeled_segments(turns, mapping)
                log(f"chunk done in {time.time() - started:.1f}s — "
                    f"{len(segments)} turns, {len({s['id'] for s in segments})} voices")
                emit({"type": "chunk_result",
                      "window_start": job.get("window_start", 0.0),
                      "segments": segments})
            else:  # final
                turns = local_turns(audio, int(job.get("num_speakers", 0)), exclusive=exclusive)
                mapping = resolve_speakers(audio, turns)
                store.save()
                segments = labeled_segments(turns, mapping)
                log(f"final done in {time.time() - started:.1f}s — "
                    f"{len(segments)} turns, {len({s['id'] for s in segments})} speakers")
                emit({"type": "result", "segments": segments})
        except Exception:  # noqa: BLE001 — job error: report, keep serving
            emit({"type": "error", "text": f"Diarization failed: {brief_traceback()}"})


if __name__ == "__main__":
    main()
