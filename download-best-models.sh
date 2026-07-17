#!/usr/bin/env bash
# =============================================================
# Meeting Transcribe — one-time model downloader + runtime setup
# Downloads ALL models and creates a self-contained Python env
# (.venv) inside the project so runtime is 100% offline.
#
# A Hugging Face token is required (pyannote is a gated repo). The token is
# YOUR credential — it is never stored in this file. Provide it either way:
#   echo 'HF_TOKEN=hf_xxx' > .env     # one-time; .env is gitignored
#   export HF_TOKEN=hf_xxx            # or just this session's environment
# With neither, the script asks for it and offers to save it to .env.
#
# Usage:
#   ./download-best-models.sh
#
# Total download: ~16-18 GB (models + Python packages; includes DiCoW ~6 GB)
# =============================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME="$SCRIPT_DIR/models"
mkdir -p "$HF_HOME"
FAILED=()

# Everything this script prints is also saved to logs/setup-run.log,
# so Claude can read the full output directly from the project folder.
# NOTE: the token is never echoed — it must not land in that log.
mkdir -p "$SCRIPT_DIR/logs"
exec > >(tee "$SCRIPT_DIR/logs/setup-run.log") 2>&1
echo "==> Run started: $(date)"

echo "==> Project:  $SCRIPT_DIR"
echo "==> Models:   $HF_HOME"

# -------------------------------------------------------------
# HF token — env wins, then .env, then ask. Never hardcoded, never logged.
# -------------------------------------------------------------
if [ -z "${HF_TOKEN:-}" ] && [ -f "$SCRIPT_DIR/.env" ]; then
  set -a; . "$SCRIPT_DIR/.env"; set +a
  [ -n "${HF_TOKEN:-}" ] && echo "==> HF token:  loaded from .env"
fi

if [ -z "${HF_TOKEN:-}" ]; then
  if [ -t 0 ]; then
    echo ""
    echo "==> A Hugging Face token is needed (pyannote is a gated repo)."
    echo "    1. Create one: https://huggingface.co/settings/tokens  (read access is enough)"
    echo "    2. Accept the terms: https://huggingface.co/pyannote/speaker-diarization-community-1"
    printf "    Paste your token (input hidden): "
    read -rs HF_TOKEN < /dev/tty
    printf "\n"
    if [ -n "${HF_TOKEN:-}" ]; then
      printf "    Save it to .env so you are not asked again? [y/N] "
      read -r save_ans < /dev/tty
      case "$save_ans" in
        [yY]*)
          umask 077
          printf 'HF_TOKEN=%s\n' "$HF_TOKEN" > "$SCRIPT_DIR/.env"
          echo "    Saved to .env (gitignored, chmod 600)."
          ;;
      esac
    fi
  fi
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo ""
  echo "!! No HF_TOKEN. The gated pyannote download will fail."
  echo "   Set it and re-run:  echo 'HF_TOKEN=hf_xxx' > .env"
  exit 1
fi
export HF_TOKEN

# Retry helper for flaky networks (DNS hiccups, etc.)
retry() { # retry <n> <cmd...>
  local n=$1; shift
  local i
  for i in $(seq 1 "$n"); do
    "$@" && return 0
    echo "   ...attempt $i/$n failed, retrying in 3s"
    sleep 3
  done
  return 1
}

# -------------------------------------------------------------
# 0. PYTHON — need 3.10+ (Xcode's python3 is 3.9: too old for MLX stack)
#    Fully automatic: if no suitable Python exists, install 3.12
#    (via Homebrew if present, else via uv — no sudo needed).
#    Everything installs into a project-local venv: .venv/
#    The app's sidecars use .venv/bin/python3 automatically.
# -------------------------------------------------------------
find_python() {
  # Prefer 3.12, accept any 3.10+
  for candidate in python3.12 /opt/homebrew/bin/python3.12 /usr/local/bin/python3.12 \
                   python3.13 python3.11 python3.10 \
                   /opt/homebrew/bin/python3 /usr/local/bin/python3 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
        command -v "$candidate"
        return 0
      fi
    fi
  done
  return 1
}

UV_BIN=""
find_uv() {
  if command -v uv >/dev/null 2>&1; then UV_BIN="$(command -v uv)"; return 0; fi
  if [ -x "$HOME/.local/bin/uv" ]; then UV_BIN="$HOME/.local/bin/uv"; return 0; fi
  return 1
}

install_python312() {
  echo ""
  echo "==> No Python 3.10+ found — installing Python 3.12 automatically..."

  # Route A: Homebrew (if the client has it)
  if command -v brew >/dev/null 2>&1; then
    echo "==> Installing via Homebrew: python@3.12"
    if retry 2 brew install python@3.12; then
      return 0
    fi
    echo "   Homebrew install failed — trying uv instead."
  fi

  # Route B: uv — tiny standalone tool, installs Python without admin rights
  if ! find_uv; then
    echo "==> Installing uv (user-local, no sudo)..."
    retry 3 sh -c 'curl -LsSf https://astral.sh/uv/install.sh | sh' \
      || { echo "!! Could not install uv"; return 1; }
    find_uv || { echo "!! uv installed but not found"; return 1; }
  fi
  echo "==> Installing Python 3.12 via uv..."
  retry 3 "$UV_BIN" python install 3.12 || { echo "!! uv python install failed"; return 1; }
  return 0
}

PYBIN="$(find_python)" || true
if [ -z "${PYBIN:-}" ]; then
  install_python312 || {
    echo ""
    echo "!! Automatic Python install failed. Install manually, then re-run:"
    echo "     brew install python@3.12"
    exit 1
  }
  PYBIN="$(find_python)" || true
fi

VENV="$SCRIPT_DIR/.venv"
# Recreate venv if it was built with an old Python
if [ -x "$VENV/bin/python3" ] && ! "$VENV/bin/python3" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
  echo "==> Existing .venv uses old Python — recreating..."
  rm -rf "$VENV"
fi

if [ ! -x "$VENV/bin/python3" ]; then
  echo "==> Creating project venv: $VENV"
  if [ -n "${PYBIN:-}" ]; then
    "$PYBIN" -m venv "$VENV" || { echo "!! venv creation failed"; exit 1; }
  elif find_uv; then
    # uv-managed Python (no system python3.12 on PATH)
    "$UV_BIN" venv --seed --python 3.12 "$VENV" || { echo "!! uv venv creation failed"; exit 1; }
  else
    echo "!! No usable Python found"; exit 1
  fi
fi

PY="$VENV/bin/python3"
PIP="$VENV/bin/pip"
echo "==> Python:   $("$PY" -c 'import sys, platform; print(sys.executable, platform.python_version())')"
"$PIP" install --quiet --upgrade pip wheel || true

pipi() { # pipi <package-spec...>
  "$PIP" install --upgrade "$@"
}

echo "==> Installing huggingface_hub CLI into venv..."
# Pin <1.0: main venv's transformers needs huggingface_hub<1.0; unpinned
# --upgrade pulls 1.x and breaks mlx-audio/transformers.
pipi "huggingface_hub[cli]<1.0" || { echo "!! could not install huggingface_hub"; exit 1; }
HF_CLI="$VENV/bin/hf"
[ -x "$HF_CLI" ] || HF_CLI="$VENV/bin/huggingface-cli"

dl() { # dl <repo_id> <label>
  echo ""
  echo "=============================================="
  echo "==> Downloading: $2"
  echo "    Repo: $1"
  echo "=============================================="
  retry 3 "$HF_CLI" download "$1" || { echo "!! FAILED: $1"; FAILED+=("$1"); }
}

# -------------------------------------------------------------
# 1. REALTIME ASR — Nemotron 3.5 Streaming 0.6B (MLX, bf16)
# -------------------------------------------------------------
dl "mlx-community/nemotron-3.5-asr-streaming-0.6b" "Nemotron 3.5 ASR Streaming 0.6B (MLX)"

# Runtime: mlx-audio with Nemotron STT support.
# Nemotron support was MERGED upstream (PR #771/#774/#775) — the old fork
# branch is gone. Latest PyPI release may or may not include it yet, so:
# 1) upgrade to latest release, 2) check the nemotron module specifically,
# 3) if still missing, install straight from upstream main.
echo ""
echo "==> Installing mlx-audio (Nemotron ASR runtime)..."
nemotron_support() {
  "$PY" -c "import mlx_audio.stt.models.nemotron_asr" 2>/dev/null
}
nemotron_files_present() {
  "$PY" -c "import mlx_audio, pathlib, sys; sys.exit(0 if (pathlib.Path(mlx_audio.__file__).parent / 'stt/models/nemotron_asr').exists() else 1)" 2>/dev/null
}
pipi "mlx-audio" || true
# --- mlx-lm / transformers compatibility fix -----------------------------
# NOTE: must run BEFORE the nemotron import check — this bug masquerades
# as missing nemotron support (the models package imports mlx_lm).
# mlx-audio's model package imports mlx_lm; mlx_lm registers a tokenizer in
# a way transformers >= 4.57 rejects ("'str' object has no attribute
# '__module__'"). Upgrade mlx-lm first; if still broken, pin transformers.
echo ""
echo "==> Checking mlx-lm / transformers compatibility..."
mlxlm_ok() {
  "$PY" -c "import mlx_lm" 2>/dev/null
}
if ! mlxlm_ok; then
  echo "==> mlx_lm import broken — upgrading mlx-lm..."
  pipi "mlx-lm" || true
fi
if ! mlxlm_ok; then
  echo "==> Still broken — pinning transformers below 4.57..."
  pipi "transformers<4.57" || true
fi
if mlxlm_ok; then
  echo "==> mlx-lm import: OK"
else
  echo "!! mlx-lm still fails to import."
  FAILED+=("mlx-lm/transformers compatibility — send the verify traceback below")
fi

# Only reinstall from git if the nemotron files are actually absent
if ! nemotron_files_present; then
  echo "==> Installed mlx-audio lacks nemotron_asr files — installing from upstream main..."
  retry 3 "$PIP" install --upgrade --force-reinstall --no-deps \
    "git+https://github.com/Blaizzy/mlx-audio.git" || true
fi

if nemotron_support; then
  echo "==> mlx-audio nemotron_asr module: OK"
else
  echo "!! mlx-audio Nemotron import still failing."
  FAILED+=("mlx-audio nemotron_asr — send the verify traceback below")
fi

# -------------------------------------------------------------
# 2. CHUNKED ASR (selectable — all 3 downloaded for A/B testing)
# -------------------------------------------------------------
dl "mlx-community/Qwen3-ASR-1.7B-bf16" "Qwen3-ASR 1.7B (MLX bf16)"
dl "mlx-community/whisper-large-v3-mlx" "Whisper large-v3 (MLX fp16)"

# Whisper runtime: mlx-whisper (Apple's official MLX Whisper — built for the
# mlx-community repo above, tokenizer bundled, no extra files needed).
echo ""
echo "==> Installing mlx-whisper (Whisper MLX runtime)..."
pipi "mlx-whisper" || FAILED+=("mlx-whisper install")
# Voxtral: Transcribe 2 batch model is API-only; open weights = 4B Realtime 2602
dl "mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16" "Voxtral Mini 4B Realtime 2602 (MLX fp16)"

# -------------------------------------------------------------
# 3. DIARIZATION — pyannote community-1 (PyTorch MPS)
#    GATED repo: accept terms + HF_TOKEN required.
# -------------------------------------------------------------
if [ -z "${HF_TOKEN:-}" ]; then
  echo ""
  echo "!! WARNING: HF_TOKEN not set — skipping pyannote (gated repo)."
  echo "   1. Create token: https://huggingface.co/settings/tokens"
  echo "   2. Accept terms: https://huggingface.co/pyannote/speaker-diarization-community-1"
  echo "   3. export HF_TOKEN=hf_xxx && re-run this script"
  FAILED+=("pyannote/speaker-diarization-community-1 (no HF_TOKEN)")
else
  dl "pyannote/speaker-diarization-community-1" "pyannote community-1 (diarization)"
fi

echo ""
echo "==> Installing pyannote.audio (diarization runtime)..."
pipi "pyannote.audio" || FAILED+=("pyannote.audio install")
# pyannote pulls its own deps — re-pin numpy afterwards (numba needs <2.5)
pipi "numpy<2.5" || true

# Speaker embedding model for profile matching (WeSpeaker, ~26MB, public)
dl "pyannote/wespeaker-voxceleb-resnet34-LM" "WeSpeaker embeddings (speaker profiles)"

# -------------------------------------------------------------
# 4. VAD — Silero VAD v6.2.1 (weights ship inside the pip package)
# -------------------------------------------------------------
echo ""
echo "==> Installing Silero VAD v6.2.1 into venv..."
pipi "silero-vad==6.2.1" || FAILED+=("silero-vad")

# Harmonize numpy AFTER all installs: torch/silero pull numpy 2.5+,
# but numba (mlx-whisper dependency) requires numpy < 2.5.
echo "==> Pinning numpy < 2.5 (numba/mlx-whisper compatibility)..."
pipi "numpy<2.5" || FAILED+=("numpy pin")

# Copy VAD weights into models/ so the project archive is complete
echo "==> Copying Silero VAD weights into project models/ ..."
"$PY" - <<'EOF' || true
import shutil, pathlib, os
try:
    import silero_vad
except ImportError:
    raise SystemExit("silero_vad not installed; skipping copy")
src = pathlib.Path(silero_vad.__file__).parent / "data"
dst = pathlib.Path(os.environ["HF_HOME"]) / "silero-vad-v6.2.1"
dst.mkdir(parents=True, exist_ok=True)
copied = 0
for f in src.rglob("*"):
    if not f.is_file() or "__pycache__" in f.parts:
        continue
    target = dst / f.relative_to(src)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(f, target)
    copied += 1
print(f"Copied {copied} files -> {dst}")
EOF

# -------------------------------------------------------------
# 4b. OVERLAP REPAIR — MossFormer2 (librimix-2spk) standalone
#     Overlap attempt #3 (2026-07-14). Uses the STANDALONE
#     alibabasglab/MossFormer2 GitHub repo's vendored PyTorch code
#     (scripts/vendor/mossformer2/) with the mossformer2-librimix-2spk
#     8 kHz / 2-speaker checkpoint. This is NOT the earlier-removed
#     clearvoice / MossFormer2_SS_16K attempt — different repo + weights.
#     Off by default in the app (Settings → Models → Overlap).
# -------------------------------------------------------------
echo ""
echo "==> Installing MossFormer2 runtime deps (pure PyTorch, no transformers)..."
pipi "rotary-embedding-torch==0.9.1" || FAILED+=("rotary-embedding-torch")
pipi "librosa==0.11.0" || FAILED+=("librosa")
pipi "soundfile==0.14.0" || FAILED+=("soundfile")

echo ""
echo "==> Downloading MossFormer2 librimix-2spk checkpoint (flat layout)..."
# Non-standard layout: the vendored loader reads config.json + the 3 .ckpt
# files directly from models/mossformer2/mossformer2-librimix-2spk/ (NOT the
# HF-hub cache), so download each file into that folder with local_dir.
HF_HUB_OFFLINE=0 "$PY" - <<'EOF' || FAILED+=("MossFormer2 librimix-2spk download")
import os, sys, traceback
try:
    from huggingface_hub import hf_hub_download
    repo = "alibabasglab/mossformer2-librimix-2spk"
    dst = os.path.join(os.environ["HF_HOME"], "mossformer2", "mossformer2-librimix-2spk")
    os.makedirs(dst, exist_ok=True)
    for fname in ("config.json", "encoder.ckpt", "decoder.ckpt", "masknet.ckpt"):
        hf_hub_download(repo_id=repo, filename=fname, local_dir=dst)
        print(f"   OK: {fname}")
except Exception:
    traceback.print_exc(file=sys.stdout)
    sys.exit(1)
EOF

# -------------------------------------------------------------
# 4c. OVERLAP REPAIR — DiCoW v3.3 Large (target-speaker ASR)
#     Second overlap engine, selectable alongside MossFormer2.
#     BUT-FIT/DiCoW_v3_3_large: diarization-conditioned Whisper —
#     given a speaker's diarization mask it transcribes only that
#     speaker, so overlaps are handled by ASR rather than by
#     separating the waveform. ~6 GB.
# -------------------------------------------------------------
echo ""
echo "==> Downloading DiCoW v3.3 Large weights (~6 GB, first run takes a while)..."
# The HF API 504s intermittently from here, so retry with backoff rather than
# giving up on the first gateway timeout. Partial downloads resume.
HF_HUB_OFFLINE=0 HF_HUB_ETAG_TIMEOUT=60 "$PY" - <<'EOF' || FAILED+=("DiCoW v3.3 Large download")
import sys, time, traceback
from huggingface_hub import snapshot_download

ATTEMPTS = 8
for attempt in range(1, ATTEMPTS + 1):
    try:
        # Whole snapshot: weights + the repo's custom modeling code, which
        # DiCoW needs (trust_remote_code) and which must exist offline.
        path = snapshot_download(repo_id="BUT-FIT/DiCoW_v3_3_large",
                                 max_workers=4)
        print(f"   OK: DiCoW v3.3 Large -> {path}")
        sys.exit(0)
    except Exception as exc:
        print(f"   attempt {attempt}/{ATTEMPTS} failed: {type(exc).__name__}: {exc}")
        if attempt == ATTEMPTS:
            traceback.print_exc(file=sys.stdout)
            sys.exit(1)
        time.sleep(min(5 * attempt, 30))
EOF

# -------------------------------------------------------------
# 4d. DiCoW RUNTIME — its own venv (.venv-dicow)
#     DiCoW's remote code only works on transformers 4.x: on the main
#     .venv's transformers 5.x, generate() dies with
#     AttributeError: '_get_initial_cache_position'. That is a hard
#     conflict with the MLX stack, so DiCoW gets a separate venv —
#     the only sidecar that does NOT run in .venv. `pandas` is an
#     undocumented import of the remote code (ImportError on load
#     without it); numpy is pinned <2.5 to match torch's ABI.
#     Dev-mode only: package-app.sh does not ship this venv yet.
# -------------------------------------------------------------
echo ""
echo "==> Setting up the DiCoW runtime venv (.venv-dicow)..."
DICOW_VENV="$SCRIPT_DIR/.venv-dicow"
DICOW_PY="$DICOW_VENV/bin/python3"

# Idempotent: skip the whole section when the venv already satisfies the pins.
dicow_venv_ok() {
  [ -x "$DICOW_PY" ] || return 1
  "$DICOW_PY" - <<'EOF' >/dev/null 2>&1
import sys
import transformers, torch, pandas, librosa, soundfile, numpy  # noqa: F401
assert transformers.__version__.startswith("4.55."), transformers.__version__
sys.exit(0)
EOF
}

if dicow_venv_ok; then
  echo "   OK: .venv-dicow already satisfies transformers 4.55 + deps — skipping"
else
  if [ ! -x "$DICOW_PY" ]; then
    echo "   creating $DICOW_VENV"
    if [ -n "${PYBIN:-}" ]; then
      "$PYBIN" -m venv "$DICOW_VENV" || FAILED+=(".venv-dicow creation")
    elif find_uv; then
      "$UV_BIN" venv --seed --python 3.12 "$DICOW_VENV" || FAILED+=(".venv-dicow creation")
    else
      echo "!! No usable Python found for .venv-dicow"
      FAILED+=(".venv-dicow creation")
    fi
  fi
  if [ -x "$DICOW_PY" ]; then
    "$DICOW_VENV/bin/pip" install --quiet --upgrade pip wheel || true
    # transformers is PINNED — do not relax it (see the header comment).
    "$DICOW_VENV/bin/pip" install --upgrade \
      "transformers==4.55.0" torch soundfile librosa pandas "numpy<2.5" \
      || FAILED+=("DiCoW venv deps (.venv-dicow)")
  fi
fi

# -------------------------------------------------------------
# 5. VERIFY — everything the app needs, checked here and now
# -------------------------------------------------------------
echo ""
echo "==> Verifying runtime imports..."
"$PY" - <<'EOF'
import importlib, sys, traceback
try:
    from importlib.metadata import version
    import mlx_audio, pathlib
    pkg = pathlib.Path(mlx_audio.__file__).parent
    has_nemotron_files = (pkg / "stt/models/nemotron_asr").exists()
    print(f"   mlx-audio: v{version('mlx-audio')} at {pkg}")
    print(f"   nemotron_asr files on disk: {'yes' if has_nemotron_files else 'NO — install did not include it'}")
except Exception:
    traceback.print_exc(file=sys.stdout)
checks = {
    "silero_vad": "Silero VAD",
    "torch": "PyTorch (Silero dependency)",
    "numpy": "NumPy",
    "mlx_audio": "mlx-audio (Nemotron runtime)",
    "mlx_audio.stt.models.nemotron_asr": "mlx-audio Nemotron model support",
    "mlx_whisper": "mlx-whisper (Whisper runtime)",
    "pyannote.audio": "pyannote.audio (diarization runtime)",
}
failed = []
for module, label in checks.items():
    try:
        importlib.import_module(module)
        print(f"   OK: {label}")
    except Exception:
        print(f"   MISSING: {label} — full traceback:")
        traceback.print_exc(file=sys.stdout)
        failed.append(label)
sys.exit(1 if failed else 0)
EOF
[ $? -eq 0 ] || FAILED+=("runtime import check — see list above")

# Real end-to-end check: load Nemotron exactly like the app's sidecar does.
# If this passes, the app's realtime ASR will start.
echo ""
echo "==> Verifying Nemotron model actually loads (takes ~10-30s first time)..."
HF_HUB_OFFLINE=1 "$PY" - <<'EOF'
import sys, traceback
try:
    from mlx_audio.stt import load
    model = load("mlx-community/nemotron-3.5-asr-streaming-0.6b")
    print("   OK: Nemotron 3.5 loaded successfully")
except Exception:
    print("   FAILED: Nemotron load — full traceback:")
    traceback.print_exc(file=sys.stdout)
    sys.exit(1)
EOF
[ $? -eq 0 ] || FAILED+=("Nemotron model load test — the app will not transcribe until this passes")

# End-to-end test of every chunked model exactly like the app's sidecar:
# load + transcribe 0.5s of silence. If these pass, the app works.
echo ""
echo "==> Verifying chunked models end-to-end (Whisper via mlx-whisper, Qwen3 via mlx-audio)..."
HF_HUB_OFFLINE=1 "$PY" - <<'EOF'
import numpy as np, tempfile, wave, os, sys, traceback

def silence_wav():
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(np.zeros(8000, dtype=np.int16).tobytes())
    return path

failed = False

# 1. Whisper large-v3 (MLX) via mlx-whisper
try:
    import mlx_whisper
    path = silence_wav()
    mlx_whisper.transcribe(path, path_or_hf_repo="mlx-community/whisper-large-v3-mlx")
    os.unlink(path)
    print("   OK: Whisper large-v3 (mlx-whisper) loads and transcribes")
except Exception:
    print("   FAILED: Whisper (mlx-whisper) — full traceback:")
    traceback.print_exc(file=sys.stdout)
    failed = True

# 2. Qwen3-ASR 1.7B via mlx-audio
try:
    from mlx_audio.stt import load
    model = load("mlx-community/Qwen3-ASR-1.7B-bf16")
    path = silence_wav()
    model.generate(path)
    os.unlink(path)
    print("   OK: Qwen3-ASR 1.7B (mlx-audio) loads and transcribes")
except Exception:
    print("   FAILED: Qwen3-ASR — full traceback:")
    traceback.print_exc(file=sys.stdout)
    failed = True

sys.exit(1 if failed else 0)
EOF
[ $? -eq 0 ] || FAILED+=("chunked model end-to-end test — see tracebacks above")

echo ""
echo "==> Verifying pyannote diarization pipeline loads (MPS)..."
HF_HUB_OFFLINE=1 "$PY" - <<'EOF'
import sys, traceback
try:
    import torch
    from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1")
    device = "cpu"
    if torch.backends.mps.is_available():
        pipeline.to(torch.device("mps"))
        device = "mps"
    print(f"   OK: pyannote community-1 pipeline loaded on {device}")
    from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
    PretrainedSpeakerEmbedding("pyannote/wespeaker-voxceleb-resnet34-LM",
                               device=torch.device(device))
    print("   OK: WeSpeaker embedding model loaded (speaker profiles)")
except Exception:
    print("   FAILED: pyannote pipeline — full traceback:")
    traceback.print_exc(file=sys.stdout)
    sys.exit(1)
EOF
[ $? -eq 0 ] || FAILED+=("pyannote pipeline load test")

# DiCoW lives in .venv-dicow, so it must be verified with THAT interpreter.
# Import check + offline snapshot resolve only — a full 6 GB model load would
# add minutes here; scripts/dicow-service.py does the real load at session start.
echo ""
echo "==> Verifying the DiCoW runtime (.venv-dicow)..."
if [ -x "$DICOW_PY" ]; then
  HF_HUB_OFFLINE=1 "$DICOW_PY" - <<'EOF'
import sys, traceback
try:
    import transformers, torch, pandas, librosa, soundfile, numpy  # noqa: F401
    print(f"   OK: transformers {transformers.__version__}, torch {torch.__version__}, "
          f"pandas {pandas.__version__}, numpy {numpy.__version__}")
    if not transformers.__version__.startswith("4.55."):
        print("   FAILED: DiCoW needs transformers 4.55.x — generate() breaks on 5.x")
        sys.exit(1)
    # Offline resolve: proves the snapshot (weights + remote code) is on disk and
    # reachable with HF_HUB_OFFLINE=1, exactly as the sidecar will read it.
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id="BUT-FIT/DiCoW_v3_3_large", local_files_only=True)
    print(f"   OK: DiCoW snapshot resolves offline -> {path}")
except Exception:
    print("   FAILED: DiCoW runtime — full traceback:")
    traceback.print_exc(file=sys.stdout)
    sys.exit(1)
EOF
  [ $? -eq 0 ] || FAILED+=("DiCoW runtime check (.venv-dicow)")
else
  echo "   MISSING: $DICOW_PY"
  FAILED+=("DiCoW runtime check (.venv-dicow missing)")
fi

# -------------------------------------------------------------
# Summary
# -------------------------------------------------------------
echo ""
echo "=============================================="
echo "Disk usage:"
du -sh "$HF_HOME" "$VENV" "$DICOW_VENV" 2>/dev/null || true
if [ ${#FAILED[@]} -gt 0 ]; then
  echo ""
  echo "FAILED steps (fix and re-run — completed downloads are skipped automatically):"
  printf '  - %s\n' "${FAILED[@]}"
  exit 1
fi
echo ""
echo "ALL OK. Runtime is self-contained in:"
echo "  $VENV      (Python + packages — used by the app automatically)"
echo "  $HF_HOME   (model weights)"
echo "=============================================="
