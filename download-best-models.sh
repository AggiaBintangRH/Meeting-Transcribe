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
  # --progress-bar off is not cosmetic: this whole script's stdout is piped to
  # tee (see the exec redirect above), and pip's animated download/progress
  # bars flood that pipe. On some Macs the pipe is non-blocking and the flood
  # overruns its buffer, so pip aborts with "OSError: [Errno 35] write could
  # not complete without blocking" mid-install. Turning the bars off keeps the
  # output to a few lines per package and avoids it entirely.
  "$PIP" install --progress-bar off --upgrade "$@"
}

echo "==> Installing huggingface_hub CLI into venv..."
# THE BOUND IS >=1.5,<2.0, AND IT IS READ OFF THE PACKAGES THAT DECLARE IT —
# `transformers` asks for `huggingface-hub<2.0,>=1.5.0` and `mlx-audio` for
# `>=1.0`, so this is their intersection, not a number chosen here.
#
# It used to be `huggingface_hub[cli]<1.0`, whose comment claimed "main venv's
# transformers needs huggingface_hub<1.0". That was TRUE WHEN WRITTEN and became
# false when transformers moved to 5.x. A pin outliving its reason is the shape
# this project keeps finding, and here it had inverted into the very thing it
# claimed to prevent: measured on the 2026-08-10 run, it DOWNGRADED 1.23.0 ->
# 0.36.2 and printed
#     ERROR: ... transformers 5.12.1 requires huggingface-hub>=1.5.0, but you
#     have huggingface-hub 0.36.2 which is incompatible.
# The very next install then dragged it back up to 1.27.0. The run still ended
# "ALL OK" because pip's resolver ERROR is not a non-zero exit — which is why
# this was invisible until the log was read line by line.
#
# `[cli]` IS DROPPED, and that is required rather than tidiness: huggingface_hub
# 1.x has NO `cli` extra (its Provides-Extra list is oauth/torch/fastai/hf-xet/
# mcp/testing/gradio/typing/quality/all/dev) because the `hf` command now ships
# in the base package. Keeping `[cli]` here would ask for an extra that does not
# exist and warn on every run. Verified: `.venv/bin/hf --version` -> 1.27.0.
pipi "huggingface_hub>=1.5,<2.0" || { echo "!! could not install huggingface_hub"; exit 1; }
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
  retry 3 "$PIP" install --progress-bar off --upgrade --force-reinstall --no-deps \
    "git+https://github.com/Blaizzy/mlx-audio.git" || true
fi

if nemotron_support; then
  echo "==> mlx-audio nemotron_asr module: OK"
else
  echo "!! mlx-audio Nemotron import still failing."
  FAILED+=("mlx-audio nemotron_asr — send the verify traceback below")
fi

# -------------------------------------------------------------
# 2. CHUNKED ASR (selectable — all 4 downloaded for A/B testing)
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
# Granite Speech 4.1 2B NAR: EN/FR/DE/ES/PT/JA ONLY — no Indonesian (English A/B option).
# Six, not five: the card lists Japanese too (rechecked 2026-07-29).
dl "mlx-community/granite-speech-4.1-2b-nar-mlx" "Granite Speech 4.1 2B NAR (MLX bf16)"

# -------------------------------------------------------------
# 2b. WORD ALIGNER — Qwen3-ForcedAligner 0.6B (MLX bf16)
#     Forced alignment: audio + KNOWN text -> per-word timestamps. It never
#     sees the ASR, so it works with ALL four chunked models above (the repo
#     name says Qwen3 only because it shares that tokenizer/encoder lineage).
#     Used for word-exact speaker attribution when the ATND beam switches
#     mid-sentence. mlx_audio.stt.load() dispatches on the config's
#     model_type=qwen3_forced_aligner — no extra pip package needed.
# -------------------------------------------------------------
dl "mlx-community/Qwen3-ForcedAligner-0.6B-bf16" "Qwen3-ForcedAligner 0.6B (MLX bf16, word timestamps)"

# -------------------------------------------------------------
# 2c. SPEAKER-ATTRIBUTED ASR — MOSS-Transcribe-Diarize 0.9B (PyTorch MPS)
#     One model returns the words, the SPEAKER LABELS and the timestamps
#     together. Selectable in two independent roles: as a chunked ASR model
#     (Settings -> Models -> Chunked) and as the diarization engine
#     (Settings -> Models -> Diarization, diarization.engine=moss). With both
#     set to MOSS it is ONE process doing one forward pass.
#     Public repo — no token gate, unlike pyannote below.
#     Runs in the MAIN .venv: it needs Python >= 3.12 and transformers 5.x,
#     which is exactly what the MLX stack already pins, so it needs no
#     interpreter of its own (unlike DiCoW, which is stuck on 4.55).
# -------------------------------------------------------------
dl "OpenMOSS-Team/MOSS-Transcribe-Diarize" "MOSS-Transcribe-Diarize 0.9B (speaker-attributed ASR)"

# The upstream helper package (prompt construction, generate wrapper and the
# [start][Sxx]text[end] transcript parser). Installable only from git.
#
# IMPORTANT — this install is PROVENANCE, not the runtime import path.
# Each MOSS service imports the VENDORED copy under its OWN
# scripts/<service>/vendor/moss_transcribe_diarize instead — two services since
# 2026-07-31: scripts/moss-asr/ (chunked-ASR role) and scripts/moss-diar/
# (diarization role) — because build.sh strips every
# VCS requirement out of the frozen list before provisioning the bundled
# interpreter:
#     build.sh:110   grep -vE '^pip==|file://|@ file://|git\+'
# so a `git+` freeze line silently disappears and the packaged .app would ship
# without the package — the DiCoW-missing-from-the-bundle failure of 2026-07-27.
# scripts/ (each service's vendor/ included) is copied into the bundle by build.sh [B3], so the
# vendored copy makes dev and the .app run the same code with no build change.
# Keep this install anyway: it is how the vendored files are refreshed, and it
# records upstream's version.
echo ""
echo "==> Installing moss_transcribe_diarize (MOSS inference helper)..."
moss_helper_present() {
  "$PY" -c "import moss_transcribe_diarize" 2>/dev/null
}
if moss_helper_present; then
  echo "    already installed — skipping"
else
  # --no-deps deliberately: every dependency it declares (torch, transformers,
  # librosa, numpy) is already in this venv at the versions the MLX stack needs,
  # and letting pip resolve them again risks moving transformers off 5.x.
  "$PIP" install --progress-bar off --no-deps \
    "git+https://github.com/OpenMOSS/MOSS-Transcribe-Diarize" \
    || { echo "!! could not install moss_transcribe_diarize"; FAILED+=("moss_transcribe_diarize"); }
fi
# The vendored copies are what actually run — fail loudly here rather than at the
# first meeting if one is missing. BOTH are checked: the two MOSS roles can run as
# two processes at once, so a tree missing from either one is a broken role.
# This ONE list is the only place the set of MOSS services is named here.
for moss_svc in moss-asr moss-diar; do
  if "$PY" -c "import sys; sys.path.insert(0, 'scripts/$moss_svc/vendor'); import moss_transcribe_diarize" 2>/dev/null; then
    echo "    vendored copy OK (scripts/$moss_svc/vendor/moss_transcribe_diarize)"
  else
    echo "!! scripts/$moss_svc/vendor/moss_transcribe_diarize is missing or broken — MOSS will not run"
    FAILED+=("scripts/$moss_svc/vendor/moss_transcribe_diarize")
  fi
done

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

# Speaker embedding model for profile matching (WeSpeaker, ~26MB, public).
# Its OWN sidecar since 2026-07-30 (scripts/wespeaker/wespeaker-service.py) — the
# weights and this line are unchanged, only the process that loads them moved out
# of the pyannote pipeline. Both run in the main .venv on pyannote.audio installed
# just above: the identity service imports PretrainedSpeakerEmbedding only and
# never instantiates the Pipeline.
dl "pyannote/wespeaker-voxceleb-resnet34-LM" "WeSpeaker embeddings (speaker profiles)"

# -------------------------------------------------------------
# 3b. DIARIZATION ENGINE 2 — spectral (vendored, CPU-only)
#     Silero VAD -> sliding-window WeSpeaker embeddings -> GMM-BIC speaker
#     counting -> spectral clustering -> Viterbi smoothing.
#     Selectable as diarization.engine = spectral.
#
#     NO MODEL DOWNLOAD OF ITS OWN, deliberately: it reuses the WeSpeaker
#     weights fetched directly above and the Silero VAD installed in section 4.
#     That reuse is the whole reason the vendored tree carries this project's
#     torch_embedder.py.
#
#     THE ENGINE IS NOT PIP-INSTALLED, and must not be. Upstream
#     (github.com/FoxNoseTech/diarize @ 4f25d27, Apache 2.0) embeds through
#     `wespeakerruntime`, which downloads its OWN ONNX copy of the same weights
#     from a hardcoded URL at runtime — a network call the 100%-offline
#     requirement forbids. scripts/spectral/vendor/diarize/ is that tree with
#     embeddings.py re-pointed at the PyTorch model (measured cosine 1.0000 vs
#     the ONNX vector). It rides scripts/ into the .app via build.sh [B3], the
#     same route as the MOSS helper, and build.sh's vendored-helper gate refuses
#     to build without it.
# -------------------------------------------------------------
echo ""
echo "==> Installing the spectral engine's one extra dependency (pydantic)..."
# The vendored tree's utils.py is built on pydantic (five validated models).
# Everything else it imports — numpy, scipy, soundfile, scikit-learn, torch,
# pyannote.audio, silero-vad — is already installed by the sections around this
# one. An ordinary PyPI package, so unlike the MOSS helper it survives build.sh's
# `git+` strip and really does reach the bundled interpreter.
pipi "pydantic" || FAILED+=("pydantic (spectral engine)")

# The vendored copy is what runs — fail loudly here rather than at the first
# meeting if it is missing or broken.
if "$PY" -c "import sys; sys.path.insert(0, 'scripts/spectral/vendor'); from diarize import diarize, DiarizeResult" 2>/dev/null; then
  echo "    vendored copy OK (scripts/spectral/vendor/diarize)"
else
  echo "!! scripts/spectral/vendor/diarize is missing or broken — the spectral engine will not run"
  FAILED+=("scripts/spectral/vendor/diarize")
fi

# -------------------------------------------------------------
# 3c. DIARIZATION ENGINE 3 — NeMo (its own venv, .venv-nemo)
#     MarbleNet VAD -> multi-scale TitaNet-Large embeddings -> NME-SC spectral
#     clustering (which estimates the speaker count itself).
#     Selectable as diarization.engine = nemo.
#
#     ITS OWN INTERPRETER, the second sidecar after DiCoW to need one.
#     nemo_toolkit 3.0.0 drags in lightning, hydra and a large pinned dependency
#     tree that conflicts with the main .venv's MLX stack; a trial install into
#     .venv was reverted on 2026-08-07 for exactly that reason. Do not "simplify"
#     this into the main venv.
#
#     THE WEIGHTS ARE FETCHED HERE AND STORED AS LOCAL .nemo FILES under
#     models/nemo/, because `ClusteringDiarizer` branches on
#     `model_path.endswith('.nemo')`: a PATH is restored off disk, while a bare
#     pretrained NAME goes to from_pretrained -> maybe_download_from_cloud ->
#     https://api.ngc.nvidia.com/... at RUNTIME. On this Mac the names would work
#     (the checkpoints sit in ~/.cache/torch/NeMo), and they would reach for the
#     network on a client machine — the invisible-until-shipped shape this
#     project keeps being bitten by.
#
#     THE URL IS NOT HARDCODED HERE. NeMo's own registry
#     (`list_available_models()` -> PretrainedModelInfo.location) is the
#     authority, and `_get_ngc_pretrained_model_info` is the exact NeMo function
#     that resolves + downloads WITHOUT instantiating the model; the .nemo file
#     is then copied out of NeMo's cache. Inventing an NGC URL here would be a
#     second copy of a fact NeMo already owns.
# -------------------------------------------------------------
echo ""
echo "==> Setting up the NeMo diarization runtime venv (.venv-nemo)..."
NEMO_VENV="$SCRIPT_DIR/.venv-nemo"
NEMO_PY="$NEMO_VENV/bin/python3"

# Idempotent: skip the install when the venv already satisfies the pins.
nemo_venv_ok() {
  [ -x "$NEMO_PY" ] || return 1
  "$NEMO_PY" - <<'EOF' >/dev/null 2>&1
import sys
import torch, omegaconf, soundfile  # noqa: F401
from nemo.collections.asr.models import ClusteringDiarizer  # noqa: F401
import nemo
assert nemo.__version__.startswith("3.0."), nemo.__version__
sys.exit(0)
EOF
}

if nemo_venv_ok; then
  echo "   OK: .venv-nemo already satisfies nemo_toolkit 3.0 + deps — skipping"
else
  if [ ! -x "$NEMO_PY" ]; then
    echo "   creating $NEMO_VENV"
    if [ -n "${PYBIN:-}" ]; then
      "$PYBIN" -m venv "$NEMO_VENV" || FAILED+=(".venv-nemo creation")
    elif find_uv; then
      "$UV_BIN" venv --seed --python 3.12 "$NEMO_VENV" || FAILED+=(".venv-nemo creation")
    else
      echo "!! No usable Python found for .venv-nemo"
      FAILED+=(".venv-nemo creation")
    fi
  fi
  if [ -x "$NEMO_PY" ]; then
    "$NEMO_VENV/bin/pip" install --quiet --upgrade pip wheel || true
    # THE FILE REDIRECT IS NOT COSMETIC (the [Errno 35] lesson, build.sh B2/B2b).
    # This script's stdout is piped to tee; nemo_toolkit pulls ~200 packages and
    # pip's one enormous "Installing collected packages: …" line through a
    # non-blocking pipe aborts the install with
    # "OSError: [Errno 35] write could not complete without blocking".
    # Writing to a file removes the non-blocking fd from pip's output path
    # entirely, and --progress-bar off keeps the log small. The tail is surfaced
    # on failure so nothing is hidden.
    NEMO_PIP_LOG="$SCRIPT_DIR/logs/setup-pip-nemo.log"
    echo "   installing nemo_toolkit==3.0.0 (log: $NEMO_PIP_LOG)"
    # torchaudio and torchcodec are DELIBERATELY ABSENT from this list, and that
    # absence is load-bearing: it makes the ffmpeg/torchcodec trap that bit
    # Whisper, pyannote and spectral structurally impossible in this venv rather
    # than merely avoided. NeMo's own audio path is soundfile/librosa. Adding
    # either package here re-opens a trap `assert_no_torchcodec_use` cannot see.
    if ! "$NEMO_VENV/bin/pip" install --progress-bar off --upgrade \
          "nemo_toolkit[asr]==3.0.0" torch omegaconf soundfile \
          >"$NEMO_PIP_LOG" 2>&1; then
      echo "!! pip install FAILED — last 30 lines of $NEMO_PIP_LOG:"
      tail -30 "$NEMO_PIP_LOG"
      FAILED+=("NeMo venv deps (.venv-nemo)")
    fi
  fi
fi

echo ""
echo "==> Fetching the NeMo diarization checkpoints into models/nemo/ ..."
if [ -x "$NEMO_PY" ]; then
  NEMO_MODEL_DIR="$SCRIPT_DIR/models/nemo"
  mkdir -p "$NEMO_MODEL_DIR"
  # Run with the sidecar's own offline env, minus HF_HUB_OFFLINE — this is the
  # one moment the network is allowed, and it is a one-time download.
  WANDB_MODE=disabled WANDB_DISABLED=true SENTRY_DSN="" \
  NEMO_ONELOGGER_ENABLED=false ONE_LOGGER_ENABLED=false \
  NEMO_MODEL_DIR="$NEMO_MODEL_DIR" \
  "$NEMO_PY" - <<'EOF' || FAILED+=("NeMo diarization checkpoints")
import os
import pathlib
import shutil
import sys

from nemo.collections.asr.models import (EncDecClassificationModel,
                                         EncDecSpeakerLabelModel)

dest = pathlib.Path(os.environ["NEMO_MODEL_DIR"])
dest.mkdir(parents=True, exist_ok=True)

# (class, pretrained name) — the classes ClusteringDiarizer itself restores
# these with (`_init_speaker_model` / `_init_vad_model`).
WANTED = ((EncDecSpeakerLabelModel, "titanet_large"),
          (EncDecClassificationModel, "vad_multilingual_marblenet"))

failed = False
for cls, name in WANTED:
    out = dest / f"{name}.nemo"
    if out.exists() and out.stat().st_size > 100_000:
        print(f"   OK: {name} already at {out} — skipping")
        continue
    try:
        # NeMo's OWN resolver: looks the name up in list_available_models(),
        # downloads from the location that registry records, and returns the
        # cached path. No URL is spelled out anywhere in this script.
        _, cached = cls._get_ngc_pretrained_model_info(name)
        shutil.copy2(cached, out)
        print(f"   OK: {name} -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    except Exception as exc:
        print(f"!! could not fetch {name}: {type(exc).__name__}: {exc}")
        failed = True

sys.exit(1 if failed else 0)
EOF
else
  echo "!! .venv-nemo is missing — cannot fetch the NeMo checkpoints"
  FAILED+=("NeMo diarization checkpoints")
fi

# The vendored inference config is what the sidecar loads (byte-identical to
# NVIDIA-NeMo/Speech @ 6c57e73e; every deviation is applied in sidecar CODE so
# each carries its measured justification). It rides scripts/ into the .app via
# build.sh [B3], the same route as the MOSS helper and the spectral tree.
if [ -f "$SCRIPT_DIR/scripts/nemo/vendor/diar_infer_general.yaml" ]; then
  echo "    vendored config OK (scripts/nemo/vendor/diar_infer_general.yaml)"
else
  echo "!! scripts/nemo/vendor/diar_infer_general.yaml is missing — the NeMo engine will not run"
  FAILED+=("scripts/nemo/vendor/diar_infer_general.yaml")
fi

# -------------------------------------------------------------
# 3d. DIARIZATION ENGINE 4 — DiariZen (its own venv, .venv-diarizen)
#     WavLM-Base+ encoder + Conformer -> END-TO-END NEURAL SEGMENTATION (EEND),
#     then WeSpeaker embeddings + agglomerative clustering.
#     Selectable as diarization.engine = diarizen.
#
#     ITS OWN INTERPRETER, AND THE ONLY PYTHON 3.11 ONE IN THE PROJECT. This is
#     not a preference: upstream pins torch 2.1.1, which has no 3.12 wheels, and
#     the vendored pyannote-audio 3.1.1 imports `torchaudio.AudioMetaData`,
#     removed in torchaudio 2.9. Both halves fail loudly if this is "simplified"
#     into a newer interpreter or the main .venv.
#
#     THE TWO PACKAGES ARE NOT ON PyPI, and that is why they are vendored under
#     scripts/diarizen/vendor/ and installed FROM THAT TREE, non-editable. An
#     editable install freezes as a `git+`/`file://` ref, and build.sh strips
#     those out of the frozen requirements — the exact failure that shipped a
#     broken DiCoW to a client machine on 2026-07-27.
#
#     LICENCE, AND IT IS A HARD CLIENT REQUIREMENT: only
#     BUT-FIT/diarizen-meeting-base is MIT. The other five DiariZen checkpoints
#     (wavlm-large-s80-md, -mlc, -v2, -origin, wavlm-base-s80-md) are
#     CC BY-NC 4.0 because they add RAMC / MSDWild / DIHARD-3 to the training
#     mix. This project ships to a paying client. DO NOT swap the checkpoint for
#     a "better" one without re-reading its licence.
# -------------------------------------------------------------
echo ""
echo "==> Setting up the DiariZen diarization runtime venv (.venv-diarizen)..."
DZ_VENV="$SCRIPT_DIR/.venv-diarizen"
DZ_PY="$DZ_VENV/bin/python3"
DZ_VENDOR="$SCRIPT_DIR/scripts/diarizen/vendor"

# Idempotent: skip the install when the venv already satisfies the pins. The
# torch check is part of the CONTRACT, not a nicety — a resolver that quietly
# moved torch off 2.1.1 leaves an engine that imports and then dies at the first
# job, which is the stop pass, which under this engine IS the labels.
dz_venv_ok() {
  [ -x "$DZ_PY" ] || return 1
  "$DZ_PY" - <<'EOF' >/dev/null 2>&1
import sys
import torch, torchaudio, soundfile  # noqa: F401
from torchaudio import AudioMetaData  # noqa: F401  (gone in torchaudio 2.9)
import pyannote.audio  # noqa: F401
from diarizen.pipelines.inference import DiariZenPipeline  # noqa: F401
assert torch.__version__.startswith("2.1."), torch.__version__
sys.exit(0)
EOF
}

if [ ! -d "$DZ_VENDOR/diarizen" ] || [ ! -d "$DZ_VENDOR/pyannote-audio" ]; then
  echo "!! scripts/diarizen/vendor is incomplete — DiariZen and its pyannote fork"
  echo "   are not on PyPI, so this tree IS the engine. Restore it from git."
  FAILED+=("scripts/diarizen/vendor")
elif dz_venv_ok; then
  echo "   OK: .venv-diarizen already satisfies torch 2.1.1 + DiariZen — skipping"
else
  if [ ! -x "$DZ_PY" ]; then
    echo "   creating $DZ_VENV (Python 3.11 — required, see the note above)"
    # 3.11 SPECIFICALLY. $PYBIN is deliberately NOT reused here: it is whatever
    # interpreter the main venv was built with (3.12), and torch 2.1.1 has no
    # 3.12 wheels, so pip would fail with an unhelpful "no matching distribution".
    if find_uv; then
      "$UV_BIN" venv --seed --python 3.11 "$DZ_VENV" || FAILED+=(".venv-diarizen creation")
    elif command -v python3.11 >/dev/null 2>&1; then
      python3.11 -m venv "$DZ_VENV" || FAILED+=(".venv-diarizen creation")
    else
      echo "!! No Python 3.11 found for .venv-diarizen."
      echo "   Install uv (https://astral.sh/uv) or python3.11, then re-run."
      FAILED+=(".venv-diarizen creation")
    fi
  fi
  if [ -x "$DZ_PY" ]; then
    "$DZ_VENV/bin/pip" install --quiet --upgrade pip wheel || true
    # THE FILE REDIRECT IS NOT COSMETIC (the [Errno 35] lesson, build.sh B2/B2b
    # and 3c above). This script's stdout is piped to tee; a large pip run's one
    # enormous "Installing collected packages: …" line through a non-blocking
    # pipe aborts the install.
    DZ_PIP_LOG="$SCRIPT_DIR/logs/setup-pip-diarizen.log"
    echo "   installing DiariZen + its pinned stack (log: $DZ_PIP_LOG)"
    : >"$DZ_PIP_LOG"

    # UPSTREAM'S OWN ORDER (vendor/README.md "Installation"), with exactly two
    # deviations, both forced and both noted where they happen. Following the
    # order matters: `pip install -e .` for pyannote-audio resolves against
    # whatever torch is already present, so installing torch LAST would let it
    # move off the pin.
    dz_pip() {
      if ! "$DZ_VENV/bin/pip" install --progress-bar off "$@" >>"$DZ_PIP_LOG" 2>&1; then
        echo "!! pip install FAILED ($1 …) — last 30 lines of $DZ_PIP_LOG:"
        tail -30 "$DZ_PIP_LOG"
        FAILED+=("DiariZen venv deps (.venv-diarizen)")
        return 1
      fi
    }

    # DEVIATION 1: upstream installs the CUDA 12.1 build from
    # download.pytorch.org. There is no CUDA on a Mac; the DEFAULT PyPI index
    # gives the arm64 build whose MPS backend this engine actually runs on. The
    # VERSIONS are upstream's, unchanged, and come from vendor/constraints.txt.
    dz_pip torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 &&
    dz_pip -r "$DZ_VENDOR/requirements.txt" -c "$DZ_VENDOR/constraints.txt" &&
    # DEVIATION 2: NON-EDITABLE, where upstream uses `-e`. An editable install
    # freezes as a `file://` ref and build.sh strips those out of the frozen
    # requirements — the exact failure that shipped a broken DiCoW to a client
    # on 2026-07-27. The constraints file is kept on the pyannote install
    # exactly as upstream has it, so its resolve cannot move torch.
    dz_pip --no-deps "$DZ_VENDOR" &&
    # `[dev,testing]` IS UPSTREAM'S, and it is kept deliberately rather than
    # trimmed: this is the exact combination the working .venv-diarizen was built
    # from and verified against, and reproducing a verified environment beats
    # shipping a smaller one nobody has run. It does cost bundle size (pytest,
    # jupyterlab and tensorboard reach the .app this way) — see the audit note if
    # that is ever worth revisiting, but re-verify the engine after trimming.
    dz_pip "$DZ_VENDOR/pyannote-audio[dev,testing]" -c "$DZ_VENDOR/constraints.txt"

    dz_venv_ok || FAILED+=("DiariZen venv verification (.venv-diarizen)")
  fi
fi

echo ""
echo "==> Fetching the DiariZen checkpoint (BUT-FIT/diarizen-meeting-base)..."
# Into the SAME models/ hub cache every other engine uses, because the sidecar
# points HF_HOME there. It also REUSES the WeSpeaker checkpoint this app already
# ships (pyannote/wespeaker-voxceleb-resnet34-LM, fetched in an earlier section),
# so there is no second copy of a 26 MB embedder.
if [ -x "$DZ_PY" ]; then
  HF_HOME="$HF_HOME" "$DZ_PY" - <<'EOF' || FAILED+=("DiariZen checkpoint")
import os, sys
# The one moment the network is allowed — a one-time download. HF_HUB_OFFLINE is
# deliberately NOT set here, and is set by the sidecar at runtime.
os.environ.pop("HF_HUB_OFFLINE", None)
try:
    from huggingface_hub import snapshot_download
    path = snapshot_download("BUT-FIT/diarizen-meeting-base")
    print(f"   OK: BUT-FIT/diarizen-meeting-base -> {path}")
except Exception as exc:
    print(f"!! could not fetch BUT-FIT/diarizen-meeting-base: {type(exc).__name__}: {exc}")
    sys.exit(1)
EOF
else
  echo "!! .venv-diarizen is missing — cannot fetch the DiariZen checkpoint"
  FAILED+=("DiariZen checkpoint")
fi

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
#     (scripts/mossformer2/vendor/mossformer2/) with the mossformer2-librimix-2spk
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
    # Both are the spectral engine's, and NOTHING ELSE in the app imports
    # pydantic — so if it is ever dropped, this is the only line that notices.
    "pydantic": "pydantic (spectral engine data models)",
    "sklearn": "scikit-learn (spectral clustering + GMM speaker counting)",
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
echo "==> Verifying chunked models end-to-end (Whisper via mlx-whisper, Qwen3 + Granite via mlx-audio)..."
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

# 3. Granite Speech 4.1 2B NAR via mlx-audio
try:
    from mlx_audio.stt import load
    model = load("mlx-community/granite-speech-4.1-2b-nar-mlx")
    path = silence_wav()
    model.generate(path)
    os.unlink(path)
    print("   OK: Granite Speech 4.1 NAR (mlx-audio) loads and transcribes")
except Exception:
    print("   FAILED: Granite Speech 4.1 NAR — full traceback:")
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

# The spectral engine, loaded exactly as its sidecar loads it: the vendored tree
# imports, Silero VAD loads, and the PyTorch WeSpeaker embedder instantiates.
# No audio is diarized here — a client machine has no fixture, and the sidecar
# does the real pass at session start.
echo ""
echo "==> Verifying the spectral diarization engine (vendored, CPU)..."
PYANNOTE_METRICS_ENABLED=false HF_HUB_OFFLINE=1 "$PY" - <<'EOF'
import sys, traceback
try:
    sys.path.insert(0, "scripts/spectral/vendor")
    import diarize
    from diarize import diarize as run_diarize  # noqa: F401
    from silero_vad import load_silero_vad
    load_silero_vad()
    from diarize.torch_embedder import Speaker
    Speaker()
    print(f"   OK: spectral engine {diarize.__version__} (vendored) + Silero VAD "
          f"+ PyTorch WeSpeaker embedder all load")
except Exception:
    print("   FAILED: spectral engine — full traceback:")
    traceback.print_exc(file=sys.stdout)
    sys.exit(1)
EOF
[ $? -eq 0 ] || FAILED+=("spectral engine load test")

# DiCoW lives in .venv-dicow, so it must be verified with THAT interpreter.
# Import check + offline snapshot resolve only — a full 6 GB model load would
# add minutes here; scripts/dicow/dicow-service.py does the real load at session start.
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

# NeMo lives in .venv-nemo, so it too must be verified with THAT interpreter.
# Import + the two LOCAL checkpoints, not a full pass: the sidecar loads them at
# session start and a diarization run here would add minutes.
echo ""
echo "==> Verifying the NeMo diarization runtime (.venv-nemo)..."
if [ -x "$NEMO_PY" ]; then
  HF_HUB_OFFLINE=1 WANDB_MODE=disabled SENTRY_DSN="" \
  NEMO_PROJECT_DIR="$SCRIPT_DIR" "$NEMO_PY" - <<'EOF'
import os, pathlib, sys, traceback
try:
    import torch, omegaconf, soundfile, nemo  # noqa: F401
    from nemo.collections.asr.models import ClusteringDiarizer  # noqa: F401
    print(f"   OK: nemo_toolkit {nemo.__version__}, torch {torch.__version__}")
    if not nemo.__version__.startswith("3.0."):
        print("   FAILED: the sidecar's overrides were measured on nemo_toolkit 3.0.x")
        sys.exit(1)
    # THE FFMPEG TRAP, checked as an ABSENCE. Neither package may be here: NeMo
    # decodes through soundfile/librosa, and torchcodec's dylibs reach ffmpeg
    # only through a Homebrew-only LC_RPATH. If either ever appears, a future
    # NeMo release could route audio through it and fail on a client Mac only.
    import importlib.util
    intruders = [m for m in ("torchaudio", "torchcodec")
                 if importlib.util.find_spec(m) is not None]
    if intruders:
        print(f"   FAILED: {intruders} installed in .venv-nemo — the ffmpeg trap is "
              f"supposed to be structurally impossible in this venv")
        sys.exit(1)
    print("   OK: no torchaudio/torchcodec in .venv-nemo (the ffmpeg trap cannot apply)")
    root = pathlib.Path(os.environ["NEMO_PROJECT_DIR"])
    for name in ("titanet_large.nemo", "vad_multilingual_marblenet.nemo"):
        path = root / "models" / "nemo" / name
        if not path.exists():
            print(f"   FAILED: missing local checkpoint {path} — without it the "
                  f"sidecar would fall back to nothing (it never uses NGC names)")
            sys.exit(1)
        print(f"   OK: {name} ({path.stat().st_size / 1e6:.1f} MB, local)")
    cfg = omegaconf.OmegaConf.load(
        root / "scripts" / "nemo" / "vendor" / "diar_infer_general.yaml")
    print(f"   OK: vendored config loads (clustering keys: "
          f"{len(cfg.diarizer.clustering.parameters)})")
except Exception:
    print("   FAILED: NeMo runtime — full traceback:")
    traceback.print_exc(file=sys.stdout)
    sys.exit(1)
EOF
  [ $? -eq 0 ] || FAILED+=("NeMo runtime check (.venv-nemo)")
else
  echo "   MISSING: $NEMO_PY"
  FAILED+=("NeMo runtime check (.venv-nemo missing)")
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
