#!/bin/bash
# Build a fully self-contained, portable Meeting Transcriber.app:
#   • release Swift build + themed icon (via package-app.sh)
#   • TWO standalone python-build-standalone interpreters:
#       - Resources/python       — the main MLX stack (every model but DiCoW)
#       - Resources/.venv-dicow  — DiCoW only; it pins transformers 4.55 while
#                                  the MLX stack needs 5.x (hard conflict, so
#                                  they can never share one interpreter)
#   • scripts/ and models/ bundled as siblings under Contents/Resources
#   • ad-hoc signed and verified
#
# Output: dist/Meeting Transcriber.app  (+ dist/README-INSTALL.txt)
#
# Env:
#   MT_SKIP_MODELS=1   skip the ~16GB models copy (fast pipeline test); BOTH
#                      portable Pythons + pip installs + import gates still run.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
CACHE="$ROOT/.build-cache"
PBS_DIR="$CACHE/python-runtime"
PBS_DICOW_DIR="$CACHE/python-runtime-dicow"
APP_NAME="Meeting Transcriber"

# Fail the build if a bundled native lib cannot actually load on a client Mac.
# Used once per bundled interpreter.
#
# TWO kinds of unportable reference, and checking only the first is how a broken
# app shipped:
#   1. LOAD COMMANDS with an ABSOLUTE Homebrew/local path (`otool -L`).
#   2. An `@rpath/...` dependency that is NOT in the bundle. `otool -L` prints
#      `@rpath/libavcodec.62.dylib` and nothing about Homebrew — the absolute path
#      is hidden in an `LC_RPATH` search path — so a lib that is completely
#      unloadable off this Mac looked clean to the old check. That is exactly
#      torchcodec: 15 libs, each with one `LC_RPATH` of
#      `/opt/homebrew/opt/ffmpeg/lib`, and ZERO `libav*` in the bundle. pyannote
#      4.x routed path-based decoding through it, so diarization died at the first
#      chunk on any Mac without Homebrew ffmpeg while this guard reported OK.
#
# Check 2 tests "is the dependency present?", NOT "does an LC_RPATH mention
# Homebrew?" — deliberately, because the latter has a real false positive:
# `scipy/linalg/_fblas...so` carries a vestigial `gcc@13` LC_RPATH from its wheel
# build while having no `@rpath` dependency at all (it links Accelerate directly,
# and scipy ships its fortran runtime in `scipy/.dylibs/`). Measured over the
# bundle's 364 libs, check 2 flags torchcodec and nothing else.
#
# A lib's own LC_ID_DYLIB (`otool -D`) is skipped: maturin-built extensions name
# themselves `@rpath/<module>.abi3.so` and libsndfile's file name differs from its
# install name, so without this tokenizers, safetensors, cryptography and
# soundfile all report a missing dependency on themselves.
#
# No `head -25` either. That sampled whichever libs `find` happened to return
# first, which is a coin flip, not a check — torchcodec's were not in the sample.
check_relocatable() {
  local root="$1" label="$2" site bundled bad=0 rpaths self dep base missing
  site="$(find "$root/lib" -maxdepth 2 -type d -name site-packages | head -1)"
  # Every lib file name present anywhere under this interpreter, for check 2.
  bundled="$(find "$root" \( -name '*.so' -o -name '*.dylib' \) -exec basename {} \; 2>/dev/null | sort -u)"
  while IFS= read -r lib; do
    [[ -z "$lib" ]] && continue

    if otool -L "$lib" 2>/dev/null | grep -qE '/opt/homebrew|/usr/local'; then
      echo "    NON-RELOCATABLE (absolute load command): $lib" >&2
      otool -L "$lib" | grep -E '/opt/homebrew|/usr/local' >&2
      bad=1
    fi

    self="$(otool -D "$lib" 2>/dev/null | tail -1 | tr -d ' \t')"
    missing=""
    for dep in $(otool -L "$lib" 2>/dev/null | grep -o '@rpath/[^ ]*' | sort -u); do
      [[ "$dep" == "$self" ]] && continue
      base="$(basename "$dep")"
      grep -qx "$base" <<< "$bundled" || missing="$missing $base"
    done
    if [[ -n "$missing" ]]; then
      echo "    UNRESOLVABLE @rpath DEPENDENCY: $lib" >&2
      echo "      not in the bundle:$missing" >&2
      rpaths="$(otool -l "$lib" 2>/dev/null | grep -A2 LC_RPATH | grep -E '^ *path ' || true)"
      [[ -n "$rpaths" ]] && echo "      LC_RPATHs it would search:" >&2 && echo "$rpaths" >&2
      bad=1
    fi
  done <<< "$(find "$site" \( -name '*.so' -o -name '*.dylib' \) 2>/dev/null || true)"

  if [[ "$bad" == "1" ]]; then
    echo "ERROR: $label libs cannot load on a Mac without Homebrew — not portable." >&2
    echo "       Fix by bundling the dependency and rewriting the path, OR by not" >&2
    echo "       using the code path that needs it (what the sidecars do for ffmpeg:" >&2
    echo "       they decode with soundfile and never hand a path to torchcodec)." >&2
    exit 1
  fi
  echo "    Relocatability OK ($label — no absolute Homebrew/local load commands, no unresolvable @rpath deps)."
}

# torchcodec ships 15 libs that need Homebrew's ffmpeg, so in the bundle they are
# dead weight that `check_relocatable` (correctly) refuses. It is PRUNED rather
# than exempted by name — exempting the single offender the guard was told to
# ignore is not a check. Removing it is safe only because nothing reaches it: every
# sidecar decodes audio itself with soundfile, and pyannote imports torchcodec
# inside a try/except, degrading to `TORCHCODEC_AVAILABLE = False` — a path we
# never take because we always pass `{"waveform": ..., "sample_rate": ...}`.
# Verified by shadowing torchcodec with an unimportable stub: torch, torchaudio and
# pyannote all still import, and diarization + embedding both still run.
#
# The gate below is what keeps that invariant true. If a sidecar ever calls
# `torchaudio.load` (a torchcodec wrapper since TorchAudio 2.9), `torchcodec`, or
# `AudioDecoder` directly, the build fails HERE rather than at the first chunk of
# a client meeting.
#
# It reads the sidecars with Python's TOKENIZER, not grep. Comments and
# docstrings are stripped first, because the files that carry this fix also
# explain it at length — a plain `grep torchcodec` matches its own documentation
# and fails the build for describing the bug it prevents. (A grep is also brittle
# here for a dumber reason: this repo's path contains a space.)
assert_no_torchcodec_use() {
  "$ROOT/.venv/bin/python3" - "$ROOT/scripts" <<'PY' || exit 1
import io, pathlib, re, sys, tokenize

BANNED = re.compile(r"torchaudio\.(load|info)\b|\btorchcodec\b|\bAudioDecoder\b")

# Token types whose text is LITERAL CONTENT, not code. STRING alone is not enough:
# since Python 3.12 (PEP 701) an f-string is tokenized as FSTRING_START /
# FSTRING_MIDDLE / FSTRING_END, and the literal parts arrive as FSTRING_MIDDLE —
# so `f"...{n} AudioDecoder"` sailed past a STRING-only filter and failed the build
# on a log message. Named via getattr because these types only exist on 3.12+.
LITERAL_TOKENS = {tokenize.COMMENT, tokenize.STRING, tokenize.NL, tokenize.NEWLINE,
                  tokenize.INDENT, tokenize.DEDENT}
for _name in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END"):
    _type = getattr(tokenize, _name, None)
    if _type is not None:
        LITERAL_TOKENS.add(_type)

# sidecar-tests.py is the HARNESS, not a sidecar. It legitimately names these
# symbols — it contains the AST check that pins this very invariant — so scanning
# it means the guard fails the build for guarding.
SKIP_FILES = {"sidecar-tests.py"}

hits = []
for path in sorted(pathlib.Path(sys.argv[1]).rglob("*.py")):
    if "vendor" in path.parts or path.name in SKIP_FILES:
        continue
    src = path.read_text(encoding="utf-8", errors="replace")
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        continue
    # `torchaudio.load` is THREE tokens (NAME "." NAME), so no single token ever
    # matches the pattern — testing tokens one at a time silently passes
    # everything. Slide a small window of concatenated CODE tokens instead;
    # STRING (docstrings) and COMMENT are dropped, which is the whole point.
    window = []   # [(row, text), ...] most recent last
    for tok in toks:
        if tok.type in LITERAL_TOKENS:
            continue
        window.append((tok.start[0], tok.string))
        window = window[-6:]
        if BANNED.search("".join(t for _, t in window)):
            # The NEWEST token is the one that completed the match, so its row is
            # the offending line. window[0] is up to 5 tokens earlier and pointed
            # at the enclosing `def` instead of the call.
            row = window[-1][0]
            line = src.splitlines()[row - 1] if row <= len(src.splitlines()) else ""
            hits.append(f"{path}:{row}: {line.strip()[:90]}")
            break
if hits:
    print("ERROR: a sidecar reaches path-based audio decoding via torchcodec.", file=sys.stderr)
    print("       torchcodec needs Homebrew ffmpeg, which no client Mac has, and", file=sys.stderr)
    print("       is pruned from the bundle. Decode with soundfile instead.", file=sys.stderr)
    for h in hits:
        print("  " + h, file=sys.stderr)
    sys.exit(1)
print("    Audio-decode gate OK (no sidecar uses torchaudio.load/torchcodec in code).")
PY
}

prune_torchcodec() {
  local root="$1" label="$2" site n
  site="$(find "$root/lib" -maxdepth 2 -type d -name site-packages | head -1)"
  [[ -d "$site/torchcodec" ]] || { echo "    torchcodec absent ($label) — nothing to prune."; return; }
  n="$(find "$site/torchcodec" \( -name '*.so' -o -name '*.dylib' \) | wc -l | tr -d ' ')"
  rm -rf "$site/torchcodec" "$site"/torchcodec-*.dist-info
  echo "    Pruned torchcodec from $label ($n unloadable native libs; needs Homebrew ffmpeg)."
}

echo "==> Project root: $ROOT"
mkdir -p "$CACHE"

# ===========================================================================
# B1 — release build + copy-on-write clone into dist/
# ===========================================================================
echo ""
echo "==> [B1] Building + signing base bundle via package-app.sh..."
"$ROOT/package-app.sh"

# Wipe dist/ by RENAMING it first, then deleting the renamed copy.
#
# A plain `rm -rf "$ROOT/dist"` is racy and really does fail: Finder (or Spotlight)
# recreates `.DS_Store` inside the directory while rm is emptying it, so the final
# rmdir gets ENOTEMPTY — `rm: .../dist: Directory not empty` — and under
# `set -e` that aborts the whole build after B1 has already done its work.
# Observed exactly that, with `.DS_Store` as the sole survivor.
#
# `mv` is atomic, so dist/ is guaranteed gone before we recreate it, and any
# `.DS_Store` written afterwards lands in the renamed directory instead. Deleting
# that leftover is then best-effort on purpose: nothing references it, so a second
# race there must not fail the build.
if [[ -d "$ROOT/dist" ]]; then
  STALE_DIST="$ROOT/.dist-stale-$$"
  mv "$ROOT/dist" "$STALE_DIST"
  rm -rf "$STALE_DIST" 2>/dev/null || rm -rf "$STALE_DIST" 2>/dev/null || \
    echo "    (note: could not fully remove $STALE_DIST — safe to delete by hand)"
fi
mkdir -p "$ROOT/dist"
cp -Rc "$ROOT/$APP_NAME.app" "$ROOT/dist/"

APP="$ROOT/dist/$APP_NAME.app"
RES="$APP/Contents/Resources"
echo "==> Bundle cloned to: $APP"

# ===========================================================================
# B2 — portable python-build-standalone interpreter (cached)
# ===========================================================================
echo ""
echo "==> [B2] Provisioning portable Python..."

# --- resolve the newest matching PBS asset from the latest release ----------
echo "    Resolving python-build-standalone asset..."
RELEASE_JSON="$CACHE/pbs-latest.json"
curl -fsSL \
  -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest" \
  -o "$RELEASE_JSON"

# All aarch64 macOS install_only_stripped 3.12.x asset URLs in this release.
# NB: the '+' between version and build date is percent-encoded as %2B in the
# download URLs, so match either form.
ASSET_URLS="$(grep -oE 'https://[^"]*cpython-3\.12\.[0-9]+(\+|%2B)[0-9]+-aarch64-apple-darwin-install_only_stripped\.tar\.gz' "$RELEASE_JSON" | sort -u || true)"
if [[ -z "$ASSET_URLS" ]]; then
  echo "ERROR: no cpython-3.12.x aarch64 install_only_stripped asset found in latest PBS release" >&2
  exit 1
fi

# Prefer 3.12.13, else the newest 3.12.x (version-sorted).
ASSET_URL="$(echo "$ASSET_URLS" | grep -E 'cpython-3\.12\.13(\+|%2B)' | head -1 || true)"
if [[ -z "$ASSET_URL" ]]; then
  ASSET_URL="$(echo "$ASSET_URLS" | sort -t. -k3 -n | tail -1)"
fi
# Local filename: decode %2B → + so the cached tarball has a clean name.
ASSET_NAME="$(basename "$ASSET_URL" | sed 's/%2B/+/g')"
ASSET_PATH="$CACHE/$ASSET_NAME"
echo "    PBS asset: $ASSET_NAME"

# --- download (skip if cached) ---------------------------------------------
if [[ ! -f "$ASSET_PATH" ]]; then
  echo "    Downloading $ASSET_NAME ..."
  curl -fSL "$ASSET_URL" -o "$ASSET_PATH"
else
  echo "    Using cached tarball."
fi

# --- freeze the current .venv, sanitize ------------------------------------
FROZEN="$CACHE/requirements-frozen.txt"
echo "    Freezing .venv packages..."
"$ROOT/.venv/bin/pip" freeze --exclude-editable > "$FROZEN.raw"
# Strip pip itself and any local/VCS refs that can't install on another machine.
grep -vE '^pip==|file://|@ file://|git\+' "$FROZEN.raw" > "$FROZEN" || true
if grep -qE 'file://|git\+' "$FROZEN"; then
  echo "ERROR: local (file://) or VCS (git+) requirements remain after strip:" >&2
  grep -nE 'file://|git\+' "$FROZEN" >&2
  exit 1
fi
echo "    $(wc -l < "$FROZEN" | tr -d ' ') packages to install."

# --- provision (keyed on freeze sha) ---------------------------------------
NEW_SHA="$(shasum "$FROZEN" | awk '{print $1}')"
SHA_FILE="$CACHE/requirements.sha"
OLD_SHA="$(cat "$SHA_FILE" 2>/dev/null || echo "")"

if [[ "$NEW_SHA" == "$OLD_SHA" && -x "$PBS_DIR/bin/python3" ]]; then
  echo "    Requirements unchanged and runtime present — skipping reinstall."
else
  echo "    (Re)extracting interpreter + installing packages..."
  rm -rf "$PBS_DIR"
  mkdir -p "$PBS_DIR"
  tar -xzf "$ASSET_PATH" -C "$PBS_DIR" --strip-components=1
  # Install all 126 packages in ONE command → pip prints one enormous
  # "Installing collected packages: …" line via its rich console. On some Macs
  # the terminal fd is non-blocking and that write dies with
  # "OSError: [Errno 35] write could not complete without blocking", aborting
  # the build. Redirecting pip to a file removes the non-blocking terminal from
  # its output path entirely (file writes always block), and -q/--progress-bar
  # off keep the log small; on failure we surface the tail so nothing is hidden.
  PIP_LOG="$(dirname "$FROZEN")/build-pip-install.log"
  if ! "$PBS_DIR/bin/python3" -m pip install --no-compile --progress-bar off -q \
        -r "$FROZEN" >"$PIP_LOG" 2>&1; then
    echo "    pip install FAILED — last 30 lines of $PIP_LOG:"
    tail -30 "$PIP_LOG"
    exit 1
  fi
  echo "    Installed $(grep -c . "$FROZEN") packages."
  echo "$NEW_SHA" > "$SHA_FILE"
fi

# --- copy the provisioned runtime into the bundle --------------------------
echo "    Copying interpreter into bundle..."
rm -rf "$RES/python"
cp -Rc "$PBS_DIR" "$RES/python"

# --- import gate (mandatory) -----------------------------------------------
echo "    Import gate..."
"$RES/python/bin/python3" -c "import mlx.core as mx; print(mx.add(mx.array(1),mx.array(1))); import torch, pyannote.audio, mlx_whisper, silero_vad; print('IMPORTS OK')"

# --- vendored-helper gate (mandatory) --------------------------------------
# Some sidecars import their OWN scripts/<service>/vendor/<package>, NOT a pip
# install in .venv, so the ONLY way that package reaches the bundle is by riding
# scripts/ in [B3]. That makes each vendor directory a silent single point of
# failure — delete one and the build still succeeds, and the feature only fails
# on the client's machine mid-meeting. This is the DiCoW lesson (2026-07-27)
# applied one step earlier: fail HERE, loudly.
# Checked against the source tree because [B3] has not copied scripts/ yet, and
# through the BUNDLED interpreter because that is the one that will run it.
#
# The reason a package is vendored differs per service, but the failure mode is
# identical, so ONE table drives ONE loop:
#   • moss-asr / moss-diar — `moss_transcribe_diarize` is installable only from
#     git, and the freeze strip above deletes `git+` lines. TWO services since
#     2026-07-31 (one-service-per-role): `moss-asr` is the chunked-ASR role,
#     `moss-diar` the diarization role, and they can run AS TWO PROCESSES AT
#     ONCE, so a tree missing from either is a broken role.
#   • spectral — `diarize` (github.com/FoxNoseTech/diarize @ 4f25d27, Apache 2.0)
#     is vendored so its ONE modification can exist at all: `embeddings.py` is
#     re-pointed at this project's PyTorch WeSpeaker instead of upstream's
#     `wespeakerruntime`, which downloads its own ONNX weights at runtime from a
#     hardcoded URL — a network call the 100 %-offline requirement forbids. A pip
#     install could therefore never replace this copy, in the bundle or in dev.
#     Its import doubles as the proof that `pydantic` + `pydantic_core` reached
#     the bundled interpreter: the vendored `utils.py` is built on pydantic and
#     NOTHING ELSE in this app imports it, so a normal import gate would miss it.
#     (They are ordinary PyPI lines — `pydantic==…`, `pydantic_core==…` — so they
#     survive the `git+` strip; verified, not assumed. `pydantic_core` is a
#     compiled extension whose only `@rpath` entry is its own LC_ID_DYLIB, the
#     maturin case `check_relocatable` already skips.) The same import also
#     covers scikit-learn (the spectral clustering) and soundfile.
#
# This table is the ONLY place the set of vendoring services is named.
# Fields: service | package dir | import statement | remediation hint
echo "    Vendored-helper gate..."
MOSS_IMPORT="from moss_transcribe_diarize import build_transcription_messages, generate_transcription, parse_transcript"
for GATE in \
  "moss-asr|moss_transcribe_diarize|$MOSS_IMPORT|section 2c" \
  "moss-diar|moss_transcribe_diarize|$MOSS_IMPORT|section 2c" \
  "spectral|diarize|from diarize import diarize, DiarizeResult|section 3b"; do
  IFS='|' read -r SVC PKG IMPORT HINT <<< "$GATE"
  if ! PYTHONPATH="$ROOT/scripts/$SVC/vendor" "$RES/python/bin/python3" \
       -c "$IMPORT; print('VENDOR OK ($SVC → $PKG)')"; then
    echo "ERROR: scripts/$SVC/vendor/$PKG does not import in the bundled runtime." >&2
    echo "       That feature would be missing from the .app. Run" >&2
    echo "       ./download-best-models.sh ($HINT) and re-vendor the package," >&2
    echo "       then rebuild." >&2
    exit 1
  fi
done

# --- portability checks -----------------------------------------------------
# Prune BEFORE checking: torchcodec's libs are genuinely unloadable on a client
# Mac, so leaving them in and exempting them would mean the guard only ever sees
# the offender it was told to ignore.
echo "    Portability checks (load commands + @rpath resolution)..."
assert_no_torchcodec_use
prune_torchcodec "$RES/python" "main runtime"
check_relocatable "$RES/python" "main runtime"

# ===========================================================================
# B2b — SECOND portable interpreter for DiCoW (.venv-dicow)
#
# DiCoW's remote code only runs on transformers 4.55; the MLX stack in the main
# runtime needs 5.x. They cannot share an interpreter, so the app ships two.
# DicowService probes exactly Resources/.venv-dicow/bin/python3 (bundled
# projectDir == Resources), which is why the copy target below is that path.
# Reuses the PBS tarball B2 already downloaded — no second download.
# ===========================================================================
echo ""
echo "==> [B2b] Provisioning portable Python for DiCoW..."

if [[ ! -x "$ROOT/.venv-dicow/bin/pip" ]]; then
  echo "ERROR: .venv-dicow is missing — cannot bundle the DiCoW runtime." >&2
  echo "       Run ./download-best-models.sh (section 4d) first, then rebuild." >&2
  exit 1
fi

FROZEN_DICOW="$CACHE/requirements-frozen-dicow.txt"
echo "    Freezing .venv-dicow packages..."
"$ROOT/.venv-dicow/bin/pip" freeze --exclude-editable > "$FROZEN_DICOW.raw"
grep -vE '^pip==|file://|@ file://|git\+' "$FROZEN_DICOW.raw" > "$FROZEN_DICOW" || true
if grep -qE 'file://|git\+' "$FROZEN_DICOW"; then
  echo "ERROR: local (file://) or VCS (git+) requirements remain after strip:" >&2
  grep -nE 'file://|git\+' "$FROZEN_DICOW" >&2
  exit 1
fi
echo "    $(wc -l < "$FROZEN_DICOW" | tr -d ' ') packages to install."

NEW_SHA_DICOW="$(shasum "$FROZEN_DICOW" | awk '{print $1}')"
SHA_FILE_DICOW="$CACHE/requirements-dicow.sha"
OLD_SHA_DICOW="$(cat "$SHA_FILE_DICOW" 2>/dev/null || echo "")"

if [[ "$NEW_SHA_DICOW" == "$OLD_SHA_DICOW" && -x "$PBS_DICOW_DIR/bin/python3" ]]; then
  echo "    Requirements unchanged and runtime present — skipping reinstall."
else
  echo "    (Re)extracting interpreter + installing packages..."
  rm -rf "$PBS_DICOW_DIR"
  mkdir -p "$PBS_DICOW_DIR"
  tar -xzf "$ASSET_PATH" -C "$PBS_DICOW_DIR" --strip-components=1
  # Same non-blocking-terminal protection as B2: pip's output goes to a file,
  # never to a terminal fd that can raise OSError [Errno 35].
  PIP_LOG_DICOW="$CACHE/build-pip-install-dicow.log"
  if ! "$PBS_DICOW_DIR/bin/python3" -m pip install --no-compile --progress-bar off -q \
        -r "$FROZEN_DICOW" >"$PIP_LOG_DICOW" 2>&1; then
    echo "    pip install FAILED — last 30 lines of $PIP_LOG_DICOW:"
    tail -30 "$PIP_LOG_DICOW"
    exit 1
  fi
  echo "    Installed $(grep -c . "$FROZEN_DICOW") packages."
  echo "$NEW_SHA_DICOW" > "$SHA_FILE_DICOW"
fi

echo "    Copying DiCoW interpreter into bundle..."
rm -rf "$RES/.venv-dicow"
cp -Rc "$PBS_DICOW_DIR" "$RES/.venv-dicow"

# Import gate (mandatory). The version assert is the build-time guard for
# DiCoW's #1 runtime gotcha: on transformers 5.x generate() dies with
# AttributeError: '_get_initial_cache_position'. pandas is DiCoW's required
# but undocumented import.
echo "    Import gate (DiCoW)..."
"$RES/.venv-dicow/bin/python3" -c "import transformers, torch, pandas; assert transformers.__version__.startswith('4.55.'), 'expected transformers 4.55.x, got ' + transformers.__version__; print('DICOW IMPORTS OK', transformers.__version__)"

echo "    Portability checks (DiCoW)..."
prune_torchcodec "$RES/.venv-dicow" "DiCoW runtime"
check_relocatable "$RES/.venv-dicow" "DiCoW runtime"

# ===========================================================================
# B3 — scripts + models (siblings under Resources/)
# ===========================================================================
echo ""
echo "==> [B3] Bundling scripts + models..."
rm -rf "$RES/scripts"
mkdir -p "$RES/scripts"
# Copy scripts/, excluding caches, the icon tool, and any *-test.py. One folder
# per service (owner, 2026-07-29) — the find below already recreates directories,
# so each service's own vendor/ rides along with no change here. The -name tests
# match BASENAMES, so scripts/tools/make-icon-variants.py is still pruned.
( cd "$ROOT/scripts" && \
  find . \( -name '__pycache__' -o -name 'make-icon-variants.py' -o -name '*-test.py' \) -prune -o -type f -print \
  | while IFS= read -r f; do
      dest="$RES/scripts/${f#./}"
      mkdir -p "$(dirname "$dest")"
      cp -c "$f" "$dest"
    done )
echo "    scripts/ copied (mossformer2/vendor/mossformer2 preserved: $([[ -d "$RES/scripts/mossformer2/vendor/mossformer2" ]] && echo yes || echo NO))"
# Reported separately because it is the ONLY way the MOSS helper reaches the
# bundle — the freeze strip in [B2] deletes its `git+` line, so there is no pip
# fallback if this copy is ever lost. The [B2] gate already refuses to build
# without it; these lines are what make their presence visible in the build log.
# One per MOSS service (two since 2026-07-31) — each owns its own vendor tree.
echo "                     (moss-asr/vendor/moss_transcribe_diarize preserved: $([[ -f "$RES/scripts/moss-asr/vendor/moss_transcribe_diarize/inference_utils.py" ]] && echo yes || echo NO))"
echo "                     (moss-diar/vendor/moss_transcribe_diarize preserved: $([[ -f "$RES/scripts/moss-diar/vendor/moss_transcribe_diarize/inference_utils.py" ]] && echo yes || echo NO))"
# Same for spectral's vendored engine: it has NO pip fallback either — upstream's
# only installable form pulls `wespeakerruntime`, which downloads ONNX weights at
# runtime, so this copy is the engine.
echo "                     (spectral/vendor/diarize preserved: $([[ -f "$RES/scripts/spectral/vendor/diarize/torch_embedder.py" ]] && echo yes || echo NO))"

if [[ "${MT_SKIP_MODELS:-0}" == "1" ]]; then
  echo "    MT_SKIP_MODELS=1 — skipping 16GB models copy."
else
  echo "    Copying models/ (~16GB, copy-on-write)..."
  rm -rf "$RES/models"
  cp -Rc "$ROOT/models" "$RES/models"
  echo "    Pruning unused / regenerable model data from bundle..."
  rm -rf "$RES/models/hub/models--openai--whisper-large-v3"
  rm -rf "$RES/models/xet"
  rm -rf "$RES/models/modules"
  rm -rf "$RES/models/speaker-profiles"
fi

# ===========================================================================
# B5 — re-sign (B1 signature is now stale after adding python/scripts/models)
# ===========================================================================
echo ""
echo "==> [B5] Signing nested Mach-Os + bundle..."
find "$RES/python" \( -name '*.so' -o -name '*.dylib' \) -print0 \
  | xargs -0 -n 50 -P 8 codesign --force --sign - 2>/dev/null || true
codesign --force --sign - "$RES/python/bin/"python3* 2>/dev/null || true
# Same treatment for the DiCoW interpreter — unsigned nested Mach-Os get the
# app killed at launch on a client Mac.
find "$RES/.venv-dicow" \( -name '*.so' -o -name '*.dylib' \) -print0 \
  | xargs -0 -n 50 -P 8 codesign --force --sign - 2>/dev/null || true
codesign --force --sign - "$RES/.venv-dicow/bin/"python3* 2>/dev/null || true
codesign --force --deep --sign - "$APP"

echo "    Verifying signature..."
codesign --verify --deep --strict "$APP"
echo "    codesign --verify --deep --strict: PASS"

echo "    spctl assessment (a 'rejected (unnotarized)' result is EXPECTED here, not a failure):"
spctl -a -vv "$APP" || true

# ===========================================================================
# B6 — install notes + manifest
# ===========================================================================
echo ""
echo "==> [B6] Writing README-INSTALL.txt + manifest..."
cat > "$ROOT/dist/README-INSTALL.txt" <<'EOF'
Meeting Transcriber — Install Notes
===================================

Requirements
------------
• Apple Silicon Mac (M1/M2/M3/M4 — arm64). This build will NOT run on Intel Macs.
• macOS 14 or newer.

First launch (Gatekeeper)
-------------------------
This app is ad-hoc signed and not notarized, so macOS will block the first
open. Do ONE of the following:

  1) Terminal (recommended):
         xattr -dr com.apple.quarantine "/Applications/Meeting Transcriber.app"
     (adjust the path to wherever you put the app), then double-click it.

  OR

  2) Right-click (Control-click) the app in Finder → Open → Open.

Where your data lives
---------------------
The app itself is read-only. Everything you create is written to:

    ~/Library/Application Support/Meeting Transcriber/

  • recordings/          your recorded audio + working files
  • logs/                sidecar + overlap-repair decision logs
  • speaker-profiles/    learned speaker voice profiles
  • hf-home/             model runtime cache

To reset the app to a clean state, delete that folder.

100% offline — no audio or transcript ever leaves your machine.
EOF
echo "    dist/README-INSTALL.txt written."

echo ""
echo "==> Manifest:"
for d in python .venv-dicow models scripts; do
  if [[ -d "$RES/$d" ]]; then
    printf "    %-10s %s\n" "$d" "$(du -sh "$RES/$d" | awk '{print $1}')"
  else
    printf "    %-10s %s\n" "$d" "(absent)"
  fi
done
printf "    %-10s %s\n" "TOTAL app" "$(du -sh "$APP" | awk '{print $1}')"
echo ""
echo "    codesign verify: $(codesign --verify --deep --strict "$APP" 2>&1 && echo PASS || echo FAIL)"
echo ""
echo "==> Done. Portable app at: $APP"
