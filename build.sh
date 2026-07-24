#!/bin/bash
# Build a fully self-contained, portable Meeting Transcriber.app:
#   • release Swift build + themed icon (via package-app.sh)
#   • a standalone python-build-standalone interpreter with every pip package
#   • scripts/ and models/ bundled as siblings under Contents/Resources
#   • ad-hoc signed and verified
#
# Output: dist/Meeting Transcriber.app  (+ dist/README-INSTALL.txt)
#
# Env:
#   MT_SKIP_MODELS=1   skip the ~16GB models copy (fast pipeline test); the
#                      portable Python + pip reinstall + import gate still run.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
CACHE="$ROOT/.build-cache"
PBS_DIR="$CACHE/python-runtime"
APP_NAME="Meeting Transcriber"

echo "==> Project root: $ROOT"
mkdir -p "$CACHE"

# ===========================================================================
# B1 — release build + copy-on-write clone into dist/
# ===========================================================================
echo ""
echo "==> [B1] Building + signing base bundle via package-app.sh..."
"$ROOT/package-app.sh"

rm -rf "$ROOT/dist"
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

# --- relocatability spot-check ---------------------------------------------
echo "    Relocatability spot-check (otool -L)..."
SITE="$(find "$RES/python/lib" -maxdepth 2 -type d -name site-packages | head -1)"
SPOT_LIBS="$(find "$SITE" \( -name '*.so' -o -name '*.dylib' \) 2>/dev/null | head -25 || true)"
BAD=0
while IFS= read -r lib; do
  [[ -z "$lib" ]] && continue
  if otool -L "$lib" 2>/dev/null | grep -qE '/opt/homebrew|/usr/local'; then
    echo "    NON-RELOCATABLE: $lib" >&2
    otool -L "$lib" | grep -E '/opt/homebrew|/usr/local' >&2
    BAD=1
  fi
done <<< "$SPOT_LIBS"
if [[ "$BAD" == "1" ]]; then
  echo "ERROR: bundled libs reference Homebrew/local paths — not portable." >&2
  exit 1
fi
echo "    Relocatability OK (no /opt/homebrew or /usr/local load commands)."

# ===========================================================================
# B3 — scripts + models (siblings under Resources/)
# ===========================================================================
echo ""
echo "==> [B3] Bundling scripts + models..."
rm -rf "$RES/scripts"
mkdir -p "$RES/scripts"
# Copy scripts/, excluding caches, the icon tool, and any *-test.py; keep vendor/.
( cd "$ROOT/scripts" && \
  find . \( -name '__pycache__' -o -name 'make-icon-variants.py' -o -name '*-test.py' \) -prune -o -type f -print \
  | while IFS= read -r f; do
      dest="$RES/scripts/${f#./}"
      mkdir -p "$(dirname "$dest")"
      cp -c "$f" "$dest"
    done )
echo "    scripts/ copied (vendor/mossformer2 preserved: $([[ -d "$RES/scripts/vendor/mossformer2" ]] && echo yes || echo NO))"

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
for d in python models scripts; do
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
