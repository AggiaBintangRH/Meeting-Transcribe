#!/bin/bash
# Package the SwiftPM MeetingTranscriber into a real, ad-hoc-signed macOS .app
# bundle with a system-themed app icon (light/dark/tinted). Runs from anywhere
# (Finder double-click included) — all paths are derived from the script dir.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PKG="$ROOT/MeetingTranscriber"
SRC="$PKG/Sources/MeetingTranscriber"
ICONSET="$SRC/Assets.xcassets/AppIcon.appiconset"
VENV_PY="$ROOT/.venv/bin/python3"
APP_NAME="Meeting Transcriber"
APP="$ROOT/$APP_NAME.app"

echo "==> Project root: $ROOT"

# --- Stage 1: ensure themed icon variants exist ---------------------------
if [[ ! -f "$ICONSET/1024-dark.png" || ! -f "$ICONSET/1024-tinted.png" ]]; then
  echo "==> Icon variants missing; generating..."
  if [[ ! -x "$VENV_PY" ]]; then
    echo "ERROR: project venv python not found at $VENV_PY" >&2
    echo "       Create the .venv and install Pillow before packaging." >&2
    exit 1
  fi
  "$VENV_PY" "$ROOT/scripts/tools/make-icon-variants.py"
fi

# --- Stage 2: release build ------------------------------------------------
echo "==> Building release..."
swift build -c release --package-path "$PKG"
BIN="$(swift build -c release --package-path "$PKG" --show-bin-path)"
echo "==> Release bin path: $BIN"

if [[ ! -f "$BIN/MeetingTranscriber" ]]; then
  echo "ERROR: built executable not found at $BIN/MeetingTranscriber" >&2
  exit 1
fi

# --- Stage 3: assemble bundle in a temp staging dir ------------------------
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
STAGE_APP="$STAGE/$APP_NAME.app"
echo "==> Staging at: $STAGE_APP"

mkdir -p "$STAGE_APP/Contents/MacOS"
mkdir -p "$STAGE_APP/Contents/Resources"

cp "$BIN/MeetingTranscriber" "$STAGE_APP/Contents/MacOS/MeetingTranscriber"

# Bundle.module resources — REQUIRED so the app finds its assets at runtime.
RES_BUNDLE="MeetingTranscriber_MeetingTranscriber.bundle"
if [[ ! -d "$BIN/$RES_BUNDLE" ]]; then
  echo "ERROR: resource bundle not found at $BIN/$RES_BUNDLE" >&2
  exit 1
fi
cp -R "$BIN/$RES_BUNDLE" "$STAGE_APP/Contents/Resources/$RES_BUNDLE"

printf 'APPL????' > "$STAGE_APP/Contents/PkgInfo"

# --- Stage 4: compile the asset catalog (themed app icon) ------------------
PARTIAL="$STAGE/assetcatalog-partial.plist"
echo "==> Compiling asset catalog with actool..."
/usr/bin/actool "$SRC/Assets.xcassets" \
  --compile "$STAGE_APP/Contents/Resources" \
  --platform macosx \
  --minimum-deployment-target 14.0 \
  --target-device mac \
  --app-icon AppIcon \
  --accent-color AccentColor \
  --output-partial-info-plist "$PARTIAL" \
  --output-format human-readable-text \
  --errors --warnings --notices

echo "==> assetutil --info on compiled Assets.car:"
/usr/bin/assetutil --info "$STAGE_APP/Contents/Resources/Assets.car" || true

# --- Stage 5: Info.plist (template + merged actool partial) ----------------
cp "$PKG/Support/Info.plist" "$STAGE_APP/Contents/Info.plist"
"$VENV_PY" - "$STAGE_APP/Contents/Info.plist" "$PARTIAL" <<'PY'
import plistlib, sys
info_path, partial_path = sys.argv[1], sys.argv[2]
with open(info_path, 'rb') as f:
    info = plistlib.load(f)
with open(partial_path, 'rb') as f:
    partial = plistlib.load(f)
info.update(partial)
with open(info_path, 'wb') as f:
    plistlib.dump(info, f)
if 'CFBundleIconName' not in info:
    print("ERROR: CFBundleIconName missing after merging actool partial plist", file=sys.stderr)
    sys.exit(1)
print("merged Info.plist keys from actool partial:", sorted(partial.keys()))
print("CFBundleIconName =", info['CFBundleIconName'])
PY

# --- Finalize: move into place, ad-hoc sign --------------------------------
echo "==> Installing bundle to: $APP"
rm -rf "$APP"
mv "$STAGE_APP" "$APP"

echo "==> Ad-hoc signing..."
codesign --force --deep --sign - "$APP"

touch "$APP"
echo ""
echo "==> Done: $APP"
echo "    If the Dock/Finder shows a stale icon, run: killall Dock Finder"
