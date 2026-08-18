#!/bin/bash
# Collect everything that can differ BETWEEN MACHINES into one zip.
#
# WHY THIS EXISTS. The owner reports a configuration that behaves well on the
# development Mac and badly on another, and the natural first guess is the audio.
# It usually is not: a recording is one of five things that differ, and it is the
# only one already saved. The other four leave no trace unless something goes and
# reads them, which is what this does.
#
#   1. SETTINGS       per machine, never in git — engine, speaker count, interval,
#                     label source, VAD, ATND addresses. The likeliest cause by far.
#   2. APP BUILD      the other Mac may simply be running an older .app; every fix
#                     of the last week is invisible until it is rebuilt.
#   3. MODELS         present, complete, and where the app actually looks.
#   4. AUDIO DEVICES  a different mic, or an Aggregate Device that is not there.
#   5. THE RECORDING  already saved by the app; included only on request, below.
#
# ⚠ AUDIO IS OPT-IN, AND THAT IS A REQUIREMENT RATHER THAN A COURTESY. Recordings
# are real client speech and this project's premise is that they never leave the
# machine — `recordings/` is gitignored for the same reason. Default: no audio.
# Pass --with-audio to include the newest recording, or --with-audio=all for every
# one. The script prints exactly what it packed, so nothing leaves unannounced.
#
# Usage:
#   ./scripts/tools/collect-diagnostics.sh                 settings + logs only
#   ./scripts/tools/collect-diagnostics.sh --with-audio    + the newest recording
set -uo pipefail

DATA="$HOME/Library/Application Support/Meeting Transcriber"
OUT="$HOME/Desktop/mt-diagnostics-$(hostname -s)-$(date +%Y%m%d-%H%M%S)"
AUDIO="none"
for arg in "$@"; do
  case "$arg" in
    --with-audio)     AUDIO="newest" ;;
    --with-audio=all) AUDIO="all" ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$OUT"
echo "==> Collecting into $OUT"

# --- 1. settings. Every domain the app has ever used, because which one is live
#        depends on how the app was built, and a diagnostic that guesses wrong
#        reports "no settings" for a machine that has plenty.
{
  for d in MeetingTranscriber com.aggia.meeting-transcriber com.meetingtranscriber.app \
           local.meeting.transcriber.mac com.example.MeetingTranscriberMac; do
    echo "===== domain: $d ====="
    defaults read "$d" 2>/dev/null || echo "(absent)"
    echo
  done
} > "$OUT/settings.txt"
echo "    settings.txt          $(grep -c '=' "$OUT/settings.txt") key(s) across all domains"

# --- 2. what build is actually installed, and what this checkout is.
{
  echo "host:        $(hostname -s)"
  echo "macOS:       $(sw_vers -productVersion) ($(uname -m))"
  echo "collected:   $(date)"
  echo
  for app in "/Applications/Meeting Transcriber.app" "$PWD/dist/Meeting Transcriber.app"; do
    echo "===== $app ====="
    if [[ -d "$app" ]]; then
      bin="$app/Contents/MacOS/MeetingTranscriber"
      echo "  binary mtime: $(stat -f '%Sm' "$bin" 2>/dev/null || echo 'MISSING')"
      echo "  binary bytes: $(stat -f '%z' "$bin" 2>/dev/null || echo '-')"
      sig=$(codesign --verify --deep --strict "$app" 2>&1)
      echo "  signature:    ${sig:-PASS}"
      # ⚠ EVERY PROBE MUST BE LONGER THAN 15 BYTES. Swift stores string literals
      # of 15 UTF-8 bytes or fewer INLINE as immediates, so they never reach the
      # string table and `strings` cannot see them. CLAUDE.md records this trap,
      # and the first version of this script walked straight into it: it probed
      # "Live only" (9 bytes), got 0, and reported a current build as missing the
      # feature it had. Probe the DESCRIPTION, never the button label.
      for probe in "Direction labels the meeting as it happens" \
                   "window(s) released, transcript untouched" \
                   "Speaker labelling has failed"; do
        printf '  has %-45s %s\n' "\"$probe\":" \
          "$(strings "$bin" 2>/dev/null | grep -cF "$probe")"
      done
    else
      echo "  (not installed here)"
    fi
    echo
  done
  echo "===== this checkout ====="
  git -C "$PWD" log --oneline -3 2>/dev/null || echo "(not a git checkout)"
  echo "  dirty files: $(git -C "$PWD" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
} > "$OUT/build.txt"
echo "    build.txt             app + checkout identity"

# --- 3. models, where the app looks for them.
{
  for hub in "$DATA/hf-home/hub" "/Applications/Meeting Transcriber.app/Contents/Resources/models/hub" \
             "$PWD/models/hub"; do
    echo "===== $hub ====="
    if [[ -d "$hub" ]]; then
      du -sh "$hub" 2>/dev/null
      ls "$hub" | grep '^models--' | sed 's/^/  /'
    else
      echo "  (absent)"
    fi
    echo
  done
} > "$OUT/models.txt"
echo "    models.txt            model inventory in all three locations"

# --- 4. audio devices. An Aggregate Device that exists on one Mac and not the
#        other silently changes which channels the app can even see.
system_profiler SPAudioDataType 2>/dev/null > "$OUT/audio-devices.txt"
echo "    audio-devices.txt     $(grep -c ':$' "$OUT/audio-devices.txt" 2>/dev/null) device entries"

# --- 5. logs. The evidence for everything else.
if [[ -d "$DATA/logs" ]]; then
  mkdir -p "$OUT/logs"
  cp "$DATA/logs/"*.log "$OUT/logs/" 2>/dev/null
  echo "    logs/                 $(ls "$OUT/logs" 2>/dev/null | wc -l | tr -d ' ') file(s)"
else
  echo "    logs/                 (none — the app has not run on this Mac)"
fi

# --- 6. the recording, only if asked.
REC="$DATA/recordings"
case "$AUDIO" in
  none)
    n=$(ls "$REC"/*.wav 2>/dev/null | wc -l | tr -d ' ')
    echo "    recordings/           SKIPPED — $n on this Mac. Re-run with --with-audio to include one."
    ;;
  newest|all)
    mkdir -p "$OUT/recordings"
    if [[ "$AUDIO" == "newest" ]]; then
      newest=$(ls -t "$REC"/*.wav 2>/dev/null | head -1)
      [[ -n "$newest" ]] && cp "$newest" "$OUT/recordings/"
    else
      cp "$REC"/*.wav "$OUT/recordings/" 2>/dev/null
    fi
    echo "    recordings/           $(ls "$OUT/recordings" 2>/dev/null | wc -l | tr -d ' ') file(s), $(du -sh "$OUT/recordings" 2>/dev/null | cut -f1)"
    ;;
esac

ZIP="$OUT.zip"
(cd "$(dirname "$OUT")" && zip -qr "$(basename "$ZIP")" "$(basename "$OUT")") && rm -rf "$OUT"
echo ""
echo "==> $ZIP"
echo "    $(du -sh "$ZIP" | cut -f1)"
echo ""
if [[ "$AUDIO" == "none" ]]; then
  echo "    No audio included. This zip holds settings, logs and machine identity only."
else
  echo "    ⚠ This zip CONTAINS MEETING AUDIO. Send it only where that is acceptable."
fi
