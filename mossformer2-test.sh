#!/bin/bash
# MossFormer2 (librimix-2spk) separation quality test:
#   ./mossformer2-test.sh <recording.wav>
# Separates the file into 2 tracks then transcribes each with Whisper
# (both steps share the project's single .venv) so you can judge both by
# ear (open the wav files) and by ASR text (printed below).
set -euo pipefail
cd "$(dirname "$0")"

if [ $# -lt 1 ] || [ ! -f "$1" ]; then
    echo "usage: ./mossformer2-test.sh <recording.wav>"
    echo "e.g.:  ./mossformer2-test.sh recordings/$(ls -t recordings 2>/dev/null | head -1)"
    exit 1
fi
AUDIO="$1"
OUT_DIR="$(mktemp -d)/mossformer2-out"

echo "separating (MossFormer2 librimix-2spk, 8kHz) ..."
.venv/bin/python3 scripts/mossformer2-test.py "$AUDIO" "$OUT_DIR"

echo ""
echo "transcribing each separated track (Whisper large-v3 MLX) ..."
for wav in "$OUT_DIR"/speaker*.wav; do
    name="$(basename "$wav" .wav)"
    .venv/bin/mlx_whisper "$wav" --model mlx-community/whisper-large-v3-mlx \
        --output-format txt --output-dir "$OUT_DIR" --verbose False >/dev/null 2>&1
    echo ""
    echo "── $name ──"
    cat "$OUT_DIR/$name.txt" 2>/dev/null || echo "(no transcript)"
done

echo ""
echo "separated audio + transcripts saved in: $OUT_DIR"
echo "(open the .wav files in Finder/QuickTime to judge by ear)"
